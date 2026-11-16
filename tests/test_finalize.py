"""All-training-match finalization keeps OOF selection separate from deployment."""

import json
import weakref

import numpy as np
import pytest
import torch

from experiments import finalize, select
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half


def _half(match, period):
    n = 8
    state = np.array([0, 0, 1, 1, 2, 2, 0, 0], np.int8)
    live = (state != 2).astype(np.int8)
    players = np.zeros((n, 2, 1, 2), np.float32)
    players[:, 0, 0] = [20, 30]
    players[:, 1, 0] = [80, 30]
    frames = FrameTable(
        np.arange(period * 100, period * 100 + n), players,
        np.ones((n, 2, 1), bool),
        np.column_stack((30 + np.arange(n), np.full(n, 30))),
        np.ones(n, bool), np.ones(n, bool), match_id=match, period=period,
    )
    targets = TrainingTargets(
        live, np.ones(n, bool), (state == 0).astype(np.int8),
        live.astype(bool), state, np.ones(n, bool),
    )
    return frames, targets


def _data(tmp_path):
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    for match in "abcd":
        for period in (1, 2):
            save_half(data_dir / f"{match}_half{period}.npz", *_half(match, period))
    return data_dir


def test_finalization_uses_every_match_for_oof_then_refits_and_exports(tmp_path, monkeypatch):
    data_dir = _data(tmp_path)
    output_dir = tmp_path / "deployment"
    original_score = select._score_records

    def prefer_neural(records, params):
        _score, by_match = original_score(records, params)
        return (1.0 if params["neural_weight"] == 1 else 0.0), by_match

    monkeypatch.setattr(select, "_score_records", prefer_neural)
    requested = {"mode": "temporal", "operations": ["time_shift"],
                 "temporal_shift_max_seconds": 0.08, "warmup_steps": 0}
    args = {"matches": list("abcd"), "n_trials": 2, "rungs": (1,),
            "decoder_trials": 3, "center_size": 4, "cache_bytes": 10_000_000,
            "augmentation_config": requested}
    report = finalize.run_finalization(data_dir, output_dir, **args)
    study_dir = tmp_path / "deployment.study" / finalize.FINAL_ID
    split = json.loads((study_dir / "splits.json").read_text())
    fold, = split["outer_folds"]
    assert split["matches"] == list("abcd")
    assert split["scope"] == "all_training_matches"
    assert split["evaluation_performed"] is False
    assert fold["outer_test"] == [finalize.FINAL_ID]
    assert fold["outer_train"] == list("abcd")
    assert sorted(match for inner in fold["inner"] for match in inner["validation"]) == list("abcd")
    assert all(not set(inner["train"]) & set(inner["validation"]) for inner in fold["inner"])
    assert not (study_dir / "outer_report.json").exists()
    frozen = json.loads((study_dir / "frozen_recipe.json").read_text())
    config = finalize.tune._augmentation_dict(requested)
    plan = json.loads((study_dir / "finalization_plan.json").read_text())
    assert plan["augmentation_config"] == config
    assert frozen["selected_model_recipe"]["training_args"]["augmentation_config"] == config
    assert frozen["outer_train"] == list("abcd")
    assert frozen["selected_pooling"] in {"mean", "ball_attention"}
    checkpoint = torch.load(study_dir / "selected.pt", weights_only=True)
    assert checkpoint["training_config"]["train_matches"] == list("abcd")
    assert checkpoint["training_config"]["validation_matches"] == []
    assert checkpoint["training_config"]["augmentation_config"] == config
    assert report["evaluation_performed"] is False
    assert report["matches"] == list("abcd")
    assert output_dir.is_dir()
    assert (output_dir / "deployment_manifest.json").is_file()
    again = finalize.run_finalization(data_dir, output_dir, **args)
    assert again["bundle_reused"] is True
    with pytest.raises(ValueError, match="plan"):
        finalize.run_finalization(data_dir, output_dir, **{**args, "n_trials": 3})
    with pytest.raises(ValueError, match="plan|fingerprint"):
        finalize.run_finalization(data_dir, output_dir, **{
            **args, "augmentation_config": {**requested, "temporal_shift_max_seconds": 0.1}})
    with pytest.raises(ValueError, match="plan|fingerprint"):
        changed = data_dir / "a_half1.npz"
        changed.write_bytes(changed.read_bytes() + b"drift")
        finalize.run_finalization(data_dir, output_dir, **args)


def test_finalization_rejects_reserved_match_and_incomplete_baselines(tmp_path):
    data_dir = _data(tmp_path)
    with pytest.raises(ValueError, match="distinct real"):
        finalize.run_finalization(
            data_dir, tmp_path / "deployment", matches=["a", "b", finalize.FINAL_ID])
    with pytest.raises(ValueError, match="both pooling baselines"):
        finalize.run_finalization(
            data_dir, tmp_path / "deployment", matches=list("abcd"), n_trials=1)


def test_reference_winner_exports_without_neural_refit(tmp_path, monkeypatch):
    data_dir = _data(tmp_path)
    original_score = select._score_records

    def prefer_reference(records, params):
        _score, by_match = original_score(records, params)
        return (1.0 if params["neural_weight"] == 0 else 0.0), by_match

    monkeypatch.setattr(select, "_score_records", prefer_reference)
    output_dir = tmp_path / "reference_bundle"
    report = finalize.run_finalization(
        data_dir, output_dir, matches=list("abcd"), n_trials=2, rungs=(1,),
        decoder_trials=3, center_size=4, cache_bytes=10_000_000,
    )
    assert report["selected_pooling"] == "reference"
    assert report["refit_progress"] == {
        "predictor": "reference", "steps": 0, "train_matches": list("abcd")}
    assert not (output_dir / "selected.pt").exists()
    assert not (tmp_path / "reference_bundle.study" / finalize.FINAL_ID / "selected.pt").exists()
    manifest = json.loads((output_dir / "deployment_manifest.json").read_text())
    assert manifest["predictor"] == "reference"
    assert manifest["evaluation_performed"] is False


def test_finalization_lock_spans_tuning_selection_and_export(tmp_path, monkeypatch):
    data_dir = _data(tmp_path)
    output_dir = tmp_path / "deployment"
    study_root = tmp_path / "deployment.study"
    cache_ref = None

    def assert_locked():
        with pytest.raises(RuntimeError, match="already has a writer"), finalize._locked(study_root):
            pass

    def fake_tune(*args, cache, **kwargs):
        nonlocal cache_ref
        assert_locked()
        cache_ref = weakref.ref(cache)
        return {"outer_train": list("abcd"),
                "best_by_pooling": {pooling: {} for pooling in finalize.POOLINGS}}

    def fake_select(*args, **kwargs):
        assert_locked()
        assert cache_ref() is None
        return {"outer_report": None, "refit_progress": {"steps": 0},
                "frozen_recipe": {"outer_train": list("abcd"),
                                  "outer_test": finalize.FINAL_ID,
                                  "selected_pooling": "reference",
                                  "selection_score": 0.5}}

    def fake_export(*args):
        assert_locked()
        return {"bundle_dir": output_dir,
                "manifest_path": output_dir / "deployment_manifest.json", "reused": False}

    monkeypatch.setattr(finalize.tune, "_run_outer", fake_tune)
    monkeypatch.setattr(finalize.select, "select_and_evaluate", fake_select)
    monkeypatch.setattr("possession.deployment.export_bundle", fake_export)
    report = finalize.run_finalization(
        data_dir, output_dir, matches=list("abcd"), n_trials=2,
        rungs=(1,), decoder_trials=3, center_size=4, cache_bytes=10_000_000)
    assert report["selected_pooling"] == "reference"
    with finalize._locked(study_root):
        pass


def test_finalization_rejects_active_lock_before_plan_or_tuning(tmp_path, monkeypatch):
    data_dir = _data(tmp_path)
    output_dir = tmp_path / "deployment"
    study_root = tmp_path / "deployment.study"
    monkeypatch.setattr(finalize.tune, "_run_outer",
                        lambda *args, **kwargs: pytest.fail("tuning started under active lock"))
    with finalize._locked(study_root), pytest.raises(RuntimeError, match="already has a writer"):
        finalize.run_finalization(
            data_dir, output_dir, matches=list("abcd"), n_trials=2,
            rungs=(1,), decoder_trials=3, center_size=4, cache_bytes=10_000_000)
    assert not (study_root / finalize.FINAL_ID / "finalization_plan.json").exists()


def test_finalization_rejects_unrelated_output_before_plan_or_tuning(tmp_path, monkeypatch):
    data_dir = _data(tmp_path)
    output_dir = tmp_path / "deployment"
    output_dir.mkdir()
    (output_dir / "unrelated.txt").write_text("keep me")
    monkeypatch.setattr(finalize.tune, "_run_outer",
                        lambda *args, **kwargs: pytest.fail("tuning started for unrelated output"))
    with pytest.raises(ValueError, match="existing bundle"):
        finalize.run_finalization(
            data_dir, output_dir, matches=list("abcd"), n_trials=2,
            rungs=(1,), decoder_trials=3, center_size=4, cache_bytes=10_000_000)
    assert not (tmp_path / "deployment.study" / finalize.FINAL_ID /
                "finalization_plan.json").exists()
    (output_dir / "unrelated.txt").unlink()
    (output_dir / "deployment_manifest.json").write_text(
        json.dumps({"predictor": "reference", "selection_fingerprint": "other"}))
    (output_dir / "frozen_recipe.json").write_text(json.dumps({
        "hpo_fingerprint_sha256": "other", "outer_train": list("abcd"),
        "outer_test": finalize.FINAL_ID, "selection_fingerprint": "other"}))
    (output_dir / "refit_progress.json").write_text("{}")
    with pytest.raises(ValueError, match="different finalization"):
        finalize.run_finalization(
            data_dir, output_dir, matches=list("abcd"), n_trials=2,
            rungs=(1,), decoder_trials=3, center_size=4, cache_bytes=10_000_000)
    assert not (tmp_path / "deployment.study" / finalize.FINAL_ID /
                "finalization_plan.json").exists()
