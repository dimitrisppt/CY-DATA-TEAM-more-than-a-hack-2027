"""Decoder selection consumes immutable inner OOF evidence before outer access."""

import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments import select, tune
from experiments.cross_validation import nested_splits
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half, sha256_file, write_json


def _half(match, period, n=8):
    state = np.resize(np.array([0, 0, 1, 1, 2, 2, 0, 0], np.int8), n)
    live = (state != 2).astype(np.int8)
    players = np.zeros((n, 2, 1, 2), np.float32)
    players[:, 0, 0] = [20, 30]
    players[:, 1, 0] = [80, 30]
    frames = FrameTable(np.arange(period * 100, period * 100 + n), players,
                        np.ones((n, 2, 1), bool), np.column_stack((30+np.arange(n),
                        np.full(n, 30))), np.ones(n, bool), np.ones(n, bool),
                        match_id=match, period=period)
    targets = TrainingTargets(live, np.ones(n, bool), (state == 0).astype(np.int8),
                              live.astype(bool), state, np.ones(n, bool))
    return frames, targets


def _artifacts(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    for match in "abcd":
        for period in (1, 2):
            frames, targets = _half(match, period)
            save_half(data / f"{match}_half{period}.npz", frames, targets)
    fold = nested_splits(list("abcd"))[0]
    split = {"matches": list("abcd"), "seed": 0, "inner_folds": 3,
             "outer_folds": nested_splits(list("abcd"))}
    outer_dir = tmp_path / "study" / "a"
    outer_dir.mkdir(parents=True)
    study_fp = tune._fingerprint(data, fold, split, seed=0, rungs=(2,), device="cpu",
                                 poolings=("mean", "ball_attention"), pruning=False,
                                 center_size=4, cache_bytes=10_000_000)
    write_json(outer_dir / "fingerprint.json", study_fp)
    write_json(outer_dir / "splits.json", split)
    best = {}
    for number, pooling in enumerate(("mean", "ball_attention")):
        artifact_dir = outer_dir / "trials" / f"trial_{number}"
        artifact_dir.mkdir(parents=True)
        hashes, checkpoints, oof = {}, [], []
        for index, inner in enumerate(fold["inner"]):
            name = f"fold_{index}.pt"
            torch.save({"training_config": {"train_matches": inner["train"],
                                            "validation_matches": inner["validation"],
                                            "outer_test": "a", "trial_number": number,
                                            "inner_fold": index},
                        "model_kwargs": {"width": 8, "pooling": pooling,
                                         "dilations": [1], "player_dim": 8,
                                         "global_dim": 9}}, artifact_dir / name)
            hashes[name] = sha256_file(artifact_dir / name)
            checkpoints.append({"inner_fold": index, "file": name,
                                "sha256": hashes[name],
                                "train_matches": inner["train"],
                                "validation_matches": inner["validation"]})
            for match in inner["validation"]:
                for period in (1, 2):
                    frames, targets = _half(match, period)
                    name = f"{match}_half{period}.npz"
                    gt = targets.state.copy()
                    with open(artifact_dir / name, "wb") as handle:
                        np.savez_compressed(handle, frame_id=frames.frame_id, gt=gt,
                            live_probability=np.where(gt == 2, .1, .9),
                            home_probability=np.where(gt == 0, .9, .1),
                            boundary=np.zeros((len(gt), 3, 3)))
                    hashes[name] = sha256_file(artifact_dir / name)
                    sidecar = f"{match}_half{period}.json"
                    write_json(artifact_dir / sidecar, {
                        "prediction_match": match, "period": period,
                        "train_matches": inner["train"], "outer_test": "a",
                        "trial_number": number, "inner_fold": index,
                        "checkpoint_file": f"fold_{index}.pt",
                        "checkpoint_sha256": hashes[f"fold_{index}.pt"],
                        "oof_file": name, "oof_sha256": hashes[name],
                        "study_fingerprint_sha256": study_fp["sha256"],
                        "arrays": ["frame_id", "gt", "live_probability",
                                   "home_probability", "boundary"],
                    })
                    hashes[sidecar] = sha256_file(artifact_dir / sidecar)
                    oof.append({"match": match, "period": period,
                                "inner_fold": index, "file": name,
                                "sha256": hashes[name], "sidecar": sidecar,
                                "sidecar_sha256": hashes[sidecar]})
        recipe = {"trial_number": number, "score": .5, "params": {
            "pooling": pooling}, "model_kwargs": {"width": 8,
            "pooling": pooling, "dilations": [1]},
            "training_args": {"steps": 2, "learning_rate": .001,
                              "weight_decay": .0001, "boundary_weight": .25,
                              "boundary_pos_weight": 1., "augment": False,
                              "center_size": 4},
            "decoder": {"switch_cost": 2., "boundary_scale": 0.},
            "study_fingerprint_sha256": study_fp["sha256"],
            "artifact_dir": str(artifact_dir.resolve()),
            "artifact_sha256": hashes, "fold_checkpoints": checkpoints,
            "oof_files": oof}
        write_json(artifact_dir / "recipe.json", recipe)
        pointer = outer_dir / "best_by_pooling" / pooling / "recipe.json"
        write_json(pointer, recipe)
        best[pooling] = recipe
    write_json(outer_dir / "summary.json", {
        "outer_test": fold["outer_test"], "outer_train": fold["outer_train"],
        "inner": fold["inner"], "fingerprint_sha256": study_fp["sha256"],
        "full_budget_steps": 2, "best_by_pooling": best,
        "trials": [{"number": number, "state": "COMPLETE", "value": .5,
                    "params": {"pooling": pooling},
                    "rungs": [{"steps_per_fold": 2}]}
                   for number, pooling in enumerate(("mean", "ball_attention"))],
    })
    return data, outer_dir


def test_selection_freezes_before_outer_access_and_excludes_outer_from_refit(tmp_path, monkeypatch):
    data, outer_dir = _artifacts(tmp_path)
    observed = {}

    def fake_refit(data_dir, directory, frozen, cache, *, device):
        assert (directory / "frozen_recipe.json").is_file()
        assert frozen["outer_test"] not in frozen["outer_train"]
        observed["refit_matches"] = frozen["outer_train"]
        return {"steps": 2, "fake": True}

    monkeypatch.setattr(select, "_ensure_refit", fake_refit)
    monkeypatch.setattr(select, "_evaluate_outer", lambda *args, **kwargs: pytest.fail(
        "outer features must not be read when evaluate_outer=False"))
    result = select.select_and_evaluate(data, outer_dir, n_trials=3,
                                        cache_bytes=10_000_000, evaluate_outer=False)
    assert result["outer_report"] is None
    assert observed["refit_matches"] == list("bcd")
    assert set(result["decoder_summary"]["by_pooling"]) == {"mean", "ball_attention"}
    assert result["frozen_recipe"]["decoder"]["boundary_scale"] == 0
    assert (outer_dir / "frozen_recipe.json").is_file()


def test_selection_rejects_wrong_oof_training_ancestry(tmp_path):
    data, outer_dir = _artifacts(tmp_path)
    sidecar = outer_dir / "trials" / "trial_0" / "b_half1.json"
    content = json.loads(sidecar.read_text())
    content["train_matches"] = ["a"]
    write_json(sidecar, content)
    with pytest.raises(ValueError, match="hash|ancestry"):
        select.select_and_evaluate(data, outer_dir, n_trials=3,
                                   cache_bytes=10_000_000, evaluate_outer=False)
    assert not (outer_dir / "frozen_recipe.json").exists()


def test_changed_source_or_prepared_data_refuses_resume(tmp_path, monkeypatch):
    data, outer_dir = _artifacts(tmp_path)
    monkeypatch.setattr(select, "_ensure_refit", lambda *args, **kwargs: {"fake": True})
    select.select_and_evaluate(data, outer_dir, n_trials=3,
                               cache_bytes=10_000_000, evaluate_outer=False)
    path = data / "b_half1.npz"
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="fingerprint"):
        select.select_and_evaluate(data, outer_dir, n_trials=3,
                                   cache_bytes=10_000_000, evaluate_outer=False)


def test_frozen_selection_refuses_new_trials(tmp_path, monkeypatch):
    data, outer_dir = _artifacts(tmp_path)
    monkeypatch.setattr(select, "_ensure_refit", lambda *args, **kwargs: {"fake": True})
    select.select_and_evaluate(data, outer_dir, n_trials=3,
                               cache_bytes=10_000_000, evaluate_outer=False)
    with pytest.raises(ValueError, match="frozen"):
        select.select_and_evaluate(data, outer_dir, n_trials=4,
                                   cache_bytes=10_000_000, evaluate_outer=False)


def test_trial_score_is_unweighted_mean_of_match_metrics():
    def record(match, n, gt_state, pred_state):
        neural = {"live_probability": np.full(n, .99),
                  "home_probability": np.full(n, .99 if pred_state == 0 else .01),
                  "boundary": np.zeros((n, 3, 3))}
        gt = np.full(n, gt_state, np.int8)
        return {"match": match, "gt": gt, "neural": neural, "reference": neural}

    records = {(match, period): record(match, n, gt, pred)
               for match, n, gt, pred in (("b", 8, 0, 0), ("c", 80, 0, 1))
               for period in (1, 2)}
    score, by_match = select._score_records(records, {
        "neural_weight": 1., "live_temperature": 1.,
        "owner_temperature": 1., "switch_cost": 2.})
    assert score == pytest.approx((by_match["b"]["spell_f1_mean"] +
                                   by_match["c"]["spell_f1_mean"]) / 2)
    assert by_match["b"]["spell_f1_mean"] > by_match["c"]["spell_f1_mean"]
    length_weighted = (8 * by_match["b"]["spell_f1_mean"] +
                       80 * by_match["c"]["spell_f1_mean"]) / 88
    assert score != pytest.approx(length_weighted)


def test_real_tiny_hpo_select_refit_and_outer_evaluation(tmp_path, monkeypatch):
    data = tmp_path / "prepared"
    data.mkdir()
    for match in "abcd":
        for period in (1, 2):
            frames, targets = _half(match, period)
            save_half(data / f"{match}_half{period}.npz", frames, targets)
    outer_dir = tmp_path / "hpo" / "a"
    requested = {"mode": "spatial", "operations": ["translation"],
                 "translation_max_meters": 0.5, "warmup_steps": 0}
    original_score = select._score_records

    def prefer_neural(records, params):
        _score, by_match = original_score(records, params)
        return (1.0 if params["neural_weight"] == 1 else 0.0), by_match

    monkeypatch.setattr(select, "_score_records", prefer_neural)
    tune.run_study(data, tmp_path / "hpo", outer_test="a", matches=list("abcd"),
                   n_trials=2, rungs=(1,), poolings=("mean", "ball_attention"),
                   center_size=4, cache_bytes=10_000_000,
                   augmentation_config=requested)
    config = tune._augmentation_dict(requested)
    study_fp = json.loads((outer_dir / "fingerprint.json").read_text())
    assert study_fp["payload"]["fixed"]["augmentation_config"] == config
    for pooling in ("mean", "ball_attention"):
        recipe = json.loads((outer_dir / "best_by_pooling" / pooling / "recipe.json").read_text())
        assert recipe["training_args"]["augmentation_config"] == config
        checkpoint = torch.load(outer_dir / "trials" / f'trial_{recipe["trial_number"]}' /
                                "fold_0.pt", weights_only=True)
        assert checkpoint["training_config"]["augmentation_config"] == config
    frozen_run = select.select_and_evaluate(
        data, outer_dir, n_trials=3, cache_bytes=10_000_000,
        evaluate_outer=False)
    assert frozen_run["outer_report"] is None
    assert (outer_dir / "frozen_recipe.json").is_file()
    assert not (outer_dir / "outer_report.json").exists()
    frozen = frozen_run["frozen_recipe"]
    assert frozen["outer_test"] == "a"
    assert frozen["outer_train"] == list("bcd")
    assert frozen["selected_model_recipe"]["training_args"]["augmentation_config"] == config
    if (outer_dir / "selected.pt").exists():
        checkpoint = torch.load(outer_dir / "selected.pt", weights_only=True)
        assert checkpoint["training_config"]["train_matches"] == list("bcd")
        assert checkpoint["training_config"]["validation_matches"] == []
        assert checkpoint["training_config"]["augmentation_config"] == config
    evaluated = select.select_and_evaluate(
        data, outer_dir, n_trials=3, cache_bytes=10_000_000,
        evaluate_outer=True)
    assert evaluated["outer_report"]["outer_test"] == "a"
    assert "a" in evaluated["outer_report"]["by_match"]
    assert evaluated["outer_report"]["official"]["spell_f1_mean"] >= 0
    cached = select.select_and_evaluate(
        data, outer_dir, n_trials=3, cache_bytes=10_000_000,
        evaluate_outer=True)
    assert cached["outer_report"] == evaluated["outer_report"]
    progress_path = outer_dir / "refit_progress.json"
    progress_bytes = progress_path.read_bytes()
    progress_path.unlink()
    with pytest.raises(ValueError, match="lacks refit artifact"):
        select.select_and_evaluate(data, outer_dir, n_trials=3,
                                   cache_bytes=10_000_000, evaluate_outer=True)
    progress_path.write_bytes(progress_bytes)
    report_path = outer_dir / "outer_report.json"
    report = json.loads(report_path.read_text())
    report["selection_score"] = -1
    write_json(report_path, report)
    with pytest.raises(ValueError, match="report hash"):
        select.select_and_evaluate(data, outer_dir, n_trials=3,
                                   cache_bytes=10_000_000, evaluate_outer=True)


def test_selected_neural_recipe_refits_all_outer_training_matches(tmp_path, monkeypatch):
    data, outer_dir = _artifacts(tmp_path)
    original_score = select._score_records

    def prefer_neural(records, params):
        _score, by_match = original_score(records, params)
        return (1.0 if params["neural_weight"] == 1 else 0.0), by_match

    monkeypatch.setattr(select, "_score_records", prefer_neural)
    result = select.select_and_evaluate(data, outer_dir, n_trials=3,
                                        cache_bytes=10_000_000, evaluate_outer=False)
    assert result["frozen_recipe"]["ensemble"]["neural_weight"] == 1
    assert result["refit_progress"]["steps"] == 2
    checkpoint = torch.load(outer_dir / "selected.pt", weights_only=True)
    config = checkpoint["training_config"]
    assert config["train_matches"] == list("bcd")
    assert config["validation_matches"] == []
    assert config["outer_test"] == "a"
    assert config["selection_fingerprint"] == result["frozen_recipe"]["selection_fingerprint"]
    forged = deepcopy(result["frozen_recipe"])
    forged["selected_model_recipe"]["training_args"]["augmentation_config"] = (
        tune._augmentation_dict({"mode": "spatial"}))
    with pytest.raises(ValueError, match="ancestry"):
        select._ensure_refit(data, outer_dir, forged, None, device="cpu")
