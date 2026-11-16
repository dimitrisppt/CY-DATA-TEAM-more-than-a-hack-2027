"""Frozen baseline all-match refit preserves selection and refit ancestry."""

import json

import numpy as np
import pytest

from experiments import final_refit
from possession.deployment import export_bundle, verify_selected_checkpoint
from possession.predict import load_model_checkpoint
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half

MATCHES = list(final_refit.EXPECTED_REFIT_MATCHES)
SELECTION_MATCHES = list(final_refit.EXPECTED_SELECTION_MATCHES)


def _recipe(steps=1):
    return {
        "model_kwargs": {"width": 8, "dilations": [1], "pooling": "mean"},
        "training_args": {"learning_rate": 1e-3, "weight_decay": 1e-4,
                          "boundary_weight": .25, "boundary_pos_weight": 1.0,
                          "center_size": 4, "augment": True},
        "augmentation_config": {"mode": "none"},
        "outage_config": {"mode": "legacy"},
        "decoder": {"switch_cost": 2.1, "boundary_scale": 0.0},
        "calibration": {"live_temperature": 1.1, "owner_temperature": .9,
                        "boundary_temperature": 1.0, "eps": 1e-6},
        "ensemble": {"neural_weight": .45},
        "steps": steps,
        "development_only": True,
    }


def _half(match, period):
    n = 8
    state = np.array([0, 0, 1, 1, 2, 2, 0, 0], np.int8)
    players = np.zeros((n, 2, 1, 2), np.float32)
    players[:, 0, 0] = [20, 30]
    players[:, 1, 0] = [80, 30]
    frames = FrameTable(
        np.arange(n), players, np.ones((n, 2, 1), bool),
        np.column_stack((30 + np.arange(n), np.full(n, 30))),
        np.ones(n, bool), np.ones(n, bool), match_id=match, period=period,
    )
    live = (state != 2).astype(np.int8)
    targets = TrainingTargets(live, np.ones(n, bool), (state == 0).astype(np.int8),
                              live.astype(bool), state, np.ones(n, bool))
    return frames, targets


def _prepared(tmp_path):
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    for match in MATCHES:
        for period in (1, 2):
            save_half(data_dir / f"{match}_half{period}.npz", *_half(match, period))
    return data_dir


def _focused(tmp_path, monkeypatch, data_dir, *, steps=1):
    focused_dir = tmp_path / "focused"
    focused_dir.mkdir()
    names = final_refit.FOCUSED_ARTIFACTS
    for name in names:
        path = focused_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen input")
    baseline = _recipe(steps)
    plan = {"sha256": "a" * 64, "payload": {
        "fold": {"outer_train": SELECTION_MATCHES,
                 "outer_test": ["DFL-MAT-J03WMX"]},
        "selected_rung": steps,
        "prepared_sha256": {
            f"{match}_half{period}.npz": final_refit.sha256_file(
                data_dir / f"{match}_half{period}.npz")
            for match in MATCHES for period in (1, 2)
        },
    }}
    monkeypatch.setattr(final_refit.frozen_transfer, "_records",
                        lambda _directory: (plan, {}, {}, {}, baseline, {}))
    monkeypatch.setattr(final_refit.frozen_transfer, "_runtime_ancestry",
                        lambda _plan: ({"experiments/final_refit.py": "b" * 64},
                                       {"runtime": "test"}))
    return focused_dir


def test_full_recipe_keeps_source_selection_matches_distinct_from_refit_matches():
    frozen = final_refit._deployment_recipe(
        _recipe(), selection_matches=SELECTION_MATCHES, refit_matches=MATCHES,
        seed=30, fingerprint="c" * 64,
    )
    assert frozen["outer_train"] == MATCHES
    assert frozen["refit_matches"] == MATCHES
    assert frozen["source_selection_matches"] == SELECTION_MATCHES
    assert frozen["selected_model_recipe"]["training_args"]["outage_config"]["mode"] == "legacy"
    assert frozen["selected_model_recipe"]["training_args"]["augmentation_config"]["mode"] == "none"


def test_final_refit_exports_loadable_bundle_and_reuses_verified_completion(tmp_path, monkeypatch):
    data_dir = _prepared(tmp_path)
    monkeypatch.setattr(final_refit, "BASELINE_STEPS", 1)
    focused_dir = _focused(tmp_path, monkeypatch, data_dir)
    output_dir = tmp_path / "bundle"
    report = final_refit.run_final_refit(focused_dir, data_dir, output_dir)
    study_dir = tmp_path / "bundle.study" / final_refit.FINAL_ID
    frozen = json.loads((study_dir / "frozen_recipe.json").read_text())
    checkpoint = study_dir / "selected.pt"
    model, payload = load_model_checkpoint(checkpoint)
    assert model is not None
    assert payload["training_config"]["train_matches"] == MATCHES
    assert payload["training_config"]["source_selection_matches"] == SELECTION_MATCHES
    assert frozen["outer_train"] == MATCHES
    assert frozen["source_selection_matches"] == SELECTION_MATCHES
    assert report["steps"] == 1
    exported = export_bundle(study_dir, output_dir)
    verified = verify_selected_checkpoint(
        output_dir / "frozen_recipe.json", frozen, output_dir / "selected.pt")
    assert exported["manifest"]["evaluation_performed"] is False
    assert verified["selection_verified"] is True

    repeated = final_refit.run_final_refit(focused_dir, data_dir, output_dir)
    assert repeated["bundle_reused"] is True


def test_final_refit_rejects_tampered_completed_checkpoint(tmp_path, monkeypatch):
    data_dir = _prepared(tmp_path)
    monkeypatch.setattr(final_refit, "BASELINE_STEPS", 1)
    focused_dir = _focused(tmp_path, monkeypatch, data_dir)
    output_dir = tmp_path / "bundle"
    final_refit.run_final_refit(focused_dir, data_dir, output_dir)
    checkpoint = tmp_path / "bundle.study" / final_refit.FINAL_ID / "selected.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checkpoint hash|artifact hash"):
        final_refit.run_final_refit(focused_dir, data_dir, output_dir)


def test_final_refit_rejects_partial_completed_artifacts(tmp_path, monkeypatch):
    data_dir = _prepared(tmp_path)
    monkeypatch.setattr(final_refit, "BASELINE_STEPS", 1)
    focused_dir = _focused(tmp_path, monkeypatch, data_dir)
    output_dir = tmp_path / "bundle"
    final_refit.run_final_refit(focused_dir, data_dir, output_dir)
    progress = tmp_path / "bundle.study" / final_refit.FINAL_ID / "refit_progress.json"
    progress.unlink()
    with pytest.raises(ValueError, match="partial final refit artifacts"):
        final_refit.run_final_refit(focused_dir, data_dir, output_dir)


def test_final_refit_rejects_runtime_ancestry_drift_before_fitting(tmp_path, monkeypatch):
    data_dir = _prepared(tmp_path)
    focused_dir = _focused(tmp_path, monkeypatch, data_dir)

    def drift(_plan):
        raise ValueError("focused execution environment differs")

    monkeypatch.setattr(final_refit.frozen_transfer, "_runtime_ancestry", drift)
    monkeypatch.setattr(final_refit, "TrainingSession",
                        lambda *args, **kwargs: pytest.fail("fit started before ancestry check"))
    output_dir = tmp_path / "bundle"
    with pytest.raises(ValueError, match="environment differs"):
        final_refit.run_final_refit(focused_dir, data_dir, output_dir)
    assert not (tmp_path / "bundle.study").exists()


def test_final_refit_rejects_changed_prepared_cache_when_resuming(tmp_path, monkeypatch):
    data_dir = _prepared(tmp_path)
    monkeypatch.setattr(final_refit, "BASELINE_STEPS", 1)
    focused_dir = _focused(tmp_path, monkeypatch, data_dir)
    output_dir = tmp_path / "bundle"
    monkeypatch.setattr(final_refit, "TrainingSession",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            RuntimeError("interrupt after immutable plan")))
    with pytest.raises(RuntimeError, match="interrupt after immutable plan"):
        final_refit.run_final_refit(focused_dir, data_dir, output_dir)
    cache = data_dir / f"{MATCHES[0]}_half1.npz"
    cache.write_bytes(cache.read_bytes() + b" changed")
    with pytest.raises(ValueError, match="prepared cache|prepared cache changed"):
        final_refit.run_final_refit(focused_dir, data_dir, output_dir)
