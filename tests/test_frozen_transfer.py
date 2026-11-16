"""Small, synthetic checks for the frozen transfer gate and native-rate mapping."""

import numpy as np
import pytest

from experiments import frozen_transfer as transfer
from experiments.skillcorner_proxy import ProxyHalf
from possession.schema import FrameTable, TrainingTargets


def _recipe():
    return {"steps": 7, "development_only": True, "model_kwargs": {"width": 4},
            "training_args": {"learning_rate": .001, "center_size": 8},
            "augmentation_config": {"mode": "none"}, "outage_config": {"mode": "legacy"},
            "decoder": {"switch_cost": 2, "boundary_scale": 0},
            "calibration": {}, "ensemble": {"neural_weight": .5}}


def test_decoder_changes_reuse_same_seeded_fit():
    baseline = _recipe()
    selected = {**baseline, "decoder": {"switch_cost": 3, "boundary_scale": 0},
                "ensemble": {"mode": "quality"}}
    assert transfer._fit_key(baseline, 20, ["a", "b"]) == transfer._fit_key(
        selected, 20, ["a", "b"])
    assert transfer._fit_key(baseline, 20, ["a", "b"]) != transfer._fit_key(
        selected, 21, ["a", "b"])


def test_selection_gate_blocks_before_any_refit(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(transfer, "_fit", lambda *a, **k: called.append(True))
    with pytest.raises(ValueError, match="selection_frozen"):
        transfer.run(tmp_path / "focus", tmp_path / "data", tmp_path / "proxy",
                     tmp_path / "out")
    with pytest.raises(ValueError, match="focused-study artifact missing"):
        transfer.run(tmp_path / "focus", tmp_path / "data", tmp_path / "proxy",
                     tmp_path / "out", selection_frozen=True)
    assert called == []


def test_changed_published_provenance_blocks_before_fit(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    transfer.write_json(out / "frozen-manifest.json", {"old": 1})
    monkeypatch.setattr(transfer, "_manifest", lambda *a, **k: {"new": 2})
    monkeypatch.setattr(transfer, "_fit", lambda *a, **k: pytest.fail("refit reached"))
    with pytest.raises(ValueError, match="provenance changed"):
        transfer.run(tmp_path, tmp_path, tmp_path, out, selection_frozen=True)


def test_source_and_environment_must_match_focused_protocol(monkeypatch):
    old = {"possession/interaction.py": "a" * 64,
           "experiments/focused_tuning.py": "b" * 64}
    environment = {"versions": {"torch": "test"}, "torch_num_threads": 4}
    plan = {"payload": {"source_sha256": old, "environment": environment}}
    live = {**old, "experiments/frozen_transfer.py": "c" * 64,
            "experiments/new_standalone.py": "d" * 64}
    monkeypatch.setattr(transfer, "_source_hashes", lambda: live)
    monkeypatch.setattr(transfer, "_execution_environment", lambda: environment)
    assert transfer._runtime_ancestry(plan) == (live, environment)

    monkeypatch.setattr(transfer, "_source_hashes", lambda: {
        **live, "possession/interaction.py": "e" * 64})
    with pytest.raises(ValueError, match="possession/interaction.py"):
        transfer._runtime_ancestry(plan)

    monkeypatch.setattr(transfer, "_source_hashes", lambda: {
        **live, "possession/new_feature.py": "e" * 64})
    with pytest.raises(ValueError, match="nonexperiment_added"):
        transfer._runtime_ancestry(plan)

    monkeypatch.setattr(transfer, "_source_hashes", lambda: live)
    monkeypatch.setattr(transfer, "_execution_environment", lambda: {
        **environment, "torch_num_threads": 8})
    with pytest.raises(ValueError, match="torch_num_threads"):
        transfer._runtime_ancestry(plan)


def test_focused_artifact_recipe_ancestry_is_checked(tmp_path):
    matches = [f"DFL-MAT-J0000{i}" for i in range(6)]
    base = {"model_kwargs": {"width": 4},
            "training_args": {"learning_rate": .001, "weight_decay": .0001,
                              "boundary_weight": .25, "boundary_pos_weight": 1.,
                              "center_size": 8}}
    payload = {"fold": {"outer_train": matches, "outer_test": ["DFL-MAT-J03WMX"]},
               "fixed": {"selected_model_recipe": base}, "selected_rung": 7}
    plan = {"sha256": transfer._digest(payload), "payload": payload}
    frozen = {"decoder": {"switch_cost": 2, "boundary_scale": 0},
              "calibration": {"live_temperature": 1},
              "ensemble": {"neural_weight": .5}}
    canonical = {"learning_rate": .001, "augmentation_config": {"mode": "none"}}
    spec = {"pooling": "mean", **canonical}
    recipe = {"steps": 7, "development_only": True,
              "model_kwargs": {"width": 4, "pooling": "mean"},
              "training_args": {**base["training_args"], "augment": True},
              "augmentation_config": canonical["augmentation_config"],
              "outage_config": {"mode": "legacy"}, **frozen}
    control = {"canonical": canonical, "decoder": {"frozen": frozen},
               "scores": {"objective": .5}}
    selection = {"protocol_sha256": plan["sha256"], "selected": "baseline",
                 "locked_before_confirmation": True, "model_spec": spec,
                 "frozen": frozen}
    confirmation = {"protocol_sha256": plan["sha256"],
                    "baseline": {"scores": {"objective": .5}},
                    "selected": {"scores": {"objective": .5}},
                    "comparison": {"eligible": False}, "no_op": True}
    summary = {"status": "complete", "protocol_sha256": plan["sha256"],
               "selection": selection, "selected_rung": 7,
               "confirmation_comparison": confirmation["comparison"],
               "confirmation_scores": {name: confirmation[name]["scores"]
                                       for name in ("baseline", "selected")},
               "model_trials": [{"canonical": canonical}],
               "baseline_search_scores": control["scores"]}
    for name, value in (("protocol", plan), ("summary", summary),
                        ("selection", selection), ("confirmation", confirmation),
                        ("baseline-recipe", recipe), ("selected-recipe", recipe)):
        transfer._save_record(tmp_path / f"{name}.json", value)
    transfer._save_record(tmp_path / "trials/trial_000/trial.json", control)
    assert transfer._records(tmp_path)[4:] == (recipe, recipe)
    altered = {**recipe, "training_args": {**recipe["training_args"],
                                           "learning_rate": .002}}
    transfer._save_record(tmp_path / "selected-recipe.json", altered)
    with pytest.raises(ValueError, match="frozen recipes differ"):
        transfer._records(tmp_path)


def _half(period, labels):
    n_model = 5
    frames = FrameTable(np.arange(n_model), np.zeros((n_model, 2, 1, 2), np.float32),
                        np.zeros((n_model, 2, 1), bool), np.zeros((n_model, 2), np.float32),
                        np.zeros(n_model, bool), np.ones(n_model, bool), 25.0,
                        (105.0, 68.0), "2016236", period)
    state = np.array(labels, np.int8)
    target = TrainingTargets(np.ones(len(state), np.int8), np.ones(len(state), bool),
                             1 - state, np.ones(len(state), bool),
                             state, np.ones(len(state), bool))
    return ProxyHalf(frames, np.arange(len(state)), np.array([0, 2, 4]), target,
                     {"period": period})


def test_match_score_uses_native_mapping_and_half_separator(monkeypatch, tmp_path):
    halves = [_half(1, [0, 1, 0]), _half(2, [1, 0, 1])]
    monkeypatch.setattr(transfer.skillcorner_proxy, "prepare_match", lambda *_: halves)
    monkeypatch.setattr(transfer, "build_features", lambda frames: type(
        "Features", (), {"global_tokens": np.zeros((len(frames.frame_id), 9))})())
    monkeypatch.setattr(transfer, "reference_scores", lambda frames: {
        "live_probability": np.ones(5), "home_probability": np.ones(5)})
    monkeypatch.setattr(transfer, "predict_features", lambda *a, **k: {
        "live_probability": np.ones(5), "home_probability": np.ones(5)})
    predictions = iter((np.array([0, 1, 1, 0, 0], np.int8),
                        np.array([1, 0, 0, 1, 1], np.int8)))
    monkeypatch.setattr(transfer, "decode_oof_half", lambda *_: next(predictions))
    model = type("Model", (), {"radius": 1})()
    report = transfer._score_match(tmp_path, "2016236", model, _recipe())
    assert report["native10_conditional"]["graded_HA_frames"] == 6
    assert report["native10_conditional"]["total_native_frames"] == 7
    assert report["provider_label_coverage"]["coverage_excluding_half_separator"] == 1
    assert set(report["native10_conditional"]["transitions"]) == {5, 10}
    assert [h["native10_conditional"]["total_native_frames"] for h in report["halves"]] == [3, 3]
