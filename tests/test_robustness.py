"""A small real paired fit and strict resume checks for the robustness protocol."""

import json

import numpy as np
import pytest

from experiments import robustness
from experiments.robustness import SCENARIOS, _frozen_plan, _mask, run_robustness
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half, write_json


def _half(match, period, n=60):
    frame_id = np.arange(period * 1000, period * 1000 + n)
    state = np.resize(np.array([0] * 15 + [1] * 15 + [2] * 10 + [0] * 20, np.int8), n)
    live = (state != 2).astype(np.int8)
    players = np.zeros((n, 2, 1, 2), np.float32)
    players[:, 0, 0] = [20, 30]
    players[:, 1, 0] = [80, 30]
    ball = np.column_stack((30 + np.arange(n) * .01, np.full(n, 30))).astype(np.float32)
    visible = np.ones(n, bool)
    visible[5:8] = False
    frames = FrameTable(frame_id, players, np.ones((n, 2, 1), bool), ball,
                        visible, np.ones(n, bool), match_id=match, period=period)
    targets = TrainingTargets(live, np.ones(n, bool), (state == 0).astype(np.int8),
                              live.astype(bool), state, np.ones(n, bool))
    return frames, targets


def _fixture(tmp_path):
    data = tmp_path / "prepared"
    baseline = tmp_path / "baseline"
    data.mkdir()
    for match in "abc":
        for period in (1, 2):
            frames, targets = _half(match, period)
            save_half(data / f"{match}_half{period}.npz", frames, targets)
        directory = baseline / match
        directory.mkdir(parents=True)
        (directory / "selected.pt").write_bytes(b"frozen baseline checkpoint fixture")
        write_json(directory / "frozen_recipe.json", {
            "outer_test": match, "outer_train": [name for name in "abc" if name != match],
            "refit_seed": 1234, "ensemble": {"neural_weight": .5},
            "calibration": {"live_temperature": 1., "owner_temperature": 1.},
            "decoder": {"switch_cost": 2., "boundary_scale": 0.},
            "selected_model_recipe": {
                "model_kwargs": {"width": 8, "dilations": [1], "pooling": "mean"},
                "training_args": {"learning_rate": .001, "weight_decay": .0001,
                                  "boundary_weight": .25, "boundary_pos_weight": 1.,
                                  "augment": True, "center_size": 20},
            },
        })
    return data, baseline


def test_masks_are_half_local_repeatable_and_input_only():
    frames, _ = _half("a", 1, n=600)
    assert set(SCENARIOS) == {"clean", "block41_seed3101", "block41_seed3102",
                             "long41_seed4101"}
    altered, evidence = _mask(frames, "a", 1, "block41_seed3101")
    again, repeated = _mask(frames, "a", 1, "block41_seed3101")
    other, different = _mask(frames, "a", 2, "block41_seed3101")
    assert evidence == repeated
    assert np.array_equal(altered.ball_valid, again.ball_valid)
    assert evidence["additional_mask_sha256"] != different["additional_mask_sha256"]
    assert not altered.ball_valid[5:8].any()
    assert frames.ball_valid.sum() > altered.ball_valid.sum()
    assert evidence["resulting_gap_statistics"]["missing_frames"] == (~altered.ball_valid).sum()
    assert evidence["additional_missing_frames"] == int(.41 * len(frames.frame_id)) - 3
    assert not np.array_equal(other.ball_valid, altered.ball_valid)


@pytest.mark.parametrize("tracking", [False, True])
def test_real_tiny_pair_scores_and_resume_rejects_changed_inputs(tmp_path, tracking):
    data, baseline = _fixture(tmp_path)
    if tracking:
        recipe_path = baseline / "a" / "frozen_recipe.json"
        recipe = json.loads(recipe_path.read_text())
        recipe["selected_model_recipe"]["training_args"]["augmentation_config"] = {
            "mode": "temporal", "operations": ["frame_dropout"],
            "clean_probability": 0, "warmup_steps": 0,
        }
        write_json(recipe_path, recipe)
    output = tmp_path / "robustness"
    with pytest.raises(ValueError, match="exactly all seven"):
        run_robustness(data, baseline, output, steps=2, matches=["a"])
    result = run_robustness(data, baseline, output, steps=2, matches=["a"],
                            cache_bytes=2_000_000, require_complete=False)
    assert result["matches_completed"] == ["a"]
    assert result["gate"]["passes"] is None
    assert result["conditions"]["blend"]["equal_match_mean_spell_f1"]["legacy"][
        "short_two_seed_mean"] >= 0
    record = json.loads((output / "a" / "result.json").read_text())
    assert set(record["scenarios"]) == set(SCENARIOS)
    assert set(record["checkpoint_sha256"]) == {"legacy", "correlated"}
    assert set(record["scenarios"]["clean"]["scores"]["legacy"]) == {
        "blend", "neural_only"}
    assert record["training_progress"]["legacy"]["training_schedule_blocks"] == \
        record["training_progress"]["correlated"]["training_schedule_blocks"]
    for progress in record["training_progress"].values():
        assert progress["center_ball_frames"] > 0
        assert 0 <= progress["center_ball_missing_frames"] <= progress["center_ball_frames"]
        assert progress["tracking_augmentation"]["operation_counts"] == (
            {"frame_dropout": 2} if tracking else {})
    assert run_robustness(data, baseline, output, steps=2, matches=["a"],
                          cache_bytes=2_000_000, require_complete=False)["matches_completed"] == ["a"]
    with (output / "a" / "legacy.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        run_robustness(data, baseline, output, steps=2, matches=["a"],
                       cache_bytes=2_000_000, require_complete=False)


def test_partial_artifact_is_rejected(tmp_path):
    data, baseline = _fixture(tmp_path)
    output = tmp_path / "robustness"
    output.mkdir()
    write_json(output / "protocol.json", _frozen_plan(
        data, baseline, ["a"], steps=2, cache_bytes=2_000_000))
    (output / "a").mkdir()
    (output / "a" / "legacy.pt").write_bytes(b"partial")
    with pytest.raises(ValueError, match="partial or unexpected robustness artifacts"):
        run_robustness(data, baseline, output, steps=2, matches=["a"],
                       cache_bytes=2_000_000, require_complete=False)


def test_fit_pair_advances_in_bounded_stages(tmp_path, monkeypatch):
    calls = []

    class Session:
        def __init__(self, *args, outage_config, **kwargs):
            self.mode = outage_config.mode
            self.steps = 0
            calls.append((self.mode, []))

        def advance_to(self, total_steps):
            calls[-1][1].append(total_steps)
            self.steps = total_steps
            return {"steps": total_steps, "center_ball_frames": total_steps,
                    "center_ball_missing_frames": 0,
                    "training_schedule_blocks": [{"half": "train_half1.npz",
                                                  "steps": total_steps}],
                    "training_window_sha256": "same-window-schedule"}

        def save_checkpoint(self, path, **kwargs):
            path.write_bytes(self.mode.encode())

    monkeypatch.setattr(robustness, "TrainingSession", Session)
    plan = {"sha256": "protocol", "payload": {"folds": {"a": {
        "outer_train": ["b", "c"], "refit_seed": 1,
        "model_kwargs": {"width": 8},
        "training_args": {"learning_rate": .001, "weight_decay": .001,
                          "boundary_weight": .25, "boundary_pos_weight": 1.,
                          "augment": True, "center_size": 20},
    }}}}
    hashes, progress, _ = robustness._fit_pair(
        tmp_path, tmp_path, "a", plan, None, steps=301)
    assert calls == [("legacy", [300, 301]), ("correlated", [300, 301])]
    assert set(hashes) == {"legacy", "correlated"}
    assert progress["legacy"]["steps"] == 301


def test_baseline_progress_hash_must_match_selected_checkpoint(tmp_path):
    data, baseline = _fixture(tmp_path)
    write_json(baseline / "a" / "refit_progress.json", {
        "checkpoint_sha256": "0" * 64})
    with pytest.raises(ValueError, match="baseline refit checkpoint hash differs"):
        _frozen_plan(data, baseline, ["a"], steps=2, cache_bytes=2_000_000)
