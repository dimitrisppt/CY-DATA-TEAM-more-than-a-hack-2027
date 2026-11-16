"""Evaluation of masked inputs against unchanged native target timelines."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import augmentation_evaluation as evaluation
from experiments.prediction_store import load_prediction_bundle
from possession.diagnostics import merge_spell_diagnostics, spell_error_diagnostics
from possession.model import PossessionModel
from possession.schema import FrameTable, TrainingTargets
from possession.storage import load_half, save_half
from possession.training_session import HalfCache


def _half(match, period, n=300):
    frame_id = np.arange(period * 1000, period * 1000 + n)
    state = np.resize(np.array([0] * 80 + [1] * 80 + [2] * 40, dtype=np.int8), n)
    state[10:15] = -1
    known = state >= 0
    alive = known & (state != 2)
    players = np.zeros((n, 2, 1, 2), np.float32)
    players[:, 0, 0] = [20, 30]
    players[:, 1, 0] = [80, 30]
    ball = np.column_stack((30 + np.arange(n) * .01, np.full(n, 30))).astype(np.float32)
    visible = np.ones(n, dtype=bool)
    visible[5:8] = False
    frames = FrameTable(frame_id, players, np.ones((n, 2, 1), bool), ball,
                        visible, np.ones(n, bool), match_id=match, period=period)
    targets = TrainingTargets(alive.astype(np.int8), known, (state == 0).astype(np.int8),
                              alive, state, known)
    return frames, targets


def test_evaluation_masks_only_inputs_and_preserves_half_local_time_axes(tmp_path, monkeypatch):
    originals = {}
    for period in (1, 2):
        frames, targets = _half("M1", period)
        originals[period] = (frames.frame_id.copy(), frames.ball_valid.copy(),
                             targets.state.copy(), targets.state_valid.copy())
        save_half(tmp_path / f"M1_half{period}.npz", frames, targets)

    observed = []
    real_build = evaluation.build_features

    def observe_features(frames):
        observed.append((frames.period, frames.frame_id.copy(), frames.ball_valid.copy(),
                         frames.graded.copy()))
        return real_build(frames)

    monkeypatch.setattr(evaluation, "build_features", observe_features)
    model = PossessionModel(width=8, dilations=(1,)).eval()
    frozen = {"ensemble": {"neural_weight": .5},
              "calibration": {"live_temperature": 1., "owner_temperature": 1.},
              "decoder": {"switch_cost": 2., "boundary_scale": 0.}}
    cache = HalfCache(tmp_path, max_bytes=2_000_000)
    scenarios = ("clean", "block41_seed3101", "block41_seed3102")
    first = evaluation.evaluate_augmented_model(
        tmp_path, "M1", model, frozen, cache, scenarios=scenarios,
        inference_center_size=128)
    second = evaluation.evaluate_augmented_model(
        tmp_path, "M1", model, frozen, cache, scenarios=scenarios,
        inference_center_size=128)

    assert first == second
    json.dumps(first, allow_nan=False)
    assert set(first) == set(scenarios)
    assert all(set(record["scores"]) == {"blend", "neural_only"}
               for record in first.values())
    for scenario in scenarios:
        record = first[scenario]
        assert set(record["masks"]) == {"1", "2"}
        for diagnostic in ("blend", "neural_only"):
            score = record["scores"][diagnostic]
            assert score["n_frames"] == 601  # two native halves and one X separator
            assert score["n_graded"] == 590
            assert set(score["transitions"]) == {"12", "25"}
            assert "pairs" not in score["transitions"]["12"]
    assert first["clean"]["masks"]["1"]["additional_missing_frames"] == 0
    assert first["block41_seed3101"]["masks"]["1"]["additional_missing_frames"] > 0
    assert (first["block41_seed3101"]["masks"]["1"]["additional_mask_sha256"]
            != first["block41_seed3102"]["masks"]["1"]["additional_mask_sha256"])

    for scenario_index, scenario in enumerate(scenarios):
        for period in (1, 2):
            seen_period, frame_id, ball_valid, graded = observed[scenario_index * 2 + period - 1]
            original_id, original_valid, _, _ = originals[period]
            assert seen_period == period
            np.testing.assert_array_equal(frame_id, original_id)
            assert graded.all()
            if scenario == "clean":
                np.testing.assert_array_equal(ball_valid, original_valid)
            else:
                assert np.count_nonzero(ball_valid) < np.count_nonzero(original_valid)
                assert not ball_valid[5:8].any()

    for period in (1, 2):
        loaded_frames, loaded_targets = load_half(tmp_path / f"M1_half{period}.npz")
        original_id, original_valid, original_state, original_state_valid = originals[period]
        np.testing.assert_array_equal(loaded_frames.frame_id, original_id)
        np.testing.assert_array_equal(loaded_frames.ball_valid, original_valid)
        np.testing.assert_array_equal(loaded_targets.state, original_state)
        np.testing.assert_array_equal(loaded_targets.state_valid, original_state_valid)


def test_unknown_scenario_is_rejected_before_loading_data(tmp_path):
    with pytest.raises(ValueError, match="unknown evaluation scenarios"):
        evaluation.evaluate_augmented_model(tmp_path, "M1", object(), {}, object(),
                                            scenarios=("clean", "unrecognized"))


def test_diagnostics_and_lossless_half_bundles_use_original_gt_and_masked_inputs(tmp_path):
    for period in (1, 2):
        save_half(tmp_path / f"M1_half{period}.npz", *_half("M1", period))
    model = PossessionModel(width=8, dilations=(1,)).eval()
    frozen = {"ensemble": {"neural_weight": .5},
              "calibration": {"live_temperature": 1., "owner_temperature": 1.},
              "decoder": {"switch_cost": 2., "boundary_scale": 0.}}
    cache = HalfCache(tmp_path, max_bytes=2_000_000)
    prediction_dir = tmp_path / "predictions"
    scenarios = ("clean", "block41_seed3101")
    result = evaluation.evaluate_augmented_model(
        tmp_path, "M1", model, frozen, cache, scenarios=scenarios,
        inference_center_size=128, include_diagnostics=True, prediction_dir=prediction_dir)
    assert set(result) == set(scenarios)
    for scenario in scenarios:
        entry = result[scenario]
        assert set(entry) == {"scores", "masks", "diagnostics", "predictions"}
        assert set(entry["predictions"]) == {"1", "2"}
        for diagnostic in ("blend", "neural_only"):
            assert entry["diagnostics"][diagnostic]["n_halves"] == 2
            assert entry["diagnostics"][diagnostic]["n_graded_frames"] == 590
        for period in (1, 2):
            record = entry["predictions"][str(period)]
            assert record["path"] == str((prediction_dir / scenario / f"half{period}.npz").resolve())
            assert (record["match"], record["period"], record["scenario"]) == (
                "M1", period, scenario)
            arrays = load_prediction_bundle(record)
            frames, targets = load_half(tmp_path / f"M1_half{period}.npz")
            np.testing.assert_array_equal(arrays["gt"], evaluation._target_codes(frames, targets))
            np.testing.assert_array_equal(arrays["frame_id"], frames.frame_id)
            altered, _ = evaluation._mask(frames, "M1", period, scenario)
            np.testing.assert_array_equal(arrays["ball_valid"], altered.ball_valid)
            features = evaluation.build_features(altered)
            expected_seconds = 10.0 * np.minimum(features.global_tokens[:, 6],
                                                 features.global_tokens[:, 7]).astype(np.float64)
            np.testing.assert_array_equal(arrays["nearest_ball_seconds"], expected_seconds)
            expected_neural = evaluation.predict_features(
                features, model, device="cpu", center_size=128, halo=model.radius)
            expected_reference = evaluation.reference_scores(altered)
            for prefix, source in (("neural", expected_neural), ("reference", expected_reference)):
                np.testing.assert_array_equal(arrays[f"{prefix}_live"],
                                              source["live_probability"])
                np.testing.assert_array_equal(arrays[f"{prefix}_home"],
                                              source["home_probability"])
        # Per-half diagnostics are pooled, with no synthetic cross-half spell.
        for diagnostic in ("blend", "neural_only"):
            parts = []
            for period in (1, 2):
                arrays = load_prediction_bundle(entry["predictions"][str(period)])
                frames, _ = load_half(tmp_path / f"M1_half{period}.npz")
                altered, _ = evaluation._mask(frames, "M1", period, scenario)
                neural = evaluation.predict_features(evaluation.build_features(altered), model,
                                                     device="cpu", center_size=128,
                                                     halo=model.radius)
                pred = evaluation._decode(neural, evaluation.reference_scores(altered), frozen,
                                          neural_only=diagnostic == "neural_only")
                parts.append(spell_error_diagnostics(arrays["gt"], pred,
                                                     arrays["ball_valid"], fps=frames.fps))
            assert entry["diagnostics"][diagnostic] == merge_spell_diagnostics(parts)
    clean_gt = load_prediction_bundle(result["clean"]["predictions"]["1"])["gt"]
    masked_gt = load_prediction_bundle(result["block41_seed3101"]["predictions"]["1"])["gt"]
    np.testing.assert_array_equal(clean_gt, masked_gt)
    assert np.count_nonzero(load_prediction_bundle(
        result["block41_seed3101"]["predictions"]["1"])["ball_valid"]) < np.count_nonzero(
            load_prediction_bundle(result["clean"]["predictions"]["1"])["ball_valid"])
    mtimes = {record["path"]: Path(record["path"]).stat().st_mtime_ns
              for scenario in scenarios for record in result[scenario]["predictions"].values()}
    again = evaluation.evaluate_augmented_model(
        tmp_path, "M1", model, frozen, cache, scenarios=scenarios,
        inference_center_size=128, include_diagnostics=True, prediction_dir=prediction_dir)
    assert again["clean"]["predictions"] == result["clean"]["predictions"]
    assert {path: Path(path).stat().st_mtime_ns for path in mtimes} == mtimes


def test_bundle_request_rejects_nonzero_boundary_scale_before_loading_data(tmp_path):
    with pytest.raises(ValueError, match="boundary_scale"):
        evaluation.evaluate_augmented_model(
            tmp_path, "M1", object(), {"decoder": {"boundary_scale": 1.}}, object(),
            prediction_dir=tmp_path / "predictions")
