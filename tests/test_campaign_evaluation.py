"""Cached OOF input, prediction, and official scoring interfaces."""

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation import possession_metrics as official
from experiments import campaign_decode, campaign_evaluation
from experiments.augmentation_evaluation import _compact
from experiments.evaluate import _join_with_x, _target_codes
from possession.model import PossessionModel
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half


def _half(match, period, n=90):
    t = np.arange(n, dtype=np.int64) + (period - 1) * 1000
    state = np.resize(np.array([0] * 20 + [1] * 20 + [2] * 10, np.int8), n)
    live = (state != 2).astype(np.int8)
    players = np.zeros((n, 2, 3, 2), np.float32)
    players[:, 0, :, 0] = 20 + t[:, None] * .02
    players[:, 1, :, 0] = 80 - t[:, None] * .02
    players[..., 1] = np.array([20, 30, 40], np.float32)[None, None, :]
    ball = np.column_stack((50 + np.sin(t / 4), np.full(n, 31))).astype(np.float32)
    ball_valid = np.ones(n, bool)
    ball_valid[10:20] = False
    frames = FrameTable(t, players, np.ones((n, 2, 3), bool), ball, ball_valid,
                        np.ones(n, bool), match_id=match, period=period)
    targets = TrainingTargets(live, np.ones(n, bool), (state == 0).astype(np.int8),
                              live.astype(bool), state, np.ones(n, bool))
    return frames, targets


def _data(tmp_path, matches=("DFL-MAT-J00001", "DFL-MAT-J00002")):
    root = tmp_path / "prepared"
    root.mkdir()
    for match in matches:
        for period in (1, 2):
            save_half(root / f"{match}_half{period}.npz", *_half(match, period))
    return root


SCENARIOS = {
    "clean": {"kind": "clean"},
    "block": {"kind": "synthetic", "seed": 45, "missing_fraction": .3,
              "min_seconds": .2, "max_seconds": 1.},
}


def _decoder():
    return campaign_decode.baseline_decoder({
        "ensemble": {"neural_weight": .5},
        "calibration": {"live_temperature": 1., "owner_temperature": 1.},
        "decoder": {"switch_cost": 2., "boundary_scale": 0.},
    })


def test_gap_quality_age_is_bidirectional_uncapped_and_finite_without_observations():
    valid = np.array([True] + [False] * 300 + [True])
    seconds = campaign_evaluation._gap_seconds(valid, 25.)
    assert seconds[150] == 6.0
    assert seconds[30] == 1.2
    absent = campaign_evaluation._gap_seconds(np.zeros(5, bool), 25.)
    np.testing.assert_array_equal(absent, np.full(5, .2, np.float32))


def test_prepare_inputs_caches_public_labels_features_and_reference_basis(tmp_path):
    data = _data(tmp_path, matches=("DFL-MAT-J00001",))
    output = tmp_path / "cache"
    records = campaign_evaluation.prepare_inputs(data, None, output,
                                                ["DFL-MAT-J00001"], SCENARIOS)
    assert len(records) == 4
    record = next(row for row in records if row["period"] == 1 and row["scenario"] == "block")
    arrays = campaign_evaluation.load_arrays(record)
    frames, targets = campaign_evaluation.load_half(data / "DFL-MAT-J00001_half1.npz")
    np.testing.assert_array_equal(arrays["gt"], _target_codes(frames, targets))
    assert arrays["ball_valid"].sum() < frames.ball_valid.sum()
    assert arrays["player_tokens"].shape == (90, 2, 3, 8)
    assert {key for key in arrays if key.startswith("ref_")} >= {
        "ref_nearest_m", "ref_second_m", "ref_fps", "ref_pitch"}
    loaded = campaign_evaluation.load_records([record])[0]["arrays"]
    assert not {"player_tokens", "player_mask", "global_tokens", "time_valid"} & set(loaded)
    assert campaign_evaluation.prepare_inputs(data, None, output,
                                              ["DFL-MAT-J00001"], SCENARIOS) == records


def test_input_orphan_is_regenerated_from_verified_sources(tmp_path):
    data = _data(tmp_path, matches=("DFL-MAT-J00001",))
    output = tmp_path / "cache"
    records = campaign_evaluation.prepare_inputs(data, None, output,
                                                ["DFL-MAT-J00001"], {"clean": SCENARIOS["clean"]})
    path = output / "input-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["records"].pop()
    campaign_evaluation._publish_input_manifest(path, manifest)
    orphan = records[-1]["input_path"]
    with open(orphan, "ab") as handle:
        handle.write(b"orphan")
    callbacks = []
    recovered = campaign_evaluation.prepare_inputs(
        data, None, output, ["DFL-MAT-J00001"], {"clean": SCENARIOS["clean"]},
        progress_callback=lambda: callbacks.append(True))
    assert len(recovered) == 2
    assert callbacks == [True]
    campaign_evaluation.load_arrays(recovered[-1])


def test_prepare_inputs_rejects_duplicate_match_ids(tmp_path):
    data = _data(tmp_path, matches=("DFL-MAT-J00001",))
    with pytest.raises(ValueError, match="duplicate match"):
        campaign_evaluation.prepare_inputs(data, None, tmp_path / "cache",
                                           ["DFL-MAT-J00001", "DFL-MAT-J00001"],
                                           {"clean": SCENARIOS["clean"]})


def test_target_replay_uses_only_input_mask(monkeypatch, tmp_path):
    data = _data(tmp_path, matches=("DFL-MAT-J00001",))
    target_dir = tmp_path / "target-inputs"
    target_dir.mkdir()
    patterns = {1: np.array([True, False, True, False], bool),
                2: np.array([False, True, False, True], bool)}
    monkeypatch.setattr("experiments.target_gap_stress._target_patterns",
                        lambda _path: (patterns, {"input_sha256": "a" * 64}))
    records = campaign_evaluation.prepare_inputs(
        data, target_dir, tmp_path / "target-cache", ["DFL-MAT-J00001"],
        {"clean": SCENARIOS["clean"], "target": {"kind": "target_replay", "seed": 5101}})
    clean = campaign_evaluation.load_arrays(next(row for row in records if row["scenario"] == "clean"))
    replay = campaign_evaluation.load_arrays(next(row for row in records
                                                  if row["scenario"] == "target"))
    np.testing.assert_array_equal(clean["gt"], replay["gt"])
    assert np.all(replay["ball_valid"] <= clean["ball_valid"])


def test_input_cache_detects_tampered_arrays(tmp_path):
    data = _data(tmp_path, matches=("DFL-MAT-J00001",))
    output = tmp_path / "cache"
    records = campaign_evaluation.prepare_inputs(data, None, output,
                                                ["DFL-MAT-J00001"], {"clean": SCENARIOS["clean"]})
    path = records[0]["input_path"]
    with open(path, "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash differs"):
        campaign_evaluation.load_arrays(records[0])


def test_prediction_bundle_reuse_tamper_and_full_match_scoring(tmp_path):
    data = _data(tmp_path)
    inputs = campaign_evaluation.prepare_inputs(data, None, tmp_path / "inputs",
                                                ["DFL-MAT-J00001", "DFL-MAT-J00002"], SCENARIOS)
    model = PossessionModel(width=8, dilations=[1])
    predictions = campaign_evaluation.predict_records(
        model, inputs, tmp_path / "predictions", {"checkpoint_sha256": "b" * 64}, center_size=24)
    assert len(predictions) == 8
    callback_count = []
    again = campaign_evaluation.predict_records(
        model, inputs, tmp_path / "predictions", {"checkpoint_sha256": "b" * 64}, center_size=24,
        progress_callback=lambda: callback_count.append(True))
    assert predictions == again
    assert len(callback_count) == len(again)
    loaded = campaign_evaluation.load_records(predictions)
    scores = campaign_evaluation.score_records(loaded, _decoder(), {"clean": .5, "block": .5})
    assert set(scores["by_match"]) == {"DFL-MAT-J00001", "DFL-MAT-J00002"}
    assert set(scores["means"]) == {"clean", "block"}
    assert 0 <= scores["objective"] <= 1
    gt, pred = [], []
    for period in (1, 2):
        arrays = campaign_evaluation.load_arrays(next(
            row for row in predictions if row["match"] == "DFL-MAT-J00001"
            and row["period"] == period and row["scenario"] == "clean"))
        gt.append(arrays["gt"])
        pred.append(campaign_decode.decode_arrays(arrays, _decoder()))
    expected = _compact(official.evaluate(_join_with_x(gt), _join_with_x(pred), bootstrap_n=0))
    assert scores["by_match"]["DFL-MAT-J00001"]["clean"] == expected
    path = predictions[0]["prediction_path"]
    sidecar = campaign_evaluation._prediction_sidecar(Path(path))
    sidecar.unlink()
    recovered = campaign_evaluation.predict_records(
        model, inputs, tmp_path / "predictions", {"checkpoint_sha256": "b" * 64}, center_size=24)
    assert recovered == predictions
    with open(path, "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash differs"):
        campaign_evaluation.load_arrays(predictions[0])


def test_score_requires_exact_grid_and_scenario_weights(tmp_path):
    data = _data(tmp_path, matches=("DFL-MAT-J00001",))
    inputs = campaign_evaluation.prepare_inputs(data, None, tmp_path / "inputs",
                                                ["DFL-MAT-J00001"], SCENARIOS)
    model = PossessionModel(width=8, dilations=[1])
    records = campaign_evaluation.predict_records(model, inputs, tmp_path / "preds", {"id": 1})
    with pytest.raises(ValueError, match="exactly once"):
        campaign_evaluation.score_records(records[:-1], _decoder(), {"clean": .5, "block": .5})
    with pytest.raises(ValueError, match="weight names"):
        campaign_evaluation.score_records(records, _decoder(), {"clean": 1.})
