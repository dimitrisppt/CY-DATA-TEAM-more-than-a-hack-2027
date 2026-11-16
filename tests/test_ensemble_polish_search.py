import json

import numpy as np
import pytest

from experiments import campaign_decode, campaign_search
from experiments import ensemble_polish_search as polish
from possession.calibrate import CalibrationConfig, apply_temperature_calibration
from possession.decode import decode_half, emissions_from_probabilities


def _decoder():
    return {
        "version": 1, "reference_mode": "default", "reference": {},
        "blend_mode": "static", "neural_weight": .5,
        "visible_neural_weight": .5, "missing_neural_weight": .5,
        "visible_live_weight": .5, "missing_live_weight": .5,
        "visible_owner_weight": .5, "missing_owner_weight": .5,
        "transition_seconds": 2., "live_temperature": 1.5, "owner_temperature": 1.,
        "boundary_temperature": 1., "switch_cost_type": "typed",
        "switch_cost": {"owner_switch": .2, "to_dead": 2., "from_dead": 2.},
        "boundary_scale": .5,
    }


def _bundle(identity, offset=0.):
    arrays = {
        "frame_id": np.array([1, 2, 3]), "gt": np.array([0, 1, 2]),
        "ball_valid": np.array([True, False, True]),
        "nearest_ball_seconds": np.array([0., .2, 0.]),
        "ref_fps": np.asarray(25.), "ref_pitch": np.array([105., 68.]),
        "ref_nearest_m": np.ones((3, 2)),
        "neural_live": np.array([.2, .4, .8]) + offset,
        "neural_home": np.array([.3, .6, .7]) + offset,
        "neural_boundary": np.ones((3, 3, 3)) * offset,
    }
    return [{"match": "m1", "period": 1, "scenario": "clean", "input_sha256": "same",
             "prediction_sha256": identity, "model_identity": identity, "arrays": arrays}]


def _score(records, decoder, weights):
    return {"objective": 1., "decoder": decoder, "record_count": len(records)}


def _setup(monkeypatch, score=_score):
    monkeypatch.setattr(polish.campaign_evaluation, "load_records", lambda rows: rows)
    monkeypatch.setattr(polish.campaign_evaluation, "score_records", score)


def _read_trials(path, count):
    return [json.loads((path / "trials" / f"trial-{i:04d}.json").read_text())["payload"]
            for i in range(count)]


def test_blended_arrays_match_campaign_search_and_share_inputs():
    left, right = _bundle("left"), _bundle("right", .1)
    pairs = polish._aligned(left, right)
    for weight in (.3, .5, .7):
        expected = campaign_search.blend_records(left, right, weight)[0]
        actual = polish._blend_pairs(pairs, weight)[0]
        for key, value in expected["arrays"].items():
            np.testing.assert_array_equal(actual["arrays"][key], value)
        assert actual["arrays"]["gt"] is left[0]["arrays"]["gt"]


def test_joint_anchors_canonical_boundary_and_baseline_preserved(tmp_path, monkeypatch):
    _setup(monkeypatch)
    baseline = _decoder()
    polish.tune_joint(_bundle("left"), _bundle("right", .1),
                      {"decoder": baseline, "blend_weight": .5}, tmp_path, n_trials=22)
    rows = _read_trials(tmp_path, 22)
    assert [row["anchor"] for row in rows[:1]] == ["baseline"]
    assert rows[0]["decoder"] == baseline
    assert rows[0]["blend_weight"] == .5
    assert [row["anchor"] for row in rows[17:22]] == [
        "owner_switch_zero", "live_temperature_3", "live_temperature_4",
        "effective_boundary_2_5", "effective_boundary_3"]
    assert rows[20]["decoder"]["boundary_scale"] == 2.
    assert rows[20]["decoder"]["boundary_temperature"] == pytest.approx(.8)
    assert rows[21]["decoder"]["boundary_scale"] == 2.
    assert rows[21]["decoder"]["boundary_temperature"] == pytest.approx(2 / 3)
    for row in rows:
        params = row["params"]
        assert .25 <= params["left_weight"] <= .85
        assert .25 <= params["neural_weight"] <= .8
        campaign_decode._validate_config(row["decoder"])


def test_joint_resume_is_deterministic_after_interruption(tmp_path, monkeypatch):
    def variable_score(records, decoder, weights):
        live = float(np.mean(records[0]["arrays"]["neural_live"]))
        costs = decoder["switch_cost"]
        return {"objective": live + .01 * decoder["neural_weight"]
                + .001 * decoder["boundary_scale"] / decoder["boundary_temperature"]
                - .0001 * costs["owner_switch"], "record_count": len(records)}

    _setup(monkeypatch, variable_score)
    selection = {"decoder": _decoder(), "blend_weight": .5}
    left, right = _bundle("left"), _bundle("right", .1)
    polish.tune_joint(left, right, selection, tmp_path / "full", n_trials=25, seed=17)

    calls = 0

    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 26:
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        polish.tune_joint(left, right, selection, tmp_path / "resume", n_trials=25,
                           seed=17, check=interrupt)
    assert len(list((tmp_path / "resume" / "trials").glob("trial-*.json"))) == 23
    polish.tune_joint(left, right, selection, tmp_path / "resume", n_trials=25, seed=17)
    full, resumed = _read_trials(tmp_path / "full", 25), _read_trials(tmp_path / "resume", 25)
    assert [row["params"] for row in full[22:]] == [row["params"] for row in resumed[22:]]
    assert [row["decoder"] for row in full[22:]] == [row["decoder"] for row in resumed[22:]]


def test_effective_boundary_scale_preserves_decode_parity():
    baseline = _decoder()
    params = {"left_weight": .5, "neural_weight": .5, "live_temperature": 1.5,
              "owner_temperature": 1., "effective_boundary_scale": 2.5,
              "owner_switch": .2, "to_dead": 2., "from_dead": 2.}
    canonical = polish._recipe(baseline, params)
    assert canonical["boundary_scale"] == 2.
    assert canonical["boundary_temperature"] == pytest.approx(.8)
    arrays = {
        "neural_live": np.array([.2, .8, .7, .15]),
        "neural_home": np.array([.15, .25, .8, .7]),
        "neural_boundary": np.arange(36, dtype=np.float64).reshape(4, 3, 3) / 20.,
        "ball_valid": np.array([True, False, False, True]),
        "nearest_ball_seconds": np.array([0., .4, .8, 0.]),
        "ref_nearest_m": np.array([[1., 5.], [2., 4.], [4., 1.], [5., 2.]]),
        "ref_second_m": np.full((4, 2), 8.), "ref_motion_factor": np.ones((4, 2)),
        "ref_owner_observed": np.ones(4, dtype=bool), "ref_activity": np.array([.1, .7, .9, .2]),
        "ref_activity_observed": np.ones(4, dtype=bool),
        "ref_ball_motion": np.array([False, True, True, False]),
        "ref_ball_speed": np.array([0., 2., 3., 0.]), "ref_outside": np.zeros(4, dtype=bool),
        "ref_fps": np.asarray(25.), "ref_pitch": np.array([105., 68.]),
    }
    actual = campaign_decode.decode_arrays(arrays, canonical)

    reference = campaign_decode.reference_from_basis(arrays)
    weight = np.full(4, baseline["neural_weight"])
    eps = 1e-6
    def blend(neural, reference_probability):
        p = np.clip(neural, eps, 1 - eps)
        r = np.clip(reference_probability, eps, 1 - eps)
        logits = weight * (np.log(p) - np.log1p(-p)) + (1 - weight) * (
            np.log(r) - np.log1p(-r))
        return campaign_decode._sigmoid(logits)

    live = blend(arrays["neural_live"], reference["live_probability"])
    home = blend(arrays["neural_home"], reference["home_probability"])
    direct_boundary = arrays["neural_boundary"] * weight[:, None, None] * 2.5
    live, home, boundary = apply_temperature_calibration(
        live, home, direct_boundary,
        CalibrationConfig(baseline["live_temperature"], baseline["owner_temperature"], 1.))
    expected = decode_half(emissions_from_probabilities(live, home), boundary,
                           switch_cost=campaign_decode._switch_matrix(baseline))
    np.testing.assert_array_equal(actual, expected)


def test_joint_rejects_grid_and_resume_tampering(tmp_path, monkeypatch):
    _setup(monkeypatch)
    with pytest.raises(ValueError, match="identical nonempty grids"):
        polish.tune_joint(_bundle("left"), [{**_bundle("right")[0], "period": 2}],
                          {"decoder": _decoder()}, tmp_path / "grid", n_trials=22)
    selection = {"decoder": _decoder(), "blend_weight": .5}
    polish.tune_joint(_bundle("left"), _bundle("right", .1), selection,
                      tmp_path / "resume", n_trials=22)
    path = tmp_path / "resume" / "trials" / "trial-0000.json"
    envelope = json.loads(path.read_text())
    envelope["payload"]["blend_weight"] = .6
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="hash differs"):
        polish.tune_joint(_bundle("left"), _bundle("right", .1), selection,
                          tmp_path / "resume", n_trials=22)


def test_joint_rejects_overlapping_fold_metadata(tmp_path, monkeypatch):
    _setup(monkeypatch)
    selection = {"decoder": _decoder(), "blend_weight": .5,
                 "train_matches": ["m1"], "validation_matches": ["m1"]}
    with pytest.raises(ValueError, match="overlap"):
        polish.tune_joint(_bundle("left"), _bundle("right", .1), selection,
                          tmp_path, n_trials=22)


def test_nested_model_identity_fold_metadata_is_checked(tmp_path, monkeypatch):
    _setup(monkeypatch)
    left = _bundle("left")
    right = _bundle("right", .1)
    left[0]["model_identity"] = {"training_matches": ["m1"], "validation_matches": ["m1"]}
    with pytest.raises(ValueError, match="training matches"):
        polish.tune_joint(left, right, {"decoder": _decoder()}, tmp_path, n_trials=22)


def test_baseline_score_mismatch_fails_before_writing_trial(tmp_path, monkeypatch):
    _setup(monkeypatch)
    selection = {"decoder": _decoder(), "blend_weight": .5,
                 "scores": {"objective": .5733205910911874}}
    with pytest.raises(ValueError, match="does not reproduce"):
        polish.tune_joint(_bundle("left"), _bundle("right", .1), selection,
                          tmp_path, n_trials=22)
    assert not list((tmp_path / "trials").glob("trial-*.json"))
