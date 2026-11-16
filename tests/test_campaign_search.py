import json

import numpy as np
import pytest

from experiments import campaign_search


def _baseline():
    return {
        "version": 1, "reference_mode": "default", "reference": {},
        "blend_mode": "static", "neural_weight": .45,
        "visible_neural_weight": .45, "missing_neural_weight": .45,
        "visible_live_weight": .45, "missing_live_weight": .45,
        "visible_owner_weight": .45, "missing_owner_weight": .45,
        "transition_seconds": 2., "live_temperature": 1., "owner_temperature": 1.,
        "boundary_temperature": 1., "switch_cost_type": "scalar", "switch_cost": 2.,
        "boundary_scale": .5,
    }


def _records():
    return [{"match": "m1", "period": 1, "scenario": "clean", "input_sha256": "i1",
             "prediction_sha256": "p1", "model_identity": {"model": "a"},
             "arrays": {"gt": np.array([0, 1])}}]


def _fast_score(records, decoder):
    return {"objective": float(decoder["neural_weight"]), "by_match": {}, "means": {}}


def _read_trials(path):
    return [json.loads((path / "trials" / f"trial-{i:04d}.json").read_text())["payload"]
            for i in range(7)]


def test_tpe_resume_recreates_same_conditional_proposals(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign_search.campaign_evaluation, "load_records", lambda rows: rows)
    monkeypatch.setattr(campaign_search, "_score", _fast_score)
    uninterrupted = tmp_path / "full"
    campaign_search.tune_decoder(_records(), _baseline(), uninterrupted, 7, 37)

    resumed = tmp_path / "resume"
    calls = 0

    def interrupt_before_later_trial():
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError("simulated stop")

    with pytest.raises(RuntimeError, match="simulated stop"):
        campaign_search.tune_decoder(_records(), _baseline(), resumed, 7, 37,
                                     check=interrupt_before_later_trial)
    campaign_search.tune_decoder(_records(), _baseline(), resumed, 7, 37)
    full, restart = _read_trials(uninterrupted), _read_trials(resumed)
    assert [r.get("tpe", {}).get("params") for r in full] == [
        r.get("tpe", {}).get("params") for r in restart]
    assert [r["decoder"] for r in full] == [r["decoder"] for r in restart]


def test_plan_tampering_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign_search.campaign_evaluation, "load_records", lambda rows: rows)
    monkeypatch.setattr(campaign_search, "_score", _fast_score)
    campaign_search.tune_decoder(_records(), _baseline(), tmp_path, 3, 2)
    plan = tmp_path / "plan.json"
    envelope = json.loads(plan.read_text())
    envelope["payload"]["seed"] = 9
    plan.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="hash differs"):
        campaign_search.tune_decoder(_records(), _baseline(), tmp_path, 3, 2)


def test_reference_only_trial_omits_irrelevant_dimensions(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign_search.campaign_evaluation, "load_records", lambda rows: rows)
    monkeypatch.setattr(campaign_search, "_score", _fast_score)
    campaign_search.tune_decoder(_records(), _baseline(), tmp_path, 2, 12,
                                 reference_only=True)
    row = json.loads((tmp_path / "trials" / "trial-0001.json").read_text())["payload"]
    assert row["kind"] == "tpe"
    assert not ({"blend_mode", "neural_weight", "boundary_scale"} & row["tpe"]["params"].keys())
    assert row["decoder"]["neural_weight"] == 0
    assert row["decoder"]["boundary_scale"] == 0


def _bundle(identity, *, offset=0, input_sha="same"):
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
    return [{"match": "m1", "period": 1, "scenario": "clean", "input_sha256": input_sha,
             "model_identity": identity, "arrays": arrays}]


def test_blend_endpoints_symmetry_and_identity():
    left, right = _bundle("left"), _bundle("right", offset=.1)
    one = campaign_search.blend_records(left, right, 1.)[0]["arrays"]
    zero = campaign_search.blend_records(left, right, 0.)[0]["arrays"]
    assert np.array_equal(one["neural_live"], left[0]["arrays"]["neural_live"])
    assert np.array_equal(zero["neural_home"], right[0]["arrays"]["neural_home"])
    forward = campaign_search.blend_records(left, right, .3)[0]["arrays"]
    reverse = campaign_search.blend_records(right, left, .7)[0]["arrays"]
    assert np.allclose(forward["neural_live"], reverse["neural_live"])
    assert np.allclose(forward["neural_boundary"], reverse["neural_boundary"])
    row = campaign_search.blend_records(left, right, .5)[0]
    assert row["left_model_identity"] == "left" and row["right_model_identity"] == "right"


def test_blend_rejects_input_grid_mismatch():
    left, right = _bundle("left"), _bundle("right", input_sha="different")
    with pytest.raises(ValueError, match="different cached inputs"):
        campaign_search.blend_records(left, right, .5)
    right = _bundle("right")
    right[0]["arrays"]["gt"][0] = 2
    with pytest.raises(ValueError, match="public input differs"):
        campaign_search.blend_records(left, right, .5)
