import numpy as np
import pytest

from experiments import coarse_context, coarse_stability


def _score(objective, shift=0.):
    means = {
        scenario: {"spell_f1_mean": value + shift,
                   "transition_f1_at_12": .4 + shift}
        for scenario, value in (("clean", .64), ("short", .49),
                                ("long", .48), ("replay", .47))
    }
    by_match = {
        f"m{i}": {scenario: {"spell_f1_mean": row["spell_f1_mean"] + .001 * i,
                             "transition_f1_at_12": row["transition_f1_at_12"]}
                  for scenario, row in means.items()}
        for i in range(6)
    }
    return {"objective": objective, "means": means, "by_match": by_match}


def _fine_arrays():
    return {
        "neural_live": np.array([.2, .25, .6, .7], dtype=np.float64),
        "neural_home": np.array([.8, .6, .4, .3], dtype=np.float64),
        "neural_boundary": np.arange(36, dtype=np.float64).reshape(4, 3, 3),
        "ball_valid": np.array([True, False, False, True]),
        "nearest_ball_seconds": np.array([0., 4., 10., 0.]),
        "ref_live": np.array([.3, .4, .5, .6]),
        "gt": np.array([0, 1, 2, 3]),
    }


def test_spec_pins_one_hertz_rf255_recipe():
    selection = {"members": [{"spec": {"outage_config": {"mode": "legacy",
                                                           "min_gap_seconds": 1.}}}]}
    spec = coarse_stability._spec(selection)
    assert spec["model_kwargs"] == coarse_context.COARSE_MODEL_KWARGS
    assert 1 + 2 * sum(spec["model_kwargs"]["dilations"]) == 255
    assert spec["training_args"] == {
        "learning_rate": 3e-4, "weight_decay": 1e-4,
        "boundary_weight": 0., "boundary_pos_weight": 1.,
        "augment": True, "center_size": 32,
    }
    assert spec["schedule"] == {
        "kind": "constant", "total_steps": 7200, "warmup_steps": 0,
        "min_lr_ratio": 1.,
    }
    assert spec["augmentation_config"] == {"mode": "none"}
    assert spec["outage_config"]["mode"] == "legacy"


def test_gate_reuses_fixed_inner_thresholds():
    raw = _score(.58)
    improved = _score(.583, shift=.002)
    gate = coarse_stability._gate(raw, improved)
    assert gate["passed"]
    assert all(gate["checks"].values())

    weak = _score(.581, shift=.002)
    assert not coarse_stability._gate(raw, weak)["checks"]["objective_gain_gt_0_002"]
    assert coarse_stability._outer_gate(_score(.58), _score(.57))["passed"]
    assert not coarse_stability._outer_gate(_score(.58), _score(.569))["passed"]


def test_ema_gate_authentication_binds_raw_scores_to_v4(monkeypatch, tmp_path):
    baseline, original = {"selection": "candidate"}, {}
    raw_inner, ema_inner = _score(.58), _score(.584, shift=.003)
    raw_outer, ema_outer = _score(.52), _score(.515)
    row = {
        "batch": "batch_1", "protocol_sha256": "protocol",
        "inner_inputs_sha256": "inner", "outer_inputs_sha256": "outer",
        "raw_v4_inner": raw_inner, "ema_inner": ema_inner,
        "inner_checks": coarse_stability._gate(raw_inner, ema_inner),
        "raw_v4_outer": raw_outer, "ema_outer": ema_outer,
        "outer_checks": coarse_stability._outer_gate(raw_outer, ema_outer),
        "artifact_sha256": {},
    }
    bag_row = {"candidate": raw_inner, "outer": {"candidate": raw_outer}}
    bag_batch = {"inner_inputs_sha256": "inner", "outer_inputs_sha256": "outer"}
    bag_result = {"batches": [bag_row, bag_row],
                  "baseline_selection": original}
    result = {"batches": [row, {**row, "batch": "batch_2"}],
              "candidate_selection": baseline, "baseline_selection": original,
              "eligible": True}
    rows_by_path = {
        str(tmp_path / "batches" / "batch_1" / "result.json"): row,
        str(tmp_path / "batches" / "batch_2" / "result.json"): result["batches"][1],
        str(tmp_path / "batch_1" / "result.json"): bag_batch,
        str(tmp_path / "batch_2" / "result.json"): bag_batch,
    }
    monkeypatch.setattr(coarse_stability, "_required",
                        lambda path: rows_by_path[str(path)])
    coarse_stability._verify_ema_result(
        tmp_path, tmp_path, result, "protocol", baseline, bag_result)

    bad_result = {**result, "batches": [
        {**row, "raw_v4_inner": _score(.581)},
        result["batches"][1],
    ]}
    rows_by_path[str(tmp_path / "batches" / "batch_1" / "result.json")] = bad_result["batches"][0]
    with pytest.raises(ValueError, match="input ancestry"):
        coarse_stability._verify_ema_result(
            tmp_path, tmp_path, bad_result, "protocol", baseline, bag_result)


def test_mix_preserves_clean_boundaries_and_reference_arrays_exactly():
    fine = _fine_arrays()
    coarse = {"live_probability": np.array([.9, .9, .9, .9]),
              "home_probability": np.array([.1, .1, .1, .1])}
    clean_arrays = {key: value.copy() for key, value in fine.items()}
    clean_arrays["ball_valid"][:] = True
    clean_arrays["nearest_ball_seconds"][:] = 0.
    records = [
        {"match": "m0", "period": 1, "scenario": "clean",
         "arrays": clean_arrays, "model_identity": {"base": "frozen"}},
        {"match": "m0", "period": 1, "scenario": "short",
         "arrays": {key: value.copy() for key, value in fine.items()},
         "model_identity": {"base": "frozen"}},
    ]
    mixed = coarse_stability._mix_coarse(records, {("m0", 1): coarse},
                                         batch_name="batch_1")
    clean, short = mixed
    for name in ("neural_live", "neural_home", "neural_boundary", "ref_live", "gt"):
        assert np.array_equal(clean["arrays"][name], records[0]["arrays"][name])
    assert np.array_equal(short["arrays"]["neural_boundary"],
                          records[1]["arrays"]["neural_boundary"])
    assert np.array_equal(short["arrays"]["ref_live"], records[1]["arrays"]["ref_live"])
    assert not np.array_equal(short["arrays"]["neural_live"],
                              records[1]["arrays"]["neural_live"])
    assert not np.array_equal(short["arrays"]["neural_home"],
                              records[1]["arrays"]["neural_home"])


def test_clean_rejects_natural_ball_gaps():
    record = {"match": "m0", "period": 1, "scenario": "clean",
              "arrays": _fine_arrays(), "model_identity": {"base": "frozen"}}
    coarse = {"live_probability": np.full(4, .9), "home_probability": np.full(4, .1)}
    with pytest.raises(ValueError, match="naturally missing"):
        coarse_stability._mix_coarse([record], {("m0", 1): coarse}, batch_name="batch_1")


def test_coarse_context_blend_is_exactly_zero_when_ball_is_valid():
    fine = _fine_arrays()
    coarse_live = np.full(4, .95)
    coarse_home = np.full(4, .05)
    blended = coarse_context.blend_coarse_context(fine, coarse_live, coarse_home)
    for index in (0, 3):
        assert blended["neural_live"][index] == fine["neural_live"][index]
        assert blended["neural_home"][index] == fine["neural_home"][index]
    assert np.array_equal(blended["neural_boundary"], fine["neural_boundary"])
    assert np.array_equal(blended["ref_live"], fine["ref_live"])


def test_seed_logit_average_and_bad_shape():
    p = np.array([.2, .75])
    q = np.array([.8, .25])
    averaged = coarse_stability._average_logits(p, q)
    assert np.allclose(averaged, [.5, .5], atol=1e-12)
    with pytest.raises(ValueError, match="different native grids"):
        coarse_stability._average_logits(p, q[:1])


def test_baseline_kind_must_be_explicit_before_input_access(tmp_path):
    with pytest.raises(ValueError, match="explicitly frozen"):
        coarse_stability.run_coarse_stability(
            tmp_path / "campaign", tmp_path / "polish", tmp_path / "bagging",
            tmp_path / "ema", tmp_path / "data", tmp_path / "target", tmp_path / "out",
            baseline_kind=None,
        )
