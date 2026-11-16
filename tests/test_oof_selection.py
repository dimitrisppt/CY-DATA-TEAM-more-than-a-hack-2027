"""Saved OOF selection uses paired halves and immutable prediction evidence."""

import copy
import json

import numpy as np
import pytest

from experiments.oof_selection import (
    LONG_SCENARIO,
    SCENARIOS,
    decode_oof_half,
    score_oof,
    select_oof_decoder,
)
from experiments.prediction_store import load_prediction_bundle, save_prediction_bundle

BASELINE = {
    "ensemble": {"neural_weight": 0.5},
    "calibration": {"live_temperature": 1.0, "owner_temperature": 1.0},
    "decoder": {"switch_cost": 2.0, "boundary_scale": 0.0},
}


@pytest.fixture
def records(tmp_path):
    matches = [f"M{index}" for index in range(6)]
    output = []
    for seed in (101, 102, 103):
        for match in matches:
            fold_index = int(match[1:]) // 2
            group = matches[2 * fold_index:2 * fold_index + 2]
            for scenario in SCENARIOS:
                for period in (1, 2):
                    gt = np.array([0] * 4 + [2] * 2 + [1] * 4, dtype=np.int8)
                    neural = {"live_probability": np.array([.9] * 4 + [.1] * 2 + [.9] * 4),
                              "home_probability": np.array([.9] * 4 + [.5] * 2 + [.1] * 4),
                              "boundary": np.zeros((10, 3, 3))}
                    reference = {"live_probability": np.array([.85] * 4 + [.15] * 2 + [.85] * 4),
                                 "home_probability": np.array([.8] * 4 + [.5] * 2 + [.2] * 4),
                                 "boundary": np.zeros((10, 3, 3))}
                    visible = np.ones(10, dtype=bool)
                    if scenario != "clean":
                        visible[2:5] = False
                    nearest = np.where(visible, 0.0, 1.0)
                    bundle = save_prediction_bundle(
                        tmp_path / f"{seed}_{match}_{scenario}_{period}.npz",
                        neural=neural, reference=reference, gt=gt, ball_valid=visible,
                        nearest_ball_seconds=nearest, frame_id=np.arange(100, 110),
                        match=match, period=period, scenario=scenario,
                    )
                    output.append({"seed": seed, "match": match, "period": period,
                                   "scenario": scenario, "prediction": bundle,
                                   "checkpoint_sha256": f"{seed * 10 + fold_index:064x}",
                                   "training_matches": [name for name in matches if name not in group],
                                   "validation_matches": group,
                                   "protocol_sha256": "a" * 64})
    return output


def test_static_selection_replays_and_resumes_from_saved_bundles(records, tmp_path):
    directory = tmp_path / "selection"
    selected = select_oof_decoder(records, baseline_frozen=BASELINE, output_dir=directory,
                                  mode="static", seed=9, n_trials=4)
    assert len(selected["trials"]) == 4
    assert selected["frozen"]["decoder"]["boundary_scale"] == 0
    assert selected["best_scores"]["objective"] == 1.0
    assert selected["best_scores"]["means"]["clean"] == 1.0
    assert set(selected["best_scores"]["by_seed"]) == {"101", "102", "103"}
    assert set(selected["best_scores"]["by_seed"]["101"]["by_match"]) == {
        f"M{index}" for index in range(6)}
    assert selected == select_oof_decoder(records, baseline_frozen=BASELINE,
                                          output_dir=directory, mode="static", seed=9, n_trials=4)
    assert json.loads((directory / "selection.json").read_text()) == selected
    json.dumps(selected, allow_nan=False)


def test_constant_quality_replays_static_and_quality_selection(records, tmp_path):
    arrays = load_prediction_bundle(records[0]["prediction"])
    static = BASELINE
    quality = {**BASELINE, "ensemble": {"mode": "quality", "visible_neural_weight": .5,
                                       "missing_neural_weight": .5, "transition_seconds": 2.0}}
    np.testing.assert_array_equal(decode_oof_half(arrays, static), decode_oof_half(arrays, quality))
    assert score_oof(records, static) == score_oof(records, quality)
    selected = select_oof_decoder(records, baseline_frozen=BASELINE,
                                  output_dir=tmp_path / "quality", mode="quality", n_trials=4)
    assert len(selected["trials"]) == 4
    assert selected["frozen"]["ensemble"]["mode"] == "quality"
    assert selected["best_scores"]["objective"] == 1.0


def test_fresh_seed_confirmation_accepts_complete_long_gap_grid(records, tmp_path):
    confirmation = copy.deepcopy([record for record in records if record["seed"] == 101])
    for record in confirmation:
        record["seed"] = 13
        record["checkpoint_sha256"] = "d" * 64
    short = score_oof(confirmation, BASELINE)
    assert set(short["by_seed"]) == {"13"}
    assert set(short["means"]["transition_f1_at_12"]) == set(SCENARIOS)

    long_records = []
    for record in confirmation:
        if record["scenario"] != "clean":
            continue
        arrays = load_prediction_bundle(record["prediction"])
        n = len(arrays["gt"])
        visible = np.zeros(n, dtype=bool)
        bundle = save_prediction_bundle(
            tmp_path / f"long_{record['match']}_{record['period']}.npz",
            neural={"live_probability": arrays["neural_live"],
                    "home_probability": arrays["neural_home"],
                    "boundary": np.zeros((n, 3, 3))},
            reference={"live_probability": arrays["reference_live"],
                       "home_probability": arrays["reference_home"],
                       "boundary": np.zeros((n, 3, 3))},
            gt=arrays["gt"], ball_valid=visible,
            nearest_ball_seconds=np.full(n, 10.0), frame_id=arrays["frame_id"],
            match=record["match"], period=record["period"], scenario=LONG_SCENARIO,
        )
        long_records.append({**record, "scenario": LONG_SCENARIO, "prediction": bundle})
    complete = confirmation + long_records
    report = score_oof(complete, BASELINE)
    assert report["objective"] == short["objective"]
    assert LONG_SCENARIO in report["means"]
    assert LONG_SCENARIO in report["means"]["transition_f1_at_12"]
    assert LONG_SCENARIO in report["by_seed"]["13"]["by_match"]["M0"]
    with pytest.raises(ValueError, match="grid"):
        score_oof(complete[:-1], BASELINE)
    with pytest.raises(ValueError, match="three"):
        select_oof_decoder(confirmation, baseline_frozen=BASELINE,
                           output_dir=tmp_path / "invalid", n_trials=4)
    extra_selection = [{**record, "seed": 101,
                        "checkpoint_sha256": f"{1010 + int(record['match'][1:]) // 2:064x}"}
                       for record in long_records]
    with pytest.raises(ValueError, match="selection permits only"):
        select_oof_decoder(records + extra_selection, baseline_frozen=BASELINE,
                           output_dir=tmp_path / "invalid2", n_trials=4)


def test_identity_and_training_proof_fail_closed(records, tmp_path):
    changed = copy.deepcopy(records)
    changed[0]["prediction"]["match"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        score_oof(changed, BASELINE)
    changed = copy.deepcopy(records)
    changed[0]["training_matches"].append(changed[0]["match"])
    with pytest.raises(ValueError, match="training"):
        score_oof(changed, BASELINE)
    changed = copy.deepcopy(records)
    changed[0]["validation_matches"] = ["M1"]
    with pytest.raises(ValueError, match="validation"):
        score_oof(changed, BASELINE)
    changed = copy.deepcopy(records)
    for item in changed:
        if item["seed"] == 103 and item["match"] == "M0":
            item["validation_matches"] = ["M0", "M2"]
            item["training_matches"] = ["M1", "M3", "M4", "M5"]
    with pytest.raises(ValueError, match="fold"):
        score_oof(changed, BASELINE)
    with pytest.raises(ValueError, match="duplicate"):
        score_oof(records + [records[0]], BASELINE)
    with pytest.raises(ValueError, match="grid"):
        score_oof(records[:-1], BASELINE)
    changed = copy.deepcopy(records)
    changed[0]["scenario"] = "long41_seed4101"
    with pytest.raises(ValueError, match="identity"):
        score_oof(changed, BASELINE)


def test_resume_rejects_changed_protocol_or_bundle(records, tmp_path):
    directory = tmp_path / "selection"
    select_oof_decoder(records, baseline_frozen=BASELINE, output_dir=directory, n_trials=4)
    changed = copy.deepcopy(records)
    changed[0]["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        select_oof_decoder(changed, baseline_frozen=BASELINE, output_dir=directory, n_trials=4)
    path = records[0]["prediction"]["path"]
    with open(path, "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        select_oof_decoder(records, baseline_frozen=BASELINE, output_dir=directory, n_trials=4)
