import numpy as np
import pytest

from experiments import live_dead_audit
from experiments.live_dead_audit import calibration_summary, full_gap_seconds, gap_strata


def test_calibration_counts_match_valid_probability_rows():
    p = np.asarray([.05, .22, .51, .78, .95, np.nan])
    y = np.asarray([0, 1, 1, 0, 1, 1])
    report = calibration_summary(p, y, bins=5)
    assert report["n"] == 5
    assert report["n_live"] == 3
    assert report["n_dead"] == 2
    assert report["n_invalid"] == 1
    assert sum(row["n"] for row in report["reliability"]) == 5
    assert report["brier"] == np.mean((p[:5] - y[:5]) ** 2)


def test_disjoint_gaps_are_stratified_by_each_full_contiguous_run():
    observed = np.asarray([1, 0, 0, 1, 0, 1, 0, 0, 0, 1], dtype=bool)
    seconds = full_gap_seconds(observed, fps=25)
    assert np.array_equal(seconds, [0, .08, .08, 0, .04, 0, .12, .12, .12, 0])
    strata = gap_strata(observed, fps=25)
    assert np.array_equal(strata, [0, 1, 1, 0, 1, 0, 1, 1, 1, 0])
    assert strata[1] == strata[2]
    assert strata[4] == 1
    assert strata[6] == strata[8]
    assert np.count_nonzero(strata) == 6


def test_calibration_rejects_finite_probability_outside_unit_interval():
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        calibration_summary(np.asarray([1.01]), np.asarray([1]))


def test_calibration_counts_all_invalid_rows():
    report = calibration_summary(np.asarray([np.nan, .4]), np.asarray([1, np.nan]))
    assert report["n_input"] == 2
    assert report["n"] == 0
    assert report["n_invalid"] == 2


def test_live_probability_reproduces_reference_blend_and_temperature(monkeypatch):
    monkeypatch.setattr(live_dead_audit.campaign_decode, "reference_from_basis",
                        lambda *_: {"live_probability": np.asarray([.5])})
    arrays = {"ref_dummy": np.asarray([0]), "neural_live": np.asarray([.8]),
              "nearest_ball_seconds": np.asarray([1.]), "ball_valid": np.asarray([False])}
    decoder = {"reference_mode": "default", "reference": {}, "transition_seconds": 2.,
               "blend_mode": "static", "neural_weight": .5, "live_temperature": 2.}
    observed = live_dead_audit.live_probability(arrays, decoder)
    expected = 1 / (1 + np.exp(-np.log(2) / 2))
    assert np.allclose(observed, [expected], rtol=0, atol=1e-12)
