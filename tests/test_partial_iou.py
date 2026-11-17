"""Check duration boundaries, abstentions and temporal overlap semantics."""

import numpy as np
import pytest

from evaluation import possession_metrics as official
from scripts.analyze_partial_iou import score_iou


def test_iou_counts_boundary_shift_and_duration_bins():
    gt = np.array([official.H] * 25 + [official.D] * 5 + [official.A] * 50)
    pred = np.array([official.H] * 20 + [official.D] * 10 + [official.A] * 50)
    scores = score_iou(gt, pred, thresholds=(.8, .9))
    assert scores["0.8"]["n_matched"] == 2
    assert scores["0.8"]["f1"] == 1
    strict = scores["0.9"]
    assert (strict["n_matched"], strict["n_spurious"], strict["n_missed"]) == (1, 1, 1)
    assert strict["f1"] == pytest.approx(.5)
    assert strict["duration_bins"]["1_to_2s"] == {"gt": 1, "matched": 0}
    assert strict["duration_bins"]["2_to_5s"] == {"gt": 1, "matched": 1}


def test_abstentions_fill_inside_runs_and_unknown_truth_splits_spells():
    gt = np.array([official.H] * 10 + [official.X] + [official.H] * 10)
    pred = np.array([official.H] * 5 + [official.X] * 5 + [official.A]
                    + [official.H] * 10)
    metrics = score_iou(gt, pred, thresholds=(.5,))["0.5"]
    assert metrics["n_matched"] == 2
    assert metrics["n_spurious"] == metrics["n_missed"] == 0
    assert metrics["duration_bins"]["under_1s"] == {"gt": 2, "matched": 2}
