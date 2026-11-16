import json

import numpy as np
import pytest

from evaluation import possession_metrics as official
from possession.diagnostics import merge_spell_diagnostics, spell_error_diagnostics


def _codes(label, n):
    return np.full(n, label, dtype=np.int8)


def test_long_spell_remains_one_global_match_across_multiple_ball_gaps():
    gt = _codes(official.H, 100)
    pred = gt.copy()
    valid = np.ones(100, dtype=bool)
    valid[10:20] = False
    valid[30:60] = False
    valid[70:90] = False
    result = spell_error_diagnostics(gt, pred, valid, fps=10)

    assert result["n_gt_spells"] == result["n_pred_spells"] == 1
    assert result["by_duration"][">=10"]["n"] == 1
    assert result["by_max_ball_gap"]["(1,3]"]["n"] == 1
    assert result["by_missing_fraction"]["(0.25,0.75]"]["n"] == 1
    assert result["by_max_ball_gap"]["(1,3]"]["recall"]["0.5"] == 1.0
    assert sum(row["n"] for row in result["by_max_ball_gap"].values()) == 1
    official_scores = official.spell_scores(gt, pred, official.valid_runs(gt))
    for threshold in official.SPELL_IOUS:
        assert result["by_duration"][">=10"]["matched"][str(threshold)] == (
            official_scores["per_iou"][threshold]["n_matched"]
        )
    assert result["frame_errors"]["by_gap_duration"]["(1,3]"]["n_frames"] == 50
    assert result["frame_errors"]["by_gap_duration"]["(0,1]"]["n_frames"] == 10
    assert result["potentially_censored"]["n"] == 1
    json.dumps(result, allow_nan=False)


def test_x_boundaries_separate_spells_and_gaps_and_abstentions_follow_official_rules():
    gt = np.array([0, 0, 3, 0, 0, 0, 2, 1, 1], dtype=np.int8)
    pred = np.array([0, 3, 0, 3, 0, 0, 2, 1, 1], dtype=np.int8)
    valid = np.array([False, False, False, False, False, True, True, True, True])
    result = spell_error_diagnostics(gt, pred, valid, fps=1)

    # The two H spells cannot merge through GT X; the predicted Xs are filled
    # only for temporal matching, while still counted as raw frame mistakes.
    assert result["n_gt_spells"] == 3
    assert result["n_pred_spells"] == 3
    assert result["by_max_ball_gap"]["(1,3]"]["n"] == 2
    assert result["by_max_ball_gap"]["0"]["n"] == 1
    assert result["by_max_ball_gap"]["(1,3]"]["matched"]["0.7"] == 2
    assert result["frame_errors"]["by_visibility"]["missing"]["n_frames"] == 4
    assert result["frame_errors"]["by_visibility"]["missing"]["n_errors"] == 2
    assert result["frame_errors"]["by_gap_duration"]["(1,3]"]["n_frames"] == 4
    assert sum(row["n_frames"] for row in result["frame_errors"]["by_gap_duration"].values()) == 4
    assert result["transition_errors_at_12"]["n_gt"] == 2


def test_global_one_to_one_match_is_preserved_in_each_gt_bucket():
    gt = np.array([0] * 5 + [2] + [0] * 20, dtype=np.int8)
    pred = _codes(official.H, len(gt))
    valid = np.ones(len(gt), dtype=bool)
    result = spell_error_diagnostics(gt, pred, valid, fps=5)
    # One predicted H spell can match at most one of the two GT H spells.
    assert result["n_gt_spells"] == 2
    assert sum(bucket["matched"]["0.3"] for bucket in result["by_duration"].values()) == 1


def test_merge_recomputes_weighted_rates_without_joining_halves():
    one = spell_error_diagnostics(_codes(0, 10), _codes(0, 10), np.ones(10, bool), fps=10)
    two = spell_error_diagnostics(_codes(0, 30), _codes(1, 30), np.zeros(30, bool), fps=10)
    three = spell_error_diagnostics(_codes(0, 10), _codes(1, 10), np.ones(10, bool), fps=10)
    merged = merge_spell_diagnostics([one, two, three])

    assert merged["n_halves"] == 3
    assert merged["n_gt_spells"] == 3
    assert merged["by_duration"]["1-3"]["n"] == 2
    assert merged["by_duration"]["1-3"]["recall"]["0.5"] == 0.5
    assert merged["by_duration"]["3-10"]["recall"]["0.5"] == 0.0
    assert merged["frame_errors"]["by_visibility"]["observed"]["error_rate"] == 0.5
    assert merged["frame_errors"]["by_visibility"]["missing"]["error_rate"] == 1.0
    assert merged["potentially_censored"]["n"] == 3
    json.dumps(merged, allow_nan=False)


def test_empty_and_short_inputs_have_json_safe_empty_bins():
    empty = spell_error_diagnostics([], [], [], fps=25)
    assert empty["n_gt_spells"] == 0
    assert empty["by_duration"]["<1"]["recall"]["0.5"] is None
    assert empty["frame_errors"]["by_visibility"]["observed"]["error_rate"] is None
    assert merge_spell_diagnostics([])["n_halves"] == 0
    short = spell_error_diagnostics([0], [3], [False])
    assert short["n_gt_spells"] == 1
    assert short["by_duration"]["<1"]["recall"]["0.5"] == 0.0
    assert short["frame_errors"]["by_visibility"]["missing"]["n_errors"] == 1
    json.dumps(empty, allow_nan=False)


@pytest.mark.parametrize("gt,pred,valid,fps", [
    ([0, 1], [0], [True, True], 25),
    ([0], [0], [], 25),
    ([[0]], [0], [True], 25),
    ([4], [0], [True], 25),
    ([np.nan], [0], [True], 25),
    ([0], [0], [1], 25),
    ([0], [0], [True], 0),
    ([0], [0], [True], np.inf),
])
def test_invalid_inputs_rejected(gt, pred, valid, fps):
    with pytest.raises(ValueError):
        spell_error_diagnostics(gt, pred, valid, fps=fps)


def test_fps_type_is_rejected():
    with pytest.raises(TypeError, match="fps"):
        spell_error_diagnostics([0], [0], [True], fps="25")
