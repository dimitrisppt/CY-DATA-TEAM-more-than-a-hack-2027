import numpy as np
import pytest

from evaluation import possession_metrics as official
from scripts.audit_iou_errors import audit_half, greedy_pairs, paired_audit


@pytest.mark.parametrize("gt,pred,category", [
    ([0] * 20 + [2] * 10, [1] * 20 + [2] * 10, "wrong_team"),
    ([0] * 20, [0] * 4 + [2] * 12 + [0] * 4, "split"),
    ([0] * 4 + [2] * 12 + [0] * 4, [0] * 20, "merge"),
    ([0] * 20 + [2] * 20, [2] * 15 + [0] * 20 + [2] * 5, "timing"),
    ([2] * 30, [2] * 10 + [0] * 10 + [2] * 10, "phantom"),
])
def test_operational_error_categories_and_official_accounting(gt, pred, category):
    rows, _, _, _ = audit_half(np.array(gt), np.array(pred))
    assert {r["category"] for r in rows} == {category}
    assert all(not r["matched"] for r in rows)


def test_greedy_matching_ties_follow_official_order():
    overlaps = [(0.5, 0, 0), (0.5, 0, 1), (0.5, 1, 0)]
    assert greedy_pairs(overlaps) == [(0, 0, .5)]
    assert sum(official.matched_gt_flags(overlaps, 2, .5)) == 1


def test_paired_changes_account_for_added_and_removed_errors():
    gt = np.array([0] * 20 + [2] * 20 + [1] * 20 + [2] * 20)
    control = np.array([0] * 20 + [2] * 20 + [0] * 20 + [2] * 20)
    candidate = np.array([0] * 20 + [2] * 20 + [1] * 20 + [0] * 20)
    left, right, changes = paired_audit(gt, control, candidate)
    assert sum(not r["matched"] for r in left) == sum(not r["matched"] for r in right) == 1
    assert changes["new_spurious"] == changes["removed_spurious"] == 1
    assert changes["recovered_gt"] == changes["recovered_short_gt"] == 1
    assert changes["lost_gt"] == 0


def test_unscored_gap_does_not_generate_spurious_events():
    gt = np.array([0] * 20 + [3] * 20 + [1] * 20)
    pred = np.array([0] * 20 + [2] * 20 + [1] * 20)
    rows, _, _, _ = audit_half(gt, pred)
    assert len(rows) == 2 and all(r["matched"] for r in rows)
