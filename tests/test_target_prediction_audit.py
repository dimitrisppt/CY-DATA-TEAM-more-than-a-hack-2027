import numpy as np
import pytest

from scripts.audit_target_predictions import summarize


def test_gaps_and_transitions_reset_at_each_published_span():
    # Missing ball across both spans must remain two separate observation gaps.
    codes = np.array([3, 0, 0, 2, 1, 1, 3], dtype=np.int8)
    report = summarize(codes, np.zeros(7, dtype=bool), fps=2, spans=((1, 3), (3, 6)))
    first, second = report["halves"]
    assert first["longest_missing_ball_runs"] == [
        {"start": 1, "stop": 3, "seconds": 1,
         "predicted_label_counts": {"H": 2, "A": 0, "D": 0, "X": 0}}]
    assert second["missing_ball_runs"] == 1
    assert second["longest_missing_ball_runs"][0]["seconds"] == 1.5
    assert first["predicted_label_changes"] == 0
    assert second["predicted_label_changes"] == 1
    assert report["accuracy_evaluated"] is False
    assert report["ground_truth_used"] is False


def test_visibility_buckets_do_not_treat_missing_ball_as_dead():
    codes = np.array([0, 1, 2, 3, 2], dtype=np.int8)
    report = summarize(codes, np.array([True, False, True, False, False]),
                       fps=25, spans=((0, 4),))
    half = report["halves"][0]
    assert half["observed_ball"]["predicted_label_counts"] == {
        "H": 1, "A": 0, "D": 1, "X": 0}
    assert half["missing_ball"]["predicted_label_counts"] == {
        "H": 0, "A": 1, "D": 0, "X": 1}
    assert report["abstentions_inside_published_spans"] == 1
    assert report["non_x_outside_published_spans"] == 1


@pytest.mark.parametrize("spans", [(), ((0, 3), (2, 4)), ((-1, 3),), ((0, 5),)])
def test_invalid_spans_are_rejected(spans):
    with pytest.raises(ValueError):
        summarize(np.zeros(4, dtype=np.int8), np.ones(4, dtype=bool), fps=25, spans=spans)


def test_fractional_labels_are_rejected_before_counting():
    with pytest.raises(ValueError, match="integer codes"):
        summarize(np.array([0., 1.5]), np.ones(2, dtype=bool), fps=25, spans=((0, 2),))
