"""Half-local error diagnostics using the official possession matching rules.

These are GT-bucket recalls, not stratified spell F1: predicted spells are matched
globally before a GT spell is assigned to a duration or ball visibility bucket.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from evaluation import possession_metrics as official

SPELL_DURATION_BINS = ("<1", "1-3", "3-10", ">=10")
BALL_GAP_BINS = ("0", "(0,1]", "(1,3]", "(3,10]", ">10")
MISSING_FRACTION_BINS = ("none", "(0,0.25]", "(0.25,0.75]", ">0.75")
IOU_KEYS = tuple(str(value) for value in official.SPELL_IOUS)


def _duration_bin(seconds: float) -> str:
    if seconds < 1:
        return "<1"
    if seconds < 3:
        return "1-3"
    if seconds < 10:
        return "3-10"
    return ">=10"


def _gap_bin(seconds: float) -> str:
    if seconds == 0:
        return "0"
    if seconds <= 1:
        return "(0,1]"
    if seconds <= 3:
        return "(1,3]"
    if seconds <= 10:
        return "(3,10]"
    return ">10"


def _fraction_bin(value: float) -> str:
    if value == 0:
        return "none"
    if value <= 0.25:
        return "(0,0.25]"
    if value <= 0.75:
        return "(0.25,0.75]"
    return ">0.75"


def _spell_bucket() -> dict:
    return {"n": 0, "matched": {key: 0 for key in IOU_KEYS},
            "recall": {key: None for key in IOU_KEYS}}


def _frame_bucket() -> dict:
    return {"n_frames": 0, "n_errors": 0, "error_rate": None}


def _empty_result() -> dict:
    return {
        "kind": "diagnostic_gt_spell_recall",
        "n_halves": 0,
        "n_graded_frames": 0,
        "n_gt_spells": 0,
        "n_pred_spells": 0,
        "by_duration": {key: _spell_bucket() for key in SPELL_DURATION_BINS},
        "by_max_ball_gap": {key: _spell_bucket() for key in BALL_GAP_BINS},
        "by_missing_fraction": {key: _spell_bucket() for key in MISSING_FRACTION_BINS},
        "potentially_censored": _spell_bucket(),
        "frame_errors": {
            "by_visibility": {key: _frame_bucket() for key in ("observed", "missing")},
            "by_gap_duration": {key: _frame_bucket() for key in BALL_GAP_BINS if key != "0"},
        },
        "transition_errors_at_12": {
            "n_gt": 0, "n_pred": 0, "n_matched": 0, "n_missed": 0,
            "n_spurious": 0, "missed_observed": 0, "missed_missing": 0,
            "spurious_observed": 0, "spurious_missing": 0,
        },
    }


def _finish(result: dict) -> dict:
    for group in ("by_duration", "by_max_ball_gap", "by_missing_fraction"):
        for bucket in result[group].values():
            bucket["recall"] = {
                key: bucket["matched"][key] / bucket["n"] if bucket["n"] else None
                for key in IOU_KEYS
            }
    bucket = result["potentially_censored"]
    bucket["recall"] = {
        key: bucket["matched"][key] / bucket["n"] if bucket["n"] else None
        for key in IOU_KEYS
    }
    for group in result["frame_errors"].values():
        for bucket in group.values():
            bucket["error_rate"] = (
                bucket["n_errors"] / bucket["n_frames"] if bucket["n_frames"] else None
            )
    return result


def _codes(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer code array")
    if not np.isfinite(array).all() or not np.isin(array, official.H + np.arange(4)).all():
        raise ValueError(f"{name} must contain only finite H/A/D/X codes 0..3")
    return array.astype(np.int8, copy=False)


def _add_spell(bucket: dict, flags: dict[str, np.ndarray], index: int) -> None:
    bucket["n"] += 1
    for key in IOU_KEYS:
        bucket["matched"][key] += int(flags[key][index])


def spell_error_diagnostics(
    gt_codes: np.ndarray, pred_codes: np.ndarray, ball_valid: np.ndarray, *, fps: float = 25.0,
) -> dict:
    """Diagnose one complete half; X boundaries stop spells and ball outage runs.

    Maximum gap duration uses the full graded missing-ball run overlapping a spell,
    while missing fraction counts only missing frames inside that spell. A spell
    touching either edge of a graded run is marked potentially censored.
    """
    gt = _codes("gt_codes", gt_codes)
    pred = _codes("pred_codes", pred_codes)
    valid = np.asarray(ball_valid)
    if valid.ndim != 1 or (valid.size and valid.dtype != np.bool_):
        raise ValueError("ball_valid must be a one-dimensional bool array")
    valid = valid.astype(bool, copy=False)
    if len(gt) != len(pred) or len(gt) != len(valid):
        raise ValueError("gt_codes, pred_codes, and ball_valid must have equal lengths")
    if isinstance(fps, (bool, str)) or not isinstance(fps, (int, float, np.number)):
        raise TypeError("fps must be finite and positive")
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")

    result = _empty_result()
    result["n_halves"] = 1
    runs = official.valid_runs(gt)
    filled = official.fill_abstentions(pred, runs)
    gt_spells = official.possession_spells(gt, runs)
    pred_spells = official.possession_spells(filled, runs)
    overlaps = official.spell_overlaps(gt_spells, pred_spells)
    flags = {key: official.matched_gt_flags(overlaps, len(gt_spells), threshold)
             for key, threshold in zip(IOU_KEYS, official.SPELL_IOUS, strict=True)}
    result["n_graded_frames"] = int(np.count_nonzero(gt != official.X))
    result["n_gt_spells"] = len(gt_spells)
    result["n_pred_spells"] = len(pred_spells)
    run_starts = {start for start, _ in runs}
    run_stops = {stop for _, stop in runs}

    # Build gaps within graded runs, so a GT X frame terminates an outage even
    # when the ball is also missing on both sides of it.
    gap_length = np.zeros(len(gt), dtype=np.int32)
    missing = (gt != official.X) & ~valid
    for run_start, run_stop in runs:
        window = missing[run_start:run_stop]
        edges = np.diff(np.r_[False, window, False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1),
                               strict=True):
            gap_length[run_start + start:run_start + stop] = int(stop - start)

    for index, (start, stop, _) in enumerate(gt_spells):
        _add_spell(result["by_duration"][_duration_bin((stop - start) / fps)], flags, index)
        max_gap = int(gap_length[start:stop].max()) if stop > start else 0
        _add_spell(result["by_max_ball_gap"][_gap_bin(max_gap / fps)], flags, index)
        fraction = float(np.count_nonzero(missing[start:stop])) / (stop - start)
        _add_spell(result["by_missing_fraction"][_fraction_bin(fraction)], flags, index)
        if start in run_starts or stop in run_stops:
            _add_spell(result["potentially_censored"], flags, index)

    errors = gt != pred  # Raw abstentions are frame errors, as in the official scorer.
    graded = gt != official.X
    for name, mask in (("observed", graded & valid), ("missing", missing)):
        bucket = result["frame_errors"]["by_visibility"][name]
        bucket["n_frames"] = int(np.count_nonzero(mask))
        bucket["n_errors"] = int(np.count_nonzero(errors & mask))
    gap_seconds = gap_length / fps
    for key, lower, upper in (("(0,1]", 0, 1), ("(1,3]", 1, 3),
                              ("(3,10]", 3, 10), (">10", 10, math.inf)):
        frame_mask = missing & (gap_seconds > lower) & (gap_seconds <= upper)
        bucket = result["frame_errors"]["by_gap_duration"][key]
        bucket["n_frames"] = int(np.count_nonzero(frame_mask))
        bucket["n_errors"] = int(np.count_nonzero(errors & frame_mask))

    gt_trans = official.transitions(gt, runs)
    pred_trans = official.transitions(filled, runs)
    pairs, missed, spurious = official.match_transitions(gt_trans, pred_trans, 12)
    transition = result["transition_errors_at_12"]
    transition.update(n_gt=len(gt_trans), n_pred=len(pred_trans), n_matched=len(pairs),
                      n_missed=len(missed), n_spurious=len(spurious))
    for name, transitions in (("missed", missed), ("spurious", spurious)):
        transition[f"{name}_missing"] = sum(bool(missing[frame]) for frame, _, _ in transitions)
        transition[f"{name}_observed"] = len(transitions) - transition[f"{name}_missing"]
    return _finish(result)


def merge_spell_diagnostics(parts: Iterable[dict]) -> dict:
    """Sum independent half diagnostics and derive rates from the pooled counts."""
    merged = _empty_result()
    for part in parts:
        if part.get("kind") != merged["kind"]:
            raise ValueError("parts must be spell error diagnostics")
        for key in ("n_halves", "n_graded_frames", "n_gt_spells", "n_pred_spells"):
            merged[key] += part[key]
        for group in ("by_duration", "by_max_ball_gap", "by_missing_fraction"):
            for name, target in merged[group].items():
                source = part[group][name]
                target["n"] += source["n"]
                for threshold in IOU_KEYS:
                    target["matched"][threshold] += source["matched"][threshold]
        target = merged["potentially_censored"]
        source = part["potentially_censored"]
        target["n"] += source["n"]
        for threshold in IOU_KEYS:
            target["matched"][threshold] += source["matched"][threshold]
        for group in ("by_visibility", "by_gap_duration"):
            for name, target in merged["frame_errors"][group].items():
                source = part["frame_errors"][group][name]
                target["n_frames"] += source["n_frames"]
                target["n_errors"] += source["n_errors"]
        for key in merged["transition_errors_at_12"]:
            merged["transition_errors_at_12"][key] += part["transition_errors_at_12"][key]
    return _finish(merged)
