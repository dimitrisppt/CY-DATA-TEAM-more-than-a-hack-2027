"""Describe target predictions against input visibility, without possession truth.

These are predicted label distributions, never accuracy or possession estimates
verified against truth. Missing-ball runs are clipped independently to each
published span; transitions never cross half boundaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from games import GAME, GAMES, require
from possession.cli import TARGET_SPANS
from possession.io_atd import read_atd
from possession.storage import sha256_file, write_json


def _counts(codes):
    return dict(zip("HADX", np.bincount(codes, minlength=4).tolist()))


def summarize(codes, ball_valid, *, fps, spans):
    codes, ball_valid = np.asarray(codes), np.asarray(ball_valid)
    if (codes.ndim != 1 or codes.dtype.kind not in "iu"
            or not np.isin(codes, [0, 1, 2, 3]).all()
            or ball_valid.shape != codes.shape or ball_valid.dtype.kind != "b"
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("aligned integer codes, boolean visibility and positive fps required")
    last_stop = 0
    included = np.zeros(len(codes), dtype=bool)
    halves = []
    for period, (start, stop) in enumerate(spans, 1):
        if not (last_stop <= start < stop <= len(codes)):
            raise ValueError("spans must be nonempty, ordered and disjoint")
        last_stop = stop
        included[start:stop] = True
        labels, visible = codes[start:stop], ball_valid[start:stop]
        edges = np.diff(np.r_[False, ~visible, False].astype(np.int8))
        gaps = [{"start": start + int(left), "stop": start + int(right),
                 "seconds": (int(right) - int(left)) / fps,
                 "predicted_label_counts": _counts(labels[left:right])}
                for left, right in zip(np.flatnonzero(edges == 1),
                                       np.flatnonzero(edges == -1), strict=True)]
        spell_count = len(official.possession_spells(labels, official.valid_runs(labels)))
        halves.append({
            "period": period, "span": [start, stop], "frames": len(labels),
            "predicted_label_counts": _counts(labels),
            "observed_ball": {"frames": int(visible.sum()),
                              "predicted_label_counts": _counts(labels[visible])},
            "missing_ball": {"frames": int((~visible).sum()),
                             "predicted_label_counts": _counts(labels[~visible])},
            "predicted_spells": spell_count,
            "predicted_label_changes": int(np.count_nonzero(labels[1:] != labels[:-1])),
            "missing_ball_runs": len(gaps),
            "longest_missing_ball_runs": sorted(gaps, key=lambda gap: -gap["seconds"])[:10],
        })
    if not halves:
        raise ValueError("at least one span required")
    return {
        "kind": "input_only_prediction_diagnostics", "ground_truth_used": False,
        "accuracy_evaluated": False, "rows": len(codes),
        "predicted_label_counts": _counts(codes),
        "included_frames": int(included.sum()),
        "abstentions_inside_published_spans": int(np.count_nonzero(codes[included] == 3)),
        "non_x_outside_published_spans": int(np.count_nonzero(codes[~included] != 3)),
        "halves": halves,
        "limits": "Predicted distributions only. Visibility is not live/dead truth. "
                  "Predicted spells treat X as a gap; the official scorer may bridge it.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw, metadata = require(["atd_raw", "atd_meta"], data_dir=args.data_dir)
    sources = {"submission": args.submission, "raw": Path(raw), "metadata": Path(metadata)}
    if args.output.resolve() in {path.resolve() for path in sources.values()}:
        raise ValueError("report must not overwrite an input")
    table = official.read_label_csv(args.submission)
    reason = official.validate_submission(table, GAMES[GAME]["n_video_frames"])
    if reason:
        raise ValueError(reason)
    frames = read_atd(raw, metadata, n_frames=len(table), pitch=GAMES[GAME]["pitch"],
                      fps=GAMES[GAME]["fps"], graded_spans=TARGET_SPANS)
    report = summarize(official.label_codes(table), frames.ball_valid,
                       fps=frames.fps, spans=TARGET_SPANS)
    report["inputs"] = {name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                        for name, path in sources.items()}
    report["audit_source_sha256"] = sha256_file(Path(__file__))
    write_json(args.output, report)
    print(f"Wrote input-only prediction diagnostics to {args.output}")


if __name__ == "__main__":
    main()
