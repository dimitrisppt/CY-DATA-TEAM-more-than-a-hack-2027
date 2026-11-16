"""Inspect observation quality without loading any possession labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from possession.outages import gap_statistics
from possession.schema import FrameTable
from possession.storage import FRAME_ARRAYS, sha256_file, write_json


def load_input_half(path) -> FrameTable:
    """Open only input arrays; a cache need not contain target_* entries."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.pop("schema_version") != 1:
            raise ValueError("unsupported cache schema version")
        metadata["pitch"] = tuple(metadata["pitch"])
        return FrameTable(**{name: data[name] for name in FRAME_ARRAYS}, **metadata)


def _runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)


def inspect_half(frames: FrameTable) -> dict:
    """Summarize ball gaps and player coverage within declared graded runs."""
    spans = []
    total = missing = available_players = possible_players = speed_pairs = jumps = 0
    gap_lengths = []
    for start, stop in zip(*_runs(frames.graded)):
        valid = frames.ball_valid[start:stop]
        player_valid = frames.player_valid[start:stop]
        starts, stops = _runs(~valid)
        gap_lengths.extend(((stops - starts) / frames.fps).tolist())
        counts = player_valid.sum(axis=-1)
        total += len(valid)
        missing += int((~valid).sum())
        available_players += int(player_valid.sum())
        possible_players += player_valid.size
        xy = frames.players_xy[start:stop]
        adjacent_valid = player_valid[1:] & player_valid[:-1]
        speed = np.linalg.norm(xy[1:] - xy[:-1], axis=-1) * frames.fps
        speed_pairs += int(adjacent_valid.sum())
        jumps += int(((speed > 15.0) & adjacent_valid).sum())
        spans.append({
            "start_frame": int(frames.frame_id[start]),
            "stop_frame_exclusive": int(frames.frame_id[stop - 1]) + 1,
            "frames": len(valid), "ball_gaps": gap_statistics(~valid, frames.fps),
            "player_count_by_team": {
                "min": counts.min(axis=0).tolist(),
                "median": np.median(counts, axis=0).tolist(),
                "p05": np.percentile(counts, 5, axis=0).tolist(),
            },
        })
    edges = [0, .5, 1, 2, 5, 10, 20, 30, 60, float("inf")]
    histogram = np.histogram(gap_lengths, bins=edges)[0]
    return {
        "match_id": frames.match_id, "period": frames.period, "fps": frames.fps,
        "frames": len(frames.frame_id), "graded_frames": total,
        "ball_missing_frames": missing,
        "ball_missing_fraction": missing / total if total else None,
        "player_observation_fraction": (
            available_players / possible_players if possible_players else None),
        "slot_speed_pairs": speed_pairs, "slot_speed_above_15_mps": jumps,
        "slot_speed_outlier_fraction": jumps / speed_pairs if speed_pairs else None,
        "gap_histogram_seconds": [
            {"lower_inclusive": low, "upper_exclusive": high if np.isfinite(high) else None,
             "count": int(count)}
            for low, high, count in zip(edges[:-1], edges[1:], histogram)
        ],
        "graded_spans": spans,
    }


def audit_prepared(data_dir) -> dict:
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*_half*.npz"))
    if not paths:
        raise ValueError("no prepared halves found")
    reports = [inspect_half(load_input_half(path)) for path in paths]
    total = sum(item["graded_frames"] for item in reports)
    missing = sum(item["ball_missing_frames"] for item in reports)
    return {
        "protocol": "observation_quality_v1; input arrays only; no labels",
        "source": "prepared", "prepared_sha256": {p.name: sha256_file(p) for p in paths},
        "graded_frames": total, "ball_missing_frames": missing,
        "ball_missing_fraction": missing / total if total else None,
        "halves": reports,
    }


def audit_target(raw, metadata) -> dict:
    from games import GAME, GAMES
    from possession.cli import TARGET_SPANS
    from possession.io_atd import read_atd

    frames = read_atd(raw, metadata, n_frames=199319, pitch=GAMES[GAME]["pitch"],
                      fps=GAMES[GAME]["fps"], graded_spans=TARGET_SPANS)
    reports = []
    for period, (start, stop) in enumerate(TARGET_SPANS, 1):
        half = frames.subset(start, stop)
        half.period = period
        reports.append(inspect_half(half))
    return {
        "protocol": "observation_quality_v1; permitted target inputs only; no labels",
        "source": "target_atd",
        "input_sha256": {"raw": sha256_file(raw), "metadata": sha256_file(metadata)},
        "halves": reports,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data-dir", type=Path)
    inputs.add_argument("--raw", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if bool(args.raw) != bool(args.metadata):
        parser.error("--raw and --metadata must be supplied together")
    input_paths = ([args.raw, args.metadata] if args.raw else
                   list(args.data_dir.glob("*_half*.npz")))
    if args.output.resolve() in {p.resolve() for p in input_paths}:
        parser.error("output must not overwrite an input")
    report = (audit_prepared(args.data_dir) if args.data_dir else
              audit_target(args.raw, args.metadata))
    write_json(args.output, report)
    print(f"Input-quality report: {args.output}")


if __name__ == "__main__":
    main()
