"""Export the frozen campaign ensemble using tactical inputs only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from evaluation import possession_metrics as official
from experiments.campaign_decode import decode_arrays, reference_basis
from experiments.campaign_evaluation import _gap_seconds
from experiments.campaign_refit import verify_bundle
from experiments.robustness import _digest, _execution_environment, _source_hashes
from games import GAME, GAMES, require
from possession.cli import TARGET_SPANS, write_submission
from possession.features import build_features
from possession.io_atd import read_atd
from possession.predict import load_model_checkpoint, predict_features
from possession.storage import sha256_file, write_json

N_FRAMES = GAMES[GAME]["n_video_frames"]
CENTER_SIZE = 2048


def blend_neural(left, right, weight):
    """Label-free equivalent of campaign_search.blend_records, including endpoints."""
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("blend weight must lie in [0,1]")
    pairs = {}
    for name in ("live_probability", "home_probability", "boundary"):
        a, b = np.asarray(left[name]), np.asarray(right[name])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"invalid prediction shape or values: {name}")
        pairs[name] = a, b
    n = len(pairs["live_probability"][0])
    for name, (a, b) in pairs.items():
        expected = (n, 3, 3) if name == "boundary" else (n,)
        if a.shape != expected:
            raise ValueError(f"invalid prediction shape: {name}")
        if name != "boundary" and (np.any((a < 0) | (a > 1))
                                   or np.any((b < 0) | (b > 1))):
            raise ValueError(f"invalid probability values: {name}")
    output = {}
    for name in ("live_probability", "home_probability"):
        a, b = pairs[name]
        p = np.clip(np.asarray(a, dtype=np.float64), 1e-7, 1 - 1e-7)
        q = np.clip(np.asarray(b, dtype=np.float64), 1e-7, 1 - 1e-7)
        logits = weight * (np.log(p) - np.log1p(-p)) + (1 - weight) * (
            np.log(q) - np.log1p(-q))
        output[name] = 1. / (1. + np.exp(-np.clip(logits, -40, 40)))
        if weight == 1:
            output[name] = a.copy()
        elif weight == 0:
            output[name] = b.copy()
    a, b = pairs["boundary"]
    output["boundary"] = weight * a + (1 - weight) * b
    return output


def predict_ensemble_half(frames, models, selection):
    """Build full-half context, blend both members, then apply the frozen decoder."""
    if selection["kind"] != "convex_pair" or len(selection["members"]) != 2:
        raise ValueError("target export requires the frozen two-member ensemble")
    features = build_features(frames)
    predictions = [predict_features(features, models[member["id"]], device="cpu",
                                    center_size=CENTER_SIZE,
                                    halo=models[member["id"]].radius)
                   for member in selection["members"]]
    scores = blend_neural(*predictions, selection["blend_weight"])
    arrays = {"neural_live": scores["live_probability"],
              "neural_home": scores["home_probability"],
              "neural_boundary": scores["boundary"],
              "ball_valid": frames.ball_valid,
              "nearest_ball_seconds": _gap_seconds(frames.ball_valid, frames.fps),
              **reference_basis(frames, features)}
    return decode_arrays(arrays, selection["decoder"])


def infer_spans(frames, models, selection, spans):
    """Reset all temporal context at half boundaries and mark every outside frame X."""
    n = len(frames.frame_id)
    previous = 0
    if not spans:
        raise ValueError("at least one half span is required")
    for start, stop in spans:
        if not previous <= start < stop <= n:
            raise ValueError("half spans must be ordered, disjoint and inside the timeline")
        previous = stop
    codes = np.full(n, 3, dtype=np.int8)
    reports = []
    for period, (start, stop) in enumerate(spans, 1):
        half = frames.subset(start, stop)
        half.period = period
        path = predict_ensemble_half(half, models, selection)
        if path.shape != (stop - start,) or not np.isin(path, [0, 1, 2]).all():
            raise ValueError("decoder returned invalid half labels")
        codes[start:stop] = path
        reports.append({"period": period, "span": [start, stop],
                        "switches": int(np.count_nonzero(np.diff(path))),
                        "ball_observed_fraction": float(half.ball_valid.mean())})
    return codes, reports


def _verified_inputs(data_dir):
    data_dir = Path(data_dir).resolve()
    manifest_path = data_dir / "input-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("files", [])
    raw, metadata = require(["atd_raw", "atd_meta"], data_dir=data_dir)
    inputs = {}
    for name, value in (("raw", raw), ("metadata", metadata)):
        path = Path(value).resolve()
        if not path.is_relative_to(data_dir):
            raise ValueError("target input escapes data directory")
        relative = path.relative_to(data_dir).as_posix()
        matches = [record for record in records if record.get("path") == relative]
        if len(matches) != 1:
            raise ValueError(f"input manifest requires exactly one record: {relative}")
        record = matches[0]
        digest = sha256_file(path)
        if path.stat().st_size != record["bytes"] or digest != record["sha256"]:
            raise ValueError(f"target input hash or size differs: {relative}")
        inputs[name] = {"path": str(path), "sha256": digest, "bytes": record["bytes"]}
    inputs["manifest"] = {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
    return inputs


def _verify_execution(manifest):
    current = _source_hashes()
    changed = [path for path, digest in manifest["source_sha256"].items()
               if current.get(path) != digest]
    if changed:
        raise ValueError(f"bundle execution source differs: {changed}")
    if _execution_environment() != manifest["runtime"]:
        raise ValueError("bundle execution environment differs")


def _validate_output(path):
    table = official.read_label_csv(path)
    error = official.validate_submission(table, N_FRAMES)
    if error is not None:
        raise ValueError(f"official submission format check failed: {error}")
    codes = official.label_codes(table)
    inside = np.zeros(N_FRAMES, dtype=bool)
    for start, stop in TARGET_SPANS:
        inside[start:stop] = True
    if np.any(codes[inside] == 3) or np.any(codes[~inside] != 3):
        raise ValueError("submission labels do not respect the published half spans")
    return codes


def infer_campaign_submission(bundle_dir, data_dir, output_path):
    """Verify the portable bundle and automatic inputs, then export without target labels."""
    started = time.perf_counter()
    bundle_dir, data_dir, output = (Path(path).resolve()
                                   for path in (bundle_dir, data_dir, output_path))
    sidecar = output.with_suffix(".manifest.json")
    if output == sidecar:
        raise ValueError("output must be a CSV path distinct from its manifest")
    if any(path.is_relative_to(base) for path in (output, sidecar)
           for base in (bundle_dir, data_dir)):
        raise ValueError("output must be outside bundle and input directories")
    bundle = verify_bundle(bundle_dir)
    manifest, recipe = bundle["manifest"], bundle["recipe"]
    _verify_execution(manifest)
    inputs = _verified_inputs(data_dir)
    signature = {"bundle_sha256": sha256_file(bundle_dir / "manifest.json"),
                 "recipe_sha256": sha256_file(bundle_dir / "recipe.json"),
                 "inputs": inputs, "selection": recipe["selection"],
                 "rows": N_FRAMES, "spans": [list(span) for span in TARGET_SPANS],
                 "source_sha256": _source_hashes(), "runtime": _execution_environment(),
                 "center_size": CENTER_SIZE, "device": "cpu"}
    fingerprint = _digest(signature)
    if output.exists() or sidecar.exists():
        if not (output.is_file() and sidecar.is_file()):
            raise ValueError("existing output is incomplete, use a new output path")
        previous = json.loads(sidecar.read_text())
        if (previous.get("request_sha256") != fingerprint
                or previous.get("output", {}).get("sha256") != sha256_file(output)):
            raise ValueError("existing output belongs to a different or changed inference run")
        _validate_output(output)
        return previous
    models = {member_id: load_model_checkpoint(path, device="cpu")[0]
              for member_id, path in bundle["member_paths"].items()}
    frames = read_atd(inputs["raw"]["path"], inputs["metadata"]["path"], n_frames=N_FRAMES,
                      pitch=GAMES[GAME]["pitch"], fps=GAMES[GAME]["fps"], graded_spans=TARGET_SPANS)
    codes, halves = infer_spans(frames, models, recipe["selection"], TARGET_SPANS)
    write_submission(output, codes, n_expected=N_FRAMES)
    _validate_output(output)
    report = {"schema": "campaign_submission_v1", "ground_truth_used": False,
              "inference_overrides": {}, "request_sha256": fingerprint, "request": signature,
              "selection_integrity": {"selection_verified": True,
                                      "bundle_sha256": signature["bundle_sha256"]},
              "output": {"path": str(output), "sha256": sha256_file(output), "rows": N_FRAMES,
                         "label_counts": dict(zip("HADX", np.bincount(codes, minlength=4).tolist()))},
              "halves": halves, "elapsed_seconds": time.perf_counter() - started,
              "format_validated": True,
              "score_note": "Format validation is not a target accuracy measurement."}
    write_json(sidecar, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    try:
        report = infer_campaign_submission(args.bundle, args.data_dir, args.output)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Campaign submission failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["output"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
