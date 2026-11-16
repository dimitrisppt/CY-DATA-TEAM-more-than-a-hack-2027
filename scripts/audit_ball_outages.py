"""Post-selection synthetic ball-outage stress audit for one frozen outer match.

This script never selects a model or changes a frozen recipe. Its masks depend
only on source ball availability, half length, period, and fixed constants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from experiments.evaluate import _join_with_x, _target_codes
from possession.calibrate import CalibrationConfig, apply_temperature_calibration
from possession.decode import decode_half, emissions_from_probabilities
from possession.ensemble import combine_scores
from possession.predict import load_model_checkpoint, predict_model
from possession.prepared import verify_prepared
from possession.reference import reference_scores
from possession.storage import load_half, sha256_file, write_json

DROP_NUMERATOR = 41
DROP_DENOMINATOR = 100
MIN_BLOCK_FRAMES = 25
MAX_BLOCK_FRAMES = 250
BASE_SEED = 2026


def generate_outage_mask(ball_valid: np.ndarray, period: int) -> tuple[np.ndarray, int]:
    """Drop min(floor(.41 * frames), observed frames) in 1–10 second blocks.

    A proposal starts at an as-yet-observed, not-yet-dropped frame, so each
    iteration makes progress. Overlap is deduplicated; the final proposal is
    truncated to the exact target. No target labels enter this function.
    """
    valid = np.asarray(ball_valid, dtype=bool)
    if valid.ndim != 1 or not isinstance(period, int) or period not in (1, 2):
        raise ValueError("ball_valid must be one-dimensional and period must be 1 or 2")
    target = min((DROP_NUMERATOR * len(valid)) // DROP_DENOMINATOR, int(valid.sum()))
    dropped = np.zeros(len(valid), dtype=bool)
    rng = np.random.default_rng(BASE_SEED + period)
    proposals = 0
    count = 0
    while count < target:
        remaining = np.flatnonzero(valid & ~dropped)
        start = int(rng.choice(remaining))
        length = int(rng.integers(MIN_BLOCK_FRAMES, MAX_BLOCK_FRAMES + 1))
        candidates = np.flatnonzero(valid[start:min(start + length, len(valid))]
                                    & ~dropped[start:min(start + length, len(valid))]) + start
        chosen = candidates[:target - count]
        dropped[chosen] = True
        count += len(chosen)
        proposals += 1
    return dropped, proposals


def _mask_sha256(mask: np.ndarray) -> str:
    packed = np.packbits(mask, bitorder="little")
    payload = len(mask).to_bytes(8, "big") + packed.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _gap_stats(missing: np.ndarray, fps: float) -> dict:
    padded = np.r_[False, np.asarray(missing, dtype=bool), False].astype(np.int8)
    changes = np.diff(padded)
    lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    if not len(lengths):
        return {"count": 0, "min_frames": 0, "median_frames": 0.0,
                "mean_frames": 0.0, "p95_frames": 0.0, "max_frames": 0,
                "max_seconds": 0.0}
    return {"count": len(lengths), "min_frames": int(lengths.min()),
            "median_frames": float(np.median(lengths)),
            "mean_frames": float(np.mean(lengths)),
            "p95_frames": float(np.percentile(lengths, 95)),
            "max_frames": int(lengths.max()),
            "max_seconds": float(lengths.max() / fps)}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _context(data_dir: Path, outer_dir: Path):
    frozen_path = outer_dir / "frozen_recipe.json"
    report_path = outer_dir / "outer_report.json"
    frozen = _read_json(frozen_path)
    report = _read_json(report_path)
    report_hash = _read_json(outer_dir / "outer_report.sha256.json")
    report_sha = sha256_file(report_path)
    frozen_sha = sha256_file(frozen_path)
    outer = frozen.get("outer_test")
    if (not isinstance(outer, str) or not outer
            or report_hash.get("sha256") != report_sha
            or report.get("outer_test") != outer
            or report.get("outer_train") != frozen.get("outer_train")
            or report.get("selection_fingerprint") != frozen.get("selection_fingerprint")
            or report.get("frozen_recipe_sha256") != frozen_sha
            or report.get("predictor") != frozen.get("selected_pooling")
            or not isinstance(report.get("official"), dict)):
        raise ValueError("frozen recipe and completed outer report disagree")

    paths = [data_dir / f"{outer}_half{period}.npz" for period in (1, 2)]
    if (set(data_dir.glob(f"{outer}_half*.npz")) != set(paths)
            or any(not path.is_file() for path in paths)):
        raise ValueError("outer match requires exactly native halves 1 and 2")
    prepared_hashes = {path.name: sha256_file(path) for path in paths}
    if report.get("prepared_sha256") != prepared_hashes:
        raise ValueError("outer prepared bytes differ from completed report")
    verify_prepared(data_dir, paths)

    summary_path = outer_dir / "summary.json"
    hpo_fp_path = outer_dir / "fingerprint.json"
    decoder_summary_path = outer_dir / "decoder_summary.json"
    summary = _read_json(summary_path)
    hpo_fp = _read_json(hpo_fp_path)
    decoder_summary = _read_json(decoder_summary_path)
    if (summary.get("outer_test") != [outer]
            or summary.get("fingerprint_sha256") != hpo_fp.get("sha256")
            or frozen.get("hpo_fingerprint_sha256") != hpo_fp.get("sha256")
            or decoder_summary.get("fingerprint_sha256")
               != frozen.get("decoder_fingerprint_sha256")):
        raise ValueError("study or decoder ancestry differs from frozen recipe")

    hpo_database = outer_dir / "study.sqlite3"
    selected_decoder = frozen.get("selected_decoder_trial", {}).get("pooling")
    if selected_decoder not in ("mean", "ball_attention"):
        raise ValueError("frozen decoder pooling is invalid")
    decoder_database = outer_dir / f"decoder_{selected_decoder}.sqlite3"
    if not hpo_database.is_file() or not decoder_database.is_file():
        raise FileNotFoundError("source HPO or decoder study database is missing")

    weight = frozen.get("ensemble", {}).get("neural_weight")
    if weight is None or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("invalid frozen neural weight")
    checkpoint_path = outer_dir / "selected.pt"
    model = None
    checkpoint_sha = None
    if weight > 0:
        if frozen.get("selected_model_recipe") is None or not checkpoint_path.is_file():
            raise ValueError("positive frozen neural weight requires selected checkpoint")
        checkpoint_sha = sha256_file(checkpoint_path)
        if report.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("selected checkpoint differs from completed outer report")
        progress = _read_json(outer_dir / "refit_progress.json")
        if progress.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("selected checkpoint differs from refit progress")
        model, payload = load_model_checkpoint(checkpoint_path, device="cpu")
        config = payload.get("training_config", {})
        if (config.get("train_matches") != frozen.get("outer_train")
                or outer in config.get("train_matches", [])
                or config.get("validation_matches") != []
                or config.get("outer_test") != outer
                or config.get("selection_fingerprint") != frozen.get("selection_fingerprint")
                or config.get("seed") != frozen.get("refit_seed")
                or config.get("decoder") != frozen.get("decoder")
                or config.get("calibration") != frozen.get("calibration")
                or config.get("ensemble") != frozen.get("ensemble")
                or config.get("steps") != frozen.get("resources", {}).get("full_budget_steps")):
            raise ValueError("selected checkpoint split or configuration differs from freeze")
    elif (checkpoint_path.exists() or report.get("checkpoint_sha256") is not None
          or frozen.get("selected_model_recipe") is not None):
        raise ValueError("reference-only freeze unexpectedly has a neural checkpoint")

    provenance = {
        "outer_report_sha256": report_sha,
        "frozen_recipe_sha256": frozen_sha,
        "hpo_summary_sha256": sha256_file(summary_path),
        "hpo_fingerprint_sha256": hpo_fp["sha256"],
        "hpo_study_sha256": sha256_file(hpo_database),
        "decoder_summary_sha256": sha256_file(decoder_summary_path),
        "decoder_fingerprint_sha256": frozen["decoder_fingerprint_sha256"],
        "selected_decoder_study_sha256": sha256_file(decoder_database),
        "selected_checkpoint_sha256": checkpoint_sha,
        "prepared_sha256": prepared_hashes,
        "audit_script_sha256": sha256_file(Path(__file__)),
    }
    return frozen, report, paths, model, provenance


def audit(data_dir, study_dir, output=None) -> dict:
    """Write one post-selection diagnostic, never feeding its score into selection."""
    data_dir = Path(data_dir).resolve()
    outer_dir = Path(study_dir).resolve()
    frozen, clean_report, paths, model, provenance = _context(data_dir, outer_dir)
    calibration = CalibrationConfig(**frozen["calibration"])
    weight = float(frozen["ensemble"]["neural_weight"])
    selected_parts, reference_parts, gt_parts, half_reports = [], [], [], []
    for period, path in enumerate(paths, 1):
        frames, targets = load_half(path)
        if (not len(frames.frame_id) or frames.match_id != frozen["outer_test"]
                or frames.period != period or frames.fps != official.FPS):
            raise ValueError("outer prepared half identity or FPS differs")
        dropped, proposals = generate_outage_mask(frames.ball_valid, period)
        stressed = frames.subset(0, len(frames.frame_id))
        stressed.ball_valid[dropped] = False
        stressed.ball_xy[dropped] = 0
        stressed.validate()
        gt = _target_codes(frames, targets)
        reference = reference_scores(stressed)
        neural = reference if model is None else predict_model(
            stressed, model, device="cpu",
            center_size=frozen["selected_model_recipe"]["training_args"]["center_size"],
            halo=model.radius,
        )
        combined = combine_scores(neural, reference, weight)
        live, home, boundary = apply_temperature_calibration(
            combined["live_probability"], combined["home_probability"],
            combined["boundary"], calibration,
        )
        selected = decode_half(
            emissions_from_probabilities(live, home),
            boundary * frozen["decoder"]["boundary_scale"],
            switch_cost=frozen["decoder"]["switch_cost"],
        )
        fixed_reference = decode_half(
            emissions_from_probabilities(reference["live_probability"],
                                         reference["home_probability"]),
            reference["boundary"], switch_cost=2.0,
        )
        target_count = min((DROP_NUMERATOR * len(dropped)) // DROP_DENOMINATOR,
                           int(frames.ball_valid.sum()))
        if (int(dropped.sum()) != target_count
                or not np.array_equal(stressed.ball_valid, frames.ball_valid & ~dropped)):
            raise AssertionError("synthetic outage mask failed its fixed count or union contract")
        half_reports.append({
            "period": period, "n_frames": len(dropped),
            "original_observed_ball_frames": int(frames.ball_valid.sum()),
            "requested_drop_frames": (DROP_NUMERATOR * len(dropped)) // DROP_DENOMINATOR,
            "actually_dropped_frames": int(dropped.sum()),
            "stressed_missing_ball_frames": int((~stressed.ball_valid).sum()),
            "stressed_missing_fraction": float(np.mean(~stressed.ball_valid)),
            "proposal_count": proposals, "mask_sha256": _mask_sha256(dropped),
            "new_outage_gaps": _gap_stats(dropped, frames.fps),
            "all_stressed_missing_gaps": _gap_stats(~stressed.ball_valid, frames.fps),
            "selected_official": official.evaluate(gt, selected, bootstrap_n=0),
            "fixed_reference_official": official.evaluate(gt, fixed_reference, bootstrap_n=0),
        })
        gt_parts.append(gt)
        selected_parts.append(selected)
        reference_parts.append(fixed_reference)

    result = {
        "protocol": "post-selection synthetic 41-percent ball-outage diagnostic v1",
        "selection_use": "diagnostic_only_no_model_change_or_promotion",
        "outer_test": frozen["outer_test"],
        "mask_rule": {"fraction_numerator": DROP_NUMERATOR,
                      "fraction_denominator": DROP_DENOMINATOR,
                      "block_frames_inclusive": [MIN_BLOCK_FRAMES, MAX_BLOCK_FRAMES],
                      "seed_rule": "2026 + half period",
                      "overlap": "deduplicated; final proposal truncated to exact drop count"},
        "provenance": provenance,
        "clean_selected_official_from_saved_outer_report": clean_report["official"],
        "clean_fixed_reference_official_from_saved_outer_report":
            clean_report["reference_official"],
        "stressed_selected_official": official.evaluate(
            _join_with_x(gt_parts), _join_with_x(selected_parts), bootstrap_n=0),
        "stressed_fixed_reference_official": official.evaluate(
            _join_with_x(gt_parts), _join_with_x(reference_parts), bootstrap_n=0),
        "halves": half_reports,
    }
    output_path = (outer_dir / "ball-outage-audit.json" if output is None
                   else Path(output).resolve())
    default_path = outer_dir / "ball-outage-audit.json"
    if (output_path != default_path and output_path.exists()
            or output_path.is_relative_to(data_dir)
            or output_path.is_relative_to(outer_dir) and output_path != default_path):
        raise ValueError("output must be the default audit file or a new external file")
    write_json(output_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="prepared DFL half cache directory")
    parser.add_argument("--study-dir", required=True, type=Path,
                        help="one outer directory containing frozen_recipe.json")
    parser.add_argument("--output", type=Path,
                        help="default: <study-dir>/ball-outage-audit.json")
    args = parser.parse_args(argv)
    result = audit(args.data_dir, args.study_dir, args.output)
    print(json.dumps({"output": str(args.output or args.study_dir / "ball-outage-audit.json"),
                      "outer_test": result["outer_test"],
                      "selection_use": result["selection_use"]}, sort_keys=True))


if __name__ == "__main__":
    main()
