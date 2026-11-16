"""Evaluate prepared external matches with the unchanged official scorer.

This module reports observed external performance only. Fold selection,
temperature choice, and switch-cost choice must happen on the designated
inner-validation match before an untouched outer match is evaluated.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from possession.calibrate import CalibrationConfig, apply_temperature_calibration
from possession.decode import decode_half, emissions_from_probabilities
from possession.prepared import verify_prepared
from possession.provenance import runtime_provenance
from possession.recipe import resolve_config
from possession.reference import reference_scores
from possession.storage import load_half, sha256_file


def _canonical_match(value):
    value = str(value)
    if re.fullmatch(r"J[0-9A-Z]{5}", value):
        return "DFL-MAT-" + value
    return value


def _target_codes(frames, targets):
    if len(frames.frame_id) != len(targets.state):
        raise ValueError("input and target length mismatch")
    known = np.asarray(targets.state_valid) & np.asarray(frames.graded)
    codes = np.full(len(frames.frame_id), official.X, dtype=np.int8)
    codes[known] = targets.state[known]
    return codes


def _edge_diagnostics(gt):
    """Count GT possessions touching half edges or unknown-label gaps.

    These counts are diagnostics, not a revised official score. A touched
    spell might be complete in reality; it simply cannot be certified from
    both observed boundaries in this half.
    """
    n = len(gt)
    runs = official.valid_runs(gt)
    spells = official.possession_spells(gt, runs)
    result = {
        "n_gt_spells": len(spells),
        "touch_half_start": 0,
        "touch_half_end": 0,
        "touch_unknown_left": 0,
        "touch_unknown_right": 0,
        "potentially_censored_gt_spells": 0,
        "bounded_gt_spells": 0,
    }
    for start, stop, _ in spells:
        half_start = start == 0
        half_end = stop == n
        unknown_left = start > 0 and gt[start - 1] == official.X
        unknown_right = stop < n and gt[stop] == official.X
        result["touch_half_start"] += int(half_start)
        result["touch_half_end"] += int(half_end)
        result["touch_unknown_left"] += int(unknown_left)
        result["touch_unknown_right"] += int(unknown_right)
        censored = half_start or half_end or unknown_left or unknown_right
        result["potentially_censored_gt_spells"] += int(censored)
        result["bounded_gt_spells"] += int(not censored)
    return result


def _join_with_x(parts):
    """Pool timelines while keeping temporal paths separate in the scorer."""
    if not parts:
        return np.empty(0, dtype=np.int8)
    separator = np.array([official.X], dtype=np.int8)
    arrays = []
    for index, part in enumerate(parts):
        if index:
            arrays.append(separator)
        arrays.append(np.asarray(part, dtype=np.int8))
    return np.concatenate(arrays)


def _period(path, match):
    match_name = re.fullmatch(re.escape(match) + r"_half(\d+)", path.stem)
    if match_name is None:
        raise ValueError(f"unexpected prepared half filename: {path.name}")
    return int(match_name.group(1))


def _load_checkpoint(path, matches, allow_training_overlap):
    import torch

    from possession.predict import _instantiate_checkpoint

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not all(
        key in payload for key in ("state_dict", "model_kwargs", "training_config")
    ):
        raise ValueError("checkpoint requires state_dict, model_kwargs, and training_config")
    training = payload["training_config"]
    if not isinstance(training, dict) or "train_matches" not in training:
        raise ValueError("checkpoint training_config requires train_matches")
    declared = training["train_matches"]
    if (not isinstance(declared, (list, tuple)) or not declared
            or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value)
                   for value in declared)):
        raise ValueError("checkpoint train_matches must be a nonempty sequence of match IDs")
    train_matches = {_canonical_match(value) for value in declared}
    validation_values = training.get("validation_matches", training.get(
        "val_matches", training.get("validation_match", training.get("val_match", []))))
    if validation_values is None:
        validation_values = []
    if isinstance(validation_values, str):
        validation_values = [validation_values]
    validation_matches = {_canonical_match(value) for value in validation_values}
    overlap = train_matches.intersection(matches)
    if overlap and not allow_training_overlap:
        raise ValueError(f"refusing to evaluate a training match: {sorted(overlap)}")
    model, _ = _instantiate_checkpoint(payload)
    calibration = CalibrationConfig(**training.get("calibration", {}))
    return (model, calibration, sorted(train_matches), sorted(overlap),
            sorted(validation_matches), sorted(validation_matches.intersection(matches)), training)


def evaluate_prepared(
    data_dir,
    *,
    matches,
    checkpoint=None,
    device="cpu",
    switch_cost=None,
    boundary_scale=None,
    neural_weight=None,
    center_size=512,
    halo=256,
    allow_training_overlap=False,
):
    """Score prepared 25 Hz halves, preserving every native source frame.

    `checkpoint=None` uses the untrained CPU reference. A checkpoint must
    declare training match IDs so an in-sample score cannot be mistaken for an
    outer evaluation. Explicit overlap is available for labeled diagnostics.
    """
    if isinstance(matches, str):
        matches = [matches]
    matches = [_canonical_match(value) for value in matches]
    if not matches or len(matches) != len(set(matches)):
        raise ValueError("matches must be nonempty and unique")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", match) for match in matches):
        raise ValueError("match IDs must contain only letters, numbers, _ or -")
    data_dir = Path(data_dir)
    model = calibration = None
    train_matches = overlap = validation_matches = validation_overlap = []
    training = {}
    if checkpoint is not None:
        (model, calibration, train_matches, overlap,
         validation_matches, validation_overlap, training) = _load_checkpoint(
            checkpoint, matches, allow_training_overlap)
    selected = resolve_config(training, switch_cost=switch_cost, boundary_scale=boundary_scale,
                              neural_weight=neural_weight, has_model=model is not None)
    switch_cost, boundary_scale = selected.switch_cost, selected.boundary_scale
    calibration = selected.calibration
    selected_paths = []
    for match in matches:
        paths = list(data_dir.glob(f"{match}_half*.npz"))
        if not paths:
            raise FileNotFoundError(f"no prepared halves for {match} in {data_dir}")
        selected_paths.extend(paths)
    dataset_provenance = verify_prepared(data_dir, selected_paths)
    report = {
        "protocol": "external prepared halves; original 25 Hz possession_metrics.evaluate",
        "predictor": "reference" if model is None else "checkpoint",
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "calibration": calibration.to_dict(),
        "ensemble": {"neural_weight": selected.neural_weight},
        "train_matches": train_matches,
        "training_overlap": overlap,
        "validation_matches": validation_matches,
        "validation_overlap": validation_overlap,
        "switch_cost": switch_cost if np.isscalar(switch_cost) else np.asarray(switch_cost).tolist(),
        "boundary_scale": float(boundary_scale),
        "half_and_match_pooling": "X separator for scorer temporal reset; native source frames retained",
        "by_match": {},
        "runtime": runtime_provenance(),
        "checkpoint_sha256": None if checkpoint is None else sha256_file(checkpoint),
        "prepared_sha256": {},
        "dataset_provenance": dataset_provenance,
    }
    all_gt, all_pred = [], []
    match_metrics = []
    for match in matches:
        paths = list(data_dir.glob(f"{match}_half*.npz"))
        if not paths:
            raise FileNotFoundError(f"no prepared halves for {match} in {data_dir}")

        paths.sort(key=lambda path, match=match: _period(path, match))
        periods = [_period(path, match) for path in paths]
        if len(periods) != len(set(periods)):
            raise ValueError(f"duplicate prepared period for {match}")
        half_reports, match_gt, match_pred = [], [], []
        for path in paths:
            report["prepared_sha256"][path.name] = sha256_file(path)
            frames, targets = load_half(path)
            if frames.match_id and frames.match_id != match:
                raise ValueError(f"cache match ID mismatch: {path}")
            if frames.period != _period(path, match):
                raise ValueError(f"cache period mismatch: {path}")
            if frames.fps != official.FPS:
                raise ValueError("official 25 Hz transition and duration metrics require 25 Hz")
            gt = _target_codes(frames, targets)
            if model is None or selected.neural_weight == 0:
                scores = reference_scores(frames)
            else:
                from possession.predict import predict_model

                scores = predict_model(frames, model, device=device,
                                       center_size=center_size, halo=halo)
                if selected.neural_weight < 1:
                    from possession.ensemble import combine_scores
                    scores = combine_scores(scores, reference_scores(frames),
                                            selected.neural_weight)
            live, home, boundary = apply_temperature_calibration(
                scores["live_probability"], scores["home_probability"],
                scores["boundary"], calibration)
            scores = {"live_probability": live, "home_probability": home,
                      "boundary": boundary}
            emission = emissions_from_probabilities(scores["live_probability"],
                                                     scores["home_probability"])
            boundary = np.asarray(scores["boundary"]) * boundary_scale
            pred = decode_half(emission, boundary, switch_cost=switch_cost)
            if len(pred) != len(gt):
                raise ValueError("predictor length differs from source timeline")
            metrics = official.evaluate(gt, pred, bootstrap_n=0)
            half_reports.append({
                "period": int(frames.period), "n_frames_native": len(gt),
                "n_unknown_state": int(np.sum(gt == official.X)),
                "official": metrics, "edge_diagnostics": _edge_diagnostics(gt),
            })
            match_gt.append(gt)
            match_pred.append(pred)

        pooled_gt = _join_with_x(match_gt)
        pooled_pred = _join_with_x(match_pred)
        match_official = official.evaluate(pooled_gt, pooled_pred, bootstrap_n=0)
        match_report = {
            "halves": half_reports,
            "n_frames_native": sum(half["n_frames_native"] for half in half_reports),
            "n_unknown_state": sum(half["n_unknown_state"] for half in half_reports),
            "n_scorer_separators": len(match_gt) - 1,
            "native_confusion": official.confusion(np.concatenate(match_gt),
                                                    np.concatenate(match_pred)).tolist(),
            "official": match_official,
            "edge_diagnostics": {
                key: sum(half["edge_diagnostics"][key] for half in half_reports)
                for key in half_reports[0]["edge_diagnostics"]
            },
        }
        report["by_match"][match] = match_report
        match_metrics.append(match_official)
        all_gt.extend(match_gt)
        all_pred.extend(match_pred)

    report["summary"] = {
        "n_matches": len(matches),
        "n_frames_native": sum(item["n_frames_native"] for item in report["by_match"].values()),
        "n_scorer_separators": len(all_gt) - 1,
        "native_confusion": official.confusion(np.concatenate(all_gt),
                                                np.concatenate(all_pred)).tolist(),
        "pooled_official": official.evaluate(_join_with_x(all_gt), _join_with_x(all_pred),
                                             bootstrap_n=0),
        "unweighted_match_mean": {
            key: float(np.mean([item[key] for item in match_metrics]))
            for key in ("spell_f1_mean", "macro_f1", "alive_dead_f1")
        },
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--matches", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--switch-cost", type=float, default=2.0)
    parser.add_argument("--boundary-scale", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = evaluate_prepared(args.data_dir, matches=args.matches,
                               checkpoint=args.checkpoint, device=args.device,
                               switch_cost=args.switch_cost, boundary_scale=args.boundary_scale)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
