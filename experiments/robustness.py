"""Paired, frozen leave-one-match-out missing-ball robustness ablation.

All seven development folds use their saved inner-selected model and decoder.
The protocol is written before the first outer score; a changed input, recipe,
source, or checkpoint makes resume fail closed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from evaluation import possession_metrics as official
from experiments.evaluate import _join_with_x, _target_codes
from experiments.select import _prepared_halves
from possession.calibrate import CalibrationConfig, apply_temperature_calibration
from possession.decode import decode_half, emissions_from_probabilities
from possession.ensemble import combine_scores
from possession.features import build_features
from possession.outages import OutageConfig, gap_statistics, generate_outage_mask
from possession.predict import load_model_checkpoint, predict_features
from possession.prepared import verify_prepared
from possession.provenance import runtime_provenance
from possession.reference import reference_scores
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache, TrainingSession

POLICIES = ("legacy", "correlated")
SCENARIOS = {
    "clean": None,
    "block41_seed3101": {"seed": 3101, "missing_fraction": .41,
                         "min_seconds": 1, "max_seconds": 10},
    "block41_seed3102": {"seed": 3102, "missing_fraction": .41,
                         "min_seconds": 1, "max_seconds": 10},
    "long41_seed4101": {"seed": 4101, "missing_fraction": .41,
                        "min_seconds": 10, "max_seconds": 30},
}
SCORE_FIELDS = ("spell_f1_mean", "macro_f1", "alive_dead_f1", "accuracy",
                "n_frames", "n_graded", "n_gt_spells", "n_pred_spells")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = [path for directory in ("possession", "experiments", "evaluation")
             for path in (root / directory).rglob("*.py")]
    paths += [root / "games.py", root / "pyproject.toml", root / "uv.lock"]
    return {str(path.relative_to(root)): sha256_file(path)
            for path in sorted(set(paths)) if path.is_file()}


def _execution_environment() -> dict:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "torch", "optuna", "scikit-learn", "numba"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    thread_names = ("OMP_NUM_THREADS", "OMP_DYNAMIC", "VECLIB_MAXIMUM_THREADS",
                    "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    return {"versions": versions,
            "thread_environment": {name: os.environ.get(name) for name in thread_names},
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads()}


def _seed(base: int, match: str, period: int) -> int:
    return int.from_bytes(hashlib.sha256(
        f"robustness-v2:{base}:{match}:{period}".encode()).digest()[:8], "big")


def _mask(frames, match: str, period: int, scenario: str):
    spec = SCENARIOS[scenario]
    if spec is None:
        additional = np.zeros(len(frames.ball_valid), dtype=bool)
    else:
        rng = np.random.default_rng(_seed(spec["seed"], match, period))
        additional = generate_outage_mask(
            frames.ball_valid, frames.fps, rng,
            missing_fraction=spec["missing_fraction"],
            min_seconds=spec["min_seconds"], max_seconds=spec["max_seconds"],
        )
    missing = ~frames.ball_valid | additional
    altered = frames.subset(0, len(frames.frame_id))
    altered.ball_valid[additional] = False
    altered.ball_xy[additional] = 0
    evidence = {
        "additional_mask_sha256": hashlib.sha256(additional.tobytes()).hexdigest(),
        "additional_missing_frames": int(additional.sum()),
        "existing_missing_frames": int((~frames.ball_valid).sum()),
        "resulting_gap_statistics": gap_statistics(missing, frames.fps),
        "seed": None if spec is None else _seed(spec["seed"], match, period),
    }
    return altered, evidence


def _compact(metrics: dict) -> dict:
    return {name: metrics[name] for name in SCORE_FIELDS}


def _decode(neural, reference, frozen, *, neural_only: bool):
    weight = 1.0 if neural_only else frozen["ensemble"]["neural_weight"]
    blended = combine_scores(neural, reference, weight)
    live, home, boundary = apply_temperature_calibration(
        blended["live_probability"], blended["home_probability"],
        blended["boundary"], CalibrationConfig(**frozen["calibration"]),
    )
    return decode_half(emissions_from_probabilities(live, home),
                       boundary * frozen["decoder"]["boundary_scale"],
                       switch_cost=frozen["decoder"]["switch_cost"])


def _frozen_plan(data_dir: Path, baseline_dir: Path, matches: list[str],
                 *, steps: int, cache_bytes: int) -> dict:
    available = sorted(path.parent.name for path in baseline_dir.glob("*/frozen_recipe.json"))
    if len(available) != len(set(available)) or not available:
        raise ValueError("baseline has no unique frozen outer recipes")
    if any(match not in available for match in matches):
        raise ValueError("requested match lacks a frozen outer recipe")
    folds = {}
    paths = []
    for match in matches:
        directory = baseline_dir / match
        frozen_path = directory / "frozen_recipe.json"
        frozen = _read_json(frozen_path)
        recipe = frozen.get("selected_model_recipe")
        if (frozen.get("outer_test") != match
                or sorted(frozen.get("outer_train", [])) != sorted(set(available) - {match})
                or not isinstance(recipe, dict)
                or not isinstance(recipe.get("model_kwargs"), dict)
                or not isinstance(recipe.get("training_args"), dict)):
            raise ValueError(f"incomplete or invalid frozen neural recipe for {match}")
        args = recipe["training_args"]
        needed = ("learning_rate", "weight_decay", "boundary_weight",
                  "boundary_pos_weight", "augment", "center_size")
        if any(name not in args for name in needed):
            raise ValueError(f"frozen training arguments are incomplete for {match}")
        if not isinstance(frozen.get("refit_seed"), int):
            raise TypeError(f"frozen refit seed is missing for {match}")
        for name in ("calibration", "decoder", "ensemble"):
            if not isinstance(frozen.get(name), dict):
                raise TypeError(f"frozen {name} is missing for {match}")
        baseline_files = ("frozen_recipe.json", "fingerprint.json",
                          "decoder_fingerprint.json", "selected.pt")
        if not (directory / "selected.pt").is_file():
            raise ValueError(f"baseline selected checkpoint is missing for {match}")
        selected_sha = sha256_file(directory / "selected.pt")
        frozen_sha = sha256_file(frozen_path)
        progress_path = directory / "refit_progress.json"
        if progress_path.is_file() and _read_json(progress_path).get(
                "checkpoint_sha256") != selected_sha:
            raise ValueError(f"baseline refit checkpoint hash differs for {match}")
        outer_path = directory / "outer_report.json"
        if outer_path.is_file():
            report = _read_json(outer_path)
            if (report.get("checkpoint_sha256") != selected_sha
                    or report.get("frozen_recipe_sha256") != frozen_sha
                    or report.get("selection_fingerprint") != frozen.get("selection_fingerprint")
                    or report.get("outer_test") != match
                    or report.get("outer_train") != frozen["outer_train"]):
                raise ValueError(f"baseline outer report ancestry differs for {match}")
            report_hash_path = directory / "outer_report.sha256.json"
            if report_hash_path.is_file() and _read_json(report_hash_path).get(
                    "sha256") != sha256_file(outer_path):
                raise ValueError(f"baseline outer report hash differs for {match}")
        baseline_files += ("refit_progress.json", "outer_report.json",
                           "outer_report.sha256.json")
        folds[match] = {
            "outer_train": frozen["outer_train"], "refit_seed": frozen["refit_seed"],
            "model_kwargs": recipe["model_kwargs"],
            "training_args": {name: args[name] for name in
                              (*needed, "augmentation_config") if name in args},
            "frozen_sha256": frozen_sha,
            "baseline_sha256": {name: sha256_file(directory / name)
                                for name in baseline_files if (directory / name).is_file()},
        }
        for source in frozen["outer_train"] + [match]:
            paths.extend(_prepared_halves(data_dir, source))
    paths = sorted(set(paths))
    provenance = verify_prepared(data_dir, paths)
    manifest = data_dir / "manifest.json"
    payload = {
        "protocol": "paired_missing_ball_robustness_v2_development",
        "matches": matches, "baseline_match_universe": available,
        "steps_per_policy_per_fold": steps,
        "cache_bytes": cache_bytes, "device": "cpu", "workers": 1,
        "policies": {mode: OutageConfig(mode=mode).to_dict() for mode in POLICIES},
        "scenarios": SCENARIOS,
        "folds": folds,
        "prepared_sha256": {path.name: sha256_file(path) for path in paths},
        "prepared_provenance": provenance,
        "manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
        "source_sha256": _source_hashes(),
        "execution_environment": _execution_environment(),
        "selection_rule": "saved inner-selected frozen recipe; no outer-based choices",
        "gate": {"short_mean_gain_gt": 0, "positive_short_matches_at_least": 4,
                 "clean_decline_at_most": .02, "long_decline_at_most": .02,
                 "development_only": True, "required_matches": 7},
    }
    return {"sha256": _digest(payload), "payload": payload}


@contextmanager
def _lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "robustness.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("robustness output already has a writer") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _fit_pair(data_dir, output_dir, match, plan, cache, *, steps):
    run_dir = output_dir / match
    run_dir.mkdir(exist_ok=True)
    frozen = plan["payload"]["folds"][match]
    checkpoints = {}
    progress = {}
    timings = {}
    for mode in POLICIES:
        print(f"[{match}] fit {mode}: 0/{steps} steps", flush=True)
        session = TrainingSession(
            data_dir, train_matches=frozen["outer_train"], seed=frozen["refit_seed"],
            device="cpu", model_kwargs=frozen["model_kwargs"],
            learning_rate=frozen["training_args"]["learning_rate"],
            weight_decay=frozen["training_args"]["weight_decay"],
            boundary_weight=frozen["training_args"]["boundary_weight"],
            boundary_pos_weight=frozen["training_args"]["boundary_pos_weight"],
            augment=frozen["training_args"]["augment"],
            center_size=frozen["training_args"]["center_size"], cache=cache,
            outage_config=OutageConfig(mode=mode),
            augmentation_config=frozen["training_args"].get("augmentation_config"),
        )
        start = time.perf_counter()
        while session.steps < steps:
            progress[mode] = session.advance_to(min(steps, session.steps + 300))
            print(f"[{match}] fit {mode}: {session.steps}/{steps} steps", flush=True)
        timings[mode] = time.perf_counter() - start
        path = run_dir / f"{mode}.pt"
        session.save_checkpoint(path, extra_training_config={
            "robustness_protocol_sha256": plan["sha256"], "outer_test": match,
            "outage_policy": mode,
        })
        checkpoints[mode] = sha256_file(path)
        if progress[mode].get("steps") != steps:
            raise ValueError("fit did not reach requested budget")
        if not all(name in progress[mode] for name in
                   ("center_ball_missing_frames", "center_ball_frames")):
            raise ValueError("TrainingSession lacks measured center-ball exposure")
    if progress["legacy"]["training_schedule_blocks"] != progress["correlated"][
            "training_schedule_blocks"]:
        raise ValueError("paired policies changed the training window schedule")
    if progress["legacy"].get("training_window_sha256") != progress["correlated"].get(
            "training_window_sha256") or not progress["legacy"].get("training_window_sha256"):
        raise ValueError("paired policies changed the training center windows")
    return checkpoints, progress, timings


def _score_pair(data_dir, match, plan, cache, checkpoints):
    frozen = _read_json(Path(plan["baseline_dir"]) / match / "frozen_recipe.json")
    models = {}
    for mode in POLICIES:
        checkpoint_path = Path(plan["output_dir"]) / match / f"{mode}.pt"
        if sha256_file(checkpoint_path) != checkpoints[mode]:
            raise ValueError(f"paired checkpoint hash changed before scoring: {match} {mode}")
        model, payload = load_model_checkpoint(
            checkpoint_path, device="cpu")
        config = payload["training_config"]
        if (config.get("outer_test") != match or match in config.get("train_matches", [])
                or config.get("robustness_protocol_sha256") != plan["sha256"]
                or config.get("outage_policy") != mode):
            raise ValueError("robustness checkpoint ancestry differs")
        models[mode] = model
    results = {}
    for scenario in SCENARIOS:
        print(f"[{match}] score {scenario}", flush=True)
        gt_parts = []
        pred_parts = {mode: {diagnostic: [] for diagnostic in ("blend", "neural_only")}
                      for mode in POLICIES}
        masks = {}
        for period, path in enumerate(_prepared_halves(data_dir, match), 1):
            frames, targets, _, _ = cache.get(path)
            if (frames.match_id != match or frames.period != period or frames.fps != official.FPS):
                raise ValueError("outer prepared half identity or FPS differs")
            altered, evidence = _mask(frames, match, period, scenario)
            masks[str(period)] = evidence
            features = build_features(altered)
            reference = reference_scores(altered)
            gt_parts.append(_target_codes(frames, targets))
            for mode in POLICIES:
                neural = predict_features(
                    features, models[mode], device="cpu",
                    center_size=frozen["selected_model_recipe"]["training_args"]["center_size"],
                    halo=models[mode].radius,
                )
                for diagnostic in ("blend", "neural_only"):
                    pred_parts[mode][diagnostic].append(_decode(
                        neural, reference, frozen, neural_only=diagnostic == "neural_only"))
        joined_gt = _join_with_x(gt_parts)
        scores = {mode: {diagnostic: _compact(official.evaluate(
            joined_gt, _join_with_x(pred_parts[mode][diagnostic]), bootstrap_n=0))
            for diagnostic in ("blend", "neural_only")} for mode in POLICIES}
        results[scenario] = {"masks": masks, "scores": scores,
                             "paired_gain": {diagnostic: float(
                                 scores["correlated"][diagnostic]["spell_f1_mean"]
                                 - scores["legacy"][diagnostic]["spell_f1_mean"])
                                 for diagnostic in ("blend", "neural_only")}}
    return results


def _verified_record(output_dir, match, protocol_sha):
    run_dir = output_dir / match
    record_path = run_dir / "result.json"
    hash_path = run_dir / "result.sha256.json"
    expected = {"legacy.pt", "correlated.pt", "result.json", "result.sha256.json"}
    if not run_dir.exists():
        return None
    actual = {path.name for path in run_dir.iterdir()}
    if actual != expected:
        raise ValueError(f"partial or unexpected robustness artifacts for {match}: {sorted(actual)}")
    if _read_json(hash_path).get("sha256") != sha256_file(record_path):
        raise ValueError(f"robustness result hash changed for {match}")
    record = _read_json(record_path)
    if (record.get("protocol_sha256") != protocol_sha or record.get("outer_test") != match
            or record.get("checkpoint_sha256") != {
                mode: sha256_file(run_dir / f"{mode}.pt") for mode in POLICIES}):
        raise ValueError(f"robustness record or checkpoint changed for {match}")
    return record


def _summary(records: list[dict], *, require_complete: bool, steps: int):
    by_match = {}
    for record in records:
        match = record["outer_test"]
        by_match[match] = {scenario: record["scenarios"][scenario]["paired_gain"]
                           for scenario in SCENARIOS}
    conditions = {}
    for diagnostic in ("blend", "neural_only"):
        per_match = {}
        for match in sorted(by_match):
            short = [by_match[match][scenario][diagnostic]
                     for scenario in ("block41_seed3101", "block41_seed3102")]
            per_match[match] = {"clean": by_match[match]["clean"][diagnostic],
                                "short_seed3101": short[0], "short_seed3102": short[1],
                                "short_mean": float(np.mean(short)),
                                "long": by_match[match]["long41_seed4101"][diagnostic]}
        means = {field: float(np.mean([item[field] for item in per_match.values()]))
                 for field in ("clean", "short_seed3101", "short_seed3102",
                               "short_mean", "long")} if per_match else {}
        absolute = {}
        for mode in POLICIES:
            by_condition = {}
            for scenario in SCENARIOS:
                values = [record["scenarios"][scenario]["scores"][mode][diagnostic][
                    "spell_f1_mean"] for record in records]
                by_condition[scenario] = float(np.mean(values))
            by_condition["short_two_seed_mean"] = float(np.mean([
                by_condition["block41_seed3101"], by_condition["block41_seed3102"]]))
            absolute[mode] = by_condition
        conditions[diagnostic] = {"per_match_paired_gain": per_match,
                                  "equal_match_mean_paired_gain": means,
                                  "equal_match_mean_spell_f1": absolute}
    blend = conditions["blend"]
    means = blend["equal_match_mean_paired_gain"]
    positive = sum(item["short_mean"] > 0 for item in
                   blend["per_match_paired_gain"].values())
    complete = require_complete and len(records) == 7 and steps == 1200
    gate = {"development_only": True, "promotion_eligible": complete,
            "positive_short_matches": positive,
            "passes": (means["short_mean"] > 0 and positive >= 4
                       and means["clean"] >= -.02 and means["long"] >= -.02)
                      if complete else None}
    return {"protocol": "paired_missing_ball_robustness_v2_development",
            "matches_completed": sorted(by_match), "required_matches_for_promotion": 7,
            "conditions": conditions, "gate": gate}


def run_robustness(data_dir, baseline_dir, output_dir, *, steps=1200, matches=None,
                   cache_bytes=900_000_000, require_complete=True):
    """Run serial paired fits, with exact-input resume and no outer score selection."""
    data_dir = Path(data_dir).resolve()
    baseline_dir = Path(baseline_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(cache_bytes, int) or isinstance(cache_bytes, bool) or cache_bytes < 1:
        raise ValueError("cache_bytes must be a positive integer")
    available = sorted(path.parent.name for path in baseline_dir.glob("*/frozen_recipe.json"))
    matches = available if matches is None else sorted(set(matches))
    if not matches or (require_complete and (len(matches) != 7 or matches != available)):
        raise ValueError("promotion protocol requires exactly all seven frozen outer matches")
    with _lock(output_dir):
        plan = _frozen_plan(data_dir, baseline_dir, matches, steps=steps,
                            cache_bytes=cache_bytes)
        protocol_path = output_dir / "protocol.json"
        if protocol_path.exists():
            if _read_json(protocol_path) != plan:
                raise ValueError("robustness protocol inputs, sources, or configuration changed")
        else:
            extra = {path.name for path in output_dir.iterdir()} - {"robustness.lock"}
            if extra:
                raise ValueError(f"output exists without frozen protocol: {sorted(extra)}")
            write_json(protocol_path, plan)
        unexpected = {path.name for path in output_dir.iterdir()} - {
            "robustness.lock", "protocol.json", "summary.json", *matches}
        if unexpected:
            raise ValueError(f"unexpected output artifacts: {sorted(unexpected)}")
        records = []
        cache = HalfCache(data_dir, max_bytes=cache_bytes)
        for match in matches:
            print(f"[{match}] fold start", flush=True)
            record = _verified_record(output_dir, match, plan["sha256"])
            if record is None:
                checkpoints, progress, timings = _fit_pair(
                    data_dir, output_dir, match, plan, cache, steps=steps)
                scored = _score_pair(data_dir, match,
                                     {**plan, "baseline_dir": str(baseline_dir),
                                      "output_dir": str(output_dir)}, cache, checkpoints)
                record = {"outer_test": match, "outer_train": plan["payload"]["folds"][match]["outer_train"],
                          "protocol_sha256": plan["sha256"],
                          "checkpoint_sha256": checkpoints,
                          "fit_seconds": timings,
                          "training_progress": progress,
                          "scenarios": scored}
                run_dir = output_dir / match
                record_path = run_dir / "result.json"
                write_json(record_path, record)
                write_json(run_dir / "result.sha256.json", {"sha256": sha256_file(record_path)})
            records.append(record)
            summary = {**_summary(records, require_complete=require_complete, steps=steps),
                       "protocol_sha256": plan["sha256"],
                       "runtime": runtime_provenance(),
                       "cache": {"hits": cache.hits, "misses": cache.misses,
                                 "peak_bytes": cache.peak_bytes}}
            write_json(output_dir / "summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/optuna-audit"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/robustness-v2"))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--matches", nargs="+")
    parser.add_argument("--cache-gb", type=float, default=.9)
    parser.add_argument("--allow-incomplete-dev", action="store_true")
    args = parser.parse_args(argv)
    if not np.isfinite(args.cache_gb) or args.cache_gb <= 0:
        parser.error("--cache-gb must be positive and finite")
    result = run_robustness(
        args.data_dir, args.baseline_dir, args.output, steps=args.steps,
        matches=args.matches, cache_bytes=int(args.cache_gb * 1_000_000_000),
        require_complete=not args.allow_incomplete_dev)
    print(json.dumps({"matches_completed": result["matches_completed"],
                      "gate": result["gate"], "summary": str(args.output / "summary.json")},
                     sort_keys=True))
    return result


if __name__ == "__main__":
    main()
