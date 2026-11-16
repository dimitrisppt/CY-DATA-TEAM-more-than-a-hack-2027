"""Resumable nested-match Optuna search using the official possession scorer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from experiments.cross_validation import _match_id, nested_splits
from experiments.evaluate import _join_with_x, _target_codes
from possession.augmentation import normalize_augmentation
from possession.decode import decode_half, emissions_from_probabilities
from possession.outages import OutageConfig
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache, TrainingSession

DEFAULT_DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_MATCHES = (
    "DFL-MAT-J03WMX", "DFL-MAT-J03WN1", "DFL-MAT-J03WOH",
    "DFL-MAT-J03WOY", "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ",
    "DFL-MAT-J03WR9",
)
SEARCH_SPACE = {
    "learning_rate": [3e-4, 5e-3, "log"],
    "weight_decay": [1e-6, 1e-2, "log"],
    "width": [32, 64],
    "boundary_pos_weight": [1.0, 100.0, "log"],
    "switch_cost": [0.0, 8.0],
}


def _optuna():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required: run uv sync --extra learn --extra tune") from exc
    return optuna


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _paths(data_dir: Path, match: str) -> list[Path]:
    paths = list(data_dir.glob(f"{match}_half*.npz"))
    if not paths:
        raise FileNotFoundError(f"no prepared halves for {match}")
    pattern = re.compile(re.escape(match) + r"_half(\d+)\.npz")
    periods = []
    for path in paths:
        found = pattern.fullmatch(path.name)
        if found is None:
            raise ValueError(f"invalid prepared filename: {path.name}")
        periods.append(int(found.group(1)))
    if len(periods) != len(set(periods)):
        raise ValueError(f"duplicate period for {match}")
    return [path for _, path in sorted(zip(periods, paths))]


def _discover_matches(data_dir: Path) -> list[str]:
    pattern = re.compile(r"(.+)_half\d+\.npz")
    ids = sorted({found.group(1) for path in data_dir.glob("*_half*.npz")
                  if (found := pattern.fullmatch(path.name)) is not None})
    if ids != sorted(DEFAULT_MATCHES):
        raise ValueError("default discovery requires exactly the seven known DFL matches; "
                         "pass matches explicitly for another dataset")
    return ids


def _outage_dict(config=None):
    return (config if isinstance(config, OutageConfig)
            else OutageConfig(**({} if config is None else config))).to_dict()


def _augmentation_dict(config=None):
    result = normalize_augmentation(config).to_dict()
    result["operations"] = list(result["operations"])
    return result


def _fingerprint(data_dir, fold, split, *, seed, rungs, device, poolings,
                 pruning, center_size, cache_bytes, outage_config=None,
                 augmentation_config=None):
    root = Path(__file__).resolve().parents[1]
    source_paths = []
    for directory in ("possession", "experiments", "evaluation"):
        source_paths.extend((root / directory).rglob("*.py"))
    source_paths.extend([root / "games.py", root / "pyproject.toml"])
    sources = {str(path.relative_to(root)): sha256_file(path)
               for path in sorted(set(source_paths)) if path.is_file()}
    prepared = {path.name: sha256_file(path)
                for match in fold["outer_train"] for path in _paths(data_dir, match)}
    manifest = data_dir / "manifest.json"
    lockfile = root / "uv.lock"
    import torch

    payload = {
        "protocol": "nested_match_hpo_v1", "outer_test": fold["outer_test"],
        "split": split, "source_sha256": sources, "prepared_sha256": prepared,
        "manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
        "uv_lock_sha256": sha256_file(lockfile) if lockfile.is_file() else None,
        "versions": {"python": platform.python_version(),
                     **{name: _version(name) for name in
                        ("optuna", "torch", "numpy", "scikit-learn", "numba")}},
        "thread_environment": {
            name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "OMP_DYNAMIC", "VECLIB_MAXIMUM_THREADS",
             "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        } | {"torch_num_threads": torch.get_num_threads(),
             "torch_num_interop_threads": torch.get_num_interop_threads()},
        "resources": {"rungs": list(rungs), "device": str(device),
                      "center_size": center_size, "cache_bytes": cache_bytes},
        "search_space": {**SEARCH_SPACE, "pooling": list(poolings)},
        "fixed": {"boundary_weight": 0.25, "augment": True,
                  "outage_config": _outage_dict(outage_config),
                  "augmentation_config": _augmentation_dict(augmentation_config),
                  "dilations": DEFAULT_DILATIONS, "temperature": 1.0,
                  "boundary_scale": 0.0},
        "seed": seed, "pruning": bool(pruning),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"sha256": hashlib.sha256(encoded.encode()).hexdigest(), "payload": payload}


def _seed_for_fold(seed: int, outer: str, fold_index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{outer}:{fold_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _suggest(trial, poolings):
    return {
        "pooling": trial.suggest_categorical("pooling", list(poolings)),
        "width": trial.suggest_categorical("width", [32, 64]),
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "boundary_pos_weight": trial.suggest_float("boundary_pos_weight", 1.0, 100.0,
                                                   log=True),
        "switch_cost": trial.suggest_float("switch_cost", 0.0, 8.0),
    }


def _baseline(pooling):
    return {"pooling": pooling, "width": 64, "learning_rate": 0.001,
            "weight_decay": 0.0001, "boundary_pos_weight": 1.0, "switch_cost": 2.0}


def _recipe(params, *, steps, center_size, seed, outage_config=None,
            augmentation_config=None):
    return {
        "model_kwargs": {"player_dim": 8, "global_dim": 9,
                         "width": params["width"], "pooling": params["pooling"],
                         "dilations": DEFAULT_DILATIONS.copy()},
        "training_args": {"steps": steps, "learning_rate": params["learning_rate"],
                          "weight_decay": params["weight_decay"],
                          "boundary_weight": 0.25,
                          "boundary_pos_weight": params["boundary_pos_weight"],
                          "augment": True, "center_size": center_size,
                          "outage_config": _outage_dict(outage_config),
                          "augmentation_config": _augmentation_dict(augmentation_config),
                          "seed_rule": "sha256(seed:outer_test:inner_fold_index) first 32 bits",
                          "study_seed": seed},
        "decoder": {"switch_cost": params["switch_cost"], "temperature": 1.0,
                    "boundary_scale": 0.0},
    }


def _score_match(data_dir, match, session, cache, *, switch_cost, center_size, device,
                 train_matches):
    from possession.predict import predict_features

    gt_parts, pred_parts, records = [], [], []
    for path in _paths(data_dir, match):
        frames, targets, features, _ = cache.get(path)
        if frames.match_id != match or frames.fps != official.FPS:
            raise ValueError(f"prepared match ID or FPS mismatch: {path}")
        gt = _target_codes(frames, targets)
        scores = predict_features(features, session.model, device=device,
                                  center_size=center_size, halo=session.halo)
        live = np.asarray(scores["live_probability"], dtype=np.float64)
        home = np.asarray(scores["home_probability"], dtype=np.float64)
        boundary = np.asarray(scores["boundary"], dtype=np.float64)
        pred = decode_half(emissions_from_probabilities(live, home),
                           boundary * 0.0, switch_cost=switch_cost)
        if len(pred) != len(gt) or boundary.shape != (len(gt), 3, 3):
            raise ValueError("prediction shape differs from prepared half")
        gt_parts.append(gt)
        pred_parts.append(pred)
        records.append({"match": match, "period": int(frames.period),
                        "train_matches": list(train_matches), "frame_id": frames.frame_id.copy(),
                        "gt": gt.copy(), "live_probability": live,
                        "home_probability": home, "boundary": boundary})
    metrics = official.evaluate(_join_with_x(gt_parts), _join_with_x(pred_parts),
                                bootstrap_n=0)
    diagnostics = {"n_frames": int(sum(len(part) for part in gt_parts)),
                   "gt_class_counts": np.bincount(np.concatenate(gt_parts),
                                                   minlength=4).tolist(),
                   "pred_class_counts": np.bincount(np.concatenate(pred_parts),
                                                     minlength=4).tolist(),
                   "official": metrics}
    return float(metrics["spell_f1_mean"]), diagnostics, records


def _cache_stats(cache):
    return {name: getattr(cache, name, None)
            for name in ("bytes", "peak_bytes", "hits", "misses", "max_bytes")}


def _run_trial(trial, fold, data_dir, cache, *, params, rungs, seed, device,
               center_size, pruning, outage_config=None, augmentation_config=None):
    optuna = _optuna()
    sessions = []
    for index, inner in enumerate(fold["inner"]):
        sessions.append(TrainingSession(
            data_dir, train_matches=inner["train"],
            seed=_seed_for_fold(seed, fold["outer_test"][0], index), device=device,
            model_kwargs={"width": params["width"], "pooling": params["pooling"],
                          "dilations": DEFAULT_DILATIONS},
            learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
            boundary_weight=0.25, boundary_pos_weight=params["boundary_pos_weight"],
            augment=True, center_size=center_size, cache=cache,
            outage_config=outage_config, augmentation_config=augmentation_config))
    rung_reports = []
    final_records = []
    for rung in rungs:
        start = time.monotonic()
        per_match = {}
        fold_progress = []
        records = []
        for index, inner in enumerate(fold["inner"]):
            progress = sessions[index].advance_to(rung)
            for match in inner["validation"]:
                match_start = time.monotonic()
                score, detail, halves = _score_match(
                    data_dir, match, sessions[index], cache,
                    switch_cost=params["switch_cost"], center_size=center_size,
                    device=device, train_matches=inner["train"])
                per_match[match] = {"spell_f1_mean": score, **detail,
                                    "inner_fold": index,
                                    "seconds": time.monotonic() - match_start}
                for half in halves:
                    half["inner_fold"] = index
                records.extend(halves)
            fold_progress.append({
                "inner_fold": index, "steps": progress["steps"],
                "train_loss": progress["train_loss"],
                "train_losses": progress["train_losses"],
                "train_counts": progress["train_counts"],
                "boundary_positive_count": progress["boundary_positive_count"],
                "boundary_negative_count": progress["boundary_negative_count"],
                "augmentation_counts": progress["augmentation_counts"],
                "n_trained_halves": len(progress["trained_halves"]),
                "cache": _cache_stats(cache),
            })
        if set(per_match) != set(fold["outer_train"]):
            raise AssertionError("all inner validation matches must be scored once")
        values = [entry["spell_f1_mean"] for entry in per_match.values()]
        score = float(np.mean(values))
        report = {"steps_per_fold": rung, "score": score,
                  "match_std": float(np.std(values)), "by_match": per_match,
                  "fold_progress": fold_progress, "cache": _cache_stats(cache),
                  "seconds": time.monotonic() - start}
        rung_reports.append(report)
        trial.set_user_attr("rungs", rung_reports)
        trial.report(score, step=rung)
        if pruning and rung != rungs[-1] and trial.should_prune():
            raise optuna.TrialPruned(f"pruned after every fold reached {rung} steps")
        final_records = records
    return score, sessions, final_records, rung_reports


def _save_trial_artifacts(directory, *, trial, fold, sessions, records, params, score,
                          rungs, center_size, seed, fingerprint_sha256, outage_config=None,
                          augmentation_config=None):
    """Publish a complete, immutable winning trial directory in one rename."""
    if directory.exists():
        raise FileExistsError(f"trial artifact directory already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        file_hashes = {}
        checkpoints = []
        for index, (inner, session) in enumerate(zip(fold["inner"], sessions)):
            name = f"fold_{index}.pt"
            session.save_checkpoint(staging / name,
                                    validation_matches=inner["validation"],
                                    extra_training_config={"trial_number": trial.number,
                                                           "outer_test": fold["outer_test"][0],
                                                           "inner_fold": index})
            file_hashes[name] = sha256_file(staging / name)
            checkpoints.append({"inner_fold": index, "file": name,
                                "sha256": file_hashes[name],
                                "train_matches": inner["train"],
                                "validation_matches": inner["validation"]})
        oof_files = []
        for record in records:
            match, period = record["match"], record["period"]
            index = record["inner_fold"]
            if index >= len(checkpoints) or index < 0:
                raise ValueError("OOF record has invalid inner fold index")
            checkpoint = checkpoints[index]
            if record["train_matches"] != checkpoint["train_matches"]:
                raise ValueError("OOF training ancestry differs from fold checkpoint")
            name = f"{match}_half{period}.npz"
            path = staging / name
            with open(path, "wb") as handle:
                np.savez_compressed(handle, **{key: record[key] for key in
                            ("frame_id", "gt", "live_probability", "home_probability", "boundary")})
            file_hashes[name] = sha256_file(path)
            sidecar_name = f"{match}_half{period}.json"
            write_json(staging / sidecar_name, {
                "prediction_match": match, "period": period,
                "train_matches": record["train_matches"],
                "outer_test": fold["outer_test"][0], "trial_number": trial.number,
                "inner_fold": index, "checkpoint_file": checkpoint["file"],
                "checkpoint_sha256": checkpoint["sha256"],
                "oof_file": name, "oof_sha256": file_hashes[name],
                "study_fingerprint_sha256": fingerprint_sha256,
                "arrays": ["frame_id", "gt", "live_probability", "home_probability", "boundary"],
                "probability_source": "native 25 Hz validation half",
            })
            file_hashes[sidecar_name] = sha256_file(staging / sidecar_name)
            oof_files.append({"match": match, "period": period, "inner_fold": index,
                              "file": name, "sha256": file_hashes[name],
                              "sidecar": sidecar_name,
                              "sidecar_sha256": file_hashes[sidecar_name]})
        metadata = {"trial_number": trial.number, "score": score, "params": params,
                    **_recipe(params, steps=rungs[-1], center_size=center_size, seed=seed,
                              outage_config=outage_config,
                              augmentation_config=augmentation_config),
                    "study_fingerprint_sha256": fingerprint_sha256,
                    "artifact_dir": str(directory),
                    "artifact_sha256": file_hashes,
                    "fold_checkpoints": checkpoints, "oof_files": oof_files}
        write_json(staging / "recipe.json", metadata)
        os.replace(staging, directory)
        return metadata
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _summary(study, fold, directory, *, fingerprint, rungs, center_size, seed):
    optuna = _optuna()
    attempted = [trial for trial in study.trials
                 if trial.state != optuna.trial.TrialState.WAITING]
    completed = [trial for trial in attempted
                 if trial.state == optuna.trial.TrialState.COMPLETE
                 and trial.user_attrs.get("full_budget_steps") == rungs[-1]]
    best_by_pooling = {}
    for trial in completed:
        pooling = trial.params["pooling"]
        if pooling not in best_by_pooling or trial.value > best_by_pooling[pooling]["score"]:
            recipe_path = directory / "trials" / f"trial_{trial.number}" / "recipe.json"
            if not recipe_path.is_file():
                raise RuntimeError(f"missing winning artifacts: {recipe_path}")
            recipe = json.loads(recipe_path.read_text())
            if (recipe["trial_number"] != trial.number or recipe["score"] != trial.value
                    or recipe["study_fingerprint_sha256"] != fingerprint["sha256"]):
                raise RuntimeError(f"winning artifact provenance mismatch: {recipe_path}")
            best_by_pooling[pooling] = recipe
    for pooling, recipe in best_by_pooling.items():
        # A completed trial is durable in SQLite before its family pointer is published.
        # Calling this on resume repairs a missing or stale pointer after interruption.
        write_json(directory / "best_by_pooling" / pooling / "recipe.json", recipe)
    winner = max(completed, key=lambda trial: trial.value) if completed else None
    summary = {
        "protocol": "nested match HPO; official unweighted match mean",
        "outer_test": fold["outer_test"], "outer_train": fold["outer_train"],
        "inner": fold["inner"], "fingerprint_sha256": fingerprint["sha256"],
        "full_budget_steps": rungs[-1], "n_trials_total": len(attempted),
        "n_complete_full_budget": len(completed),
        "architecture_comparison": {
            "paired_baselines": "one fixed baseline per pooling, each trained to full budget",
            "family_best": "selected from completed full-budget trials only",
            "adaptive_budget_note": "family bests may have unequal search histories with or without pruning",
        },
        "best": None if winner is None else best_by_pooling[winner.params["pooling"]],
        "best_by_pooling": best_by_pooling,
        "trials": [{"number": trial.number, "state": trial.state.name,
                    "value": trial.value, "params": trial.params,
                    "baseline_full_budget": trial.user_attrs.get("baseline_full_budget", False),
                    "rungs": trial.user_attrs.get("rungs", [])}
                   for trial in attempted],
    }
    write_json(directory / "summary.json", summary)
    return summary


def _run_outer(data_dir, output_dir, fold, *, n_trials, rungs, seed, device,
               poolings, pruning, center_size, cache_bytes, cache, split,
               outage_config=None, augmentation_config=None):
    optuna = _optuna()
    outer = fold["outer_test"][0]
    directory = output_dir / outer
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "run.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"study lock exists; inspect interrupted run: {lock}") from exc
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(f"pid={os.getpid()}\n")
        fingerprint = _fingerprint(data_dir, fold, split, seed=seed, rungs=rungs,
                                   device=device, poolings=poolings, pruning=pruning,
                                   center_size=center_size, cache_bytes=cache_bytes,
                                   outage_config=outage_config,
                                   augmentation_config=augmentation_config)
        fp_path = directory / "fingerprint.json"
        split_path = directory / "splits.json"
        database = directory / "study.sqlite3"
        if database.exists() and (not fp_path.is_file() or not split_path.is_file()):
            raise RuntimeError("existing study lacks fingerprint or splits; refusing resume")
        if fp_path.is_file() and json.loads(fp_path.read_text()) != fingerprint:
            raise ValueError("study fingerprint changed; refusing resume")
        if split_path.is_file() and json.loads(split_path.read_text()) != split:
            raise ValueError("study splits changed; refusing resume")
        if not fp_path.exists():
            write_json(fp_path, fingerprint)
            write_json(split_path, split)
        storage = f"sqlite:///{database}"
        study_name = f"nested-{outer}"
        pruner = (optuna.pruners.SuccessiveHalvingPruner(min_resource=rungs[0],
                                                         reduction_factor=2)
                  if pruning else optuna.pruners.NopPruner())
        study = optuna.create_study(study_name=study_name, storage=storage,
                                    load_if_exists=True, direction="maximize",
                                    pruner=pruner,
                                    sampler=optuna.samplers.TPESampler(seed=seed,
                                        multivariate=True, n_startup_trials=3))
        previous_fp = study.user_attrs.get("fingerprint_sha256")
        if previous_fp is not None and previous_fp != fingerprint["sha256"]:
            raise ValueError("study storage fingerprint changed; refusing resume")
        study.set_user_attr("fingerprint_sha256", fingerprint["sha256"])
        if any(trial.state == optuna.trial.TrialState.RUNNING for trial in study.trials):
            raise RuntimeError("interrupted RUNNING trial; explicit recovery required")
        if not study.trials:
            for pooling in poolings:
                study.enqueue_trial(_baseline(pooling))
        finished = sum(trial.state in (optuna.trial.TrialState.COMPLETE,
                                       optuna.trial.TrialState.PRUNED,
                                       optuna.trial.TrialState.FAIL)
                       for trial in study.trials)
        if (directory / "frozen_recipe.json").exists():
            if n_trials > finished:
                raise RuntimeError("outer recipe is frozen; refusing new HPO trials")
            summary_path = directory / "summary.json"
            if not summary_path.is_file():
                raise RuntimeError("outer recipe is frozen but the study summary is missing")
            summary = json.loads(summary_path.read_text())
            if summary.get("fingerprint_sha256") != fingerprint["sha256"]:
                raise RuntimeError("frozen study summary fingerprint mismatch")
            return summary
        if finished > n_trials:
            raise ValueError("n_trials is smaller than the number already attempted")
        for _ in range(n_trials - finished):
            if (directory / "frozen_recipe.json").exists():
                raise RuntimeError("outer recipe is frozen; refusing new HPO trials")
            next_number = max((trial.number for trial in study.trials), default=-1) + 1
            study = optuna.load_study(study_name=study_name, storage=storage,
                                      sampler=optuna.samplers.TPESampler(
                                          seed=seed + next_number, multivariate=True,
                                          n_startup_trials=3), pruner=pruner)
            trial = study.ask()
            try:
                params = _suggest(trial, poolings)
                baseline_full_budget = (trial.number < len(poolings)
                                        and params == _baseline(poolings[trial.number]))
                trial.set_user_attr("baseline_full_budget", baseline_full_budget)
                score, sessions, records, _reports = _run_trial(
                    trial, fold, data_dir, cache, params=params, rungs=rungs,
                    seed=seed, device=device, center_size=center_size,
                    pruning=pruning and not baseline_full_budget, outage_config=outage_config,
                    augmentation_config=augmentation_config)
                trial.set_user_attr("full_budget_steps", rungs[-1])
                old_family = [old for old in study.trials
                              if old.state == optuna.trial.TrialState.COMPLETE
                              and old.user_attrs.get("full_budget_steps") == rungs[-1]
                              and old.params.get("pooling") == params["pooling"]]
                if not old_family or score > max(old.value for old in old_family):
                    _save_trial_artifacts(directory / "trials" / f"trial_{trial.number}",
                                          trial=trial, fold=fold, sessions=sessions,
                                          records=records, params=params, score=score,
                                          rungs=rungs, center_size=center_size, seed=seed,
                                          fingerprint_sha256=fingerprint["sha256"],
                                          outage_config=outage_config,
                                          augmentation_config=augmentation_config)
                study.tell(trial, score)
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            except Exception:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                raise
            _summary(study, fold, directory, fingerprint=fingerprint, rungs=rungs,
                     center_size=center_size, seed=seed)
        return _summary(study, fold, directory, fingerprint=fingerprint,
                        rungs=rungs, center_size=center_size, seed=seed)
    finally:
        lock.unlink(missing_ok=True)


def run_study(data_dir, output_dir, *, outer_test=None, matches=None, n_trials=8,
              rungs=(600, 1200), seed=0, device="cpu",
              poolings=("mean", "ball_attention"), pruning=False, center_size=512,
              cache_bytes=2_000_000_000, outage_config=None,
              augmentation_config=None):
    """Tune one or all outer folds; n_trials is the desired total on resume.

    Only inner-training and inner-validation prepared halves are read. The
    outer test match is named in split metadata and remains untouched.
    """
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    outage_config = _outage_dict(outage_config)
    augmentation_config = _augmentation_dict(augmentation_config)
    ids = [_match_id(value) for value in (matches if matches is not None
                                         else _discover_matches(data_dir))]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate matches")
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise ValueError("n_trials must be positive")
    if (not rungs or any(not isinstance(rung, int) or isinstance(rung, bool) or rung < 1
                         for rung in rungs)
            or tuple(rungs) != tuple(sorted(set(rungs)))):
        raise ValueError("rungs must be strictly increasing positive integers")
    if not poolings or len(poolings) != len(set(poolings)) or any(
            value not in ("mean", "ball_attention") for value in poolings):
        raise ValueError("poolings must contain unique supported values")
    if n_trials < len(poolings):
        raise ValueError("n_trials must cover every pooling baseline at full budget")
    if not isinstance(center_size, int) or center_size < 1:
        raise ValueError("center_size must be positive")
    folds = nested_splits(ids, seed=seed)
    split = {"matches": sorted(ids), "seed": seed, "inner_folds": 3,
             "outer_folds": folds}
    targets = [fold for fold in folds if outer_test is None or
               fold["outer_test"][0] == _match_id(outer_test)]
    if not targets:
        raise ValueError(f"outer test match is not in matches: {outer_test}")
    cache = HalfCache(data_dir, max_bytes=cache_bytes)
    reports = {}
    for fold in targets:
        reports[fold["outer_test"][0]] = _run_outer(
            data_dir, output_dir, fold, n_trials=n_trials, rungs=tuple(rungs),
            seed=seed, device=device, poolings=tuple(poolings), pruning=pruning,
            center_size=center_size, cache_bytes=cache_bytes, cache=cache, split=split,
            outage_config=outage_config, augmentation_config=augmentation_config)
    return reports[targets[0]["outer_test"][0]] if outer_test is not None else reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--outer-test")
    parser.add_argument("--matches", nargs="+")
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--rungs", nargs="+", type=int, default=[600, 1200])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--poolings", nargs="+", default=["mean", "ball_attention"])
    parser.add_argument("--pruning", action="store_true")
    parser.add_argument("--center-size", type=int, default=512)
    parser.add_argument("--cache-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--outage-policy", choices=("legacy", "correlated"), default="legacy")
    parser.add_argument("--augmentation-policy", choices=("none", "spatial", "temporal", "combined"),
                        default="none")
    parser.add_argument("--augmentation-config", type=Path,
                        help="JSON object with fixed augmentation settings")
    args = parser.parse_args(argv)
    augmentation_config = (json.loads(args.augmentation_config.read_text())
                           if args.augmentation_config else {"mode": args.augmentation_policy})
    report = run_study(args.data_dir, args.output_dir, outer_test=args.outer_test,
                       matches=args.matches, n_trials=args.n_trials,
                       rungs=tuple(args.rungs), seed=args.seed, device=args.device,
                       poolings=tuple(args.poolings), pruning=args.pruning,
                       center_size=args.center_size, cache_bytes=args.cache_bytes,
                       outage_config={"mode": args.outage_policy},
                       augmentation_config=augmentation_config)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
