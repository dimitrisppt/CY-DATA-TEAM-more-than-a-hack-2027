"""Freeze decoder selection on OOF matches before touching the outer match."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from evaluation import possession_metrics as official
from experiments import tune
from experiments.evaluate import _join_with_x, _target_codes
from possession.calibrate import CalibrationConfig, apply_temperature_calibration
from possession.decode import decode_half, emissions_from_probabilities
from possession.ensemble import combine_scores
from possession.predict import load_model_checkpoint, predict_features
from possession.prepared import verify_prepared
from possession.provenance import runtime_provenance
from possession.reference import reference_scores
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache, TrainingSession

DECODER_SEARCH_SPACE = {
    "neural_weight": [0.0, 1.0],
    "live_temperature": [0.5, 2.0, "log"],
    "owner_temperature": [0.5, 2.0, "log"],
    "switch_cost": [0.0, 8.0],
    "boundary_scale": 0.0,
    "sampler": "serial multivariate TPE; five startup trials",
}


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _expanded_model_kwargs(recipe_kwargs):
    """Match the defaults recorded by TrainingSession checkpoints."""
    kwargs = {"player_dim": 8, "global_dim": 9, "width": 64,
              "dilations": [1, 2, 4, 8, 16, 32, 64, 128]}
    kwargs.update(recipe_kwargs)
    kwargs["dilations"] = list(kwargs["dilations"])
    return kwargs


@contextmanager
def _locked(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "decoder.lock"
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("decoder selection already has a writer") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _prepared_halves(data_dir: Path, match: str) -> list[Path]:
    paths = [data_dir / f"{match}_half{period}.npz" for period in (1, 2)]
    actual = set(data_dir.glob(f"{match}_half*.npz"))
    if set(paths) != actual or any(not path.is_file() for path in paths):
        raise ValueError(f"expected exactly native halves 1 and 2 for {match}")
    return paths


def _load_context(data_dir: Path, outer_dir: Path):
    summary = _json(outer_dir / "summary.json")
    study_fp = _json(outer_dir / "fingerprint.json")
    split = _json(outer_dir / "splits.json")
    if (not isinstance(summary.get("outer_test"), list)
            or len(summary["outer_test"]) != 1):
        raise ValueError("HPO summary requires one outer test")
    outer = summary["outer_test"][0]
    matching = [fold for fold in split.get("outer_folds", [])
                if fold.get("outer_test") == [outer]]
    if len(matching) != 1:
        raise ValueError("HPO split lacks unique outer fold")
    fold = matching[0]
    outer_train = fold.get("outer_train")
    if (not isinstance(outer_train, list) or len(outer_train) != len(set(outer_train))
            or outer in outer_train or not outer_train):
        raise ValueError("invalid outer training split")
    if (summary.get("outer_train") != outer_train or summary.get("inner") != fold.get("inner")
            or summary.get("fingerprint_sha256") != study_fp.get("sha256")):
        raise ValueError("HPO summary, splits, and fingerprint disagree")
    inner = fold.get("inner")
    if not isinstance(inner, list) or not inner:
        raise ValueError("HPO inner folds are missing")
    validation_to_fold = {}
    for index, group in enumerate(inner):
        train, validation = group.get("train"), group.get("validation")
        if (not isinstance(train, list) or not isinstance(validation, list)
                or not validation or sorted(train) != sorted(set(outer_train) - set(validation))
                or set(train) & set(validation) or outer in train + validation):
            raise ValueError("invalid HPO inner split")
        for match in validation:
            if match in validation_to_fold:
                raise ValueError("inner validation match appears twice")
            validation_to_fold[match] = index
    if set(validation_to_fold) != set(outer_train):
        raise ValueError("inner validation must cover every outer training match once")
    payload = study_fp.get("payload")
    if not isinstance(payload, dict):
        raise TypeError("HPO fingerprint payload is missing")
    resources = payload.get("resources", {})
    poolings = payload.get("search_space", {}).get("pooling")
    if (not isinstance(poolings, list) or not poolings
            or set(poolings) != {"mean", "ball_attention"}):
        raise ValueError("decoder selection requires both pooling families")
    current = tune._fingerprint(
        data_dir, fold, split, seed=payload["seed"], rungs=resources["rungs"],
        device=resources["device"], poolings=poolings,
        pruning=payload["pruning"], center_size=resources["center_size"],
        cache_bytes=resources["cache_bytes"],
        outage_config=payload.get("fixed", {}).get("outage_config"),
        augmentation_config=payload.get("fixed", {}).get("augmentation_config"),
    )
    if current != study_fp:
        raise ValueError("HPO source or prepared fingerprint changed; refusing selection")
    if summary.get("full_budget_steps") != resources["rungs"][-1]:
        raise ValueError("HPO full budget differs from fingerprint")
    best = summary.get("best_by_pooling")
    if not isinstance(best, dict) or set(best) != set(poolings):
        raise ValueError("HPO needs a complete best artifact for each pooling")
    completed = [trial for trial in summary.get("trials", [])
                 if trial.get("state") == "COMPLETE" and trial.get("rungs")
                 and trial["rungs"][-1].get("steps_per_fold")
                    == summary["full_budget_steps"]]
    for pooling in poolings:
        candidates = [trial for trial in completed
                      if trial.get("params", {}).get("pooling") == pooling]
        if not candidates:
            raise ValueError(f"no completed full-budget HPO trial for {pooling}")
        pointer = best[pooling]
        winner = max(candidates, key=lambda trial: (trial["value"], -trial["number"]))
        if (pointer.get("trial_number") != winner["number"]
                or pointer.get("params") != winner["params"]
                or pointer.get("score") != winner["value"]):
            raise ValueError(f"HPO {pooling} best artifact is not the complete winner")
    train_paths = [path for match in outer_train for path in _prepared_halves(data_dir, match)]
    verify_prepared(data_dir, train_paths)
    return summary, study_fp, split, fold, validation_to_fold, poolings


def _checked_recipe(outer_dir, pooling, pointer, study_fp, fold, validation_to_fold):
    pointer_path = outer_dir / "best_by_pooling" / pooling / "recipe.json"
    if _json(pointer_path) != pointer:
        raise ValueError(f"HPO {pooling} pointer differs from summary")
    number = pointer.get("trial_number")
    if not isinstance(number, int) or isinstance(number, bool) or number < 0:
        raise ValueError("HPO recipe trial number is invalid")
    artifact_dir = Path(pointer.get("artifact_dir", "")).resolve()
    expected_dir = (outer_dir / "trials" / f"trial_{number}").resolve()
    if artifact_dir != expected_dir or _json(artifact_dir / "recipe.json") != pointer:
        raise ValueError("HPO immutable trial recipe or artifact path differs")
    if (pointer.get("study_fingerprint_sha256") != study_fp["sha256"]
            or pointer.get("params", {}).get("pooling") != pooling
            or pointer.get("model_kwargs", {}).get("pooling") != pooling
            or not math.isfinite(pointer.get("score", float("nan")))):
        raise ValueError("HPO recipe fingerprint, pooling, or score differs")
    hashes = pointer.get("artifact_sha256")
    folds = pointer.get("fold_checkpoints")
    oof = pointer.get("oof_files")
    if not isinstance(hashes, dict) or not isinstance(folds, list) or not isinstance(oof, list):
        raise TypeError("HPO recipe artifact manifest is missing")
    expected_names = set()
    if len(folds) != len(fold["inner"]):
        raise ValueError("HPO fold checkpoint count differs")
    for index, item in enumerate(folds):
        name = item.get("file")
        if (item.get("inner_fold") != index or name != f"fold_{index}.pt"
                or item.get("train_matches") != fold["inner"][index]["train"]
                or item.get("validation_matches") != fold["inner"][index]["validation"]):
            raise ValueError("HPO fold checkpoint ancestry differs")
        expected_names.add(name)
    expected_oof = {(match, period) for match in fold["outer_train"] for period in (1, 2)}
    seen_oof = set()
    for item in oof:
        match, period = item.get("match"), item.get("period")
        if (match, period) not in expected_oof or (match, period) in seen_oof:
            raise ValueError("HPO OOF match/period coverage differs")
        seen_oof.add((match, period))
        index = validation_to_fold[match]
        name = f"{match}_half{period}.npz"
        sidecar = f"{match}_half{period}.json"
        if (item.get("inner_fold") != index or item.get("file") != name
                or item.get("sidecar") != sidecar):
            raise ValueError("HPO OOF manifest ancestry differs")
        expected_names.update((name, sidecar))
    if seen_oof != expected_oof or set(hashes) != expected_names:
        raise ValueError("HPO OOF artifacts are incomplete")
    for name, expected in hashes.items():
        if (Path(name).name != name or not isinstance(expected, str)
                or sha256_file(artifact_dir / name) != expected):
            raise ValueError(f"HPO artifact hash differs: {name}")
    for item in folds:
        name = item["file"]
        if item.get("sha256") != hashes[name]:
            raise ValueError("HPO fold checkpoint hash differs")
        checkpoint = torch.load(artifact_dir / name, map_location="cpu", weights_only=True)
        config = checkpoint.get("training_config", {})
        if (config.get("train_matches") != item["train_matches"]
                or config.get("validation_matches") != item["validation_matches"]
                or config.get("outer_test") != fold["outer_test"][0]
                or config.get("trial_number") != number
                or config.get("inner_fold") != item["inner_fold"]
                or ("outage_config" in pointer["training_args"]
                    and config.get("outage_config") != pointer["training_args"]["outage_config"])
                or ("augmentation_config" in pointer["training_args"]
                    and ("augmentation_config" not in config
                         or tune._augmentation_dict(config["augmentation_config"])
                         != pointer["training_args"]["augmentation_config"]))
                or checkpoint.get("model_kwargs")
                   != _expanded_model_kwargs(pointer["model_kwargs"])):
            raise ValueError("HPO checkpoint training ancestry differs")
    for item in oof:
        if (item.get("sha256") != hashes[item["file"]]
                or item.get("sidecar_sha256") != hashes[item["sidecar"]]):
            raise ValueError("HPO OOF hash manifest differs")
        sidecar = _json(artifact_dir / item["sidecar"])
        match, index = item["match"], item["inner_fold"]
        if (sidecar.get("prediction_match") != match
                or sidecar.get("period") != item["period"]
                or sidecar.get("train_matches") != fold["inner"][index]["train"]
                or sidecar.get("outer_test") != fold["outer_test"][0]
                or sidecar.get("trial_number") != number
                or sidecar.get("inner_fold") != index
                or sidecar.get("checkpoint_file") != f"fold_{index}.pt"
                or sidecar.get("checkpoint_sha256") != hashes[f"fold_{index}.pt"]
                or sidecar.get("oof_file") != item["file"]
                or sidecar.get("oof_sha256") != hashes[item["file"]]
                or sidecar.get("study_fingerprint_sha256") != study_fp["sha256"]
                or set(sidecar.get("arrays", [])) != {
                    "frame_id", "gt", "live_probability", "home_probability", "boundary"}):
            raise ValueError("HPO OOF sidecar ancestry differs")
    return artifact_dir


def _load_oof(data_dir, outer_dir, summary, study_fp, fold, validation_to_fold,
              poolings, cache):
    references = {}
    targets = {}
    for match in fold["outer_train"]:
        for period, path in enumerate(_prepared_halves(data_dir, match), 1):
            frames, labels, _, _ = cache.get(path)
            if (frames.match_id != match or frames.period != period
                    or frames.fps != official.FPS):
                raise ValueError("prepared half identity or FPS differs")
            key = (match, period)
            references[key] = reference_scores(frames)
            targets[key] = (frames.frame_id.copy(), _target_codes(frames, labels))
    result = {}
    artifact_fingerprints = {}
    for pooling in poolings:
        recipe = summary["best_by_pooling"][pooling]
        artifact_dir = _checked_recipe(outer_dir, pooling, recipe, study_fp,
                                       fold, validation_to_fold)
        records = {}
        for item in recipe["oof_files"]:
            match, period = item["match"], item["period"]
            key = (match, period)
            with np.load(artifact_dir / item["file"], allow_pickle=False) as arrays:
                if set(arrays.files) != {
                    "frame_id", "gt", "live_probability", "home_probability", "boundary"}:
                    raise ValueError("HPO OOF array names differ")
                frame_id = arrays["frame_id"]
                gt = arrays["gt"]
                neural = {field: arrays[field].copy() for field in
                          ("live_probability", "home_probability", "boundary")}
            expected_frame_id, expected_gt = targets[key]
            if (not np.array_equal(frame_id, expected_frame_id)
                    or not np.array_equal(gt, expected_gt)):
                raise ValueError("HPO OOF frame IDs or ground truth differ from prepared source")
            combine_scores(neural, references[key], 1.0)  # Validate all head shapes/values.
            records[key] = {"match": match, "period": period, "gt": expected_gt,
                            "neural": neural, "reference": references[key]}
        result[pooling] = records
        artifact_fingerprints[pooling] = {
            "recipe_sha256": sha256_file(artifact_dir / "recipe.json"),
            "artifact_sha256": recipe["artifact_sha256"],
        }
    return result, artifact_fingerprints


def _score_records(records, params):
    calibration = CalibrationConfig(
        live_temperature=params["live_temperature"],
        owner_temperature=params["owner_temperature"],
    )
    by_match = {}
    for match in sorted({key[0] for key in records}):
        gt_parts, pred_parts = [], []
        for period in (1, 2):
            record = records[(match, period)]
            blended = combine_scores(record["neural"], record["reference"],
                                     params["neural_weight"])
            live, home, boundary = apply_temperature_calibration(
                blended["live_probability"], blended["home_probability"],
                blended["boundary"], calibration,
            )
            pred = decode_half(emissions_from_probabilities(live, home),
                               boundary * 0.0, switch_cost=params["switch_cost"])
            gt_parts.append(record["gt"])
            pred_parts.append(pred)
        metrics = official.evaluate(_join_with_x(gt_parts), _join_with_x(pred_parts),
                                    bootstrap_n=0)
        by_match[match] = {"spell_f1_mean": float(metrics["spell_f1_mean"]),
                           "official": metrics}
    score = float(np.mean([entry["spell_f1_mean"] for entry in by_match.values()]))
    return score, by_match


def _suggest(trial):
    return {
        "neural_weight": trial.suggest_float("neural_weight", 0.0, 1.0),
        "live_temperature": trial.suggest_float("live_temperature", 0.5, 2.0, log=True),
        "owner_temperature": trial.suggest_float("owner_temperature", 0.5, 2.0, log=True),
        "switch_cost": trial.suggest_float("switch_cost", 0.0, 8.0),
    }


def _trial_summary(study):
    optuna = tune._optuna()
    attempts = [trial for trial in study.trials
                if trial.state != optuna.trial.TrialState.WAITING]
    return [{"number": trial.number, "state": trial.state.name,
             "value": trial.value, "params": trial.params,
             "by_match": trial.user_attrs.get("by_match")}
            for trial in attempts]


def _decoder_fingerprint(study_fp, summary, artifact_fingerprints, *, seed,
                         device, cache_bytes):
    payload = {
        "protocol": "OOF single-weight decoder selection v1",
        "hpo_study_fingerprint_sha256": study_fp["sha256"],
        "hpo_summary_sha256": _digest(summary),
        "oof_artifacts": artifact_fingerprints,
        "search_space": DECODER_SEARCH_SPACE,
        "seed": seed, "device": str(device), "cache_bytes": cache_bytes,
    }
    return {"sha256": _digest(payload), "payload": payload}


def _run_decoder_studies(outer_dir, summary, poolings, records, fingerprint,
                         *, n_trials, seed):
    optuna = tune._optuna()
    family_reports = {}
    for pooling in poolings:
        database = outer_dir / f"decoder_{pooling}.sqlite3"
        storage = f"sqlite:///{database}"
        study_name = f"decoder-{summary['outer_test'][0]}-{pooling}"
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True,
                                             n_startup_trials=5)
        study = optuna.create_study(study_name=study_name, storage=storage,
                                    load_if_exists=True, direction="maximize",
                                    sampler=sampler)
        previous = study.user_attrs.get("fingerprint_sha256")
        if previous is not None and previous != fingerprint["sha256"]:
            raise ValueError("decoder study fingerprint changed; refusing resume")
        study.set_user_attr("fingerprint_sha256", fingerprint["sha256"])
        if any(trial.state == optuna.trial.TrialState.RUNNING for trial in study.trials):
            raise RuntimeError("interrupted RUNNING decoder trial requires recovery")
        if not study.trials:
            neural_cost = summary["best_by_pooling"][pooling]["decoder"]["switch_cost"]
            for anchor in (
                {"neural_weight": 0.0, "live_temperature": 1.0,
                 "owner_temperature": 1.0, "switch_cost": 2.0},
                {"neural_weight": 1.0, "live_temperature": 1.0,
                 "owner_temperature": 1.0, "switch_cost": neural_cost},
                {"neural_weight": 0.5, "live_temperature": 1.0,
                 "owner_temperature": 1.0, "switch_cost": 2.0},
            ):
                study.enqueue_trial(anchor)
        completed = sum(trial.state in (optuna.trial.TrialState.COMPLETE,
                                        optuna.trial.TrialState.FAIL,
                                        optuna.trial.TrialState.PRUNED)
                        for trial in study.trials)
        if completed > n_trials:
            raise ValueError("n_trials is smaller than attempted decoder trials")
        for _ in range(n_trials - completed):
            next_number = max((trial.number for trial in study.trials), default=-1) + 1
            study = optuna.load_study(study_name=study_name, storage=storage,
                                      sampler=optuna.samplers.TPESampler(
                                          seed=seed + next_number, multivariate=True,
                                          n_startup_trials=5))
            trial = study.ask()
            try:
                params = _suggest(trial)
                score, by_match = _score_records(records[pooling], params)
                trial.set_user_attr("by_match", by_match)
                study.tell(trial, score)
            except Exception:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                raise
        trials = _trial_summary(study)
        winners = [trial for trial in trials if trial["state"] == "COMPLETE"]
        if not winners:
            raise ValueError("decoder study has no complete trials")
        family_reports[pooling] = {
            "n_trials_total": len(trials), "trials": trials,
            "best": max(winners, key=lambda trial: (trial["value"], -trial["number"])),
        }
    reference_params = {"neural_weight": 0.0, "live_temperature": 1.0,
                        "owner_temperature": 1.0, "switch_cost": 2.0}
    baseline_score, baseline_by_match = _score_records(records[poolings[0]], reference_params)
    report = {
        "protocol": "inner OOF decoder selection; official unweighted match mean",
        "selection_status": "inner selection criterion, not independent stack validation",
        "outer_test": summary["outer_test"][0],
        "outer_train": summary["outer_train"],
        "fingerprint_sha256": fingerprint["sha256"],
        "reference_baseline": {"score": baseline_score, "by_match": baseline_by_match},
        "by_pooling": family_reports,
    }
    write_json(outer_dir / "decoder_summary.json", report)
    return _json(outer_dir / "decoder_summary.json")


def _winner(decoder_summary):
    candidates = []
    for pooling, family in decoder_summary["by_pooling"].items():
        for trial in family["trials"]:
            if trial["state"] == "COMPLETE":
                reference_priority = int(trial["params"]["neural_weight"] != 0)
                family_priority = 0 if pooling == "mean" else 1
                candidates.append((trial["value"], -reference_priority,
                                   -family_priority, -trial["number"], pooling, trial))
    if not candidates:
        raise ValueError("no completed decoder candidate")
    _, _, _, _, pooling, trial = max(candidates)
    return pooling, trial


def _refit_seed(seed: int, outer: str) -> int:
    digest = hashlib.sha256(f"{seed}:{outer}:refit".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _freeze(outer_dir, summary, decoder_summary, fingerprint, *, seed,
            device, cache_bytes, n_trials):
    pooling, trial = _winner(decoder_summary)
    weight = float(trial["params"]["neural_weight"])
    recipe = None if weight == 0 else summary["best_by_pooling"][pooling]
    frozen = {
        "protocol": "frozen inner-selected decoder before outer evaluation",
        "outer_test": summary["outer_test"][0],
        "outer_train": summary["outer_train"],
        "selected_pooling": "reference" if recipe is None else pooling,
        "selected_model_recipe": recipe,
        "selected_decoder_trial": {"pooling": pooling, **trial},
        "selection_score": trial["value"],
        "selection_score_type": "inner OOF official unweighted match mean",
        "ensemble": {"neural_weight": weight},
        "calibration": CalibrationConfig(
            live_temperature=trial["params"]["live_temperature"],
            owner_temperature=trial["params"]["owner_temperature"],
        ).to_dict(),
        "decoder": {"switch_cost": trial["params"]["switch_cost"],
                    "boundary_scale": 0.0},
        "selection_fingerprint": fingerprint["sha256"],
        "hpo_fingerprint_sha256": fingerprint["payload"]["hpo_study_fingerprint_sha256"],
        "decoder_fingerprint_sha256": fingerprint["sha256"],
        "refit_seed": _refit_seed(seed, summary["outer_test"][0]),
        "resources": {"n_trials_per_pooling": n_trials, "device": str(device),
                      "cache_bytes": cache_bytes,
                      "full_budget_steps": summary["full_budget_steps"]},
    }
    write_json(outer_dir / "frozen_recipe.json", frozen)
    return _json(outer_dir / "frozen_recipe.json")


def _validate_frozen(outer_dir, frozen, decoder_summary, fingerprint,
                     *, summary, poolings, n_trials, seed, device, cache_bytes):
    if (frozen.get("selection_fingerprint") != fingerprint["sha256"]
            or decoder_summary.get("fingerprint_sha256") != fingerprint["sha256"]
            or frozen.get("resources") != {
                "n_trials_per_pooling": n_trials, "device": str(device),
                "cache_bytes": cache_bytes,
                "full_budget_steps": summary["full_budget_steps"]}
            or frozen.get("outer_test") != summary["outer_test"][0]
            or frozen.get("outer_train") != summary["outer_train"]
            or frozen.get("refit_seed") != _refit_seed(seed, summary["outer_test"][0])):
        raise ValueError("frozen recipe settings or artifacts differ")
    if set(decoder_summary.get("by_pooling", {})) != set(poolings):
        raise ValueError("frozen decoder pooling families differ")
    optuna = tune._optuna()
    for pooling in poolings:
        family = decoder_summary["by_pooling"][pooling]
        if family.get("n_trials_total") != n_trials:
            raise ValueError("frozen decoder trial count differs")
        study = optuna.load_study(
            study_name=f"decoder-{summary['outer_test'][0]}-{pooling}",
            storage=f"sqlite:///{outer_dir / f'decoder_{pooling}.sqlite3'}")
        if (study.user_attrs.get("fingerprint_sha256") != fingerprint["sha256"]
                or _trial_summary(study) != family["trials"]):
            raise ValueError("frozen decoder summary differs from persistent study")
    pooling, trial = _winner(decoder_summary)
    recipe = None if trial["params"]["neural_weight"] == 0 else summary["best_by_pooling"][pooling]
    expected_calibration = CalibrationConfig(
        live_temperature=trial["params"]["live_temperature"],
        owner_temperature=trial["params"]["owner_temperature"],
    ).to_dict()
    if (frozen.get("selected_pooling") != ("reference" if recipe is None else pooling)
            or frozen.get("selected_model_recipe") != recipe
            or frozen.get("selected_decoder_trial") != {"pooling": pooling, **trial}
            or frozen.get("selection_score") != trial["value"]
            or frozen.get("ensemble") != {"neural_weight": trial["params"]["neural_weight"]}
            or frozen.get("calibration") != expected_calibration
            or frozen.get("decoder") != {"switch_cost": trial["params"]["switch_cost"],
                                         "boundary_scale": 0.0}):
        raise ValueError("frozen selection differs from persistent decoder winner")


def _ensure_refit(data_dir, outer_dir, frozen, cache, *, device):
    progress_path = outer_dir / "refit_progress.json"
    checkpoint_path = outer_dir / "selected.pt"
    if frozen["selected_model_recipe"] is None:
        if checkpoint_path.exists():
            raise ValueError("reference selection unexpectedly has a neural checkpoint")
        progress = {"predictor": "reference", "steps": 0,
                    "train_matches": frozen["outer_train"]}
        if progress_path.exists() and _json(progress_path) != progress:
            raise ValueError("reference refit progress differs from frozen recipe")
        write_json(progress_path, progress)
        return progress
    recipe = frozen["selected_model_recipe"]
    if checkpoint_path.exists():
        if not progress_path.exists():
            raise ValueError("selected checkpoint lacks refit progress")
        progress = _json(progress_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = payload.get("training_config", {})
        if (config.get("train_matches") != frozen["outer_train"]
                or config.get("validation_matches") != []
                or config.get("outer_test") != frozen["outer_test"]
                or config.get("selection_fingerprint") != frozen["selection_fingerprint"]
                or config.get("selection_matches") != frozen["outer_train"]
                or config.get("decoder") != frozen["decoder"]
                or config.get("calibration") != frozen["calibration"]
                or config.get("ensemble") != frozen["ensemble"]
                or config.get("steps") != frozen["resources"]["full_budget_steps"]
                or config.get("seed") != frozen["refit_seed"]
                or ("outage_config" in recipe["training_args"]
                    and config.get("outage_config") != recipe["training_args"]["outage_config"])
                or ("augmentation_config" in recipe["training_args"]
                    and ("augmentation_config" not in config
                         or tune._augmentation_dict(config["augmentation_config"])
                         != recipe["training_args"]["augmentation_config"]))
                or payload.get("model_kwargs")
                   != _expanded_model_kwargs(recipe["model_kwargs"])
                or progress.get("checkpoint_sha256") != sha256_file(checkpoint_path)):
            raise ValueError("selected checkpoint ancestry differs from frozen recipe")
        return progress
    if progress_path.exists():
        raise ValueError("refit progress exists without selected checkpoint")
    args = recipe["training_args"]
    session = TrainingSession(
        data_dir, train_matches=frozen["outer_train"], seed=frozen["refit_seed"],
        device=device, model_kwargs=recipe["model_kwargs"],
        learning_rate=args["learning_rate"], weight_decay=args["weight_decay"],
        boundary_weight=args["boundary_weight"],
        boundary_pos_weight=args["boundary_pos_weight"],
        augment=args["augment"], center_size=args["center_size"], cache=cache,
        outage_config=args.get("outage_config"),
        augmentation_config=args.get("augmentation_config"),
    )
    progress = session.advance_to(frozen["resources"]["full_budget_steps"])
    session.save_checkpoint(
        checkpoint_path, extra_training_config={
            "decoder": frozen["decoder"], "calibration": frozen["calibration"],
            "ensemble": frozen["ensemble"],
            "selection_matches": frozen["outer_train"],
            "outer_test": frozen["outer_test"],
            "selection_fingerprint": frozen["selection_fingerprint"],
        },
    )
    progress = {**progress, "predictor": "selected_neural",
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "train_matches": frozen["outer_train"]}
    write_json(progress_path, progress)
    return progress


def _evaluate_outer(data_dir, outer_dir, frozen, cache, *, device):
    # Called only after frozen_recipe.json has been published.
    if not (outer_dir / "frozen_recipe.json").is_file():
        raise RuntimeError("outer evaluation requires a frozen recipe")
    outer = frozen["outer_test"]
    paths = _prepared_halves(data_dir, outer)
    provenance = verify_prepared(data_dir, paths)
    checkpoint_path = outer_dir / "selected.pt"
    model = None
    if frozen["selected_model_recipe"] is not None:
        model, payload = load_model_checkpoint(checkpoint_path, device=device)
        if set(payload["training_config"]["train_matches"]) & {outer}:
            raise ValueError("outer test appears in selected checkpoint training matches")
    calibration = CalibrationConfig(**frozen["calibration"])
    gt_parts, pred_parts, baseline_parts, half_reports = [], [], [], []
    prepared_hashes = {}
    for period, path in enumerate(paths, 1):
        prepared_hashes[path.name] = sha256_file(path)
        frames, targets, features, _ = cache.get(path)
        if (frames.match_id != outer or frames.period != period
                or frames.fps != official.FPS):
            raise ValueError("outer prepared half identity or FPS differs")
        gt = _target_codes(frames, targets)
        reference = reference_scores(frames)
        neural = (reference if model is None else predict_features(
            features, model, device=device,
            center_size=frozen["selected_model_recipe"]["training_args"]["center_size"],
            halo=model.radius,
        ))
        combined = combine_scores(neural, reference, frozen["ensemble"]["neural_weight"])
        live, home, boundary = apply_temperature_calibration(
            combined["live_probability"], combined["home_probability"],
            combined["boundary"], calibration,
        )
        pred = decode_half(emissions_from_probabilities(live, home),
                           boundary * frozen["decoder"]["boundary_scale"],
                           switch_cost=frozen["decoder"]["switch_cost"])
        baseline = decode_half(emissions_from_probabilities(
            reference["live_probability"], reference["home_probability"]),
            reference["boundary"], switch_cost=2.0)
        gt_parts.append(gt)
        pred_parts.append(pred)
        baseline_parts.append(baseline)
        half_reports.append({"period": period, "n_frames_native": len(gt),
                             "official": official.evaluate(gt, pred, bootstrap_n=0),
                             "reference_official": official.evaluate(gt, baseline,
                                                                     bootstrap_n=0)})
    official_result = official.evaluate(_join_with_x(gt_parts), _join_with_x(pred_parts),
                                        bootstrap_n=0)
    baseline_result = official.evaluate(_join_with_x(gt_parts),
                                        _join_with_x(baseline_parts), bootstrap_n=0)
    return {
        "protocol": "single untouched outer match; frozen inner-selected recipe",
        "outer_test": outer, "outer_train": frozen["outer_train"],
        "selection_score": frozen["selection_score"],
        "selection_score_type": frozen["selection_score_type"],
        "predictor": frozen["selected_pooling"],
        "frozen_recipe_sha256": sha256_file(outer_dir / "frozen_recipe.json"),
        "selection_fingerprint": frozen["selection_fingerprint"],
        "checkpoint_sha256": (sha256_file(checkpoint_path) if model is not None else None),
        "prepared_sha256": prepared_hashes,
        "dataset_provenance": provenance,
        "runtime": runtime_provenance(),
        "by_match": {outer: {"halves": half_reports,
                             "n_frames_native": sum(len(gt) for gt in gt_parts),
                             "official": official_result,
                             "reference_official": baseline_result}},
        "official": official_result,
        "reference_official": baseline_result,
    }


def _cached_outer_report(data_dir, outer_dir, frozen):
    report_path = outer_dir / "outer_report.json"
    report_hash = _json(outer_dir / "outer_report.sha256.json")
    if report_hash.get("sha256") != sha256_file(report_path):
        raise ValueError("cached outer report hash differs")
    report = _json(report_path)
    checkpoint_path = outer_dir / "selected.pt"
    expected_checkpoint = (sha256_file(checkpoint_path) if checkpoint_path.exists() else None)
    paths = _prepared_halves(data_dir, frozen["outer_test"])
    if (report.get("outer_test") != frozen["outer_test"]
            or report.get("outer_train") != frozen["outer_train"]
            or report.get("selection_fingerprint") != frozen["selection_fingerprint"]
            or report.get("frozen_recipe_sha256") != sha256_file(outer_dir / "frozen_recipe.json")
            or report.get("checkpoint_sha256") != expected_checkpoint
            or report.get("prepared_sha256") != {path.name: sha256_file(path)
                                                    for path in paths}):
        raise ValueError("cached outer report artifact or configuration differs")
    return report


def select_and_evaluate(
    data_dir, outer_dir, *, n_trials=32, seed=0, device="cpu",
    cache_bytes=2_000_000_000, evaluate_outer=True,
):
    """Select decoder on immutable OOF artifacts, freeze, refit, then score outer.

    `n_trials` is the desired total *per pooling* and may increase on an
    unfinished study. Once frozen, all selection settings and trial counts are
    fixed. `evaluate_outer=False` freezes and refits without loading outer data.
    """
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 3:
        raise ValueError("n_trials must be at least three for decoder anchors")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(cache_bytes, int) or isinstance(cache_bytes, bool) or cache_bytes < 1:
        raise ValueError("cache_bytes must be positive")
    data_dir, outer_dir = Path(data_dir).resolve(), Path(outer_dir).resolve()
    with _locked(outer_dir):
        summary, study_fp, _, fold, validation_to_fold, poolings = _load_context(
            data_dir, outer_dir)
        cache = HalfCache(data_dir, max_bytes=cache_bytes)
        records, artifact_fingerprints = _load_oof(
            data_dir, outer_dir, summary, study_fp, fold, validation_to_fold,
            poolings, cache)
        fingerprint = _decoder_fingerprint(
            study_fp, summary, artifact_fingerprints, seed=seed,
            device=device, cache_bytes=cache_bytes)
        fp_path = outer_dir / "decoder_fingerprint.json"
        existing_databases = list(outer_dir.glob("decoder_*.sqlite3"))
        if existing_databases and not fp_path.is_file():
            raise ValueError("decoder studies lack fingerprint; refusing resume")
        if fp_path.exists() and _json(fp_path) != fingerprint:
            raise ValueError("decoder fingerprint changed; refusing resume")
        if not fp_path.exists():
            write_json(fp_path, fingerprint)
        frozen_path = outer_dir / "frozen_recipe.json"
        decoder_summary_path = outer_dir / "decoder_summary.json"
        report_path = outer_dir / "outer_report.json"
        checkpoint_path = outer_dir / "selected.pt"
        progress_path = outer_dir / "refit_progress.json"
        if (report_path.exists() or checkpoint_path.exists() or progress_path.exists()) \
                and not frozen_path.exists():
            raise ValueError("completed artifacts lack frozen recipe; refusing resume")
        if frozen_path.exists():
            frozen = _json(frozen_path)
            decoder_summary = _json(decoder_summary_path)
            _validate_frozen(
                outer_dir, frozen, decoder_summary, fingerprint, summary=summary,
                poolings=poolings, n_trials=n_trials, seed=seed, device=device,
                cache_bytes=cache_bytes)
        else:
            decoder_summary = _run_decoder_studies(
                outer_dir, summary, poolings, records, fingerprint,
                n_trials=n_trials, seed=seed)
            frozen = _freeze(outer_dir, summary, decoder_summary, fingerprint,
                             seed=seed, device=device, cache_bytes=cache_bytes,
                             n_trials=n_trials)
        # The first operation allowed to access outer prepared features follows
        # the durable frozen_recipe.json write above.
        if report_path.exists() and (
            not progress_path.exists() or
            (frozen["selected_model_recipe"] is not None and not checkpoint_path.exists())
        ):
            raise ValueError("completed outer report lacks refit artifact; refusing retrain")
        refit_progress = _ensure_refit(data_dir, outer_dir, frozen, cache, device=device)
        outer_report = None
        if evaluate_outer:
            if report_path.exists():
                outer_report = _cached_outer_report(data_dir, outer_dir, frozen)
            else:
                outer_report = _evaluate_outer(data_dir, outer_dir, frozen, cache,
                                               device=device)
                write_json(report_path, outer_report)
                write_json(outer_dir / "outer_report.sha256.json",
                           {"sha256": sha256_file(report_path)})
                outer_report = _json(report_path)
        return {"frozen_recipe": frozen, "decoder_summary": decoder_summary,
                "refit_progress": refit_progress, "outer_report": outer_report}
