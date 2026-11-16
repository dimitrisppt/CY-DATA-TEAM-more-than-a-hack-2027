"""Development-only, seeded learning-curve study for the frozen WMX baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.augmentation_evaluation import evaluate_augmented_model
from experiments.augmentation_tuning import (
    FAMILIES,
    _load_record,
    _plan,
    _save_record,
)
from experiments.robustness import SCENARIOS, _digest, _lock
from experiments.tune import _seed_for_fold
from possession.augmentation import AugmentationConfig
from possession.predict import load_model_checkpoint
from possession.storage import sha256_file
from possession.training_session import HalfCache, TrainingSession

SCENARIO_NAMES = tuple(SCENARIOS)
DIAGNOSTICS = ("blend", "neural_only")


def _positive_ints(name, values):
    values = tuple(values)
    if (not values or any(isinstance(value, bool) or not isinstance(value, int)
                          or value < 0 for value in values) or len(set(values)) != len(values)):
        raise ValueError(f"{name} must be nonempty, unique nonnegative integers")
    return values


def _protocol(data_dir, recipe_path, *, seeds, rungs, cache_bytes,
              inference_center_size):
    seeds = _positive_ints("seeds", seeds)
    rungs = _positive_ints("rungs", rungs)
    if rungs[0] < 1 or rungs != tuple(sorted(rungs)):
        raise ValueError("rungs must be positive and increase strictly")
    # Reuse the policy study's recipe, split, provenance, environment, and source
    # validation. Its search settings are discarded; this is a separate protocol.
    base = _plan(data_dir, recipe_path, steps=rungs[-1], n_trials=len(FAMILIES),
                 seed=seeds[0], confirmation_seed=next(n for n in range(1, max(seeds) + 2)
                                                        if n not in seeds),
                 outer_seeds=(20, 21), cache_bytes=cache_bytes, warmup_steps=0,
                 inference_center_size=inference_center_size)["payload"]
    fold = base["fold"]
    if fold["outer_test"] != ["DFL-MAT-J03WMX"] or len(fold["outer_train"]) != 6 \
            or len(fold["inner"]) != 3:
        raise ValueError("convergence requires the WMX recipe's six-match, three-fold split")
    payload = {
        "protocol": "wmx_baseline_convergence_v1_development",
        "fold": fold, "fixed": base["fixed"],
        "tracking_augmentation_config": AugmentationConfig().to_dict(),
        "outage_policy": {"mode": "legacy"},
        "resources": {"seeds": list(seeds), "rungs": list(rungs), "cache_bytes": cache_bytes,
                      "device": "cpu", "workers": 1,
                      "inference_center_size": inference_center_size},
        "scenarios": list(SCENARIO_NAMES), "scenario_specs": SCENARIOS,
        "selection": {"objective": ".5 clean + .5 mean of two short-gap seeds",
                      "weighting": "equal validation matches within seed, equal seeds",
                      "rule": "smallest rung within 0.002 of best mean objective",
                      "development_only": True, "global_default_promotion": False},
        "source_recipe_sha256": base["source_recipe_sha256"],
        "prepared_sha256": base["prepared_sha256"],
        "prepared_provenance": base["prepared_provenance"],
        "source_sha256": base["source_sha256"],
        "environment": base["environment"],
    }
    payload = json.loads(json.dumps(payload))
    return {"sha256": _digest(payload), "payload": payload}


def _publish(path, value):
    existing = _load_record(path)
    if existing is not None:
        if existing != value:
            raise ValueError(f"completed record differs: {path}")
        return existing
    _save_record(path, value)
    return value


def _job(plan, seed, fold_index, inner, rung):
    return {"protocol_sha256": plan["sha256"], "seed": seed,
            "training_seed": _seed_for_fold(seed, plan["payload"]["fold"]["outer_test"][0],
                                            fold_index),
            "fold_index": fold_index, "train_matches": inner["train"],
            "validation_matches": inner["validation"], "steps": rung,
            "scenarios": list(SCENARIO_NAMES)}


def _session(data_dir, plan, cache, job):
    fixed = plan["payload"]["fixed"]["selected_model_recipe"]
    return TrainingSession(
        data_dir, train_matches=job["train_matches"], seed=job["training_seed"],
        device="cpu", cache=cache, model_kwargs=fixed["model_kwargs"],
        **fixed["training_args"], augment=True, outage_config={"mode": "legacy"},
        augmentation_config=plan["payload"]["tracking_augmentation_config"],
    )


def _verify_fit(directory, job, fit, plan):
    checkpoint = directory / "model.pt"
    if fit["job"] != job or fit["progress"]["steps"] != job["steps"]:
        raise ValueError("cached fit ancestry differs")
    if not checkpoint.is_file() or fit["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("cached checkpoint hash differs")
    model, payload = load_model_checkpoint(checkpoint)
    expected = {"train_matches": job["train_matches"],
                "validation_matches": job["validation_matches"],
                "seed": job["training_seed"], "steps": job["steps"],
                "augmentation_config": plan["payload"]["tracking_augmentation_config"],
                "convergence_protocol_sha256": plan["sha256"]}
    if any(payload["training_config"].get(key) != value for key, value in expected.items()):
        raise ValueError("cached checkpoint training configuration differs")
    if payload["training_config"].get("outage_config", {}).get("mode") != "legacy":
        raise ValueError("cached checkpoint outage policy differs")
    return model, payload


def _verify_replay(session, fit, payload):
    if session.advance_to(fit["job"]["steps"]) != fit["progress"]:
        raise ValueError("deterministic replay progress or training-window hash differs")
    current = session.model.state_dict()
    saved = payload["state_dict"]
    if set(current) != set(saved) or any(not torch.equal(current[key].cpu(), saved[key])
                                         for key in current):
        raise ValueError("deterministic replay model tensors differ")


def _score(data_dir, directory, plan, cache, fit, model):
    job = fit["job"]
    started = time.monotonic()
    by_match = {}
    for match in job["validation_matches"]:
        by_match[match] = evaluate_augmented_model(
            data_dir, match, model, plan["payload"]["fixed"], cache,
            scenarios=SCENARIO_NAMES,
            inference_center_size=plan["payload"]["resources"]["inference_center_size"],
            include_diagnostics=True)
    result = {"job": job, "checkpoint_sha256": fit["checkpoint_sha256"],
              "by_match": by_match, "score_seconds": time.monotonic() - started}
    return _publish(directory / "result.json", result)


def _trajectory(data_dir, directory, plan, cache, *, seed, fold_index, inner):
    session = None
    prior = []
    results = []
    for rung in plan["payload"]["resources"]["rungs"]:
        job = _job(plan, seed, fold_index, inner, rung)
        rung_dir = directory / f"rung_{rung}"
        rung_dir.mkdir(parents=True, exist_ok=True)
        _publish(rung_dir / "job.json", job)
        fit_path = rung_dir / "fit.json"
        fit = _load_record(fit_path)
        checkpoint = rung_dir / "model.pt"
        if fit is not None and session is not None:
            raise ValueError("noncontiguous completed rungs")
        if fit is None:
            if checkpoint.exists():
                raise ValueError("checkpoint without completed fit record; inspect incomplete rung")
            if session is None:
                session = _session(data_dir, plan, cache, job)
                for prior_fit, prior_payload in prior:
                    _verify_replay(session, prior_fit, prior_payload)
            started = time.monotonic()
            progress = session.advance_to(rung)
            fit_seconds = time.monotonic() - started
            session.save_checkpoint(checkpoint, validation_matches=inner["validation"],
                                    extra_training_config={
                                        "convergence_protocol_sha256": plan["sha256"]})
            fit = _publish(fit_path, {"job": job, "progress": progress,
                                      "fit_seconds": fit_seconds,
                                      "checkpoint_sha256": sha256_file(checkpoint)})
        model, payload = _verify_fit(rung_dir, job, fit, plan)
        if session is None:
            prior.append((fit, payload))
        result = _load_record(rung_dir / "result.json")
        if result is None:
            result = _score(data_dir, rung_dir, plan, cache, fit, model)
        elif (result["job"] != job or result["checkpoint_sha256"] != fit["checkpoint_sha256"]
              or set(result["by_match"]) != set(inner["validation"])
              or any(set(scores) != set(SCENARIO_NAMES)
                     for scores in result["by_match"].values())):
            raise ValueError("cached score ancestry or scenarios differ")
        results.append({"fit": fit, "result": result})
    return results


def _aggregate(raw, plan):
    rungs = plan["payload"]["resources"]["rungs"]
    seeds = plan["payload"]["resources"]["seeds"]
    expected_matches = set(plan["payload"]["fold"]["outer_train"])
    by_rung = {}
    for rung in rungs:
        seed_scores = {}
        for seed in seeds:
            by_match = {}
            for fold in raw[str(seed)]:
                result = fold[str(rung)]["result"]
                if set(by_match) & set(result["by_match"]):
                    raise ValueError("duplicate validation match")
                by_match.update(result["by_match"])
            if set(by_match) != expected_matches:
                raise ValueError("validation matches do not partition the six development matches")
            scores = {}
            for diagnostic in DIAGNOSTICS:
                scores[diagnostic] = {
                    scenario: float(np.mean([by_match[match][scenario]["scores"][diagnostic]
                                              ["spell_f1_mean"] for match in sorted(by_match)]))
                    for scenario in SCENARIO_NAMES
                }
                scores[diagnostic]["short_mean"] = float(np.mean([
                    scores[diagnostic]["block41_seed3101"],
                    scores[diagnostic]["block41_seed3102"]]))
                scores[diagnostic]["objective"] = (.5 * scores[diagnostic]["clean"]
                                                    + .5 * scores[diagnostic]["short_mean"])
            seed_scores[str(seed)] = scores
        mean = {diagnostic: {metric: float(np.mean([
            seed_scores[str(seed)][diagnostic][metric] for seed in seeds]))
            for metric in (*SCENARIO_NAMES, "short_mean", "objective")}
            for diagnostic in DIAGNOSTICS}
        std = {diagnostic: {metric: float(np.std([
            seed_scores[str(seed)][diagnostic][metric] for seed in seeds], ddof=1))
            if len(seeds) > 1 else 0.0
            for metric in (*SCENARIO_NAMES, "short_mean", "objective")}
            for diagnostic in DIAGNOSTICS}
        by_rung[str(rung)] = {"by_seed": seed_scores, "mean": mean, "std_across_seeds": std}
    for index, rung in enumerate(rungs):
        if index == 0:
            by_rung[str(rung)]["paired_gain_from_previous"] = None
            continue
        previous = rungs[index - 1]
        by_rung[str(rung)]["paired_gain_from_previous"] = {
            diagnostic: {
                metric: {"by_seed": {str(seed): (
                    by_rung[str(rung)]["by_seed"][str(seed)][diagnostic][metric]
                    - by_rung[str(previous)]["by_seed"][str(seed)][diagnostic][metric])
                    for seed in seeds},
                    "mean": (by_rung[str(rung)]["mean"][diagnostic][metric]
                             - by_rung[str(previous)]["mean"][diagnostic][metric]),
                    "std_across_seeds": float(np.std([
                        by_rung[str(rung)]["by_seed"][str(seed)][diagnostic][metric]
                        - by_rung[str(previous)]["by_seed"][str(seed)][diagnostic][metric]
                        for seed in seeds], ddof=1)) if len(seeds) > 1 else 0.0}
                for metric in (*SCENARIO_NAMES, "short_mean", "objective")}
            for diagnostic in DIAGNOSTICS}
    best = max(by_rung[str(rung)]["mean"]["blend"]["objective"] for rung in rungs)
    chosen = next(rung for rung in rungs
                  if by_rung[str(rung)]["mean"]["blend"]["objective"] >= best - .002 - 1e-12)
    return {"rungs": by_rung, "selection": {"rung": chosen,
            "best_mean_objective": best, "within_best_tolerance": .002,
            "development_only": True, "global_default_promoted": False}}


def run_convergence(data_dir, recipe_path, output_dir, *, seeds=(10, 11, 12),
                    rungs=(1200, 2400, 3600), cache_bytes=2_000_000_000,
                    inference_center_size=2048):
    """Fit three continuous baseline trajectories per seed and score each rung."""
    data_dir = Path(data_dir).resolve()
    recipe_path = Path(recipe_path).resolve()
    output_dir = Path(output_dir).resolve()
    with _lock(output_dir):
        plan = _protocol(data_dir, recipe_path, seeds=seeds, rungs=rungs,
                         cache_bytes=cache_bytes,
                         inference_center_size=inference_center_size)
        _publish(output_dir / "protocol.json", plan)
        cache = HalfCache(data_dir, max_bytes=cache_bytes)
        raw = {}
        for seed in plan["payload"]["resources"]["seeds"]:
            folds = []
            for index, inner in enumerate(plan["payload"]["fold"]["inner"]):
                print(f"[convergence] seed={seed} fold={index}", flush=True)
                entries = _trajectory(data_dir, output_dir / f"seed_{seed}" / f"fold_{index}",
                                      plan, cache, seed=seed, fold_index=index, inner=inner)
                folds.append({str(rung): entry for rung, entry in zip(
                    plan["payload"]["resources"]["rungs"], entries)})
            raw[str(seed)] = folds
        aggregate = _aggregate(raw, plan)
        summary = {"protocol_sha256": plan["sha256"], "status": "complete",
                   "interpretation": "development diagnostics on six previously used public matches",
                   "raw": raw, **aggregate}
        return _publish(output_dir / "summary.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/dfl"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    parser.add_argument("--rungs", type=int, nargs="+", default=[1200, 2400, 3600])
    parser.add_argument("--cache-gb", type=float, default=2.)
    parser.add_argument("--inference-center-size", type=int, default=2048)
    args = parser.parse_args(argv)
    run_convergence(args.data_dir, args.recipe, args.output, seeds=tuple(args.seeds),
                    rungs=tuple(args.rungs), cache_bytes=int(args.cache_gb * 1_000_000_000),
                    inference_center_size=args.inference_center_size)


if __name__ == "__main__":
    main()
