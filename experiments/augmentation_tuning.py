"""Bounded Optuna policy tuning with paired match folds and a frozen decoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.augmentation_evaluation import evaluate_augmented_model
from experiments.cross_validation import nested_splits
from experiments.robustness import (
    SCENARIOS,
    _digest,
    _execution_environment,
    _lock,
    _source_hashes,
)
from experiments.tune import _optuna, _paths, _seed_for_fold
from possession.augmentation import AugmentationConfig
from possession.predict import load_model_checkpoint
from possession.prepared import verify_prepared
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache, TrainingSession

FAMILIES = {
    "geometry": ("translation", "rotation", "reflection"),
    "noise": ("coordinate_noise",),
    "occlusion": ("player_dropout", "spatial_occlusion"),
    "time": ("time_warp", "time_shift"),
    "sensor": ("frame_dropout", "frame_hold", "temporal_blur"),
    "combined": (),
}
SEARCH_SCENARIOS = ("clean", "block41_seed3101", "block41_seed3102")
SPACE = {"family": list(FAMILIES), "strength": [.1, .25, .5],
         "clean_probability": [.5, .75], "max_transforms": [1, 2]}


def _read(path):
    return json.loads(Path(path).read_text())


def _checked_write(path, value):
    path = Path(path)
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"configuration changed: {path}")
    else:
        write_json(path, value)


def _save_record(path, record):
    write_json(path, record)
    write_json(Path(path).with_suffix(".sha256.json"), {"sha256": sha256_file(path)})


def _load_record(path):
    path = Path(path)
    if not path.exists():
        return None
    digest_path = path.with_suffix(".sha256.json")
    if not digest_path.exists() or _read(digest_path).get("sha256") != sha256_file(path):
        raise ValueError(f"result hash differs: {path}")
    return _read(path)


def _config(params, warmup_steps):
    return AugmentationConfig(
        mode="combined", operations=FAMILIES[params["family"]],
        strength=params["strength"], clean_probability=params["clean_probability"],
        max_transforms=params["max_transforms"], warmup_steps=warmup_steps,
    ).to_dict()


def _means(by_match):
    if not by_match:
        raise ValueError("at least one validation match is required")
    result = {}
    for diagnostic in ("blend", "neural_only"):
        scores = {
            scenario: float(np.mean([record[scenario]["scores"][diagnostic]["spell_f1_mean"]
                                    for record in by_match.values()]))
            for scenario in next(iter(by_match.values()))
        }
        scores["short_mean"] = float(np.mean([scores[name] for name in SEARCH_SCENARIOS[1:]]))
        scores["objective"] = .5 * scores["clean"] + .5 * scores["short_mean"]
        result[diagnostic] = scores
    return result


def _comparison(control, candidate):
    gains = {name: candidate["means"]["blend"][name] - control["means"]["blend"][name]
             for name in control["means"]["blend"]}
    constraints = [-.01 - gains["clean"], -gains["short_mean"]]
    constraints = [0.0 if abs(value) < 1e-12 else value for value in constraints]
    return {"gains": gains, "constraints": constraints,
            "eligible": max(constraints) <= 0 and gains["objective"] > 1e-12}


def _assert_paired(control, candidate):
    if set(control["by_match"]) != set(candidate["by_match"]):
        raise ValueError("paired evaluation matches differ")
    if len(control["fits"]) != len(candidate["fits"]):
        raise ValueError("paired fold count differs")
    for left, right in zip(control["fits"], candidate["fits"]):
        for field in ("steps", "training_window_sha256", "training_schedule_blocks", "trained_halves"):
            if left["progress"][field] != right["progress"][field]:
                raise ValueError(f"paired training differs: {field}")
        for field in ("train_matches", "validation_matches", "seed"):
            if left["job"][field] != right["job"][field]:
                raise ValueError(f"paired training differs: {field}")
    for match, scenarios in control["by_match"].items():
        if set(scenarios) != set(candidate["by_match"][match]):
            raise ValueError("paired evaluation scenarios differ")
        for scenario, record in scenarios.items():
            if record["masks"] != candidate["by_match"][match][scenario]["masks"]:
                raise ValueError("paired evaluation masks differ")


def _plan(data_dir, recipe_path, *, steps, n_trials, seed, confirmation_seed,
          outer_seeds, cache_bytes, warmup_steps, inference_center_size):
    for name, value, minimum in (("steps", steps, 1), ("n_trials", n_trials, len(FAMILIES)),
                                 ("cache_bytes", cache_bytes, 1), ("warmup_steps", warmup_steps, 0),
                                 ("inference_center_size", inference_center_size, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    if seed == confirmation_seed or len(outer_seeds) != len(set(outer_seeds)) or not outer_seeds:
        raise ValueError("confirmation seed must be fresh and outer seeds nonempty/unique")
    original = _read(recipe_path)
    recipe = original["selected_model_recipe"]
    if recipe is None or not recipe["training_args"]["augment"]:
        raise ValueError("a neural recipe with existing training augmentation is required")
    matches = sorted(original["outer_train"] + [original["outer_test"]])
    fold = next(f for f in nested_splits(matches, seed=0)
                if f["outer_test"] == [original["outer_test"]])
    if fold["outer_train"] != sorted(original["outer_train"]):
        raise ValueError("frozen recipe training matches differ from split")
    paths = [path for match in matches for path in _paths(data_dir, match)]
    provenance = verify_prepared(data_dir, paths)
    fixed = {key: original[key] for key in ("decoder", "calibration", "ensemble")}
    fixed["selected_model_recipe"] = {
        "model_kwargs": recipe["model_kwargs"],
        "training_args": {key: recipe["training_args"][key] for key in (
            "learning_rate", "weight_decay", "boundary_weight", "boundary_pos_weight", "center_size")},
    }
    payload = {
        "protocol": "augmentation_policy_tuning_v1_development", "fold": fold,
        "fixed": fixed, "source_recipe_sha256": sha256_file(recipe_path),
        "resources": {"steps": steps, "n_trials": n_trials, "cache_bytes": cache_bytes,
                      "device": "cpu", "workers": 1, "inference_center_size": inference_center_size},
        "seed": seed, "confirmation_seed": confirmation_seed, "outer_seeds": list(outer_seeds),
        "warmup_steps": warmup_steps, "search_space": SPACE, "families": FAMILIES,
        "outage_policy": "legacy", "search_scenarios": SEARCH_SCENARIOS,
        "scenario_specs": SCENARIOS, "outer_scenarios": list(SCENARIOS),
        "selection": {"objective": ".5 clean + .5 mean of two short-gap seeds; equal match means",
                      "clean_decline_at_most": .01, "short_decline_at_most": 0,
                      "objective_gain_gt": 0, "pruning": False,
                      "fallback": "highest-objective noncontrol for diagnosis only",
                      "global_default_promotion": False},
        "prepared_sha256": {path.name: sha256_file(path) for path in paths},
        "prepared_provenance": provenance, "source_sha256": _source_hashes(),
        "environment": _execution_environment(),
    }
    # Canonical JSON types make disk and in-memory fingerprints identical.
    payload = json.loads(json.dumps(payload))
    return {"sha256": _digest(payload), "payload": payload}


def _fit_score(data_dir, directory, plan, cache, *, config, train_matches,
               validation_matches, seed, scenarios):
    directory.mkdir(parents=True, exist_ok=True)
    settings = plan["payload"]
    resources = settings["resources"]
    fixed = settings["fixed"]
    job = {"protocol_sha256": plan["sha256"], "augmentation_config": config,
           "train_matches": list(train_matches), "validation_matches": list(validation_matches),
           "seed": seed, "steps": resources["steps"], "scenarios": list(scenarios)}
    if set(train_matches) & set(validation_matches):
        raise ValueError("training and validation matches overlap")
    _checked_write(directory / "job.json", job)
    result = _load_record(directory / "result.json")
    checkpoint = directory / "model.pt"
    ready = _load_record(directory / "fit.json")
    if result is not None:
        if (result["job"] != job or ready is None
                or result["checkpoint_sha256"] != sha256_file(checkpoint)
                or ready["checkpoint_sha256"] != result["checkpoint_sha256"]):
            raise ValueError("cached fit ancestry or checkpoint changed")
        return result
    if ready is None:
        if checkpoint.exists():
            raise ValueError("checkpoint without completed fit record; inspect incomplete job")
        session = TrainingSession(
            data_dir, train_matches=train_matches, seed=seed, device="cpu", cache=cache,
            model_kwargs=fixed["selected_model_recipe"]["model_kwargs"],
            **fixed["selected_model_recipe"]["training_args"],
            augment=True, outage_config={"mode": "legacy"}, augmentation_config=config,
        )
        started = time.monotonic()
        while session.steps < resources["steps"]:
            progress = session.advance_to(min(resources["steps"], session.steps + 300))
            print(f"[{directory.name}] fit {session.steps}/{resources['steps']}", flush=True)
        fit_seconds = time.monotonic() - started
        session.save_checkpoint(checkpoint, validation_matches=validation_matches,
                                extra_training_config={"augmentation_tuning_protocol_sha256": plan["sha256"]})
        ready = {"job": job, "progress": progress, "fit_seconds": fit_seconds,
                 "checkpoint_sha256": sha256_file(checkpoint)}
        _save_record(directory / "fit.json", ready)
        model = session.model
    else:
        if ready["job"] != job or ready["checkpoint_sha256"] != sha256_file(checkpoint):
            raise ValueError("cached checkpoint ancestry differs")
        model, payload = load_model_checkpoint(checkpoint)
        expected = {"train_matches": list(train_matches), "validation_matches": list(validation_matches),
                    "seed": seed, "steps": resources["steps"], "augmentation_config": config,
                    "augmentation_tuning_protocol_sha256": plan["sha256"]}
        if any(payload["training_config"].get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint training configuration differs")
    started = time.monotonic()
    by_match = {}
    for match in validation_matches:
        print(f"[{directory.name}] score {match}: {','.join(scenarios)}", flush=True)
        by_match[match] = evaluate_augmented_model(
            data_dir, match, model, fixed, cache, scenarios=scenarios,
            inference_center_size=resources["inference_center_size"])
    result = {**ready, "by_match": by_match, "score_seconds": time.monotonic() - started}
    _save_record(directory / "result.json", result)
    return result


def _fit_cv(data_dir, directory, plan, cache, *, config, seed):
    fold = plan["payload"]["fold"]
    fits, by_match = [], {}
    for index, inner in enumerate(fold["inner"]):
        fit = _fit_score(
            data_dir, directory / f"fold_{index}", plan, cache, config=config,
            train_matches=inner["train"], validation_matches=inner["validation"],
            seed=_seed_for_fold(seed, fold["outer_test"][0], index), scenarios=SEARCH_SCENARIOS)
        fits.append(fit)
        by_match.update(fit["by_match"])
    if set(by_match) != set(fold["outer_train"]):
        raise ValueError("inner validation must cover every search match exactly once")
    return {"augmentation_config": config, "seed": seed, "fits": fits,
            "by_match": by_match, "means": _means(by_match)}


def _suggest_params(study, index, seed):
    if index < len(FAMILIES):
        family = list(FAMILIES)[index]
        return {"family": family, "strength": .5, "clean_probability": .5,
                "max_transforms": 2 if family == "combined" else 1}
    optuna = _optuna()
    # Only completed trials enter durable storage. A disposable ask reproduces
    # the next proposal from immutable history, even after an interrupted fit.
    proposal = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=_seed_for_fold(seed, "augmentation_policy", index), n_startup_trials=len(FAMILIES),
        multivariate=True))
    proposal.add_trials(study.get_trials())
    trial = proposal.ask()
    return {key: trial.suggest_categorical(key, values) for key, values in SPACE.items()}


def _append_trial(study, index, record):
    optuna = _optuna()
    attrs = {"constraints": record["comparison"]["constraints"],
             "augmentation_config": record["cv"]["augmentation_config"],
             "means": record["cv"]["means"], "record_sha256": _digest(record)}
    constraints = dict(zip(("clean_decline", "short_decline"), attrs["constraints"]))
    if index < len(study.trials):
        existing = study.trials[index]
        if (existing.params != record["params"] or existing.user_attrs != attrs
                or existing.constraints != constraints
                or existing.value != record["cv"]["means"]["blend"]["objective"]):
            raise ValueError("Optuna history differs from verified trial artifacts")
        return
    if index != len(study.trials):
        raise ValueError("Optuna history is not contiguous")
    study.add_trial(optuna.trial.create_trial(
        params=record["params"], distributions={key: optuna.distributions.CategoricalDistribution(values)
                                               for key, values in SPACE.items()},
        value=record["cv"]["means"]["blend"]["objective"], user_attrs=attrs,
        constraints=constraints))


def run_augmentation_study(data_dir, recipe_path, output_dir, *, steps=1200, n_trials=11,
                           seed=0, confirmation_seed=1, outer_seeds=(20, 21),
                           cache_bytes=2_000_000_000, warmup_steps=400,
                           inference_center_size=2048):
    data_dir, recipe_path, output_dir = Path(data_dir).resolve(), Path(recipe_path).resolve(), Path(output_dir).resolve()
    with _lock(output_dir):
        plan = _plan(data_dir, recipe_path, steps=steps, n_trials=n_trials, seed=seed,
                     confirmation_seed=confirmation_seed, outer_seeds=outer_seeds,
                     cache_bytes=cache_bytes, warmup_steps=warmup_steps,
                     inference_center_size=inference_center_size)
        _checked_write(output_dir / "protocol.json", plan)
        cache = HalfCache(data_dir, max_bytes=cache_bytes)
        no_tracking = AugmentationConfig().to_dict()
        print("[search] fresh paired control", flush=True)
        control = _fit_cv(data_dir, output_dir / "control", plan, cache,
                          config=no_tracking, seed=seed)
        _save_record(output_dir / "control.json", control)
        optuna = _optuna()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(study_name="augmentation_policy_v1", direction="maximize",
                                    storage=f"sqlite:///{output_dir / 'study.db'}", load_if_exists=True)
        if len(study.trials) > n_trials:
            raise ValueError("Optuna study exceeds declared trial budget")
        records = []
        for index in range(n_trials):
            directory = output_dir / "trials" / f"trial_{index:03d}"
            result_path = directory / "trial.json"
            record = _load_record(result_path)
            if record is None:
                params_path = directory / "params.json"
                params = _read(params_path) if params_path.exists() else _suggest_params(study, index, seed)
                _checked_write(params_path, params)
                config = _config(params, warmup_steps)
                print(f"[search] trial {index + 1}/{n_trials}: {params}", flush=True)
                previous = next((item for item in records if item["cv"]["augmentation_config"] == config), None)
                cv = previous["cv"] if previous else _fit_cv(
                    data_dir, directory, plan, cache, config=config, seed=seed)
                _assert_paired(control, cv)
                record = {"trial": index, "protocol_sha256": plan["sha256"], "params": params,
                          "cv": cv, "comparison": _comparison(control, cv),
                          "reused_from_trial": previous["trial"] if previous else None}
                _save_record(result_path, record)
            else:
                if record["trial"] != index or record["protocol_sha256"] != plan["sha256"]:
                    raise ValueError("trial ancestry differs")
                # Verify every reused model file before trusting its cached scores.
                for fold_index, fit in enumerate(record["cv"]["fits"]):
                    source_index = record["reused_from_trial"]
                    parent = directory if source_index is None else output_dir / "trials" / f"trial_{source_index:03d}"
                    if sha256_file(parent / f"fold_{fold_index}" / "model.pt") != fit["checkpoint_sha256"]:
                        raise ValueError("trial checkpoint changed")
                _assert_paired(control, record["cv"])
            _append_trial(study, index, record)
            records.append(record)
            print(f"[search] trial {index}: {record['cv']['means']['blend']} "
                  f"eligible={record['comparison']['eligible']}", flush=True)
            _save_record(output_dir / "search-progress.json", {
                "protocol_sha256": plan["sha256"], "completed_trials": len(records),
                "control": control["means"],
                "trials": [{"trial": r["trial"], "params": r["params"],
                            "means": r["cv"]["means"], "comparison": r["comparison"],
                            "reused_from_trial": r["reused_from_trial"]} for r in records]})
        eligible = [record for record in records if record["comparison"]["eligible"]]
        winner = max(eligible or records, key=lambda record: (
            record["cv"]["means"]["blend"]["objective"], -record["trial"]))
        selection = {"protocol_sha256": plan["sha256"], "trial": winner["trial"],
                     "augmentation_config": winner["cv"]["augmentation_config"],
                     "eligible_in_search": bool(eligible), "search_comparison": winner["comparison"],
                     "locked_before_confirmation_and_outer": True}
        _checked_write(output_dir / "selection.json", selection)
        _checked_write(output_dir / "candidate-config.json", selection["augmentation_config"])
        print(f"[confirmation] frozen trial {winner['trial']}, fresh seed {confirmation_seed}", flush=True)
        confirmation_control = _fit_cv(
            data_dir, output_dir / "confirmation" / "control", plan, cache,
            config=no_tracking, seed=confirmation_seed)
        confirmation_candidate = _fit_cv(
            data_dir, output_dir / "confirmation" / "candidate", plan, cache,
            config=selection["augmentation_config"], seed=confirmation_seed)
        _assert_paired(confirmation_control, confirmation_candidate)
        confirmation = {"control": confirmation_control, "candidate": confirmation_candidate,
                        "comparison": _comparison(confirmation_control, confirmation_candidate)}
        _save_record(output_dir / "confirmation.json", confirmation)
        # Persist the seed-repeat result before any new outer predictions.
        fold = plan["payload"]["fold"]
        outer = []
        for outer_seed in outer_seeds:
            paired = {}
            for name, config in (("control", no_tracking), ("candidate", selection["augmentation_config"])):
                print(f"[outer] seed={outer_seed} policy={name}", flush=True)
                fit = _fit_score(
                    data_dir, output_dir / "outer" / f"seed_{outer_seed}" / name, plan, cache,
                    config=config, train_matches=fold["outer_train"], validation_matches=fold["outer_test"],
                    seed=_seed_for_fold(outer_seed, "augmentation_outer", 0), scenarios=tuple(SCENARIOS))
                paired[name] = {"fits": [fit], "by_match": fit["by_match"], "means": _means(fit["by_match"])}
            _assert_paired(paired["control"], paired["candidate"])
            outer.append({"seed": outer_seed, **paired,
                          "comparison": _comparison(paired["control"], paired["candidate"])})
            _save_record(output_dir / "outer-progress.json", outer)
        summary = {
            "protocol_sha256": plan["sha256"], "status": "complete",
            "interpretation": "development search, fresh-seed repeat and one match held out from this search",
            "global_default_changed": False, "selection": selection,
            "search_control": control["means"], "search_candidate": winner["cv"]["means"],
            "confirmation": {"control": confirmation_control["means"],
                             "candidate": confirmation_candidate["means"],
                             "comparison": confirmation["comparison"]},
            "outer": [{"seed": entry["seed"], "control": entry["control"]["means"],
                       "candidate": entry["candidate"]["means"], "comparison": entry["comparison"]}
                      for entry in outer],
            "completed_trials": n_trials, "unique_trial_policies": len({
                _digest(record["cv"]["augmentation_config"]) for record in records}),
            "cache_peak_array_bytes": cache.peak_bytes,
        }
        completed = _load_record(output_dir / "summary.json")
        if completed is not None:
            # Cached replay allocates no training arrays. Retain the completed
            # run's resource observation after verifying all fits and scores.
            summary["cache_peak_array_bytes"] = completed["cache_peak_array_bytes"]
            if summary != completed:
                raise ValueError("completed summary differs from verified results")
        _save_record(output_dir / "summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/dfl"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confirmation-seed", type=int, default=1)
    parser.add_argument("--outer-seeds", type=int, nargs="+", default=[20, 21])
    parser.add_argument("--cache-gb", type=float, default=2.)
    args = parser.parse_args(argv)
    run_augmentation_study(args.data_dir, args.recipe, args.output, steps=args.steps,
                           n_trials=args.trials, seed=args.seed, confirmation_seed=args.confirmation_seed,
                           outer_seeds=args.outer_seeds, cache_bytes=int(args.cache_gb * 1_000_000_000))


if __name__ == "__main__":
    main()
