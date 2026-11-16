"""Development-only temporal, decoder, quality-blend, and interaction study."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.augmentation_evaluation import evaluate_augmented_model
from experiments.augmentation_tuning import FAMILIES, _load_record, _plan, _save_record
from experiments.oof_selection import SCENARIOS as SEARCH_SCENARIOS
from experiments.oof_selection import decode_oof_half, score_oof, select_oof_decoder
from experiments.prediction_store import load_prediction_bundle
from experiments.robustness import SCENARIOS, _digest, _lock
from experiments.tune import _optuna, _seed_for_fold
from possession.augmentation import AugmentationConfig
from possession.diagnostics import spell_error_diagnostics
from possession.predict import load_model_checkpoint
from possession.storage import sha256_file
from possession.training_session import HalfCache, TrainingSession

ALL_SCENARIOS = tuple(SCENARIOS)
POLICIES = ("none", "time_warp", "time_shift")
STRENGTHS = (.1, .25, .5)
SKIPS = (.5, .75)


def _publish(path, value):
    existing = _load_record(path)
    if existing is not None:
        if existing != value:
            raise ValueError(f"completed record differs: {path}")
        return existing
    _save_record(path, value)
    return value


def _protocol(data_dir, recipe_path, convergence_dir, *, n_trials, decoder_trials,
              seeds, confirmation_seed, cache_bytes):
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or not 3 <= n_trials <= 6:
        raise ValueError("n_trials must be between 3 and 6")
    if (isinstance(decoder_trials, bool) or not isinstance(decoder_trials, int)
            or not 4 <= decoder_trials <= 16):
        raise ValueError("decoder_trials must be between 4 and 16")
    seeds = tuple(seeds)
    if (not seeds or len(seeds) != len(set(seeds))
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
            or isinstance(confirmation_seed, bool) or not isinstance(confirmation_seed, int)
            or confirmation_seed in seeds):
        raise ValueError("search seeds must be unique and confirmation seed fresh")
    convergence = _load_record(convergence_dir / "protocol.json")
    summary = _load_record(convergence_dir / "summary.json")
    if convergence is None or summary is None:
        raise ValueError("verified convergence protocol and summary are required")
    if convergence["sha256"] != _digest(convergence["payload"]):
        raise ValueError("convergence protocol digest differs")
    if (summary["protocol_sha256"] != convergence["sha256"]
            or summary["status"] != "complete"
            or tuple(convergence["payload"]["resources"]["seeds"]) != seeds):
        raise ValueError("convergence summary or seed ancestry differs")
    rung = summary["selection"]["rung"]
    if rung not in convergence["payload"]["resources"]["rungs"]:
        raise ValueError("selected convergence rung is outside its protocol")
    rung_scores = summary["rungs"]
    ordered_rungs = convergence["payload"]["resources"]["rungs"]
    best = max(rung_scores[str(step)]["mean"]["blend"]["objective"]
               for step in ordered_rungs)
    expected_rung = next(step for step in ordered_rungs
                         if rung_scores[str(step)]["mean"]["blend"]["objective"]
                         >= best - .002 - 1e-12)
    if (rung != expected_rung or summary["selection"]["best_mean_objective"] != best
            or summary["selection"]["global_default_promoted"] is not False):
        raise ValueError("convergence budget selection differs from declared rule")
    base = _plan(
        data_dir, recipe_path, steps=rung, n_trials=len(FAMILIES), seed=seeds[0],
        confirmation_seed=confirmation_seed, outer_seeds=(20, 21),
        cache_bytes=cache_bytes, warmup_steps=400,
        inference_center_size=convergence["payload"]["resources"]["inference_center_size"])["payload"]
    for key in ("fold", "fixed", "source_recipe_sha256", "prepared_sha256",
                "prepared_provenance", "source_sha256", "environment"):
        if base[key] != convergence["payload"][key]:
            raise ValueError(f"convergence source/data/recipe ancestry differs: {key}")
    if (base["fold"]["outer_test"] != ["DFL-MAT-J03WMX"]
            or len(base["fold"]["outer_train"]) != 6
            or len(base["fold"]["inner"]) != 3):
        raise ValueError("focused study requires WMX's six-match inner folds")
    model = base["fixed"]["selected_model_recipe"]["model_kwargs"]
    if model.get("width") != 32 or model.get("pooling", "mean") != "mean":
        raise ValueError("focused study requires the frozen width-32 mean-pooling baseline")
    # Summary JSON is not enough to establish the selected budget: verify its
    # checkpoint and fit/result records for every paired search trajectory.
    for seed in seeds:
        folds = summary["raw"][str(seed)]
        if len(folds) != 3:
            raise ValueError("convergence fold count differs")
        for index, fold in enumerate(folds):
            entry = fold[str(rung)]
            directory = convergence_dir / f"seed_{seed}" / f"fold_{index}" / f"rung_{rung}"
            if (_load_record(directory / "fit.json") != entry["fit"]
                    or _load_record(directory / "result.json") != entry["result"]
                    or sha256_file(directory / "model.pt") != entry["fit"]["checkpoint_sha256"]):
                raise ValueError("selected convergence checkpoint ancestry differs")
            expected_seed = _seed_for_fold(seed, "DFL-MAT-J03WMX", index)
            if (entry["fit"]["job"]["training_seed"] != expected_seed
                    or entry["fit"]["job"]["steps"] != rung):
                raise ValueError("selected convergence training job differs")
    payload = {
        "protocol": "focused_temporal_oof_v1_development", "fold": base["fold"],
        "fixed": base["fixed"], "source_recipe_sha256": base["source_recipe_sha256"],
        "source_sha256": base["source_sha256"],
        "prepared_sha256": base["prepared_sha256"],
        "prepared_provenance": base["prepared_provenance"], "environment": base["environment"],
        "convergence_protocol_sha256": convergence["sha256"],
        "convergence_summary_sha256": sha256_file(convergence_dir / "summary.json"),
        "selected_rung": rung, "convergence_selection": summary["selection"],
        "resources": {"n_trials": n_trials, "decoder_trials": decoder_trials,
                      "seeds": list(seeds), "confirmation_seed": confirmation_seed,
                      "cache_bytes": cache_bytes,
                      "inference_center_size": base["resources"]["inference_center_size"],
                      "device": "cpu", "workers": 1},
        "policy_space": {"policy": POLICIES, "learning_rate_log": [.0005, .003],
                         "strength": STRENGTHS, "skip": SKIPS, "warmup_steps": 400},
        "scenarios": ALL_SCENARIOS, "selection_scenarios": SEARCH_SCENARIOS,
        "selection_gate": {"objective_gain_gt": 0, "short_gain_gt": 0,
                           "clean_loss_at_most": .01, "long_loss_at_most": .01,
                           "short_transition_loss_at_most": .01,
                           "positive_short_matches_at_least": 4,
                           "development_only": True, "global_default_promotion": False},
    }
    payload = json.loads(json.dumps(payload))
    return {"sha256": _digest(payload), "payload": payload}


def _canonical(params):
    policy = params["policy"]
    if policy not in POLICIES:
        raise ValueError("unknown temporal policy")
    learning_rate = float(params["learning_rate"])
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if policy == "none":
        config = AugmentationConfig().to_dict()
    else:
        config = AugmentationConfig(
            mode="temporal", operations=(policy,), strength=params["strength"],
            clean_probability=params["skip"], max_transforms=1,
            warmup_steps=400).to_dict()
    return {"policy": policy, "learning_rate": learning_rate,
            "augmentation_config": config}


def _anchor_params(index, old_lr):
    return {"policy": POLICIES[index], "learning_rate": old_lr,
            "strength": .5, "skip": .5}


def _suggest_params(study, index, seed):
    optuna = _optuna()
    disposable = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(
            seed=seed + index, n_startup_trials=3, multivariate=True))
    disposable.add_trials(study.get_trials())
    trial = disposable.ask()
    return {"policy": trial.suggest_categorical("policy", list(POLICIES)),
            "learning_rate": trial.suggest_float("learning_rate", .0005, .003, log=True),
            "strength": trial.suggest_categorical("strength", list(STRENGTHS)),
            "skip": trial.suggest_categorical("skip", list(SKIPS))}


def _append_trial(study, index, record, old_lr):
    optuna = _optuna()
    attrs = {"record_sha256": _digest(record), "reused_from_trial": record["reused_from_trial"]}
    if index < len(study.trials):
        trial = study.trials[index]
        if (trial.params != record["params"] or trial.user_attrs != attrs
                or trial.value != record["scores"]["objective"]):
            raise ValueError("Optuna trial differs from verified artifact")
        return
    if index != len(study.trials):
        raise ValueError("Optuna history is not contiguous")
    lower, upper = min(.0005, old_lr), max(.003, old_lr)
    study.add_trial(optuna.trial.create_trial(
        params=record["params"],
        distributions={"policy": optuna.distributions.CategoricalDistribution(POLICIES),
                       "learning_rate": optuna.distributions.FloatDistribution(
                           lower, upper, log=True),
                       "strength": optuna.distributions.CategoricalDistribution(STRENGTHS),
                       "skip": optuna.distributions.CategoricalDistribution(SKIPS)},
        value=record["scores"]["objective"], user_attrs=attrs))


def _job(plan, model_spec, seed, fold_index, inner):
    return {"protocol_sha256": plan["sha256"], "model_spec": model_spec,
            "seed": seed,
            "training_seed": _seed_for_fold(seed, "DFL-MAT-J03WMX", fold_index),
            "fold_index": fold_index, "train_matches": inner["train"],
            "validation_matches": inner["validation"],
            "steps": plan["payload"]["selected_rung"], "scenarios": list(ALL_SCENARIOS)}


def _session(data_dir, plan, cache, job):
    fixed = plan["payload"]["fixed"]["selected_model_recipe"]
    model_kwargs = {**fixed["model_kwargs"], "pooling": job["model_spec"]["pooling"]}
    training_args = {**fixed["training_args"],
                     "learning_rate": job["model_spec"]["learning_rate"]}
    return TrainingSession(
        data_dir, train_matches=job["train_matches"], seed=job["training_seed"],
        device="cpu", cache=cache, model_kwargs=model_kwargs, **training_args,
        augment=True, outage_config={"mode": "legacy"},
        augmentation_config=job["model_spec"]["augmentation_config"])


def _verify_fit(directory, plan, job, fit):
    checkpoint = directory / "model.pt"
    if (fit["job"] != job or fit["progress"]["steps"] != job["steps"]
            or not checkpoint.is_file() or sha256_file(checkpoint) != fit["checkpoint_sha256"]):
        raise ValueError("focused fit/checkpoint ancestry differs")
    model, payload = load_model_checkpoint(checkpoint)
    config = payload["training_config"]
    expected = {"train_matches": job["train_matches"],
                "validation_matches": job["validation_matches"],
                "seed": job["training_seed"], "steps": job["steps"],
                "learning_rate": job["model_spec"]["learning_rate"],
                "augmentation_config": job["model_spec"]["augmentation_config"],
                "focused_protocol_sha256": plan["sha256"]}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("focused checkpoint training configuration differs")
    if (config.get("outage_config", {}).get("mode") != "legacy"
            or payload["model_kwargs"].get("pooling", "mean") != job["model_spec"]["pooling"]):
        raise ValueError("focused checkpoint model or outage policy differs")
    return model


def _verify_result(result, fit, inner):
    if (result["job"] != fit["job"]
            or result["checkpoint_sha256"] != fit["checkpoint_sha256"]
            or set(result["by_match"]) != set(inner["validation"])):
        raise ValueError("focused result ancestry differs")
    for match, scenarios in result["by_match"].items():
        if set(scenarios) != set(ALL_SCENARIOS):
            raise ValueError("focused result scenarios differ")
        for scenario, evidence in scenarios.items():
            if (set(evidence["predictions"]) != {"1", "2"}
                    or set(evidence["diagnostics"]) != {"blend", "neural_only"}):
                raise ValueError("focused result diagnostics or half bundles differ")
            for period in (1, 2):
                prediction = evidence["predictions"][str(period)]
                if (prediction["match"] != match or prediction["period"] != period
                        or prediction["scenario"] != scenario):
                    raise ValueError("focused prediction identity differs")
                load_prediction_bundle(prediction)


def _fit(data_dir, directory, plan, cache, model_spec, seed, fold_index, inner):
    directory.mkdir(parents=True, exist_ok=True)
    job = _job(plan, model_spec, seed, fold_index, inner)
    _publish(directory / "job.json", job)
    fit = _load_record(directory / "fit.json")
    checkpoint = directory / "model.pt"
    if fit is None:
        orphan_payload = None
        if checkpoint.exists():
            _, orphan_payload = load_model_checkpoint(checkpoint)
        session = _session(data_dir, plan, cache, job)
        started = time.monotonic()
        while session.steps < job["steps"]:
            progress = session.advance_to(min(job["steps"], session.steps + 300))
        fit_seconds = time.monotonic() - started
        if orphan_payload is None:
            session.save_checkpoint(checkpoint, validation_matches=inner["validation"],
                                    extra_training_config={"focused_protocol_sha256": plan["sha256"]})
        else:
            current, saved = session.model.state_dict(), orphan_payload["state_dict"]
            if (set(current) != set(saved)
                    or any(not torch.equal(current[key].cpu(), saved[key]) for key in current)):
                raise ValueError("orphan checkpoint differs from deterministic seed replay")
        fit = _publish(directory / "fit.json", {
            "job": job, "progress": progress, "fit_seconds": fit_seconds,
            "checkpoint_sha256": sha256_file(checkpoint),
            "replayed_orphan_checkpoint": orphan_payload is not None})
    model = _verify_fit(directory, plan, job, fit)
    result = _load_record(directory / "result.json")
    if result is None:
        started = time.monotonic()
        by_match = {}
        for match in inner["validation"]:
            by_match[match] = evaluate_augmented_model(
                data_dir, match, model, plan["payload"]["fixed"], cache,
                scenarios=ALL_SCENARIOS,
                inference_center_size=plan["payload"]["resources"]["inference_center_size"],
                include_diagnostics=True, prediction_dir=directory / "predictions" / match)
        result = _publish(directory / "result.json", {
            "job": job, "checkpoint_sha256": fit["checkpoint_sha256"],
            "by_match": by_match, "score_seconds": time.monotonic() - started})
    _verify_result(result, fit, inner)
    return {"fit": fit, "result": result}


def _run_model(data_dir, directory, plan, cache, model_spec, seeds):
    entries = {}
    for seed in seeds:
        folds = []
        for index, inner in enumerate(plan["payload"]["fold"]["inner"]):
            print(f"[focused] {directory.name} seed={seed} fold={index}", flush=True)
            folds.append(_fit(data_dir, directory / f"seed_{seed}" / f"fold_{index}",
                              plan, cache, model_spec, seed, index, inner))
        entries[str(seed)] = folds
    return entries


def _oof_records(entries, scenarios):
    records = []
    for seed, folds in entries.items():
        for entry in folds:
            fit, result = entry["fit"], entry["result"]
            job = fit["job"]
            for match, by_scenario in result["by_match"].items():
                for scenario in scenarios:
                    for period, prediction in by_scenario[scenario]["predictions"].items():
                        records.append({
                            "seed": int(seed), "match": match, "scenario": scenario,
                            "period": int(period), "prediction": prediction,
                            "training_matches": job["train_matches"],
                            "validation_matches": job["validation_matches"],
                            "checkpoint_sha256": fit["checkpoint_sha256"],
                            "protocol_sha256": job["protocol_sha256"],
                        })
    return records


def _paired(baseline, candidate):
    if set(baseline) != set(candidate):
        raise ValueError("paired search seeds differ")
    for seed in baseline:
        if len(baseline[seed]) != len(candidate[seed]):
            raise ValueError("paired fold counts differ")
        for left, right in zip(baseline[seed], candidate[seed], strict=True):
            left_job, right_job = left["fit"]["job"], right["fit"]["job"]
            for key in ("seed", "training_seed", "fold_index", "train_matches",
                        "validation_matches", "steps"):
                if left_job[key] != right_job[key]:
                    raise ValueError(f"paired fit differs: {key}")
            for key in ("training_window_sha256", "training_schedule_blocks", "trained_halves"):
                if left["fit"]["progress"][key] != right["fit"]["progress"][key]:
                    raise ValueError(f"paired windows differ: {key}")
            for match, by_scenario in left["result"]["by_match"].items():
                for scenario, record in by_scenario.items():
                    if record["masks"] != right["result"]["by_match"][match][scenario]["masks"]:
                        raise ValueError("paired evaluation masks differ")


def _gate(control, candidate, matches, seeds):
    left, right = control["means"], candidate["means"]
    gains = {key: right[key] - left[key]
             for key in (*ALL_SCENARIOS, "short_mean", "objective")}
    short_transition_left = float(np.mean([
        left["transition_f1_at_12"][name] for name in SEARCH_SCENARIOS[1:]]))
    short_transition_right = float(np.mean([
        right["transition_f1_at_12"][name] for name in SEARCH_SCENARIOS[1:]]))
    gains["short_transition_f1_at_12"] = short_transition_right - short_transition_left
    per_match = {}
    for match in matches:
        def match_short(scores, match=match):
            return float(np.mean([
                scores["by_seed"][str(seed)]["by_match"][match][scenario]["spell_f1_mean"]
                for seed in seeds for scenario in SEARCH_SCENARIOS[1:]]))
        per_match[match] = match_short(candidate) - match_short(control)
    eligible = (gains["objective"] > 0 and gains["short_mean"] > 0
                and gains["clean"] >= -.01 - 1e-12
                and gains["long41_seed4101"] >= -.01 - 1e-12
                and gains["short_transition_f1_at_12"] >= -.01 - 1e-12
                and sum(gain > 0 for gain in per_match.values()) >= 4)
    return {"gains": gains, "short_gain_by_match": per_match,
            "positive_short_matches": sum(gain > 0 for gain in per_match.values()),
            "eligible": eligible}


def _model_spec(canonical, pooling="mean"):
    return {"pooling": pooling, "learning_rate": canonical["learning_rate"],
            "augmentation_config": canonical["augmentation_config"]}


def _recipe(plan, model_spec, frozen):
    selected = plan["payload"]["fixed"]["selected_model_recipe"]
    return {"model_kwargs": {**selected["model_kwargs"], "pooling": model_spec["pooling"]},
            "training_args": {**selected["training_args"],
                              "learning_rate": model_spec["learning_rate"], "augment": True},
            "augmentation_config": model_spec["augmentation_config"],
            "outage_config": {"mode": "legacy"},
            "decoder": frozen["decoder"], "calibration": frozen["calibration"],
            "ensemble": frozen["ensemble"], "steps": plan["payload"]["selected_rung"],
            "development_only": True}


def _short_spell_recall(entries, frozen):
    """Pool short-spell recall after decoding saved OOF halves with this recipe."""
    totals = {scenario: {"n": 0, "matched_at_iou_0_5": 0}
              for scenario in ALL_SCENARIOS}
    for record in _oof_records(entries, ALL_SCENARIOS):
        arrays = load_prediction_bundle(record["prediction"])
        predicted = decode_oof_half(arrays, frozen)
        diagnostics = spell_error_diagnostics(
            arrays["gt"], predicted, arrays["ball_valid"])["by_duration"]
        counts = totals[record["scenario"]]
        for bucket in ("<1", "1-3"):
            counts["n"] += diagnostics[bucket]["n"]
            counts["matched_at_iou_0_5"] += diagnostics[bucket]["matched"]["0.5"]
    for counts in totals.values():
        counts["recall_at_iou_0_5"] = (counts["matched_at_iou_0_5"] / counts["n"]
                                       if counts["n"] else None)
    return totals


def _variability(scores):
    by_seed = scores["by_seed"]
    seeds = sorted(by_seed, key=int)
    by_seed_objective = {}
    by_match_short = {}
    matches = sorted(next(iter(by_seed.values()))["by_match"])
    for seed in seeds:
        records = by_seed[seed]["by_match"]
        by_seed_objective[seed] = float(np.mean([
            .5 * records[match]["clean"]["spell_f1_mean"]
            + .25 * sum(records[match][scenario]["spell_f1_mean"]
                          for scenario in SEARCH_SCENARIOS[1:])
            for match in matches]))
    for match in matches:
        by_match_short[match] = float(np.mean([
            by_seed[seed]["by_match"][match][scenario]["spell_f1_mean"]
            for seed in seeds for scenario in SEARCH_SCENARIOS[1:]]))
    return {"objective_by_seed": by_seed_objective,
            "objective_std_across_seeds": float(np.std(list(by_seed_objective.values()), ddof=1))
            if len(seeds) > 1 else 0.0,
            "short_f1_by_match": by_match_short}


def _model_resources(plan, model_spec, entries):
    # Count architecture parameters from the frozen model kwargs; fit and score
    # wall times are retained per seed/fold in the verified records.
    from possession.model import PossessionModel

    kwargs = {**plan["payload"]["fixed"]["selected_model_recipe"]["model_kwargs"],
              "pooling": model_spec["pooling"]}
    model = PossessionModel(**kwargs)
    return {"parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "fit_seconds_total": float(sum(entry["fit"]["fit_seconds"]
                                           for folds in entries.values() for entry in folds)),
            "score_seconds_total": float(sum(entry["result"]["score_seconds"]
                                             for folds in entries.values() for entry in folds))}


def run_focused_study(data_dir, recipe_path, convergence_dir, output_dir, *, n_trials=6,
                      decoder_trials=16, seeds=(10, 11, 12), confirmation_seed=13,
                      cache_bytes=2_000_000_000):
    """Run all predeclared focused phases without using external proxy scores."""
    data_dir, recipe_path = Path(data_dir).resolve(), Path(recipe_path).resolve()
    convergence_dir, output_dir = Path(convergence_dir).resolve(), Path(output_dir).resolve()
    with _lock(output_dir):
        plan = _protocol(data_dir, recipe_path, convergence_dir, n_trials=n_trials,
                         decoder_trials=decoder_trials, seeds=seeds,
                         confirmation_seed=confirmation_seed, cache_bytes=cache_bytes)
        _publish(output_dir / "protocol.json", plan)
        cache = HalfCache(data_dir, max_bytes=cache_bytes)
        baseline_frozen = {key: plan["payload"]["fixed"][key]
                           for key in ("ensemble", "calibration", "decoder")}
        old_lr = plan["payload"]["fixed"]["selected_model_recipe"]["training_args"][
            "learning_rate"]
        optuna = _optuna()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            study_name="focused-model-v1", direction="maximize",
            storage=f"sqlite:///{output_dir / 'study.sqlite3'}", load_if_exists=True)
        if len(study.trials) > n_trials:
            raise ValueError("focused Optuna history exceeds trial budget")
        trials = []
        for index in range(n_trials):
            directory = output_dir / "trials" / f"trial_{index:03d}"
            record = _load_record(directory / "trial.json")
            if record is None:
                params_path = directory / "params.json"
                params = _load_record(params_path)
                if params is None:
                    params = (_anchor_params(index, old_lr) if index < 3 else
                              _suggest_params(study, index, seeds[0]))
                    _publish(params_path, params)
                canonical = _canonical(params)
                reused = next((trial for trial in trials
                               if trial["canonical"] == canonical), None)
                if reused is None:
                    entries = _run_model(data_dir, directory / "models", plan, cache,
                                         _model_spec(canonical), seeds)
                    if trials:
                        _paired(trials[0]["entries"], entries)
                    records = _oof_records(entries, SEARCH_SCENARIOS)
                    decoder = select_oof_decoder(
                        records, mode="static", seed=1000 + index,
                        n_trials=decoder_trials, baseline_frozen=baseline_frozen,
                        output_dir=directory / "decoder_search")
                    _publish(directory / "decoder.json", decoder)
                else:
                    entries, decoder = reused["entries"], reused["decoder"]
                scores = score_oof(_oof_records(entries, ALL_SCENARIOS), decoder["frozen"])
                record = {"trial": index, "protocol_sha256": plan["sha256"],
                          "params": params, "canonical": canonical,
                          "reused_from_trial": reused["trial"] if reused else None,
                          "entries": entries, "decoder": decoder, "scores": scores}
                _publish(directory / "trial.json", record)
            else:
                if (record["trial"] != index or record["protocol_sha256"] != plan["sha256"]
                        or record["canonical"] != _canonical(record["params"])):
                    raise ValueError("focused trial ancestry differs")
                source = (directory if record["reused_from_trial"] is None else
                          output_dir / "trials" / f"trial_{record['reused_from_trial']:03d}")
                entries = _run_model(data_dir, source / "models", plan, cache,
                                     _model_spec(record["canonical"]), seeds)
                if entries != record["entries"]:
                    raise ValueError("focused cached fits differ from trial")
                records = _oof_records(entries, SEARCH_SCENARIOS)
                if record["reused_from_trial"] is None:
                    decoder = select_oof_decoder(
                        records, mode="static", seed=1000 + index,
                        n_trials=decoder_trials, baseline_frozen=baseline_frozen,
                        output_dir=directory / "decoder_search")
                    if decoder != record["decoder"]:
                        raise ValueError("focused cached decoder differs")
                if score_oof(_oof_records(entries, ALL_SCENARIOS),
                             record["decoder"]["frozen"]) != record["scores"]:
                    raise ValueError("focused cached scores differ")
            _append_trial(study, index, record, old_lr)
            trials.append(record)
            print(f"[focused] trial={index} objective={record['scores']['objective']:.4f}",
                  flush=True)
        control = trials[0]
        matches = plan["payload"]["fold"]["outer_train"]
        comparisons = {}
        for trial in trials[1:]:
            _paired(control["entries"], trial["entries"])
            comparisons[str(trial["trial"])] = _gate(control["scores"], trial["scores"],
                                                     matches, seeds)
        eligible = [trial for trial in trials[1:]
                    if comparisons[str(trial["trial"])]["eligible"]]
        policy_winner = max(eligible, key=lambda trial: (
            trial["scores"]["objective"], -trial["trial"])) if eligible else control
        best_mean = max(trials, key=lambda trial: (
            trial["scores"]["objective"], -trial["trial"]))
        policy_selection = {"protocol_sha256": plan["sha256"],
                            "selected_trial": policy_winner["trial"],
                            "best_mean_trial": best_mean["trial"],
                            "comparisons": comparisons, "development_only": True,
                            "global_default_promoted": False}
        _publish(output_dir / "policy-selection.json", policy_selection)

        quality = select_oof_decoder(
            _oof_records(policy_winner["entries"], SEARCH_SCENARIOS), mode="quality",
            seed=2000, n_trials=decoder_trials, baseline_frozen=baseline_frozen,
            output_dir=output_dir / "quality_decoder_search")
        _publish(output_dir / "quality-decoder.json", quality)
        quality_scores = score_oof(_oof_records(policy_winner["entries"], ALL_SCENARIOS),
                                   quality["frozen"])
        interaction_spec = _model_spec(policy_winner["canonical"], pooling="interaction")
        interaction_entries = _run_model(data_dir, output_dir / "interaction" / "models",
                                         plan, cache, interaction_spec, seeds)
        _paired(control["entries"], interaction_entries)
        interaction_records = _oof_records(interaction_entries, SEARCH_SCENARIOS)
        interaction_decoders = {}
        interaction_scores = {}
        for mode in ("static", "quality"):
            decoder = select_oof_decoder(
                interaction_records, mode=mode, seed=3000 if mode == "static" else 3001,
                n_trials=decoder_trials, baseline_frozen=baseline_frozen,
                output_dir=output_dir / "interaction" / f"{mode}_decoder_search")
            _publish(output_dir / "interaction" / f"{mode}-decoder.json", decoder)
            interaction_decoders[mode] = decoder
            interaction_scores[mode] = score_oof(
                _oof_records(interaction_entries, ALL_SCENARIOS), decoder["frozen"])
        candidates = [{"name": "policy", "model": policy_winner,
                       "frozen": policy_winner["decoder"]["frozen"],
                       "scores": policy_winner["scores"]},
                      {"name": "quality", "model": policy_winner,
                       "frozen": quality["frozen"], "scores": quality_scores}]
        candidates += [{"name": f"interaction_{mode}", "model": None,
                        "frozen": interaction_decoders[mode]["frozen"],
                        "scores": interaction_scores[mode]} for mode in ("static", "quality")]
        architecture_comparisons = {}
        for candidate in candidates:
            architecture_comparisons[candidate["name"]] = _gate(
                control["scores"], candidate["scores"], matches, seeds)
        allowed = [candidate for candidate in candidates
                   if architecture_comparisons[candidate["name"]]["eligible"]]
        winner = max(allowed, key=lambda candidate: (
            candidate["scores"]["objective"], -candidates.index(candidate))) if allowed else None
        selected_name = winner["name"] if winner else "baseline"
        selected_spec = (interaction_spec if winner and selected_name.startswith("interaction")
                         else _model_spec(winner["model"]["canonical"]) if winner
                         else _model_spec(control["canonical"]))
        selected_frozen = winner["frozen"] if winner else control["decoder"]["frozen"]
        selected_entries = (interaction_entries if winner and
                            selected_name.startswith("interaction") else
                            winner["model"]["entries"] if winner else control["entries"])
        selected_scores = winner["scores"] if winner else control["scores"]
        selection = {"protocol_sha256": plan["sha256"], "selected": selected_name,
                     "model_spec": selected_spec, "frozen": selected_frozen,
                     "architecture_comparisons": architecture_comparisons,
                     "locked_before_confirmation": True, "development_only": True,
                     "global_default_promoted": False}
        _publish(output_dir / "selection.json", selection)
        _publish(output_dir / "baseline-recipe.json", _recipe(
            plan, _model_spec(control["canonical"]), control["decoder"]["frozen"]))
        _publish(output_dir / "selected-recipe.json", _recipe(plan, selected_spec,
                                                               selected_frozen))
        confirmation = {}
        for label, model_spec, frozen in (
                ("baseline", _model_spec(control["canonical"]),
                 control["decoder"]["frozen"]),
                ("selected", selected_spec, selected_frozen)):
            if label == "selected" and model_spec == _model_spec(control["canonical"]):
                entries = confirmation["baseline"]["entries"]
            else:
                entries = _run_model(data_dir, output_dir / "confirmation" / label,
                                     plan, cache, model_spec, (confirmation_seed,))
            scores = score_oof(_oof_records(entries, ALL_SCENARIOS), frozen)
            confirmation[label] = {"entries": entries, "scores": scores}
        _paired(confirmation["baseline"]["entries"], confirmation["selected"]["entries"])
        confirmation_comparison = _gate(
            confirmation["baseline"]["scores"], confirmation["selected"]["scores"],
            matches, (confirmation_seed,))
        _publish(output_dir / "confirmation.json", {
            "protocol_sha256": plan["sha256"], **confirmation,
            "comparison": confirmation_comparison,
            "no_op": selected_name == "baseline"})
        summary = {"protocol_sha256": plan["sha256"], "status": "complete",
                   "selected_rung": plan["payload"]["selected_rung"],
                   "policy_selection": policy_selection, "selection": selection,
                   "model_trials": [{"trial": trial["trial"], "canonical": trial["canonical"],
                                     "objective": trial["scores"]["objective"],
                                     "reused_from_trial": trial["reused_from_trial"]}
                                    for trial in trials],
                   "quality_scores": quality_scores,
                   "interaction_scores": interaction_scores,
                   "baseline_search_scores": control["scores"],
                   "selected_search_scores": selected_scores,
                   "search_variability": {
                       "baseline": _variability(control["scores"]),
                       "selected": _variability(selected_scores)},
                   "short_spell_diagnostics": {
                       "baseline": _short_spell_recall(
                           control["entries"], control["decoder"]["frozen"]),
                       "selected": _short_spell_recall(selected_entries, selected_frozen)},
                   "model_resources": {
                       "baseline": _model_resources(
                           plan, _model_spec(control["canonical"]), control["entries"]),
                       "selected": _model_resources(plan, selected_spec, selected_entries),
                       "interaction": _model_resources(
                           plan, interaction_spec, interaction_entries)},
                   "confirmation_comparison": confirmation_comparison,
                   "confirmation_scores": {key: value["scores"]
                                           for key, value in confirmation.items()},
                   "global_defaults_changed": False,
                   "external_proxy_scored": False}
        return _publish(output_dir / "summary.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/dfl"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--convergence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--decoder-trials", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    parser.add_argument("--confirmation-seed", type=int, default=13)
    parser.add_argument("--cache-gb", type=float, default=2.)
    args = parser.parse_args(argv)
    run_focused_study(args.data_dir, args.recipe, args.convergence, args.output,
                      n_trials=args.trials, decoder_trials=args.decoder_trials,
                      seeds=tuple(args.seeds), confirmation_seed=args.confirmation_seed,
                      cache_bytes=int(args.cache_gb * 1_000_000_000))


if __name__ == "__main__":
    main()
