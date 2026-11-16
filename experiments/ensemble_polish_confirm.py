"""No-training confirmation of a frozen pair on fresh outage masks.

This is development evidence only. It reuses campaign checkpoints, match IDs,
and target replay seed 5102. The short and long synthetic masks use new seeds.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import campaign_evaluation, campaign_protocol, campaign_search
from experiments.tune import _seed_for_fold
from possession.predict import load_model_checkpoint
from possession.storage import sha256_file
from possession.training_session import HalfCache

FRESH_SCENARIOS = {
    "clean": {"kind": "clean"},
    "short": {"kind": "synthetic", "seed": 73103, "missing_fraction": .41,
              "min_seconds": 1., "max_seconds": 10.},
    "long": {"kind": "synthetic", "seed": 74103, "missing_fraction": .41,
             "min_seconds": 10., "max_seconds": 30.},
    "replay": {"kind": "target_replay", "seed": 5102},
}
_SCHEMA = "ensemble_polish_confirm_v1"
_STEPS = 7200
_RAW_HALF_CACHE_BYTES = 1_000_000_000


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False))


def _selection(value, name):
    if not isinstance(value, dict) or not isinstance(value.get("members"), list):
        raise TypeError(f"{name} must contain members")
    members = value["members"]
    if not 1 <= len(members) <= 2:
        raise ValueError(f"{name} must contain one or two campaign members")
    ids = []
    for member in members:
        if (not isinstance(member, dict) or not isinstance(member.get("id"), str)
                or not isinstance(member.get("spec"), dict)
                or member.get("steps") != _STEPS):
            raise ValueError(f"{name} member must have id, spec, and 7200 steps")
        ids.append(member["id"])
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} contains duplicate members")
    if not isinstance(value.get("decoder"), dict):
        raise TypeError(f"{name} must contain a decoder")
    if len(members) == 2:
        weight = value.get("blend_weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
            raise ValueError(f"{name} pair needs a blend_weight in [0, 1]")
    elif value.get("blend_weight") not in (None, 0, 1):
        raise ValueError(f"{name} single member cannot have a blend weight")
    return _canonical(value)


def _protocol(campaign_dir):
    path = Path(campaign_dir) / "protocol.json"
    payload = campaign_protocol.read_record(path)
    if payload is None:
        raise FileNotFoundError(f"campaign protocol not found: {path}")
    return payload, _digest(payload)


def _folds(protocol):
    try:
        folds = protocol["fold"]["inner"]
    except (KeyError, TypeError) as exc:
        raise ValueError("campaign protocol lacks inner folds") from exc
    if len(folds) != 3:
        raise ValueError("campaign protocol must have three inner folds")
    if any(set(row["train"]) & set(row["validation"]) for row in folds):
        raise ValueError("campaign fold train and validation matches overlap")
    validation = [match for row in folds for match in row["validation"]]
    if len(validation) != 6 or len(set(validation)) != 6:
        raise ValueError("inner folds must cover six distinct development matches")
    return folds


def _fit_model(campaign_dir, protocol_sha, member, seed, fold_index, fold, scope,
               check=None):
    campaign_dir = Path(campaign_dir)
    actual_seed = _seed_for_fold(seed, "DFL-MAT-J03WMX", fold_index)
    if scope == "inner":
        fit_dir = campaign_dir / "fits" / "inner" / member["id"] / f"seed_{seed}" / f"fold_{fold_index}"
        train_matches, validation_matches = fold["train"], fold["validation"]
    else:
        fit_dir = campaign_dir / "fits" / "outer" / member["id"] / f"seed_{seed}" / "fold_0"
        train_matches, validation_matches = fold["train"], fold["validation"]
    destination = fit_dir / f"rung_{_STEPS}"
    fit_path, model_path = destination / "fit.json", destination / "model.pt"
    fit = campaign_protocol.read_record(fit_path)
    if fit is None:
        raise FileNotFoundError(f"required existing fit record not found: {fit_path}")
    expected_identity = {"protocol_sha256": protocol_sha,
                         "spec_sha256": _digest(member["spec"]),
                         "train_matches": train_matches, "seed": actual_seed,
                         "name": member["id"]}
    if fit.get("identity") != expected_identity or fit.get("steps") != _STEPS:
        raise ValueError(f"existing fit identity differs: {fit_path}")
    if check is not None:
        check()
    checkpoint_sha = sha256_file(model_path)
    if checkpoint_sha != fit.get("checkpoint_sha256"):
        raise ValueError(f"checkpoint checksum differs: {model_path}")
    model, payload = load_model_checkpoint(model_path)
    config = payload.get("training_config", {})
    expected_model_kwargs = {"player_dim": 8, "global_dim": 9, "width": 64,
                             "dilations": [1, 2, 4, 8, 16, 32, 64, 128]}
    expected_model_kwargs.update(member["spec"].get("model_kwargs", {}))
    if (payload.get("model_kwargs") != expected_model_kwargs
            or config.get("campaign_spec") != member["spec"]
            or config.get("seed") != actual_seed
            or config.get("steps") != _STEPS
            or config.get("train_matches") != train_matches
            or config.get("validation_matches") != []):
        del model, payload
        raise ValueError(f"checkpoint recipe, seed, or fold differs: {model_path}")
    return model, {"protocol_sha256": protocol_sha, "checkpoint_sha256": checkpoint_sha,
                   "train_matches": train_matches, "validation_matches": validation_matches,
                   "scope": scope, "fold": fold_index, "seed": seed,
                   "actual_train_seed": actual_seed,
                   "fresh_inputs_sha256": None}


def _predict_selection(campaign_dir, protocol_sha, selection, seed, folds, inputs,
                      fresh_inputs_sha, output_dir, scope, check=None):
    grouped = {}
    for record in inputs:
        grouped.setdefault(record["match"], []).append(record)
    bundles = []
    fold_rows = folds if scope == "inner" else [{"train": folds[0]["train"],
                                                  "validation": ["DFL-MAT-J03WMX"]}]
    for member in selection["members"]:
        records = []
        for index, fold in enumerate(fold_rows):
            if check is not None:
                check()
            model, identity = _fit_model(campaign_dir, protocol_sha, member, seed,
                                         index if scope == "inner" else 0, fold, scope, check)
            identity["fresh_inputs_sha256"] = fresh_inputs_sha
            selected = [r for match in fold["validation"] for r in grouped.get(match, [])]
            if not selected or {r["match"] for r in selected} != set(fold["validation"]):
                raise ValueError("input records do not cover this validation fold")
            prediction_dir = (Path(output_dir) / scope / member["id"] / f"seed_{seed}"
                              / f"fold_{index if scope == 'inner' else 0}")
            records.extend(campaign_evaluation.predict_records(
                model, selected, prediction_dir, identity,
                progress_callback=check))
            del model
            gc.collect()
        bundles.append(records)
    loaded = [campaign_evaluation.load_records(records) for records in bundles]
    if len(loaded) == 1:
        return loaded[0], bundles
    return campaign_search.blend_records(loaded[0], loaded[1], float(selection["blend_weight"])), bundles


def _score_selection(campaign_dir, protocol_sha, selection, seed, folds, inputs,
                     fresh_inputs_sha, output_dir, scope, check=None):
    combined, bundles = _predict_selection(campaign_dir, protocol_sha, selection, seed,
                                           folds, inputs, fresh_inputs_sha, output_dir,
                                           scope, check)
    if check is not None:
        check()
    score = campaign_evaluation.score_records(combined, selection["decoder"],
                                               campaign_protocol.WEIGHTS)
    return score, bundles


def _selection_identity(selection):
    return _digest(_canonical(selection))


def _validate_saved(row, *, seed, baseline_sha, candidate_sha, fresh_inputs_sha):
    expected = {"seed": seed, "baseline_selection_sha256": baseline_sha,
                "candidate_selection_sha256": candidate_sha,
                "fresh_inputs_sha256": fresh_inputs_sha}
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("saved confirmation result identity differs")
    for key in ("baseline", "candidate"):
        campaign_evaluation.load_records([record for bundle in row[key + "_records"]
                                          for record in bundle])


def _bagging(campaign_dir, protocol_sha, selection, folds, inputs, fresh_inputs_sha,
             output_dir, check=None):
    if len(selection["members"]) != 2:
        return {"available": False, "reason": "candidate selection is not a pair"}
    seed_bundles = []
    for seed in campaign_protocol.CONFIRMATION_SEEDS:
        _, bundles = _predict_selection(campaign_dir, protocol_sha, selection, seed,
                                        folds, inputs, fresh_inputs_sha, output_dir,
                                        "inner", check)
        seed_bundles.append([campaign_evaluation.load_records(bundle) for bundle in bundles])
        del bundles
        gc.collect()
    averaged = [campaign_search.blend_records(seed_bundles[0][i], seed_bundles[1][i], .5)
                for i in range(2)]
    combined = campaign_search.blend_records(averaged[0], averaged[1],
                                              float(selection["blend_weight"]))
    if check is not None:
        check()
    score = campaign_evaluation.score_records(combined, selection["decoder"],
                                               campaign_protocol.WEIGHTS)
    singles = []
    for seed_records in seed_bundles:
        paired = campaign_search.blend_records(seed_records[0], seed_records[1],
                                                float(selection["blend_weight"]))
        singles.append(campaign_evaluation.score_records(paired, selection["decoder"],
                                                          campaign_protocol.WEIGHTS))
    matches = sorted(score["by_match"])
    delta = {match: float(
        sum(campaign_protocol.WEIGHTS[name] * score["by_match"][match][name]["spell_f1_mean"]
            for name in campaign_protocol.WEIGHTS)
        - np.mean([sum(campaign_protocol.WEIGHTS[name]
                       * item["by_match"][match][name]["spell_f1_mean"]
                       for name in campaign_protocol.WEIGHTS) for item in singles]))
        for match in matches}
    del seed_bundles, averaged, combined
    gc.collect()
    return {"available": True, "score": score,
            "mean_single_seed_objective": float(np.mean([item["objective"] for item in singles])),
            "objective_delta_vs_mean_singles": float(score["objective"]
                                                       - np.mean([item["objective"] for item in singles])),
            "per_match_objective_delta_vs_mean_singles": delta,
            "not_independent_confirmation_seeds": True, "automatic_promotion": False}


def confirm_pair(campaign_dir, data_dir, target_dir, output_dir,
                 baseline_selection, candidate_selection, *, check=None) -> dict:
    """Compare frozen selections on existing cross-fitted checkpoints.

    The two synthetic outage masks are newly sampled and fixed. The model seeds,
    six inner-fold matches, and target replay mask are reused campaign evidence.
    No target possession labels are read, and no training session is created.
    """
    if check is not None:
        check()
    baseline = _selection(baseline_selection, "baseline_selection")
    candidate = _selection(candidate_selection, "candidate_selection")
    if [(m["id"], m["spec"], m["steps"]) for m in baseline["members"]] != [
            (m["id"], m["spec"], m["steps"]) for m in candidate["members"]]:
        raise ValueError("candidate may change only decoder settings and pair blend weight")
    protocol, protocol_sha = _protocol(campaign_dir)
    if set(campaign_protocol.WEIGHTS) != set(FRESH_SCENARIOS):
        raise ValueError("scenario definitions differ from frozen campaign weights")
    folds = _folds(protocol)
    baseline_sha, candidate_sha = _selection_identity(baseline), _selection_identity(candidate)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = {"schema": _SCHEMA, "development_only": True,
            "historically_reused": {"matches": sorted(m for row in folds for m in row["validation"]),
                                    "model_seeds": list(campaign_protocol.CONFIRMATION_SEEDS),
                                    "target_replay_seed": 5102},
            "fresh_scenarios": FRESH_SCENARIOS, "protocol_sha256": protocol_sha,
            "baseline_selection": baseline, "baseline_selection_sha256": baseline_sha,
            "candidate_selection": candidate, "candidate_selection_sha256": candidate_sha,
            "steps": _STEPS, "raw_half_cache_max_bytes": _RAW_HALF_CACHE_BYTES}
    campaign_protocol.publish(output / "plan.json", plan)
    inputs_dir = output / "fresh_inputs"
    cache = HalfCache(data_dir, max_bytes=_RAW_HALF_CACHE_BYTES)
    matches = sorted(m for row in folds for m in row["validation"])
    inputs = campaign_evaluation.prepare_inputs(
        data_dir, target_dir, inputs_dir, matches, FRESH_SCENARIOS, cache,
        progress_callback=check)
    if cache.peak_bytes > _RAW_HALF_CACHE_BYTES:
        raise RuntimeError("raw half cache exceeded its one GB limit")
    manifest = json.loads((inputs_dir / "input-manifest.json").read_text())
    fresh_inputs_sha = manifest["signature_sha256"]
    del cache
    gc.collect()
    results = []
    for seed in campaign_protocol.CONFIRMATION_SEEDS:
        if check is not None:
            check()
        path = output / f"seed_{seed}.json"
        previous = campaign_protocol.read_record(path)
        if previous is not None:
            _validate_saved(previous, seed=seed, baseline_sha=baseline_sha,
                            candidate_sha=candidate_sha, fresh_inputs_sha=fresh_inputs_sha)
            results.append(previous)
            continue
        baseline_score, baseline_records = _score_selection(
            campaign_dir, protocol_sha, baseline, seed, folds, inputs, fresh_inputs_sha,
            output / "predictions", "inner", check)
        gc.collect()
        candidate_score, candidate_records = _score_selection(
            campaign_dir, protocol_sha, candidate, seed, folds, inputs, fresh_inputs_sha,
            output / "predictions", "inner", check)
        row = {"seed": seed, "baseline_selection_sha256": baseline_sha,
               "candidate_selection_sha256": candidate_sha,
               "fresh_inputs_sha256": fresh_inputs_sha,
               "baseline": baseline_score, "candidate": candidate_score,
               "baseline_records": baseline_records, "candidate_records": candidate_records,
               "development_only": True}
        campaign_protocol.publish(path, row)
        results.append(row)
        del baseline_records, candidate_records
        gc.collect()
    if check is not None:
        check()
    gate = campaign_protocol.paired_gate([row["baseline"] for row in results],
                                         [row["candidate"] for row in results])
    outer_baseline, outer_candidate = [], []
    outer_fold = {"train": protocol["fold"]["outer_train"],
                  "validation": ["DFL-MAT-J03WMX"]}
    outer_matches = outer_fold["validation"]
    outer_inputs = campaign_evaluation.prepare_inputs(
        data_dir, target_dir, output / "outer_inputs", outer_matches, FRESH_SCENARIOS,
        HalfCache(data_dir, max_bytes=_RAW_HALF_CACHE_BYTES), progress_callback=check)
    outer_manifest = json.loads((output / "outer_inputs" / "input-manifest.json").read_text())
    outer_input_sha = outer_manifest["signature_sha256"]
    for seed in campaign_protocol.OUTER_SEEDS:
        path = output / f"outer_seed_{seed}.json"
        saved = campaign_protocol.read_record(path)
        if saved is not None:
            if (saved.get("baseline_selection_sha256") != baseline_sha
                    or saved.get("candidate_selection_sha256") != candidate_sha
                    or saved.get("fresh_inputs_sha256") != outer_input_sha):
                raise ValueError("saved outer result identity differs")
            for records in saved["baseline_records"] + saved["candidate_records"]:
                campaign_evaluation.load_records(records)
            outer_baseline.append(saved["baseline"])
            outer_candidate.append(saved["candidate"])
            continue
        base_score, base_records = _score_selection(
            campaign_dir, protocol_sha, baseline, seed, [outer_fold], outer_inputs,
            outer_input_sha, output / "predictions", "outer", check)
        gc.collect()
        candidate_score, candidate_records = _score_selection(
            campaign_dir, protocol_sha, candidate, seed, [outer_fold], outer_inputs,
            outer_input_sha, output / "predictions", "outer", check)
        saved = {"seed": seed, "baseline_selection_sha256": baseline_sha,
                 "candidate_selection_sha256": candidate_sha,
                 "fresh_inputs_sha256": outer_input_sha,
                 "baseline": base_score, "candidate": candidate_score,
                 "baseline_records": base_records, "candidate_records": candidate_records,
                 "development_only": True}
        campaign_protocol.publish(path, saved)
        outer_baseline.append(base_score)
        outer_candidate.append(candidate_score)
        del base_records, candidate_records
        gc.collect()
    outer_delta = float(np.mean([c["objective"] - b["objective"]
                                 for b, c in zip(outer_baseline, outer_candidate, strict=True)]))
    outer = {"seeds": list(campaign_protocol.OUTER_SEEDS), "baseline": outer_baseline,
             "candidate": outer_candidate, "objective_delta": outer_delta,
             "guard_passed": outer_delta >= -.01, "gate_passed": outer_delta >= -.01,
             "historically_reused_match": True,
             "used_for_tuning": False, "test_claim": False}
    bagging = _bagging(campaign_dir, protocol_sha, candidate, folds, inputs,
                       fresh_inputs_sha, output / "predictions", check)
    result = {"development_only": True, "protocol_sha256": protocol_sha,
              "baseline_selection_sha256": baseline_sha,
              "candidate_selection_sha256": candidate_sha,
              "fresh_inputs_sha256": fresh_inputs_sha,
              "confirmation": {"seeds": list(campaign_protocol.CONFIRMATION_SEEDS),
                               "baseline": [row["baseline"] for row in results],
                               "candidate": [row["candidate"] for row in results],
                               "gate": gate},
              "gate": gate, "outer": outer, "seed_bagging": bagging,
              "outer_wmx": outer, "seed_bagging_diagnostic": bagging,
              "eligible": bool(gate["passed"] and outer["gate_passed"]),
              "historically_reused_matches": sorted(m for row in folds for m in row["validation"]),
              "fresh_scenarios": FRESH_SCENARIOS,
              "notes": ["Fresh short and long synthetic masks. Model seeds, matches, and replay mask are reused.",
                        "Development evidence only. No independent test claim."],
              "interpretation": "Fresh synthetic masks, reused model seeds and matches. Development evidence only."}
    return campaign_protocol.publish(output / "summary.json", result)
