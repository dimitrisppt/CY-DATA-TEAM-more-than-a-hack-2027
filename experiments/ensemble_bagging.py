"""Preregistered, no-training confirmation of seed bagging on fresh masks."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from experiments import campaign_evaluation, campaign_protocol
from experiments.ensemble_polish_confirm import (
    _digest,
    _folds,
    _predict_selection,
    _selection,
)
from experiments.ensemble_polish_search import blend_loaded
from experiments.robustness import _execution_environment, _source_hashes
from experiments.tune import _seed_for_fold
from possession.prepared import verify_prepared
from possession.storage import sha256_file
from possession.training_session import HalfCache

MASK_BATCHES = (
    {"name": "batch_1", "short_seed": 73104, "long_seed": 74104},
    {"name": "batch_2", "short_seed": 73105, "long_seed": 74105},
)
_SCENARIO_FIXED = {
    "clean": {"kind": "clean"},
    "replay": {"kind": "target_replay", "seed": 5102},
}
_RAW_HALF_CACHE_BYTES = 1_000_000_000


def _scenarios(batch):
    return {
        "clean": dict(_SCENARIO_FIXED["clean"]),
        "short": {"kind": "synthetic", "seed": batch["short_seed"],
                  "missing_fraction": .41, "min_seconds": 1., "max_seconds": 10.},
        "long": {"kind": "synthetic", "seed": batch["long_seed"],
                 "missing_fraction": .41, "min_seconds": 10., "max_seconds": 30.},
        "replay": dict(_SCENARIO_FIXED["replay"]),
    }


def _required(path):
    value = campaign_protocol.read_record(path)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _selection_pair(value, name):
    result = _selection(value, name)
    if len(result["members"]) != 2:
        raise ValueError(f"{name} must be a two-member pair")
    return result


def _authenticated_inputs(campaign_dir, polish_dir, data_dir, target_dir):
    campaign_dir, polish_dir = Path(campaign_dir).resolve(), Path(polish_dir).resolve()
    data_dir, target_dir = Path(data_dir).resolve(), Path(target_dir).resolve()
    campaign_protocol = _required(campaign_dir / "protocol.json")
    baseline = _selection_pair(_required(campaign_dir / "selection.json"),
                               "original campaign selection")
    campaign_summary = _required(campaign_dir / "summary.json")
    if (campaign_summary.get("status") != "complete"
            or campaign_summary.get("candidate_eligible_for_review") is not True
            or campaign_summary.get("protocol_sha256") != _digest(campaign_protocol)
            or campaign_summary.get("selection") != baseline):
        raise ValueError("original campaign is not the completed confirmed selection")
    polish_protocol = _required(polish_dir / "protocol.json")
    polish_selection_record = _required(polish_dir / "selection.json")
    polish_summary = _required(polish_dir / "summary.json")
    candidate = _selection_pair(polish_selection_record.get("selection"),
                                "polish selection")
    if (polish_selection_record.get("protocol_sha256") != _digest(polish_protocol)
            or polish_protocol.get("campaign_protocol_sha256") != _digest(campaign_protocol)
            or polish_protocol.get("baseline_selection") != baseline
            or polish_selection_record.get("baseline_selection_sha256") != _digest(baseline)
            or polish_summary.get("status") != "complete"
            or polish_summary.get("protocol_sha256") != _digest(polish_protocol)
            or polish_summary.get("selection") != candidate):
        raise ValueError("polish protocol, selection, or completed summary identity differs")
    if [(m["id"], m["spec"], m["steps"]) for m in baseline["members"]] != [
            (m["id"], m["spec"], m["steps"]) for m in candidate["members"]]:
        raise ValueError("polish candidate member specs or order differ from the original pair")
    confirmation = polish_summary.get("confirmation", {})
    if (confirmation.get("candidate_selection_sha256") != _digest(candidate)
            or confirmation.get("baseline_selection_sha256") != _digest(baseline)):
        raise ValueError("polish confirmation does not authenticate these frozen selections")
    diagnostic = confirmation.get("seed_bagging")
    if (not isinstance(diagnostic, dict) or diagnostic.get("available") is not True
            or diagnostic.get("not_independent_confirmation_seeds") is not True
            or not isinstance(diagnostic.get("score"), dict)):
        raise ValueError("polish summary lacks the completed seed-bagging diagnostic")
    current_sources = _source_hashes()
    for label, source_snapshot in (("campaign", campaign_protocol.get("source_sha256", {})),
                                   ("polish", polish_protocol.get("source_sha256", {}))):
        changed = [name for name, digest in source_snapshot.items()
                   if current_sources.get(name) != digest]
        if changed:
            raise ValueError(f"{label} source snapshot differs: {changed}")
    current_runtime = _execution_environment()
    if (current_runtime != campaign_protocol.get("environment")
            or current_runtime != polish_protocol.get("runtime")):
        raise ValueError("campaign, polish, and current runtime differ")
    cache_paths = [data_dir / row["cache"]
                   for row in campaign_protocol.get("prepared", {}).get("caches", [])]
    if not cache_paths or verify_prepared(data_dir, cache_paths) != campaign_protocol.get("prepared"):
        raise ValueError("prepared public data differs from original campaign")
    target_manifest = target_dir / "input-manifest.json"
    if sha256_file(target_manifest) != campaign_protocol.get("target_input_manifest_sha256"):
        raise ValueError("target automatic-input manifest differs from original campaign")
    return campaign_dir, polish_dir, campaign_protocol, baseline, polish_protocol, candidate, diagnostic


def _manifest_identity(records, *, member, seed, scope, folds, protocol_sha,
                       fresh_inputs_sha, campaign_dir):
    if len(records) != len(member["members"]):
        raise ValueError("prediction member bundle count differs")
    fold_rows = folds
    for member_records, selected_member in zip(records, member["members"], strict=True):
        seen = set()
        for record in member_records:
            identity = record.get("model_identity", {})
            fold_index = identity.get("fold")
            if not isinstance(fold_index, int) or isinstance(fold_index, bool):
                raise TypeError("prediction record has no valid fold index")
            if not 0 <= fold_index < len(fold_rows):
                raise ValueError("prediction record fold index is out of range")
            fold = fold_rows[fold_index]
            key = (record["match"], record["period"], record["scenario"])
            if key in seen:
                raise ValueError("duplicate prediction record")
            seen.add(key)
            fit_root = Path(campaign_dir) / "fits" / scope / selected_member["id"] / f"seed_{seed}"
            fit_root = fit_root / f"fold_{fold_index}" / "rung_7200"
            fit = _required(fit_root / "fit.json")
            if sha256_file(fit_root / "model.pt") != fit.get("checkpoint_sha256"):
                raise ValueError("saved prediction checkpoint checksum differs")
            actual_seed = _seed_for_fold(seed, "DFL-MAT-J03WMX", fold_index)
            expected = {"protocol_sha256": protocol_sha,
                        "checkpoint_sha256": fit["checkpoint_sha256"],
                        "train_matches": fold["train"],
                        "validation_matches": fold["validation"], "scope": scope,
                        "fold": fold_index, "seed": seed,
                        "actual_train_seed": actual_seed,
                        "fresh_inputs_sha256": fresh_inputs_sha}
            if identity != expected or fit.get("identity", {}).get("name") != selected_member["id"]:
                raise ValueError("prediction record identity differs from frozen fit")
            if record.get("input_identity") != fresh_inputs_sha:
                raise ValueError("prediction record input identity differs from fresh mask batch")
            if (record["match"] not in fold["validation"]
                    or record["period"] not in (1, 2)
                    or record["scenario"] not in campaign_protocol.WEIGHTS):
                raise ValueError("prediction record is outside the declared input grid")
            if (fit.get("identity", {}).get("protocol_sha256") != protocol_sha
                    or fit.get("identity", {}).get("spec_sha256") != _digest(selected_member["spec"])
                    or fit.get("identity", {}).get("train_matches") != fold["train"]
                    or fit.get("identity", {}).get("seed") != actual_seed
                    or fit.get("steps") != 7200):
                raise ValueError("prediction record's checkpoint fit ancestry differs")
        expected_grid = {(match, period, scenario)
                         for fold in fold_rows for match in fold["validation"]
                         for period in (1, 2) for scenario in campaign_protocol.WEIGHTS}
        actual_grid = {(row["match"], row["period"], row["scenario"])
                       for row in member_records}
        if actual_grid != expected_grid:
            raise ValueError("prediction records do not cover the complete full-match grid")
        campaign_evaluation.load_records(member_records)


def _seed_records(campaign_dir, protocol_sha, selection, seed, scope, folds,
                  input_records, inputs_sha, prediction_root, batch_dir, check):
    manifest_path = batch_dir / f"{scope}_seed_{seed}.json"
    existing = campaign_protocol.read_record(manifest_path)
    selection_sha = _digest(selection)
    if existing is not None:
        if (existing.get("selection_sha256") != selection_sha
                or existing.get("protocol_sha256") != protocol_sha
                or existing.get("fresh_inputs_sha256") != inputs_sha
                or existing.get("seed") != seed or existing.get("scope") != scope):
            raise ValueError("saved prediction manifest identity differs")
        _manifest_identity(existing["records"], member=selection, seed=seed, scope=scope,
                           folds=folds, protocol_sha=protocol_sha,
                           fresh_inputs_sha=inputs_sha, campaign_dir=campaign_dir)
        return existing["records"]
    if check is not None:
        check()
    _, bundles = _predict_selection(
        campaign_dir, protocol_sha, selection, seed, folds, input_records, inputs_sha,
        prediction_root, scope, check)
    _manifest_identity(bundles, member=selection, seed=seed, scope=scope,
                       folds=folds, protocol_sha=protocol_sha,
                       fresh_inputs_sha=inputs_sha, campaign_dir=campaign_dir)
    campaign_protocol.publish(manifest_path, {
        "scope": scope, "seed": seed, "selection_sha256": selection_sha,
        "protocol_sha256": protocol_sha, "fresh_inputs_sha256": inputs_sha,
        "records": bundles})
    return bundles


def _load_architectures(bundles):
    return [campaign_evaluation.load_records(records) for records in bundles]


def _score_baseline(raw_by_seed, selection, check=None):
    scores = []
    for architectures in raw_by_seed:
        if check is not None:
            check()
        combined = blend_loaded(architectures[0], architectures[1],
                                float(selection["blend_weight"]))
        scores.append(campaign_evaluation.score_records(combined, selection["decoder"],
                                                          campaign_protocol.WEIGHTS))
    return scores


def _score_bagged(raw_by_seed, selection, check=None):
    averaged_architectures = [blend_loaded(raw_by_seed[0][index], raw_by_seed[1][index], .5)
                              for index in range(2)]
    combined = blend_loaded(averaged_architectures[0], averaged_architectures[1],
                            float(selection["blend_weight"]))
    if check is not None:
        check()
    return campaign_evaluation.score_records(combined, selection["decoder"],
                                              campaign_protocol.WEIGHTS)


def _missing_spell(score, match=None):
    names = ("short", "long", "replay")
    if match is None:
        return float(sum(campaign_protocol.WEIGHTS[name] * score["means"][name]["spell_f1_mean"]
                         for name in names) / .5)
    return float(sum(campaign_protocol.WEIGHTS[name]
                     * score["by_match"][match][name]["spell_f1_mean"]
                     for name in names) / .5)


def _checks(baselines, candidate):
    if len(baselines) != 2:
        raise ValueError("two individual-seed baselines are required")
    mean_scenario = {
        name: float(np.mean([score["means"][name]["spell_f1_mean"] for score in baselines]))
        for name in campaign_protocol.WEIGHTS}
    missing_delta = _missing_spell(candidate) - float(sum(
        campaign_protocol.WEIGHTS[name] * mean_scenario[name]
        for name in ("short", "long", "replay")) / .5)
    matches = sorted(candidate["by_match"])
    per_match_missing = {match: _missing_spell(candidate, match) - float(np.mean(
        [_missing_spell(score, match) for score in baselines])) for match in matches}
    candidate_transition = sum(campaign_protocol.WEIGHTS[name]
                               * candidate["means"][name]["transition_f1_at_12"]
                               for name in ("short", "long", "replay")) / .5
    baseline_transition = float(np.mean([sum(
        campaign_protocol.WEIGHTS[name] * score["means"][name]["transition_f1_at_12"]
        for name in ("short", "long", "replay")) / .5 for score in baselines]))
    checks = {
        "beats_both_seed_baselines_by_margin": all(
            candidate["objective"] - score["objective"] > .002 for score in baselines),
        "missing_aggregate_improved": missing_delta > 0,
        "four_matches_missing_improved": len(matches) == 6 and sum(
            value > 0 for value in per_match_missing.values()) >= 4,
        "clean_preserved": candidate["means"]["clean"]["spell_f1_mean"]
        - mean_scenario["clean"] >= -.01,
        "short_preserved": candidate["means"]["short"]["spell_f1_mean"]
        - mean_scenario["short"] >= -.01,
        "long_preserved": candidate["means"]["long"]["spell_f1_mean"]
        - mean_scenario["long"] >= -.01,
        "replay_preserved": candidate["means"]["replay"]["spell_f1_mean"]
        - mean_scenario["replay"] >= -.01,
        "missing_transitions_preserved": candidate_transition - baseline_transition >= -.01,
    }
    return {"passed": bool(all(checks.values())), "checks": checks,
            "missing_delta": float(missing_delta),
            "missing_transition_delta": float(candidate_transition - baseline_transition),
            "missing_by_match_deltas": per_match_missing,
            "baseline_mean_scenario_spell_f1": mean_scenario,
            "seed_objective_deltas": [float(candidate["objective"] - score["objective"])
                                       for score in baselines]}


def _outer_checks(baselines, candidate):
    if len(baselines) != 2:
        raise ValueError("two individual-seed outer baselines are required")
    deltas = [float(candidate["objective"] - score["objective"]) for score in baselines]
    checks = {"beats_best_outer_seed_within_margin":
              candidate["objective"] >= max(score["objective"] for score in baselines) - .01}
    return {"passed": bool(all(checks.values())), "checks": checks,
            "seed_objective_deltas": deltas}


def _input_bundle(data_dir, target_dir, batch_dir, matches, scenarios, check):
    cache = HalfCache(data_dir, max_bytes=_RAW_HALF_CACHE_BYTES)
    records = campaign_evaluation.prepare_inputs(
        data_dir, target_dir, batch_dir / "inputs", matches, scenarios, cache,
        progress_callback=check)
    if cache.peak_bytes > _RAW_HALF_CACHE_BYTES:
        raise RuntimeError("raw half cache exceeded its one GB limit")
    manifest_path = batch_dir / "inputs" / "input-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del cache
    gc.collect()
    return records, manifest["signature_sha256"]


def _verify_batch_result(row, protocol_sha, baseline_sha, candidate_sha,
                         inner_inputs_sha, outer_inputs_sha, campaign_dir,
                         protocol, folds, batch):
    if (row.get("protocol_sha256") != protocol_sha
            or row.get("baseline_selection_sha256") != baseline_sha
            or row.get("candidate_selection_sha256") != candidate_sha
            or row.get("inner_inputs_sha256") != inner_inputs_sha
            or row.get("outer_inputs_sha256") != outer_inputs_sha
            or row.get("candidate_selection") is None
            or _digest(row["candidate_selection"]) != candidate_sha):
        raise ValueError("saved batch result identity differs")
    expected_mask_seeds = {"short": batch["short_seed"], "long": batch["long_seed"],
                           "replay": 5102}
    if row.get("mask_seeds") != expected_mask_seeds:
        raise ValueError("saved batch mask seeds differ")
    for scope, seeds, fold_rows, inputs_sha in (
            ("inner", campaign_protocol.CONFIRMATION_SEEDS, folds, inner_inputs_sha),
            ("outer", campaign_protocol.OUTER_SEEDS,
             [{"train": protocol["fold"]["outer_train"],
               "validation": ["DFL-MAT-J03WMX"]}], outer_inputs_sha)):
        for seed in seeds:
            manifest_path = Path(row["prediction_manifest_dir"]) / f"{scope}_seed_{seed}.json"
            manifest = _required(manifest_path)
            _manifest_identity(manifest["records"], member=row["candidate_selection"],
                               seed=seed, scope=scope, folds=fold_rows,
                               protocol_sha=_digest(protocol), fresh_inputs_sha=inputs_sha,
                               campaign_dir=campaign_dir)
    return row


def _final_payload(rows, baseline, candidate, plan_sha):
    eligible = all(row["checks"]["passed"] and row["outer"]["checks"]["passed"]
                   for row in rows)
    summary = {"batch_objective_deltas_vs_mean_individual_seeds": [
        float(row["candidate"]["objective"]
              - np.mean([score["objective"] for score in row["baseline"]])) for row in rows],
        "mean_bagged_objective": float(np.mean([row["candidate"]["objective"] for row in rows])),
        "mean_baseline_objective": float(np.mean([score["objective"]
                                                   for row in rows for score in row["baseline"]])),
        "all_inner_gates_passed": all(row["checks"]["passed"] for row in rows),
        "all_outer_gates_passed": all(row["outer"]["checks"]["passed"] for row in rows)}
    return {"eligible": bool(eligible), "candidate_selection": candidate,
            "baseline_selection": baseline,
            "batches": [{"mask_seeds": row["mask_seeds"], "baseline": row["baseline"],
                         "candidate": row["candidate"], "checks": row["checks"],
                         "outer": row["outer"]} for row in rows],
            "summary": summary, "protocol_sha256": plan_sha,
            "notes": ["These are two fresh outage-mask batches, not independent model-seed replications.",
                      "All model seeds, development matches, clean inputs, and replay inputs are reused.",
                      "Development evidence only. No independent test claim.",
                      "Candidate remains the exact decoder-polish selection diagnosed before this run."]}


def _validate_completed_result(done, rows, baseline, candidate, plan_sha):
    expected = _final_payload(rows, baseline, candidate, plan_sha)
    if done != expected:
        raise ValueError("completed result differs from revalidated batch records")
    return done


def _run_batch(campaign_dir, data_dir, target_dir, output_dir, batch,
               protocol_sha, protocol, folds, baseline, candidate, check):
    batch_dir = Path(output_dir) / batch["name"]
    result_path = batch_dir / "result.json"
    inner_matches = sorted(match for fold in folds for match in fold["validation"])
    outer_fold = {"train": protocol["fold"]["outer_train"],
                  "validation": ["DFL-MAT-J03WMX"]}
    outer_folds = [outer_fold]
    scenarios = _scenarios(batch)
    inner_inputs, inner_sha = _input_bundle(data_dir, target_dir, batch_dir / "inner",
                                             inner_matches, scenarios, check)
    outer_inputs, outer_sha = _input_bundle(data_dir, target_dir, batch_dir / "outer",
                                             outer_fold["validation"], scenarios, check)
    baseline_sha, candidate_sha = _digest(baseline), _digest(candidate)
    previous = campaign_protocol.read_record(result_path)
    if previous is not None:
        return _verify_batch_result(previous, protocol_sha, baseline_sha, candidate_sha,
                                    inner_sha, outer_sha, campaign_dir, protocol, folds, batch)
    prediction_root = batch_dir / "predictions"
    inner_by_seed = []
    for seed in campaign_protocol.CONFIRMATION_SEEDS:
        records = _seed_records(campaign_dir, _digest(protocol), candidate, seed, "inner",
                                folds, inner_inputs, inner_sha, prediction_root, batch_dir, check)
        inner_by_seed.append(_load_architectures(records))
        gc.collect()
    baseline_scores = _score_baseline(inner_by_seed, baseline, check)
    candidate_score = _score_bagged(inner_by_seed, candidate, check)
    gate = _checks(baseline_scores, candidate_score)
    del inner_by_seed
    gc.collect()

    outer_by_seed = []
    for seed in campaign_protocol.OUTER_SEEDS:
        records = _seed_records(campaign_dir, _digest(protocol), candidate, seed, "outer",
                                outer_folds, outer_inputs, outer_sha, prediction_root,
                                batch_dir, check)
        outer_by_seed.append(_load_architectures(records))
        gc.collect()
    outer_baseline = _score_baseline(outer_by_seed, baseline, check)
    outer_candidate = _score_bagged(outer_by_seed, candidate, check)
    outer_checks = _outer_checks(outer_baseline, outer_candidate)
    del outer_by_seed
    gc.collect()
    payload = {"mask_seeds": {"short": batch["short_seed"], "long": batch["long_seed"],
                              "replay": 5102},
               "baseline": baseline_scores, "candidate": candidate_score,
               "checks": gate,
               "outer": {"baseline": outer_baseline, "candidate": outer_candidate,
                         "checks": outer_checks},
               "inner_inputs_sha256": inner_sha, "outer_inputs_sha256": outer_sha,
               "baseline_selection_sha256": baseline_sha,
               "candidate_selection_sha256": candidate_sha,
               "protocol_sha256": protocol_sha,
               "candidate_selection": candidate,
               "prediction_manifest_dir": str(batch_dir),
               "development_only": True}
    gc.collect()
    return campaign_protocol.publish(result_path, payload)


def run_bagging_confirmation(campaign_dir, polish_dir, data_dir, target_dir,
                             output_dir, *, check=None) -> dict:
    """Confirm the frozen two-seed bagged pair on two new mask batches.

    Training checkpoints and development matches are reused. This evaluates two
    outage-mask batches, not two independent model-seed replications.
    """
    if check is not None:
        check()
    (campaign_dir, polish_dir, campaign_protocol_payload, baseline,
     polish_protocol, candidate, diagnostic) = _authenticated_inputs(
         campaign_dir, polish_dir, data_dir, target_dir)
    folds = _folds(campaign_protocol_payload)
    baseline_sha, candidate_sha = _digest(baseline), _digest(candidate)
    output = Path(output_dir).resolve()
    for base in (campaign_dir, polish_dir, Path(data_dir).resolve(), Path(target_dir).resolve()):
        if output.is_relative_to(base) or base.is_relative_to(output):
            raise ValueError("bagging confirmation output must be separate from all inputs")
    output.mkdir(parents=True, exist_ok=True)
    campaign_hash_names = ("protocol.json", "selection.json", "summary.json",
                           "confirmation.json", "outer.json", "final.json")
    polish_hash_names = ("protocol.json", "selection.json", "summary.json")
    plan = {
        "schema": "ensemble_bagging_confirmation_v1", "development_only": True,
        "campaign_dir": str(campaign_dir), "polish_dir": str(polish_dir),
        "data_dir": str(Path(data_dir).resolve()), "target_dir": str(Path(target_dir).resolve()),
        "campaign_protocol_sha256": _digest(campaign_protocol_payload),
        "polish_protocol_sha256": _digest(polish_protocol),
        "prepared_provenance": campaign_protocol_payload["prepared"],
        "target_input_manifest_sha256": campaign_protocol_payload["target_input_manifest_sha256"],
        "campaign_record_sha256": {name: sha256_file(Path(campaign_dir) / name)
                                    for name in campaign_hash_names},
        "polish_record_sha256": {name: sha256_file(Path(polish_dir) / name)
                                  for name in polish_hash_names},
        "baseline_selection": baseline, "baseline_selection_sha256": baseline_sha,
        "candidate_selection": candidate, "candidate_selection_sha256": candidate_sha,
        "diagnostic_objective": diagnostic["score"]["objective"],
        "diagnostic_objective_delta": diagnostic["objective_delta_vs_mean_singles"],
        "source_sha256": _source_hashes(), "runtime": _execution_environment(),
        "mask_batches": [{**batch, "scenarios": _scenarios(batch)} for batch in MASK_BATCHES],
        "confirmation_training_seeds_reused": list(campaign_protocol.CONFIRMATION_SEEDS),
        "outer_training_seeds_reused": list(campaign_protocol.OUTER_SEEDS),
        "gate": {"beats_each_inner_seed_by": .002,
                 "missing_aggregate_gain_gt": 0., "four_of_six_missing_matches_gt": 0.,
                 "scenario_loss_at_most": .01, "missing_transition_loss_at_most": .01,
                 "outer_loss_at_most": .01, "all_checks_in_both_mask_batches": True},
        "interpretation": "Two fresh mask batches, reused model seeds and matches. Development evidence only.",
    }
    campaign_protocol.publish(output / "protocol.json", plan)
    plan_sha = _digest(plan)
    done = campaign_protocol.read_record(output / "result.json")
    if done is not None:
        if done.get("protocol_sha256") != plan_sha:
            raise ValueError("completed bagging confirmation protocol differs")
        inner_matches = sorted(match for fold in folds for match in fold["validation"])
        outer_fold = [{"train": campaign_protocol_payload["fold"]["outer_train"],
                       "validation": ["DFL-MAT-J03WMX"]}]
        rows = []
        for batch in MASK_BATCHES:
            batch_dir = output / batch["name"]
            scenarios = _scenarios(batch)
            _, inner_sha = _input_bundle(data_dir, target_dir, batch_dir / "inner",
                                          inner_matches, scenarios, check)
            _, outer_sha = _input_bundle(data_dir, target_dir, batch_dir / "outer",
                                          outer_fold[0]["validation"], scenarios, check)
            row = _required(batch_dir / "result.json")
            _verify_batch_result(row, plan_sha, baseline_sha, candidate_sha,
                                 inner_sha, outer_sha, campaign_dir,
                                 campaign_protocol_payload, folds, batch)
            rows.append(row)
        return _validate_completed_result(done, rows, baseline, candidate, plan_sha)

    rows = []
    for batch in MASK_BATCHES:
        if check is not None:
            check()
        rows.append(_run_batch(campaign_dir, data_dir, target_dir, output, batch,
                               plan_sha, campaign_protocol_payload, folds,
                               baseline, candidate, check))
    if check is not None:
        check()
    result = _final_payload(rows, baseline, candidate, plan_sha)
    return campaign_protocol.publish(output / "result.json", result)
