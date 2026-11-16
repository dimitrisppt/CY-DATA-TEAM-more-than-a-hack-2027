"""Fixed tail-EMA stability comparison for the frozen four-network candidate."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from experiments import campaign_evaluation, campaign_protocol, campaign_search, ensemble_bagging
from experiments.campaign_decode import _validate_config
from experiments.ensemble_polish_confirm import _folds
from experiments.robustness import _execution_environment, _source_hashes
from experiments.tune import _seed_for_fold
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache

_SCHEMA = "ema_stability_v1"
_TAIL_START = 7000
_TAIL_END = 7200
_EMA_DECAY = .99
_CACHE_LIMIT = 1_000_000_000
_INNER_BATCH_NAMES = ("batch_1", "batch_2")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode()).hexdigest()


def _required(path: Path) -> dict:
    value = campaign_protocol.read_record(path)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _disjoint_folds(folds: list[dict]) -> None:
    seen = set()
    for row in folds:
        train, validation = set(row["train"]), set(row["validation"])
        if train & validation or seen & validation:
            raise ValueError("EMA fold train and validation matches must be disjoint")
        seen |= validation
    if len(seen) != 6:
        raise ValueError("EMA evaluation requires six unique inner validation matches")


def _verify_input_cache(input_root: Path, expected_signature: str,
                        expected_matches: set[str]) -> tuple[list[dict], dict]:
    manifest_path = input_root / "inputs" / "input-manifest.json"
    sidecar_path = manifest_path.with_suffix(".sha256.json")
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"reused EMA input cache is incomplete: {input_root}")
    if json.loads(sidecar_path.read_text()).get("sha256") != sha256_file(manifest_path):
        raise ValueError("reused input-manifest checksum differs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("signature_sha256") != expected_signature:
        raise ValueError("reused batch input signature differs from bagging proof")
    if _digest(manifest.get("signature")) != expected_signature:
        raise ValueError("reused input signature digest differs")
    indexed = {}
    records = []
    for raw in manifest.get("records", []):
        key = (raw.get("match"), raw.get("period"), raw.get("scenario"))
        if key in indexed:
            raise ValueError("reused input manifest contains duplicate rows")
        if raw.get("input_identity") != expected_signature:
            raise ValueError("reused input identity differs")
        path = campaign_evaluation._input_path(input_root / "inputs", *key)
        if not path.is_file() or sha256_file(path) != raw.get("input_sha256"):
            raise ValueError(f"reused input array hash differs: {path}")
        record = {**raw, "input_path": str(path)}
        indexed[key] = record
        records.append(record)
    scenarios = set(campaign_protocol.WEIGHTS)
    expected = {(match, period, scenario) for match in expected_matches
                for period in (1, 2) for scenario in scenarios}
    if set(indexed) != expected:
        raise ValueError("reused input cache does not cover the frozen match/scenario grid")
    return [indexed[key] for key in sorted(indexed)], manifest


def _gate(raw: dict, candidate: dict) -> dict:
    raw_matches = set(raw.get("by_match", {}))
    candidate_matches = set(candidate.get("by_match", {}))
    if raw_matches != candidate_matches or len(raw_matches) != 6:
        raise ValueError("inner EMA comparison requires the same six match scores")
    weights = campaign_protocol.WEIGHTS
    missing_names = ("short", "long", "replay")

    def missing(score, match=None):
        values = score["means"] if match is None else score["by_match"][match]
        return float(sum(weights[name] * values[name]["spell_f1_mean"]
                         for name in missing_names) / .5)

    delta_by_match = {match: missing(candidate, match) - missing(raw, match)
                      for match in sorted(raw_matches)}
    missing_delta = missing(candidate) - missing(raw)
    scenario_deltas = {name: float(candidate["means"][name]["spell_f1_mean"]
                                   - raw["means"][name]["spell_f1_mean"])
                       for name in weights}
    transition_delta = float(sum(weights[name]
                                 * (candidate["means"][name]["transition_f1_at_12"]
                                    - raw["means"][name]["transition_f1_at_12"])
                                 for name in missing_names) / .5)
    objective_delta = float(candidate["objective"] - raw["objective"])
    checks = {
        "objective_gain_gt_0_002": objective_delta - .002 > 1e-12,
        "missing_aggregate_improved": missing_delta > 0.,
        "four_of_six_missing_matches_improved": sum(v > 0. for v in delta_by_match.values()) >= 4,
        "scenario_spell_f1_loss_at_most_0_01": all(v >= -.01 for v in scenario_deltas.values()),
        "missing_transition_f1_loss_at_most_0_01": transition_delta >= -.01,
    }
    return {"passed": bool(all(checks.values())), "checks": checks,
            "objective_delta": objective_delta, "missing_aggregate_delta": missing_delta,
            "missing_by_match_deltas": delta_by_match,
            "scenario_spell_f1_deltas": scenario_deltas,
            "missing_transition_f1_delta": transition_delta}


def _outer_gate(raw: dict, candidate: dict) -> dict:
    delta = float(candidate["objective"] - raw["objective"])
    checks = {"outer_objective_loss_at_most_0_01":
              delta >= -.01 or np.isclose(delta, -.01, rtol=0., atol=1e-12)}
    return {"passed": bool(all(checks.values())), "checks": checks,
            "objective_delta": delta}


def _score_records(records_by_member: list[list[dict]], selection: dict) -> dict:
    if len(records_by_member) != 2:
        raise ValueError("frozen EMA candidate must have two architecture record bundles")
    left, right = (campaign_evaluation.load_records(bundle) for bundle in records_by_member)
    combined = campaign_search.blend_records(left, right, float(selection["blend_weight"]))
    return campaign_evaluation.score_records(combined, selection["decoder"],
                                               campaign_protocol.WEIGHTS)


def _score_bagged(records_by_seed: dict[int, list[list[dict]]], selection: dict) -> dict:
    seed_values = sorted(records_by_seed)
    if len(seed_values) != 2:
        raise ValueError("EMA bagging requires two fixed training seeds")
    architecture_averages = []
    for member_index in range(2):
        first = campaign_evaluation.load_records(records_by_seed[seed_values[0]][member_index])
        second = campaign_evaluation.load_records(records_by_seed[seed_values[1]][member_index])
        architecture_averages.append(campaign_search.blend_records(first, second, .5))
    combined = campaign_search.blend_records(architecture_averages[0],
                                               architecture_averages[1],
                                               float(selection["blend_weight"]))
    return campaign_evaluation.score_records(combined, selection["decoder"],
                                               campaign_protocol.WEIGHTS)


def _prediction_manifest_path(output_dir: Path, batch_name: str, scope: str,
                              seed: int, member_id: str, fold_index: int) -> Path:
    return (output_dir / "predictions" / batch_name / scope / f"seed_{seed}"
            / member_id / f"fold_{fold_index:02d}.json")


def _validate_prediction_grid(records: list[dict], fold: dict, input_records: list[dict],
                              identity: dict) -> None:
    expected = {(row["match"], row["period"], row["scenario"]): row
                for row in input_records if row["match"] in set(fold["validation"])}
    actual = {(row["match"], row["period"], row["scenario"]): row for row in records}
    if actual.keys() != expected.keys():
        raise ValueError("saved EMA predictions do not cover their validation grid")
    for key, record in actual.items():
        source = expected[key]
        if (record.get("model_identity") != identity
                or record.get("input_identity") != source.get("input_identity")
                or record.get("input_sha256") != source.get("input_sha256")):
            raise ValueError("saved EMA prediction has changed model/input identity")
    campaign_evaluation.load_records(records)


def _batch_artifact_hashes(output_dir: Path, batch_name: str, fits: dict,
                           selection: dict) -> dict:
    hashes = {}
    for _, fit_path, _ in fits.values():
        relative = fit_path.relative_to(output_dir).as_posix()
        hashes[relative] = sha256_file(fit_path)
    for scope, seeds, fold_count in (("inner", (8101, 8102), 3),
                                     ("outer", (9101, 9102), 1)):
        for seed in seeds:
            for member in selection["members"]:
                for fold_index in range(fold_count):
                    path = _prediction_manifest_path(output_dir, batch_name, scope, seed,
                                                     member["id"], fold_index)
                    if not path.is_file():
                        raise FileNotFoundError(f"EMA prediction manifest is missing: {path}")
                    hashes[path.relative_to(output_dir).as_posix()] = sha256_file(path)
    return dict(sorted(hashes.items()))


def _validate_selection(candidate: dict, baseline: dict) -> None:
    if (candidate.get("kind") != "convex_pair" or baseline.get("kind") != "convex_pair"
            or [row.get("id") for row in candidate.get("members", [])]
            != [row.get("id") for row in baseline.get("members", [])]
            or candidate.get("members") != baseline.get("members")):
        raise ValueError("EMA may not change frozen pair members or training specs")
    if any(row.get("steps") != _TAIL_END for row in candidate["members"]):
        raise ValueError("EMA candidate selection must retain 7200-step member specs")
    _validate_config(candidate["decoder"])


def _fit_identity(campaign_dir: Path, protocol_sha: str, member: dict, seed: int,
                  scope: str, fold_index: int, fold: dict) -> tuple[dict, Path, Path, Path]:
    fit_fold = 0 if scope == "outer" else fold_index
    fit_dir = (campaign_dir / "fits" / scope / member["id"] / f"seed_{seed}"
               / f"fold_{fit_fold}")
    destination = fit_dir / "rung_7200"
    fit = _required(destination / "fit.json")
    expected = {"protocol_sha256": protocol_sha, "spec_sha256": _digest(member["spec"]),
                "train_matches": fold["train"],
                "seed": _seed_for_fold(seed, "DFL-MAT-J03WMX", fit_fold),
                "name": member["id"]}
    if fit.get("identity") != expected or fit.get("steps") != _TAIL_END:
        raise ValueError("original fit identity differs from frozen EMA tail task")
    checkpoint = destination / "model.pt"
    if sha256_file(checkpoint) != fit.get("checkpoint_sha256"):
        raise ValueError("raw 7200-step checkpoint hash differs from fit record")
    state = fit_dir / f"state_{_TAIL_START:06d}.pt"
    state_sidecar = state.with_suffix(state.suffix + ".sha256.json")
    end_state = fit_dir / f"state_{_TAIL_END:06d}.pt"
    end_sidecar = end_state.with_suffix(end_state.suffix + ".sha256.json")
    if not state.is_file() or not state_sidecar.is_file():
        raise FileNotFoundError(f"saved 7000-step state is missing: {state}")
    if json.loads(state_sidecar.read_text()).get("sha256") != sha256_file(state):
        raise ValueError("saved 7000-step state hash differs")
    if not end_state.is_file() or not end_sidecar.is_file():
        raise FileNotFoundError(f"saved 7200-step state is missing: {end_state}")
    if json.loads(end_sidecar.read_text()).get("sha256") != sha256_file(end_state):
        raise ValueError("saved 7200-step state hash differs")
    return fit, state, checkpoint, destination


def _read_original_seed_records(bagging_dir: Path, batch_name: str, scope: str, seed: int,
                               selection: dict, protocol: dict, campaign_dir: Path,
                               folds: list[dict], input_sha: str) -> list[list[dict]]:
    manifest = _required(bagging_dir / batch_name / f"{scope}_seed_{seed}.json")
    if (manifest.get("seed") != seed or manifest.get("scope") != scope
            or manifest.get("fresh_inputs_sha256") != input_sha
            or manifest.get("selection_sha256") != _digest(selection)
            or manifest.get("protocol_sha256") != _digest(protocol)):
        raise ValueError("saved v4 prediction manifest identity differs")
    ensemble_bagging._manifest_identity(
        manifest["records"], member=selection, seed=seed, scope=scope,
        folds=folds, protocol_sha=_digest(protocol), fresh_inputs_sha=input_sha,
        campaign_dir=campaign_dir)
    return manifest["records"]


def _ema_model_records(output_dir: Path, batch_name: str, scope: str, seed: int,
                       selection: dict, folds: list[dict], input_records: list[dict],
                       input_sha: str, fits: dict, protocol_sha: str,
                       check: Callable[[], None]) -> list[list[dict]]:
    grouped = {}
    for record in input_records:
        grouped.setdefault(record["match"], []).append(record)
    fold_rows = folds if scope == "inner" else [{"train": folds[0]["train"],
                                                   "validation": ["DFL-MAT-J03WMX"]}]
    all_members = []
    for member in selection["members"]:
        member_records = []
        for index, fold in enumerate(fold_rows):
            fit_key = (scope, seed, member["id"], index if scope == "inner" else 0)
            fit_row, fit_path, model_path = fits[fit_key]
            identity = {"protocol_sha256": protocol_sha,
                        "ema_fit_sha256": sha256_file(fit_path),
                        "ema_sha256": fit_row["ema_sha256"],
                        "ema_checkpoint_sha256": fit_row["checkpoint_sha256"],
                        "raw_checkpoint_sha256": fit_row["inputs"]["baseline_checkpoint_sha256"],
                        "scope": scope, "seed": seed, "fold": index,
                        "train_matches": fold["train"],
                        "validation_matches": fold["validation"],
                        "input_manifest_sha256": input_sha}
            record_path = (output_dir / "predictions" / batch_name / scope / f"seed_{seed}"
                           / member["id"] / f"fold_{index:02d}.json")
            saved = campaign_protocol.read_record(record_path)
            if saved is not None:
                if saved.get("identity") != identity:
                    raise ValueError("saved EMA prediction identity differs")
                records = saved["records"]
                _validate_prediction_grid(records, fold, input_records, identity)
            else:
                check()
                from possession.predict import load_model_checkpoint

                net, _ = load_model_checkpoint(model_path, device="cpu")
                selected = [record for match in fold["validation"]
                            for record in grouped.get(match, [])]
                if {record["match"] for record in selected} != set(fold["validation"]):
                    raise ValueError("EMA input records do not cover fold validation matches")
                prediction_root = record_path.parent / record_path.stem
                records = campaign_evaluation.predict_records(
                    net, selected, prediction_root, identity, center_size=2048,
                    progress_callback=check)
                _validate_prediction_grid(records, fold, input_records, identity)
                del net
                campaign_protocol.publish(record_path, {"identity": identity, "records": records})
            member_records.extend(records)
        all_members.append(member_records)
    return all_members


class _Budget:
    def __init__(self, output: Path, max_seconds: int, callback=None):
        if (not isinstance(max_seconds, int) or isinstance(max_seconds, bool)
                or not 1 <= max_seconds <= 5400):
            raise ValueError("EMA stability budget must be between 1 and 5400 seconds")
        self.output, self.callback = output, callback
        path = output / "clock.json"
        previous = campaign_protocol.read_record(path)
        now = time.time()
        if previous is None:
            self.clock = campaign_protocol.publish(path, {
                "started_unix": now, "max_seconds": max_seconds,
                "deadline_unix": now + max_seconds})
        else:
            if previous.get("max_seconds") != max_seconds:
                raise ValueError("resume cannot extend the EMA stability budget")
            self.clock = previous

    def check(self):
        if time.time() >= self.clock["deadline_unix"]:
            raise campaign_protocol.BudgetExceeded("EMA stability wall-clock budget expired")
        if self.callback is not None:
            self.callback()

    def status(self, stage: str, **detail):
        write_json(self.output / "progress.json", {
            "schema": _SCHEMA, "stage": stage, "updated_unix": time.time(),
            "deadline_unix": self.clock["deadline_unix"],
            "seconds_remaining": max(0., self.clock["deadline_unix"] - time.time()),
            **detail})


def run_ema_stability(campaign_dir, polish_dir, bagging_dir, data_dir, target_dir,
                      output_dir, *, max_seconds: int = 5400,
                      check: Callable[[], None] | None = None) -> dict:
    """Run fixed 7000-to-7200 EMA tails and compare on authenticated bagging masks."""
    campaign_dir, polish_dir, bagging_dir, data_dir, target_dir, output_dir = (
        Path(value).resolve() for value in (campaign_dir, polish_dir, bagging_dir,
                                             data_dir, target_dir, output_dir))
    for source in (campaign_dir, polish_dir, bagging_dir, data_dir, target_dir):
        if output_dir.is_relative_to(source) or source.is_relative_to(output_dir):
            raise ValueError("EMA output must be separate from all input directories")
    output_dir.mkdir(parents=True, exist_ok=True)
    budget = _Budget(output_dir, max_seconds, check)
    budget.check()
    (campaign_dir, polish_dir, campaign_payload, baseline, polish_payload,
     candidate, _diagnostic) = ensemble_bagging._authenticated_inputs(
         campaign_dir, polish_dir, data_dir, target_dir)
    folds = _folds(campaign_payload)
    _disjoint_folds(folds)
    _validate_selection(candidate, baseline)
    bagging_protocol = _required(bagging_dir / "protocol.json")
    bagging_result = _required(bagging_dir / "result.json")
    bagging_sha = _digest(bagging_protocol)
    if (bagging_protocol.get("schema") != "ensemble_bagging_confirmation_v1"
            or bagging_result.get("protocol_sha256") != bagging_sha
            or bagging_result.get("eligible") is not True
            or bagging_protocol.get("campaign_protocol_sha256") != _digest(campaign_payload)
            or bagging_protocol.get("polish_protocol_sha256") != _digest(polish_payload)
            or bagging_protocol.get("candidate_selection") != candidate
            or bagging_result.get("candidate_selection") != candidate):
        raise ValueError("bagging evidence differs from original frozen candidate")
    current_sources = _source_hashes()
    if (bagging_protocol.get("runtime") != _execution_environment()
            or any(current_sources.get(name) != digest
                   for name, digest in bagging_protocol.get("source_sha256", {}).items())):
        raise ValueError("bagging source/runtime differs from current EMA runner")
    if bagging_result.get("batches") is None or len(bagging_result["batches"]) != 2:
        raise ValueError("bagging result must contain both frozen batches")
    if tuple(row["name"] for row in bagging_protocol.get("mask_batches", [])) != _INNER_BATCH_NAMES:
        raise ValueError("bagging mask batches differ from frozen pair")

    batch_rows, batch_inputs = [], {}
    for batch_config, summary in zip(bagging_protocol["mask_batches"],
                                     bagging_result["batches"], strict=True):
        name = batch_config["name"]
        raw_batch = _required(bagging_dir / name / "result.json")
        inner_sha, outer_sha = raw_batch.get("inner_inputs_sha256"), raw_batch.get("outer_inputs_sha256")
        inner_records, inner_manifest = _verify_input_cache(
            bagging_dir / name / "inner", inner_sha,
            {match for fold in folds for match in fold["validation"]})
        outer_records, outer_manifest = _verify_input_cache(
            bagging_dir / name / "outer", outer_sha, {"DFL-MAT-J03WMX"})
        ensemble_bagging._verify_batch_result(
            raw_batch, bagging_sha, _digest(baseline), _digest(candidate), inner_sha, outer_sha,
            campaign_dir, campaign_payload, folds, batch_config)
        batch_rows.append((name, raw_batch, summary))
        batch_inputs[name] = {"inner": (inner_records, inner_sha, inner_manifest),
                              "outer": (outer_records, outer_sha, outer_manifest)}
    ensemble_bagging._validate_completed_result(
        bagging_result, [row for _, row, _ in batch_rows], baseline, candidate, bagging_sha)

    task_specs = []
    original_fits = {}
    outer_folds = [{"train": campaign_payload["fold"]["outer_train"],
                    "validation": ["DFL-MAT-J03WMX"]}]
    for scope, seeds, scope_folds in (("inner", (8101, 8102), folds),
                                      ("outer", (9101, 9102), outer_folds)):
        for seed in seeds:
            for fold_index, fold in enumerate(scope_folds):
                for member in candidate["members"]:
                    fit, state, raw_checkpoint, _ = _fit_identity(
                        campaign_dir, _digest(campaign_payload), member, seed,
                        scope, fold_index, fold)
                    key = (scope, seed, member["id"], fold_index)
                    original_fits[key] = (fit, state, raw_checkpoint)
                    task_specs.append((scope, seed, fold_index, fold, member, key))

    source_sha = _source_hashes()
    protocol = {"schema": _SCHEMA, "development_only": True,
                "campaign_protocol_sha256": _digest(campaign_payload),
                "polish_protocol_sha256": _digest(polish_payload),
                "bagging_protocol_sha256": bagging_sha,
                "bagging_result_sha256": sha256_file(bagging_dir / "result.json"),
                "campaign_dir": str(campaign_dir), "polish_dir": str(polish_dir),
                "bagging_dir": str(bagging_dir), "data_dir": str(data_dir),
                "target_input_manifest_sha256": sha256_file(target_dir / "input-manifest.json"),
                "baseline_selection": baseline, "candidate_selection": candidate,
                "baseline_selection_sha256": _digest(baseline),
                "candidate_selection_sha256": _digest(candidate),
                "members": candidate["members"], "inner_folds": folds,
                "tail_tasks": [{"scope": scope, "seed": seed, "fold": fold_index,
                                "member_id": member["id"],
                                "original_fit_sha256": fit["checkpoint_sha256"],
                                "start_state_sha256": sha256_file(state),
                                "end_state_sha256": sha256_file(state.parent / "state_007200.pt")}
                               for scope, seed, fold_index, fold, member, key in task_specs
                               for fit, state, _ in (original_fits[key],)],
                "inner_seeds": [8101, 8102], "outer_seeds": [9101, 9102],
                "tail": {"start_steps": _TAIL_START, "end_steps": _TAIL_END,
                         "updates": _TAIL_END - _TAIL_START, "ema_decay": _EMA_DECAY},
                "input_reuse": {name: {scope: {"signature_sha256": values[1],
                                               "manifest_sha256": sha256_file(
                                                   bagging_dir / name / scope / "inputs" / "input-manifest.json")}
                                        for scope, values in scopes.items()}
                                for name, scopes in batch_inputs.items()},
                "gate": {"inner_objective_gain_gt": .002,
                         "missing_aggregate_gain_gt": 0., "positive_matches_at_least": 4,
                         "scenario_loss_at_most": .01, "missing_transition_loss_at_most": .01,
                         "outer_objective_loss_at_most": .01},
                "source_sha256": source_sha, "runtime": _execution_environment(),
                "max_seconds": max_seconds,
                "interpretation": "Fixed EMA tail comparison on reused development matches and outage masks. No target labels."}
    protocol_path = output_dir / "protocol.json"
    campaign_protocol.publish(protocol_path, protocol)
    protocol_sha = _digest(protocol)
    preflight = {"protocol_sha256": protocol_sha,
                 "bagging_batches": [{"name": name,
                                      "raw_batch_sha256": sha256_file(bagging_dir / name / "result.json"),
                                      "inner_inputs_sha256": row["inner_inputs_sha256"],
                                      "outer_inputs_sha256": row["outer_inputs_sha256"]}
                                     for name, row, _ in batch_rows],
                 "source_sha256": source_sha, "runtime": _execution_environment(),
                 "target_ground_truth_used": False, "inputs_regenerated": False}
    campaign_protocol.publish(output_dir / "preflight.json", preflight)

    budget.status("tails")
    cache = HalfCache(data_dir, max_bytes=_CACHE_LIMIT)
    fit_rows = {}
    total = len(task_specs)
    for number, (scope, seed, fold_index, fold, member, fit_key) in enumerate(task_specs, 1):
        budget.check()
        fit, state_path, baseline_checkpoint = original_fits[fit_key]
        target = output_dir / "fits" / scope / member["id"] / f"seed_{seed}" / f"fold_{fold_index:02d}"
        from experiments.ema_training import replay_tail_ema

        ema_fit = replay_tail_ema(
            data_dir=data_dir, spec=member["spec"], identity=fit["identity"],
            state_path=state_path, baseline_checkpoint=baseline_checkpoint,
            baseline_sha256=fit["checkpoint_sha256"], output_dir=target,
            validation_matches=fold["validation"], decay=_EMA_DECAY,
            start_steps=_TAIL_START, end_steps=_TAIL_END, cache=cache,
            progress_callback=budget.check)
        ema_fit_path = target / "fit.json"
        parity = {"raw_weights_match_endpoint_state": True,
                  "raw_weights_match_baseline_checkpoint": True,
                  "raw_training_progress_matches_endpoint": True,
                  "compared_optimizer_rng_transcript_and_counters": True,
                  "start_steps": _TAIL_START, "end_steps": _TAIL_END}
        if (ema_fit.get("identity") != fit["identity"]
                or ema_fit.get("inputs", {}).get("baseline_checkpoint_sha256")
                != fit["checkpoint_sha256"]
                or ema_fit.get("inputs", {}).get("start_state_sha256") != sha256_file(state_path)
                or ema_fit.get("inputs", {}).get("spec_sha256") != _digest(member["spec"])
                or ema_fit.get("validation_matches") != fold["validation"]
                or ema_fit.get("tail_ema", {}).get("decay") != _EMA_DECAY
                or ema_fit.get("tail_ema", {}).get("updates") != _TAIL_END - _TAIL_START
                or ema_fit.get("raw_parity") != parity
                or ema_fit.get("tail_ema", {}).get("raw_parity") != parity
                or ema_fit.get("steps") != _TAIL_END
                or Path(ema_fit.get("model_path", "")).resolve() != (target / "model.pt").resolve()
                or not (target / "model.pt").is_file()
                or sha256_file(target / "model.pt") != ema_fit.get("checkpoint_sha256")):
            raise ValueError("EMA fit or raw-training parity identity differs")
        fit_rows[(scope, seed, member["id"], fold_index)] = (ema_fit, ema_fit_path, target / "model.pt")
        budget.status("tails", completed=number, total=total, scope=scope,
                      seed=seed, member=member["id"], fold=fold_index)
    if cache.peak_bytes > _CACHE_LIMIT:
        raise RuntimeError("EMA tail raw-half cache exceeded its one GB cap")
    del cache

    completed_batches = []
    for name, raw_batch, summary in batch_rows:
        budget.check()
        result_path = output_dir / "batches" / name / "result.json"
        input_cache = batch_inputs[name]
        inner_records, inner_sha, _ = input_cache["inner"]
        outer_records, outer_sha, _ = input_cache["outer"]
        ema_inner_by_seed, v4_inner_by_seed = {}, {}
        for seed in (8101, 8102):
            budget.check()
            raw_bundles = _read_original_seed_records(
                bagging_dir, name, "inner", seed, candidate, campaign_payload,
                campaign_dir, folds, inner_sha)
            v4_inner_by_seed[seed] = _score_records(raw_bundles, candidate)
            ema_inner_by_seed[seed] = _ema_model_records(
                output_dir, name, "inner", seed, candidate, folds, inner_records,
                inner_sha, fit_rows, protocol_sha, budget.check)
            budget.status("inner_predictions", batch=name, seed=seed)
        raw_inner_score = summary["candidate"]
        ema_inner_score = _score_bagged(ema_inner_by_seed, candidate)
        checks = _gate(raw_inner_score, ema_inner_score)

        ema_outer_by_seed, v4_outer_by_seed = {}, {}
        outer_folds = [{"train": campaign_payload["fold"]["outer_train"],
                        "validation": ["DFL-MAT-J03WMX"]}]
        for seed in (9101, 9102):
            budget.check()
            raw_bundles = _read_original_seed_records(
                bagging_dir, name, "outer", seed, candidate, campaign_payload,
                campaign_dir, outer_folds, outer_sha)
            v4_outer_by_seed[seed] = _score_records(raw_bundles, candidate)
            ema_outer_by_seed[seed] = _ema_model_records(
                output_dir, name, "outer", seed, candidate, outer_folds, outer_records,
                outer_sha, fit_rows, protocol_sha, budget.check)
        raw_outer_score = summary["outer"]["candidate"]
        ema_outer_score = _score_bagged(ema_outer_by_seed, candidate)
        outer_checks = _outer_gate(raw_outer_score, ema_outer_score)
        row = {"schema": _SCHEMA, "protocol_sha256": protocol_sha, "batch": name,
               "inner_inputs_sha256": inner_sha, "outer_inputs_sha256": outer_sha,
               "raw_v4_inner": raw_inner_score, "ema_inner": ema_inner_score,
               "raw_v4_inner_by_seed": v4_inner_by_seed,
               "ema_inner_by_seed": {seed: _score_records(bundle, candidate)
                                     for seed, bundle in ema_inner_by_seed.items()},
               "inner_checks": checks,
               "raw_v4_outer": raw_outer_score, "ema_outer": ema_outer_score,
               "raw_v4_outer_by_seed": v4_outer_by_seed,
               "ema_outer_by_seed": {seed: _score_records(bundle, candidate)
                                     for seed, bundle in ema_outer_by_seed.items()},
               "outer_checks": outer_checks, "development_only": True,
               "target_ground_truth_used": False,
               "artifact_sha256": _batch_artifact_hashes(output_dir, name, fit_rows, candidate)}
        budget.check()
        published = campaign_protocol.publish(result_path, row)
        completed_batches.append(published)
        budget.status("batch_complete", batch=name, inner_passed=checks["passed"],
                      outer_passed=outer_checks["passed"])

    if len(completed_batches) != 2:
        raise RuntimeError("EMA stability did not complete both frozen batches")
    result = {"schema": _SCHEMA, "protocol_sha256": protocol_sha,
              "candidate_selection": candidate, "baseline_selection": baseline,
              "batches": completed_batches,
              "eligible": all(row["inner_checks"]["passed"] and row["outer_checks"]["passed"]
                              for row in completed_batches),
              "summary": {"inner_gates_passed": all(row["inner_checks"]["passed"]
                                                    for row in completed_batches),
                          "outer_gates_passed": all(row["outer_checks"]["passed"]
                                                    for row in completed_batches),
                          "mean_inner_objective_delta": float(np.mean([
                              row["inner_checks"]["objective_delta"] for row in completed_batches]))},
              "notes": ["EMA tails reuse saved 7000-step training states and replay exactly 200 updates.",
                        "The development input caches and v4 predictions are reused.",
                        "Per-seed scores are diagnostics. They are not independent ensemble replications.",
                        "No target possession labels or all-seven-match fits were used."]}
    budget.check()
    campaign_protocol.publish(output_dir / "result.json", result)
    budget.status("complete", eligible=result["eligible"], batches=2)
    return result
