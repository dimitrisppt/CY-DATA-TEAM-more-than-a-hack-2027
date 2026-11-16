"""Restartable Optuna decoder tuning and safe in-memory neural blending."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.distributions import distribution_to_json, json_to_distribution

from experiments import campaign_decode, campaign_evaluation, campaign_protocol


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _record_inputs(records: list[dict]) -> list[dict]:
    """Bind durable search state to prediction/input provenance, not loaded arrays."""
    result = []
    for record in records:
        result.append({key: value for key, value in record.items() if key != "arrays"})
    return sorted(result, key=lambda row: (row["match"], row["period"], row["scenario"]))


def _reference_only(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.update({"blend_mode": "static", "neural_weight": 0.0,
                   "visible_neural_weight": 0.0, "missing_neural_weight": 0.0,
                   "visible_live_weight": 0.0, "missing_live_weight": 0.0,
                   "visible_owner_weight": 0.0, "missing_owner_weight": 0.0,
                   "boundary_scale": 0.0})
    campaign_decode._validate_config(result)
    return result


def _neural_only(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.update({"blend_mode": "static", "neural_weight": 1.0,
                   "visible_neural_weight": 1.0, "missing_neural_weight": 1.0,
                   "visible_live_weight": 1.0, "missing_live_weight": 1.0,
                   "visible_owner_weight": 1.0, "missing_owner_weight": 1.0})
    campaign_decode._validate_config(result)
    return result


def _anchors(baseline: dict, reference_only: bool, extra_anchors) -> list[dict]:
    candidates = ([{"name": "baseline", "decoder": _reference_only(baseline)}]
                  if reference_only else [
                      {"name": "baseline", "decoder": baseline},
                      {"name": "neural_only", "decoder": _neural_only(baseline)},
                      {"name": "reference_only", "decoder": _reference_only(baseline)},
                  ])
    for index, anchor in enumerate(extra_anchors):
        if isinstance(anchor, dict) and set(anchor) == {"name", "decoder"}:
            name, decoder = anchor["name"], anchor["decoder"]
        else:
            name, decoder = f"extra_{index}", anchor
        if not isinstance(name, str) or not name:
            raise ValueError("extra anchor name must be a nonempty string")
        campaign_decode._validate_config(decoder)
        candidates.append({"name": name, "decoder": _canonical(decoder)})
    unique, seen = [], set()
    for row in candidates:
        key = json.dumps(_canonical(row["decoder"]), sort_keys=True, separators=(",", ":"))
        if key not in seen:
            unique.append({"name": row["name"], "decoder": _canonical(row["decoder"])})
            seen.add(key)
    return unique


def _rebuild_study(seed: int, proposal_index: int, history: list[dict]) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed + proposal_index, multivariate=True,
                                         group=True, n_startup_trials=4)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for row in history:
        distributions = {key: json_to_distribution(value)
                         for key, value in row["distributions"].items()}
        study.add_trial(optuna.trial.create_trial(
            params=row["params"], distributions=distributions, value=float(row["value"])))
    return study


def _score(records, decoder):
    return campaign_evaluation.score_records(records, decoder, campaign_protocol.WEIGHTS)


def tune_decoder(records: list[dict], baseline_config: dict, output_dir, n_trials: int,
                 seed: int, *, reference_only: bool = False,
                 check: Callable[[], None] | None = None, extra_anchors=()) -> dict:
    """Tune decoder against loaded public OOF records, with hash-checked resume.

    ``n_trials`` includes explicit anchor trials. ``check`` is called before
    preload, after preload, and before every unfinished trial for budget stops.
    """
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    campaign_decode._validate_config(baseline_config)
    baseline = _canonical(baseline_config)
    extra_anchors = list(extra_anchors)
    anchors = _anchors(baseline, reference_only, extra_anchors)
    if len(anchors) > n_trials:
        raise ValueError("anchor count exceeds n_trials")
    output_dir = Path(output_dir)
    trial_dir = output_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    if check is not None:
        check()
    loaded = campaign_evaluation.load_records(records)
    if check is not None:
        check()
    protocol = {"schema": "campaign_decoder_search_v1",
                "records": _record_inputs(records), "baseline_config": baseline,
                "n_trials": n_trials, "seed": seed,
                "extra_anchors": _canonical(list(extra_anchors)),
                "reference_only": bool(reference_only), "anchors": anchors,
                "scenario_weights": campaign_protocol.WEIGHTS}
    plan_path = output_dir / "plan.json"
    campaign_protocol.publish(plan_path, protocol)

    existing_files = sorted(trial_dir.glob("trial-*.json"))
    found = {}
    for path in existing_files:
        payload = campaign_protocol.read_record(path)
        if payload is None:
            raise ValueError(f"empty completed trial record: {path}")
        index = payload.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index in found:
            raise ValueError(f"invalid or duplicate trial index: {path}")
        if index < 0 or index >= n_trials:
            raise ValueError(f"completed trial outside requested range: {path}")
        found[index] = payload
    if found and set(found) != set(range(max(found) + 1)):
        raise ValueError("completed decoder trial records contain a hole")

    completed: list[dict] = []
    tpe_history: list[dict] = []
    for index in range(n_trials):
        path = trial_dir / f"trial-{index:04d}.json"
        if index in found:
            row = found[index]
            expected_base = {"schema": "campaign_decoder_trial_v1", "index": index,
                             "protocol_sha256": _plan_hash(protocol)}
            if any(row.get(key) != value for key, value in expected_base.items()):
                raise ValueError(f"completed trial protocol differs: {path}")
            if row.get("kind") == "tpe":
                tpe_history.append(row["tpe"])
            completed.append(row)
            continue
        if check is not None:
            check()
        if index < len(anchors):
            anchor = anchors[index]
            decoder = anchor["decoder"]
            score = _score(loaded, decoder)
            row = {"schema": "campaign_decoder_trial_v1", "index": index,
                   "protocol_sha256": _plan_hash(protocol), "kind": "anchor",
                   "anchor": anchor["name"], "decoder": decoder, "score": score,
                   "value": float(score["objective"])}
        else:
            study = _rebuild_study(seed, index, tpe_history)
            trial = study.ask()
            decoder = campaign_decode.sample_decoder(trial, reference_only=reference_only)
            score = _score(loaded, decoder)
            frozen = study.tell(trial, float(score["objective"]))
            frozen = study.trials[-1]
            tpe_row = {"params": frozen.params,
                       "distributions": {key: distribution_to_json(value)
                                         for key, value in frozen.distributions.items()},
                       "value": float(score["objective"])}
            row = {"schema": "campaign_decoder_trial_v1", "index": index,
                   "protocol_sha256": _plan_hash(protocol), "kind": "tpe",
                   "decoder": decoder, "score": score, "value": float(score["objective"]),
                   "tpe": tpe_row}
            tpe_history.append(tpe_row)
        campaign_protocol.publish(path, row)
        completed.append(row)
    best = max(completed, key=lambda row: (float(row["value"]), -row["index"]))
    return {"decoder": best["decoder"], "scores": best["score"],
            "trials_complete": len(completed)}


def _plan_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def blend_records(left_loaded: list[dict], right_loaded: list[dict], weight: float) -> list[dict]:
    """Logit blend aligned neural arrays; ``weight`` is the left-model weight."""
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("blend weight must lie in [0,1]")
    def index(records):
        result = {}
        for record in records:
            key = (record["match"], record["period"], record["scenario"])
            if key in result:
                raise ValueError("duplicate record in model bundle")
            result[key] = record
        return result
    left, right = index(left_loaded), index(right_loaded)
    if not left or left.keys() != right.keys():
        raise ValueError("model bundles must have identical nonempty grids")
    result = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a.get("input_sha256") != b.get("input_sha256"):
            raise ValueError("model bundles have different cached inputs")
        aa, ba = a.get("arrays"), b.get("arrays")
        if not isinstance(aa, dict) or not isinstance(ba, dict):
            raise TypeError("blend_records requires loaded arrays")
        shared = ("frame_id", "gt", "ball_valid", "nearest_ball_seconds")
        shared += tuple(sorted(k for k in aa if k.startswith("ref_")))
        for name in shared:
            if name not in aa or name not in ba or not np.array_equal(aa[name], ba[name], equal_nan=True):
                raise ValueError(f"model bundle public input differs: {name}")
        for name in ("neural_live", "neural_home", "neural_boundary"):
            if name not in aa or name not in ba or aa[name].shape != ba[name].shape:
                raise ValueError(f"model bundle prediction shape differs: {name}")
        blended = {name: np.asarray(value).copy() for name, value in aa.items()}
        for name in ("neural_live", "neural_home"):
            p = np.clip(np.asarray(aa[name], dtype=np.float64), 1e-7, 1 - 1e-7)
            q = np.clip(np.asarray(ba[name], dtype=np.float64), 1e-7, 1 - 1e-7)
            logits = weight * (np.log(p) - np.log1p(-p)) + (1 - weight) * (
                np.log(q) - np.log1p(-q))
            blended[name] = 1. / (1. + np.exp(-np.clip(logits, -40, 40)))
            if weight == 1:
                blended[name] = np.asarray(aa[name]).copy()
            elif weight == 0:
                blended[name] = np.asarray(ba[name]).copy()
        blended["neural_boundary"] = (weight * np.asarray(aa["neural_boundary"])
                                      + (1 - weight) * np.asarray(ba["neural_boundary"]))
        identity = {"kind": "logit_blend", "weight_left": float(weight),
                    "left": a.get("model_identity"), "right": b.get("model_identity")}
        result.append({**a, "arrays": blended, "left_model_identity": a.get("model_identity"),
                       "right_model_identity": b.get("model_identity"),
                       "model_identity": identity})
    return result
