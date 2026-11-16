"""Small, restartable joint search over a frozen neural pair and decoder."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.distributions import FloatDistribution, distribution_to_json, json_to_distribution

from experiments import campaign_decode, campaign_evaluation, campaign_protocol, campaign_search
from possession.storage import write_json

_SCHEMA = "ensemble_polish_search_v1"
SEARCH_BOUNDS = {
    "left_weight": (.25, .85), "neural_weight": (.25, .8),
    "live_temperature": (1., 4.), "owner_temperature": (.6, 1.5),
    "effective_boundary_scale": (.5, 3.), "owner_switch": (0., .5),
    "to_dead": (1., 5.), "from_dead": (1., 5.),
}
_DIMENSIONS = SEARCH_BOUNDS


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _record_inputs(records: list[dict]) -> list[dict]:
    return campaign_search._record_inputs(records)


def _optional_disjoint(records: list[dict], selection: dict) -> None:
    pairs = (("train_matches", "validation_matches"),
             ("training_matches", "validation_matches"),
             ("training_match_ids", "validation_match_ids"),
             ("outer_train", "outer_test"))
    values = [selection, selection.get("fold", {}), *records]
    for value in values:
        for train_key, validation_key in pairs:
            if train_key in value and validation_key in value:
                train, validation = set(value[train_key]), set(value[validation_key])
                if train & validation:
                    raise ValueError("training and validation matches overlap")
    for record in records:
        identity = record.get("model_identity")
        if not isinstance(identity, dict):
            continue
        match = record.get("match")
        train = identity.get("train_matches", identity.get("training_matches"))
        validation = identity.get("validation_matches")
        if train is not None and match in set(train):
            raise ValueError("record match appears in model training matches")
        if validation is not None and match not in set(validation):
            raise ValueError("record match is absent from model validation matches")


def _aligned(left: list[dict], right: list[dict]) -> list[tuple[dict, dict]]:
    def index(records):
        result = {}
        for record in records:
            key = (record["match"], record["period"], record["scenario"])
            if key in result:
                raise ValueError("duplicate record in model bundle")
            result[key] = record
        return result

    a, b = index(left), index(right)
    if not a or a.keys() != b.keys():
        raise ValueError("model bundles must have identical nonempty grids")
    pairs = []
    for key in sorted(a):
        x, y = a[key], b[key]
        if not isinstance(x.get("input_sha256"), str) or not x["input_sha256"]:
            raise ValueError("model bundle records require an input hash")
        if x.get("input_sha256") != y.get("input_sha256"):
            raise ValueError("model bundles have different cached inputs")
        aa, ba = x.get("arrays"), y.get("arrays")
        if not isinstance(aa, dict) or not isinstance(ba, dict):
            raise TypeError("tune_joint requires loaded arrays")
        shared = ("frame_id", "gt", "ball_valid", "nearest_ball_seconds")
        shared += tuple(sorted(k for k in aa if k.startswith("ref_")))
        for name in shared:
            if name not in aa or name not in ba or not np.array_equal(aa[name], ba[name], equal_nan=True):
                raise ValueError(f"model bundle public input differs: {name}")
        for name in ("neural_live", "neural_home", "neural_boundary"):
            if name not in aa or name not in ba or aa[name].shape != ba[name].shape:
                raise ValueError(f"model bundle prediction shape differs: {name}")
        pairs.append((x, y))
    return pairs


def _blend_pairs(pairs: list[tuple[dict, dict]], weight: float) -> list[dict]:
    """Match blend_records while sharing immutable cached inputs between trials."""
    output = []
    for left, right in pairs:
        aa, ba = left["arrays"], right["arrays"]
        arrays = aa.copy()
        for name in ("neural_live", "neural_home"):
            p = np.clip(np.asarray(aa[name], dtype=np.float64), 1e-7, 1 - 1e-7)
            q = np.clip(np.asarray(ba[name], dtype=np.float64), 1e-7, 1 - 1e-7)
            logits = weight * (np.log(p) - np.log1p(-p)) + (1 - weight) * (
                np.log(q) - np.log1p(-q))
            blended = 1. / (1. + np.exp(-np.clip(logits, -40, 40)))
            if weight == 1:
                blended = np.asarray(aa[name]).copy()
            elif weight == 0:
                blended = np.asarray(ba[name]).copy()
            arrays[name] = blended
        arrays["neural_boundary"] = (weight * np.asarray(aa["neural_boundary"])
                                     + (1 - weight) * np.asarray(ba["neural_boundary"]))
        identity = {"kind": "logit_blend", "weight_left": float(weight),
                    "left": left.get("model_identity"), "right": right.get("model_identity")}
        output.append({**left, "arrays": arrays,
                       "left_model_identity": left.get("model_identity"),
                       "right_model_identity": right.get("model_identity"),
                       "model_identity": identity})
    return output


def blend_loaded(left: list[dict], right: list[dict], weight: float) -> list[dict]:
    """Blend two loaded bundles with the exact campaign_search semantics."""
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("blend weight must lie in [0,1]")
    return _blend_pairs(_aligned(left, right), float(weight))


def _effective_scale(config: dict) -> float:
    return float(config["boundary_scale"]) / float(config["boundary_temperature"])


def _set_scale(config: dict, effective: float) -> None:
    if effective > 2.:
        config["boundary_scale"] = 2.
        config["boundary_temperature"] = 2. / effective
    else:
        config["boundary_scale"] = effective
        config["boundary_temperature"] = 1.


def _typed_cost(config: dict) -> dict[str, float]:
    old = config["switch_cost"]
    if isinstance(old, dict):
        return {key: float(old.get(key, 0.)) for key in ("owner_switch", "to_dead", "from_dead")}
    return {key: float(old) for key in ("owner_switch", "to_dead", "from_dead")}


def _recipe(baseline: dict, params: dict) -> dict:
    config = copy.deepcopy(baseline)
    for key in ("neural_weight", "visible_neural_weight", "missing_neural_weight",
                "visible_live_weight", "missing_live_weight", "visible_owner_weight",
                "missing_owner_weight"):
        config[key] = float(params["neural_weight"])
    config["live_temperature"] = float(params["live_temperature"])
    config["owner_temperature"] = float(params["owner_temperature"])
    costs = _typed_cost(config)
    costs.update({key: float(params[key]) for key in ("owner_switch", "to_dead", "from_dead")})
    config["switch_cost_type"], config["switch_cost"] = "typed", costs
    _set_scale(config, float(params["effective_boundary_scale"]))
    campaign_decode._validate_config(config)
    return config


def _params_for_anchor(weight: float, decoder: dict) -> dict:
    costs = _typed_cost(decoder)
    return {"left_weight": float(weight), "neural_weight": float(decoder["neural_weight"]),
            "live_temperature": float(decoder["live_temperature"]),
            "owner_temperature": float(decoder["owner_temperature"]),
            "effective_boundary_scale": _effective_scale(decoder), **costs}


def _in_bounds(params: dict) -> bool:
    return all(name in params and low <= float(params[name]) <= high
               for name, (low, high) in _DIMENSIONS.items())


def _history_row(params: dict, value: float) -> dict | None:
    if not _in_bounds(params):
        return None
    distributions = {key: distribution_to_json(FloatDistribution(low, high))
                     for key, (low, high) in _DIMENSIONS.items()}
    return {"params": {key: float(params[key]) for key in _DIMENSIONS},
            "distributions": distributions, "value": float(value)}


def _study(seed: int, proposal_index: int, history: list[dict]) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed + proposal_index, multivariate=True,
                                         group=True, n_startup_trials=4)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for row in history:
        distributions = {key: json_to_distribution(value)
                         for key, value in row["distributions"].items()}
        study.add_trial(optuna.trial.create_trial(params=row["params"],
                                                 distributions=distributions,
                                                 value=float(row["value"])))
    return study


def _source_hashes() -> dict:
    base = Path(__file__).resolve().parent
    paths = [Path(__file__), base / "campaign_search.py", base / "campaign_decode.py",
             base / "campaign_evaluation.py", base / "campaign_protocol.py"]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def tune_joint(left_records: list[dict], right_records: list[dict], baseline_selection: dict,
               output_dir, *, n_trials: int = 64, seed: int = 20261116,
               check: Callable[[], None] | None = None) -> dict:
    """Tune left-model blend weight and decoder against cached public OOF arrays.

    Trial budget includes the frozen baseline, 16 other pair weights, and five
    decoder anchors. The function performs no fitting or target-label access.
    """
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 22:
        raise ValueError("n_trials must be an integer of at least 22")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(baseline_selection, dict) or not isinstance(baseline_selection.get("decoder"), dict):
        raise TypeError("baseline_selection must contain a decoder mapping")
    decoder = _canonical(baseline_selection["decoder"])
    campaign_decode._validate_config(decoder)
    base_weight = baseline_selection.get("blend_weight", .5)
    if isinstance(base_weight, bool) or not np.isfinite(base_weight) or not 0 <= base_weight <= 1:
        raise ValueError("baseline blend_weight must lie in [0,1]")
    _optional_disjoint(left_records + right_records, baseline_selection)
    output_dir = Path(output_dir)
    trial_dir = output_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    if check is not None:
        check()
    left_loaded = campaign_evaluation.load_records(left_records)
    right_loaded = campaign_evaluation.load_records(right_records)
    pairs = _aligned(left_loaded, right_loaded)
    if check is not None:
        check()

    weight_anchors = [round(.30 + .025 * i, 3) for i in range(17) if abs(.30 + .025 * i - .5) > 1e-9]
    anchors = [{"name": "baseline", "params": _params_for_anchor(float(base_weight), decoder),
                "decoder": decoder}]
    anchors.extend({"name": f"pair_weight_{weight:.3f}",
                    "params": _params_for_anchor(weight, decoder), "decoder": decoder}
                   for weight in weight_anchors)
    variants = (("owner_switch_zero", "owner_switch", 0.), ("live_temperature_3", "live_temperature", 3.),
                ("live_temperature_4", "live_temperature", 4.),
                ("effective_boundary_2_5", "effective_boundary_scale", 2.5),
                ("effective_boundary_3", "effective_boundary_scale", 3.))
    # The boundary and temperature anchor names encode canonical recipe values.
    for name, key, value in variants:
        params = _params_for_anchor(float(base_weight), decoder)
        params[key] = value
        recipe = _recipe(decoder, params)
        anchors.append({"name": name, "params": params, "decoder": recipe})
    if len(anchors) > n_trials:
        raise ValueError("anchor count exceeds n_trials")

    protocol = {"schema": _SCHEMA, "records_left": _record_inputs(left_records),
                "records_right": _record_inputs(right_records), "baseline_selection": _canonical(baseline_selection),
                "n_trials": n_trials, "seed": seed, "anchors": anchors,
                "dimensions": _DIMENSIONS, "scenario_weights": campaign_protocol.WEIGHTS,
                "source_sha256": _source_hashes()}
    protocol_sha = _digest(protocol)
    campaign_protocol.publish(output_dir / "plan.json", protocol)
    found = {}
    for path in sorted(trial_dir.glob("trial-*.json")):
        row = campaign_protocol.read_record(path)
        if row is None:
            raise ValueError(f"empty completed trial record: {path}")
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index in found or not 0 <= index < n_trials:
            raise ValueError(f"invalid or duplicate trial index: {path}")
        found[index] = row
    if found and set(found) != set(range(max(found) + 1)):
        raise ValueError("completed joint trial records contain a hole")

    completed, history = [], []
    for index in range(n_trials):
        path = trial_dir / f"trial-{index:04d}.json"
        if index in found:
            row = found[index]
            if row.get("schema") != _SCHEMA or row.get("protocol_sha256") != protocol_sha or row.get("index") != index:
                raise ValueError(f"completed trial protocol differs: {path}")
            if row.get("tpe"):
                history.append(row["tpe"])
            elif row.get("kind") == "anchor":
                hist = _history_row(row["params"], row["value"])
                if hist is not None:
                    history.append(hist)
            if index == 0:
                expected_objective = baseline_selection.get("scores", {}).get("objective")
                if expected_objective is not None and not np.isclose(
                        float(row["value"]), float(expected_objective), rtol=0., atol=1e-12):
                    raise ValueError("joint baseline score does not reproduce frozen selection objective")
            completed.append(row)
            if index == 2:
                _write_timing(output_dir, completed, n_trials)
            continue
        if check is not None:
            check()
        started = time.perf_counter()
        if index < len(anchors):
            anchor = anchors[index]
            params, recipe = anchor["params"], anchor["decoder"]
            kind, anchor_name = "anchor", anchor["name"]
        else:
            study = _study(seed, index, history)
            trial = study.ask()
            params = {name: trial.suggest_float(name, low, high)
                      for name, (low, high) in _DIMENSIONS.items()}
            recipe = _recipe(decoder, params)
            kind, anchor_name = "tpe", None
        blended = _blend_pairs(pairs, float(params["left_weight"]))
        score = campaign_evaluation.score_records(blended, recipe, campaign_protocol.WEIGHTS)
        value = float(score["objective"])
        if index == 0:
            expected_objective = baseline_selection.get("scores", {}).get("objective")
            if expected_objective is not None and not np.isclose(
                    value, float(expected_objective), rtol=0., atol=1e-12):
                raise ValueError("joint baseline score does not reproduce frozen selection objective")
        row = {"schema": _SCHEMA, "index": index, "protocol_sha256": protocol_sha,
               "kind": kind, "anchor": anchor_name, "params": params, "decoder": recipe,
               "blend_weight": float(params["left_weight"]),
               "effective_boundary_scale": _effective_scale(recipe), "score": score,
               "value": value, "elapsed_seconds": time.perf_counter() - started}
        if kind == "tpe":
            frozen = study.tell(trial, value)
            frozen = study.trials[-1]
            row["tpe"] = {"params": frozen.params,
                          "distributions": {key: distribution_to_json(item)
                                            for key, item in frozen.distributions.items()},
                          "value": value}
            history.append(row["tpe"])
        else:
            hist = _history_row(params, value)
            if hist is not None:
                row["tpe"] = hist
                history.append(hist)
        campaign_protocol.publish(path, row)
        completed.append(row)
        if index == 2:
            _write_timing(output_dir, completed, n_trials)

    best = max(completed, key=lambda row: (float(row["value"]), -row["index"]))
    baseline_row = completed[0]
    expected_objective = baseline_selection.get("scores", {}).get("objective")
    if expected_objective is not None and not np.isclose(
            float(baseline_row["value"]), float(expected_objective), rtol=0., atol=1e-12):
        raise ValueError("joint baseline score does not reproduce frozen selection objective")
    result = {**{key: value for key, value in baseline_selection.items()
                 if key not in ("decoder", "blend_weight", "scores")},
            "schema": _SCHEMA, "decoder": best["decoder"],
            "blend_weight": best["blend_weight"], "effective_boundary_scale": best["effective_boundary_scale"],
            "scores": best["score"], "best_trial": best["index"],
            "baseline_score": baseline_row["score"], "trials_complete": len(completed),
            "trial_timings": [row["elapsed_seconds"] for row in completed],
            "source_plan_digest": protocol_sha}
    campaign_protocol.publish(output_dir / "result.json", result)
    _write_timing(output_dir, completed, n_trials)
    return result


def _write_timing(output_dir: Path, completed: list[dict], n_trials: int) -> None:
    timings = [float(row["elapsed_seconds"]) for row in completed]
    observed = timings[:3] if len(timings) >= 3 else timings
    average = float(np.mean(observed)) if observed else None
    write_json(output_dir / "timing.json", {
        "schema": _SCHEMA, "trials_complete": len(completed), "n_trials": n_trials,
        "observed_trials": len(observed), "mean_seconds_per_trial": average,
        "estimated_seconds_remaining": (max(0, n_trials - len(completed)) * average
                                         if average is not None else None),
    })
