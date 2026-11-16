"""Bounded decoder and blend selection from saved development OOF predictions."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from experiments.augmentation_evaluation import _compact
from experiments.evaluate import _join_with_x
from possession.calibrate import CalibrationConfig, apply_temperature_calibration
from possession.decode import decode_half, emissions_from_probabilities
from possession.ensemble import combine_scores
from possession.quality_blend import QualityBlendConfig, combine_quality_scores
from possession.storage import sha256_file, write_json

SCENARIOS = ("clean", "block41_seed3101", "block41_seed3102")
LONG_SCENARIO = "long41_seed4101"
PROTOCOL = "development_oof_decoder_selection_v1"
SOURCE_FILES = (
    "evaluation/possession_metrics.py", "experiments/oof_selection.py",
    "experiments/prediction_store.py", "experiments/augmentation_evaluation.py",
    "experiments/evaluate.py", "possession/calibrate.py", "possession/decode.py",
    "possession/ensemble.py", "possession/quality_blend.py", "possession/_viterbi.py",
)


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _record_key(record: dict) -> tuple[int, str, str, int]:
    return record["seed"], record["match"], record["scenario"], record["period"]


def _checked_records(records: list[dict], *, selection: bool = False) -> tuple[dict, dict]:
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list of OOF half records")
    indexed = {}
    protocols = set()
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("each OOF record must be a mapping")
        seed, match, scenario, period = _record_key(record)
        if isinstance(seed, bool) or not isinstance(seed, int) or not isinstance(match, str) or not match:
            raise ValueError("invalid OOF seed or match")
        if scenario not in (*SCENARIOS, LONG_SCENARIO) or period not in (1, 2):
            raise ValueError("OOF records permit clean, two short-gap, and long-gap scenarios")
        prediction = record["prediction"]
        if not isinstance(prediction, dict):
            raise TypeError("prediction must be a bundle record")
        if any(prediction.get(name) != record[name] for name in ("match", "period", "scenario")):
            raise ValueError("OOF wrapper and prediction bundle identity differ")
        if (not isinstance(prediction.get("sha256"), str)
                or len(prediction["sha256"]) != 64
                or not isinstance(prediction.get("path"), str)
                or not prediction["path"]):
            raise ValueError("prediction bundle needs path and SHA256")
        if not isinstance(prediction.get("schema_version"), int):
            raise TypeError("prediction bundle schema_version missing")
        training = record["training_matches"]
        validation = record["validation_matches"]
        if (not isinstance(training, list) or not isinstance(validation, list)
                or any(not isinstance(name, str) or not name for name in (*training, *validation))
                or len(training) != len(set(training))
                or len(validation) != len(set(validation))
                or match not in validation or set(training) & set(validation)):
            raise ValueError("OOF training or validation match proof differs")
        if not isinstance(record.get("checkpoint_sha256"), str) or len(record["checkpoint_sha256"]) != 64:
            raise ValueError("checkpoint_sha256 missing")
        protocol_sha = record.get("protocol_sha256")
        if not isinstance(protocol_sha, str) or len(protocol_sha) != 64:
            raise ValueError("protocol_sha256 missing")
        protocols.add(protocol_sha)
        key = _record_key(record)
        if key in indexed:
            raise ValueError(f"duplicate OOF period: {key}")
        indexed[key] = record
    if len(protocols) != 1:
        raise ValueError("OOF records must share one protocol")
    seeds = sorted({key[0] for key in indexed})
    matches = sorted({key[1] for key in indexed})
    scenarios = tuple(name for name in (*SCENARIOS, LONG_SCENARIO)
                      if any(key[2] == name for key in indexed))
    if (len(matches) != 6 or len(seeds) not in ((3,) if selection else (1, 3))):
        raise ValueError("OOF grid requires six matches and one or three paired seeds")
    if (scenarios not in (SCENARIOS, (*SCENARIOS, LONG_SCENARIO))
            or (selection and scenarios != SCENARIOS)):
        raise ValueError("OOF selection permits only clean and two short-gap scenarios")
    expected = {(seed, match, scenario, period) for seed in seeds for match in matches
                for scenario in scenarios for period in (1, 2)}
    if set(indexed) != expected:
        raise ValueError("OOF seed/match/scenario grid is not rectangular or has missing periods")
    fold_groups = {}
    for seed in seeds:
        groups = set()
        for match in matches:
            related = [indexed[(seed, match, scenario, period)] for scenario in scenarios
                       for period in (1, 2)]
            group = tuple(sorted(related[0]["validation_matches"]))
            training = tuple(sorted(related[0]["training_matches"]))
            if (len(group) != 2 or not set(group) <= set(matches) or len(training) != 4
                    or set(training) != set(matches) - set(group)
                    or any(tuple(sorted(item["validation_matches"])) != group
                           or tuple(sorted(item["training_matches"])) != training
                           for item in related)):
                raise ValueError("unexpected match training or validation fold proof")
            if any(item["checkpoint_sha256"] != related[0]["checkpoint_sha256"]
                   for item in related):
                raise ValueError("checkpoint differs across scenarios or periods")
            groups.add(group)
            if any(tuple(sorted(indexed[(seed, held_out, scenarios[0], 1)][
                "validation_matches"])) != group for held_out in group):
                raise ValueError("validation fold groups do not pair held-out matches")
            if any(indexed[(seed, held_out, scenario, period)]["checkpoint_sha256"]
                   != related[0]["checkpoint_sha256"]
                   for held_out in group for scenario in scenarios for period in (1, 2)):
                raise ValueError("checkpoint differs across validation fold matches")
            previous = fold_groups.setdefault(match, group)
            if previous != group:
                raise ValueError("validation folds differ across seeds")
        if len(groups) != 3 or set().union(*(set(group) for group in groups)) != set(matches):
            raise ValueError("validation fold groups must partition six matches")
    identities = [{"seed": seed, "match": match, "scenario": scenario, "period": period,
                   "prediction": indexed[(seed, match, scenario, period)]["prediction"],
                   "checkpoint_sha256": indexed[(seed, match, scenario, period)]["checkpoint_sha256"],
                   "training_matches": sorted(indexed[(seed, match, scenario, period)]["training_matches"]),
                   "validation_matches": sorted(
                       indexed[(seed, match, scenario, period)]["validation_matches"])}
                  for seed, match, scenario, period in sorted(indexed)]
    return indexed, {"seeds": seeds, "matches": matches, "scenarios": list(scenarios),
                     "source_protocol_sha256": protocols.pop(), "records": identities}


def _bundle_arrays(record: dict) -> dict:
    from experiments.prediction_store import load_prediction_bundle

    arrays = load_prediction_bundle(record["prediction"])
    expected = ("neural_live", "neural_home", "reference_live", "reference_home",
                "gt", "ball_valid", "nearest_ball_seconds", "frame_id")
    if not isinstance(arrays, dict) or any(name not in arrays for name in expected):
        raise ValueError("prediction bundle arrays are incomplete")
    n = len(arrays["gt"])
    if any(np.asarray(arrays[name]).shape != (n,) for name in expected):
        raise ValueError("prediction bundle arrays must be aligned vectors")
    frame_id = np.asarray(arrays["frame_id"])
    if (n == 0 or frame_id.dtype.kind not in "iu"
            or not np.all(np.diff(frame_id) == 1)):
        raise ValueError("prediction bundle frame_id must be nonempty and contiguous")
    return arrays


def decode_oof_half(arrays: dict, frozen: dict) -> np.ndarray:
    """Replay a frozen decoder on one stored half, without reading labels."""
    neural = {"live_probability": arrays["neural_live"],
              "home_probability": arrays["neural_home"],
              "boundary": np.zeros((len(arrays["neural_live"]), 3, 3), dtype=np.float64)}
    reference = {"live_probability": arrays["reference_live"],
                 "home_probability": arrays["reference_home"],
                 "boundary": np.zeros((len(arrays["reference_live"]), 3, 3), dtype=np.float64)}
    ensemble = frozen["ensemble"]
    if ensemble.get("mode", "static") == "quality":
        blended = combine_quality_scores(
            neural, reference, arrays["ball_valid"], arrays["nearest_ball_seconds"],
            QualityBlendConfig.from_dict({key: ensemble[key] for key in (
                "visible_neural_weight", "missing_neural_weight", "transition_seconds")}),
        )
    else:
        blended = combine_scores(neural, reference, ensemble["neural_weight"])
    live, home, _ = apply_temperature_calibration(
        blended["live_probability"], blended["home_probability"], None,
        CalibrationConfig(**frozen["calibration"]),
    )
    if frozen["decoder"]["boundary_scale"] != 0:
        raise ValueError("OOF decoder must fix boundary_scale at zero")
    return decode_half(emissions_from_probabilities(live, home),
                       switch_cost=frozen["decoder"]["switch_cost"])


def score_oof(records: list[dict], frozen: dict) -> dict:
    """Score paired halves by seed, match, and scenario with official metrics."""
    indexed, grid = _checked_records(records)
    by_seed = {}
    objective_values = []
    timelines = {}
    for seed in grid["seeds"]:
        by_match = {}
        for match in grid["matches"]:
            by_scenario = {}
            for scenario in grid["scenarios"]:
                gt_parts, pred_parts = [], []
                for period in (1, 2):
                    arrays = _bundle_arrays(indexed[(seed, match, scenario, period)])
                    timeline = hashlib.sha256(arrays["frame_id"].tobytes()
                                              + arrays["gt"].tobytes()).hexdigest()
                    key = match, period
                    if key in timelines and timelines[key] != timeline:
                        raise ValueError("OOF labels or frame IDs differ across paired records")
                    timelines[key] = timeline
                    gt_parts.append(arrays["gt"])
                    pred_parts.append(decode_oof_half(arrays, frozen))
                metrics = official.evaluate(_join_with_x(gt_parts), _join_with_x(pred_parts),
                                            bootstrap_n=0)
                by_scenario[scenario] = _compact(metrics)
            short = float(np.mean([by_scenario[name]["spell_f1_mean"] for name in SCENARIOS[1:]]))
            objective_values.append(.5 * by_scenario["clean"]["spell_f1_mean"] + .5 * short)
            by_match[match] = by_scenario
        by_seed[str(seed)] = {"by_match": by_match}
    means = {scenario: float(np.mean([
        by_seed[str(seed)]["by_match"][match][scenario]["spell_f1_mean"]
        for seed in grid["seeds"] for match in grid["matches"]])) for scenario in grid["scenarios"]}
    means["short_mean"] = float(np.mean([means[name] for name in SCENARIOS[1:]]))
    means["objective"] = float(np.mean(objective_values))
    means["transition_f1_at_12"] = {scenario: float(np.mean([
        by_seed[str(seed)]["by_match"][match][scenario]["transitions"]["12"]["f1"]
        for seed in grid["seeds"] for match in grid["matches"]])) for scenario in grid["scenarios"]}
    return {"objective": means["objective"], "means": means, "by_seed": by_seed}


def _frozen(baseline: dict, mode: str, params: dict) -> dict:
    frozen = copy.deepcopy(baseline)
    if mode == "quality":
        frozen["ensemble"] = {"mode": "quality",
                              "visible_neural_weight": params["visible_neural_weight"],
                              "missing_neural_weight": params["missing_neural_weight"],
                              "transition_seconds": 2.0}
    else:
        frozen["ensemble"] = {"neural_weight": params["neural_weight"]}
    frozen["calibration"] = {**frozen["calibration"],
                             "live_temperature": params["live_temperature"],
                             "owner_temperature": params["owner_temperature"]}
    frozen["decoder"] = {**frozen["decoder"], "switch_cost": params["switch_cost"],
                         "boundary_scale": 0.0}
    return frozen


@contextmanager
def _single_writer(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "oof_selection.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("OOF selection already has a writer") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def select_oof_decoder(
    records: list[dict], *, mode: str = "static", seed: int = 0, n_trials: int = 16,
    baseline_frozen: dict, output_dir,
) -> dict:
    """Select one frozen decoder from 16 or fewer development OOF trials."""
    if mode not in ("static", "quality"):
        raise ValueError("mode must be static or quality")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or not 4 <= n_trials <= 16:
        raise ValueError("n_trials must be between 4 and 16")
    if not isinstance(baseline_frozen, dict) or any(
        name not in baseline_frozen for name in ("ensemble", "calibration", "decoder")
    ):
        raise ValueError("baseline_frozen needs ensemble, calibration, and decoder")
    indexed, grid = _checked_records(records, selection=True)
    baseline_weight = baseline_frozen["ensemble"]["neural_weight"]
    base = {"live_temperature": baseline_frozen["calibration"]["live_temperature"],
            "owner_temperature": baseline_frozen["calibration"]["owner_temperature"],
            "switch_cost": baseline_frozen["decoder"]["switch_cost"]}
    if (not 0 <= baseline_weight <= 1 or not 0 <= base["switch_cost"] <= 5
            or any(not .5 <= base[name] <= 2 for name in ("live_temperature", "owner_temperature"))):
        raise ValueError("baseline_frozen lies outside decoder search space")
    for prediction in (record["prediction"] for record in indexed.values()):
        # The store also validates the hash when loading. Verify all inputs now,
        # before any trial or result is written.
        if sha256_file(prediction["path"]) != prediction["sha256"]:
            raise ValueError("prediction bundle hash differs")
    # An OOF condition changes input observations, never the source labels or
    # timeline. Compare one half at a time without retaining the full grid.
    for match in grid["matches"]:
        for period in (1, 2):
            expected = None
            for oof_seed in grid["seeds"]:
                for scenario in SCENARIOS:
                    arrays = _bundle_arrays(indexed[(oof_seed, match, scenario, period)])
                    pair = (arrays["frame_id"], arrays["gt"])
                    if expected is None:
                        expected = tuple(value.copy() for value in pair)
                    elif any(not np.array_equal(left, right)
                             for left, right in zip(expected, pair, strict=True)):
                        raise ValueError("OOF labels or frame IDs differ across paired records")
    root = Path(__file__).resolve().parents[1]
    source = {name: sha256_file(root / name) for name in SOURCE_FILES}
    payload = {"protocol": PROTOCOL, "mode": mode, "seed": seed, "n_trials": n_trials,
               "grid": grid, "baseline_frozen": baseline_frozen, "source_sha256": source,
               "search_space": {"neural_weights": [0, 1], "switch_cost": [0, 5],
                                "temperatures": [0.5, 2], "transition_seconds": 2.0}}
    protocol = {"sha256": _digest(payload), "payload": payload}
    output_dir = Path(output_dir)
    with _single_writer(output_dir):
        protocol_path = output_dir / "protocol.json"
        if protocol_path.exists():
            if json.loads(protocol_path.read_text()) != protocol:
                raise ValueError("OOF selection protocol changed; refusing resume")
        else:
            write_json(protocol_path, protocol)
        from experiments.tune import _optuna

        optuna = _optuna()
        study = optuna.create_study(
            study_name=f"oof-{mode}", storage=f"sqlite:///{output_dir / 'study.sqlite3'}",
            load_if_exists=True, direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=4),
        )
        previous = study.user_attrs.get("protocol_sha256")
        if previous is not None and previous != protocol["sha256"]:
            raise ValueError("OOF study protocol changed")
        study.set_user_attr("protocol_sha256", protocol["sha256"])
        if any(trial.state == optuna.trial.TrialState.RUNNING for trial in study.trials):
            raise RuntimeError("interrupted RUNNING OOF trial requires recovery")
        completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
        if any(trial.state not in (optuna.trial.TrialState.COMPLETE,
                                   optuna.trial.TrialState.WAITING) for trial in study.trials):
            raise RuntimeError("failed OOF trial requires recovery")
        if completed > n_trials:
            raise ValueError("study has more trials than requested")
        if not study.trials:
            if mode == "static":
                anchors = [{**base, "neural_weight": weight}
                           for weight in (baseline_weight, 0.0, 1.0, 0.5)]
            else:
                anchors = [{**base, "visible_neural_weight": visible,
                            "missing_neural_weight": missing}
                           for visible, missing in ((baseline_weight, baseline_weight),
                                                    (0.0, 0.0), (1.0, 1.0), (0.25, 0.75))]
            for anchor in anchors:
                study.enqueue_trial(anchor)

        def suggest(trial):
            params = {"live_temperature": trial.suggest_float("live_temperature", .5, 2, log=True),
                      "owner_temperature": trial.suggest_float("owner_temperature", .5, 2, log=True),
                      "switch_cost": trial.suggest_float("switch_cost", 0, 5)}
            if mode == "static":
                params["neural_weight"] = trial.suggest_float("neural_weight", 0, 1)
            else:
                params["visible_neural_weight"] = trial.suggest_float(
                    "visible_neural_weight", 0, 1)
                params["missing_neural_weight"] = trial.suggest_float(
                    "missing_neural_weight", 0, 1)
            return params

        while completed < n_trials:
            study = optuna.load_study(
                study_name=f"oof-{mode}", storage=f"sqlite:///{output_dir / 'study.sqlite3'}",
                sampler=optuna.samplers.TPESampler(
                    seed=seed + completed, multivariate=True, n_startup_trials=4),
            )
            trial = study.ask()
            params = suggest(trial)
            frozen = _frozen(baseline_frozen, mode, params)
            scores = score_oof(records, frozen)
            trial.set_user_attr("scores", scores)
            study.tell(trial, scores["objective"])
            completed += 1
        trials = [{"number": trial.number, "value": trial.value, "params": trial.params,
                   "scores": trial.user_attrs["scores"]}
                  for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        winner = max(trials, key=lambda trial: (trial["value"], -trial["number"]))
        result = {"protocol_sha256": protocol["sha256"], "mode": mode,
                  "frozen": _frozen(baseline_frozen, mode, winner["params"]),
                  "best_trial": winner["number"], "best_scores": winner["scores"],
                  "trials": trials}
        result_path = output_dir / "selection.json"
        if result_path.exists() and json.loads(result_path.read_text()) != result:
            raise ValueError("existing OOF selection differs from persistent study")
        write_json(result_path, result)
        return result
