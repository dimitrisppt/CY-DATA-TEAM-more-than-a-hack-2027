"""Fixed-protocol timing comparisons with disjoint base, calibration and test matches."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from experiments.campaign_protocol import read_record
from experiments.timing_validation import (
    HISTORICAL_REUSE_CAVEAT,
    build_holdout_plan,
    canonical_sha256,
    freeze_protocol,
    validate_prediction_records,
)
from possession.storage import sha256_file, write_json
from possession.timing import (
    FAMILIES,
    TimingConfig,
    TimingModel,
    associate_targets,
    fit_timing_model,
    make_boundary_batch,
    refine_labels,
    shift_labels,
)

WEIGHTS = {"clean": .50, "short": .20, "long": .15, "replay": .15}
OFFSET_GRID = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
OFFSET_PENALTY = .0005
METHODS = ("identity", "global_offset", "family_offsets", "contextual")
EXPECTED_COMPONENTS = {
    scope: sorted([f"{member}_seed_{seed}" for seed in seeds
                   for member in ("mean_007", "ball_attention_002", "coarse_player_only")]
                  + ["fixed_features_reference_blend"])
    for scope, seeds in (("inner", (8101, 8102)), ("outer", (9101, 9102)))
}
FEATURE_KEYS = ("neural_live", "neural_home", "neural_boundary", "ball_valid",
                "nearest_ball_seconds", "fine_coarse_disagreement")


def _join(parts):
    return np.concatenate([parts[0], np.array([official.X], dtype=np.uint8), parts[1]])


def _row_key(row):
    return row["batch"], row["match"], row["scenario"], row["half"]


def _load(row, *, features=False):
    path = Path(row["path"])
    if sha256_file(path) != row["sha256"]:
        raise ValueError("timing array checksum differs")
    with np.load(path, allow_pickle=False) as stored:
        needed = ["gt", "v6_labels"]
        if features:
            needed += [key for key in FEATURE_KEYS if key in stored]
        arrays = {key: stored[key] for key in needed}
        if features and all(key in stored for key in ("fine_live", "coarse_live", "fine_home", "coarse_home")):
            arrays["fine_coarse_disagreement"] = (
                np.abs(stored["fine_live"] - stored["coarse_live"])
                + np.abs(stored["fine_home"] - stored["coarse_home"])) / 2.
    arrays["labels"] = arrays.pop("v6_labels")
    if arrays["labels"].shape != arrays["gt"].shape:
        raise ValueError("timing label timelines differ")
    return arrays


def _grid(rows):
    actual = {_row_key(row) for row in rows}
    expected = {(batch, match, scenario, half)
                for batch in {row["batch"] for row in rows}
                for match in {row["match"] for row in rows}
                for scenario in WEIGHTS for half in (1, 2)}
    if not rows or len(actual) != len(rows) or actual != expected:
        raise ValueError("timing rows must cover every batch, match, scenario and half")


def _score(labels_by_key, truth_by_key, *, full=True):
    if set(labels_by_key) != set(truth_by_key):
        raise ValueError("prediction and truth record sets differ")
    grouped = sorted({key[:3] for key in truth_by_key})
    records = []
    for batch, match, scenario in grouped:
        keys = [(batch, match, scenario, half) for half in (1, 2)]
        truth = _join([truth_by_key[key] for key in keys])
        labels = _join([labels_by_key[key] for key in keys])
        if full:
            metrics = official.evaluate(truth, labels, bootstrap_n=0)
            metrics = {name: metrics[name] for name in (
                "spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells",
                "fragmentation", "transitions", "spell_f1", "n_graded")}
            metrics["transitions"] = {
                tolerance: {key: value for key, value in scores.items()
                            if key not in ("pairs", "missed", "spurious")}
                for tolerance, scores in metrics["transitions"].items()}
        else:
            runs = official.valid_runs(truth)
            filled = official.fill_abstentions(labels, runs)
            metrics = {"spell_f1_mean": official.spell_scores(truth, filled, runs)["mean_f1"]}
        records.append({"batch": batch, "match": match, "scenario": scenario, **metrics})
    means = {scenario: float(np.mean([r["spell_f1_mean"] for r in records
                                     if r["scenario"] == scenario])) for scenario in WEIGHTS}
    return {"objective": float(sum(WEIGHTS[name] * means[name] for name in WEIGHTS)),
            "missing_ball": float(sum(WEIGHTS[name] * means[name]
                                      for name in ("short", "long", "replay")) / .5),
            "means": means, "records": records}


def fit_offsets(rows):
    """Fit simple baselines solely on calibration matches, with shrinkage toward zero."""
    truth, base = {}, {}
    for row in rows:
        arrays = _load(row)
        truth[_row_key(row)], base[_row_key(row)] = arrays["gt"], arrays["labels"]
    cache = {}

    def objective(offsets):
        key = tuple(int(v) for v in offsets)
        if key not in cache:
            pred = {k: shift_labels(labels, dict(zip(FAMILIES, key)), radius=25)
                    for k, labels in base.items()}
            score = _score(pred, truth, full=False)["objective"]
            cache[key] = {"objective": score,
                          "penalized": score - OFFSET_PENALTY * float(np.mean(np.abs(key)))}
        return cache[key]["penalized"], -sum(abs(x) for x in key), tuple(-x for x in key)

    global_offset = max(OFFSET_GRID, key=lambda x: objective((x, x, x)))
    family = (global_offset,) * 3
    for _ in range(2):
        previous = family
        for group in range(3):
            choices = [tuple(value if i == group else family[i] for i in range(3))
                       for value in OFFSET_GRID]
            family = max(choices, key=objective)
        if family == previous:
            break
    return {"global": int(global_offset), "family": list(family),
            "evaluations": [{"offsets": list(key), **value} for key, value in cache.items()],
            "penalty": OFFSET_PENALTY, "grid": list(OFFSET_GRID)}


def _audit(rows, assessment, universe, components):
    _grid(rows)
    audits = []
    for batch in sorted({r["batch"] for r in rows}):
        selected = [r for r in rows if r["batch"] == batch]
        for row in selected:
            prov = row["provenance"]
            if prov["feature_sha256"] != row["sha256"]:
                raise ValueError("array checksum is not bound to its training ancestry")
            if prov["split_id"] != f"{row['scope']}_fold_{row['fold']}":
                raise ValueError("array fold is not bound to its training ancestry")
            if (prov["match"], prov["half"], prov["scenario"]) != (
                    row["match"], row["half"], row["scenario"]):
                raise ValueError("array record and ancestry sample identity differ")
        for scope in sorted({r["scope"] for r in selected}):
            scoped = [r for r in selected if r["scope"] == scope]
            audits.extend(validate_prediction_records(
                [r["provenance"] for r in scoped], outer_assessment_matches=assessment,
                expected_matches=sorted({r["match"] for r in scoped}),
                allowed_matches=universe, expected_components=components[scope]))
    return audits


def _make_splits(records):
    if {row["scope"] for row in records} != {"inner", "outer"}:
        raise ValueError("timing records have an unrecognized or absent scope")
    inner = [row for row in records if row["scope"] == "inner"]
    universe = sorted({row["match"] for row in records})
    pairs = {}
    for row in inner:
        pairs.setdefault(row["fold"], set()).add(row["match"])
    if len(pairs) != 3 or any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("expected three existing base-model folds, each excluding two matches")
    if sum(len(pair) for pair in pairs.values()) != len(set.union(*pairs.values())):
        raise ValueError("assessment pairs overlap")
    splits = []
    inner_matches = sorted({row["match"] for row in inner})
    for fold, pair in sorted(pairs.items()):
        for assess in sorted(pair):
            calibrate = sorted(pair - {assess})
            base = sorted(set(inner_matches) - pair)
            plan = build_holdout_plan(base, calibrate, [assess],
                                      historical_selection_matches=universe)
            splits.append({"id": f"paired_{fold}_{assess}", "kind": "paired_holdout",
                           "base_train": base, "calibration": calibrate,
                           "assessment": [assess], "plan": plan})
    guard = sorted({row["match"] for row in records if row["scope"] == "outer"})
    if len(guard) != 1 or set(guard) & set(inner_matches):
        raise ValueError("expected one separate historical guard match")
    # Calibration features here use several OOF base fits. They all exclude the
    # guard and their own sample, but their training matches overlap calibration.
    # A three-way holdout plan would therefore misdescribe this separate design.
    splits.append({"id": "historical_guard", "kind": "oof_training_guard",
                   "base_train": inner_matches, "calibration": inner_matches,
                   "assessment": guard, "plan": None})
    return splits


def _prediction_summary(batch, target, prediction):
    matched = np.asarray(target.matched, dtype=bool)
    if not np.any(matched):
        return {"boundaries": len(matched), "matched": 0}
    probs = prediction.probabilities
    offsets = np.arange(probs.shape[1]) - probs.shape[1] // 2
    chosen = probs.argmax(axis=1)
    targets = target.offsets
    actual = targets + probs.shape[1] // 2
    indices = np.flatnonzero(matched)
    nll = -np.log(np.maximum(probs[indices, actual[indices]], 1e-15))
    cdf = probs.cumsum(axis=1)
    low = offsets[(cdf >= .05).argmax(axis=1)]
    high = offsets[(cdf >= .95).argmax(axis=1)]
    return {"boundaries": len(matched), "matched": int(matched.sum()),
            "ambiguous": int(np.sum(target.ambiguous)),
            "matched_nll": float(np.mean(nll)),
            "matched_mode_error_frames": float(np.mean(np.abs(offsets[chosen[matched]]
                                                                  - targets[matched]))),
            "matched_90pct_coverage": float(np.mean((targets[matched] >= low[matched])
                                                    & (targets[matched] <= high[matched]))),
            "matched_90pct_width_frames": float(np.mean(high[matched] - low[matched]))}


def _run_split(split, records, output, config, protocol_sha, universe, components, status):
    train = split.get("training_records")
    if train is None:
        train = [row for row in records if row["match"] in split["calibration"]
                 and row["scope"] == "inner"]
    assess = split.get("assessment_records")
    if assess is None:
        assess = [row for row in records if row["match"] in split["assessment"]]
    if {row["match"] for row in train} != set(split["calibration"]):
        raise ValueError("calibration rows differ from declared training matches")
    if {row["match"] for row in assess} != set(split["assessment"]):
        raise ValueError("assessment rows differ from declared assessment matches")
    training_audit = _audit(train, split["assessment"], universe, components)
    assessment_audit = _audit(assess, split["assessment"], universe, components)
    if set(split["calibration"]) & set(split["assessment"]):
        raise ValueError("calibrator training overlaps assessment matches")
    directory = output / "splits" / split["id"]
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status("fitting_offsets", split=split["id"])
    offsets = fit_offsets(train)
    status("building_boundary_training", split=split["id"])
    batches, targets, matches, weights = [], [], [], []
    for row in train:
        arrays = _load(row, features=True)
        features = {k: arrays[k] for k in FEATURE_KEYS if k in arrays}
        batch = make_boundary_batch(arrays["labels"], features, radius=config.radius)
        target = associate_targets(batch, arrays["gt"])
        batches.append(batch)
        targets.append(target)
        matches.append(row["match"])
        weights.append(np.full(len(batch.boundaries), WEIGHTS[row["scenario"]], dtype=float))
    # Allocate each match/scenario its declared mass, regardless of how many
    # boundaries or repeated outage batches it contains.
    for match in set(matches):
        for scenario in WEIGHTS:
            indices = [i for i, row in enumerate(train)
                       if row["match"] == match and row["scenario"] == scenario]
            count = sum(len(batches[i].boundaries) for i in indices)
            if count:
                for i in indices:
                    weights[i] /= count
    status("fitting_contextual", split=split["id"])
    model = fit_timing_model(batches, targets, match_ids=matches, weights=weights, config=config)
    model_path = directory / "model.npz"
    model.save(model_path)
    model = TimingModel.load(model_path)
    del batches, targets, arrays, features, batch, target
    gc.collect()
    status("assessing", split=split["id"])
    truth = {}
    predictions = {name: {} for name in METHODS}
    diagnostics = []
    for row in assess:
        arrays = _load(row, features=True)
        labels = arrays["labels"]
        features = {k: arrays[k] for k in FEATURE_KEYS if k in arrays}
        batch = make_boundary_batch(labels, features, radius=config.radius)
        predicted = model.predict(batch)
        outputs = {
            "identity": labels.copy(),
            "global_offset": shift_labels(labels, offsets["global"], radius=config.radius),
            "family_offsets": shift_labels(labels, dict(zip(FAMILIES, offsets["family"])),
                                            radius=config.radius),
            "contextual": refine_labels(labels, batch, predicted, config=config),
        }
        for name, value in outputs.items():
            if value.shape != labels.shape or not np.array_equal(value == official.X,
                                                                  labels == official.X):
                raise ValueError("calibrator changed timeline or excluded-frame mask")
            predictions[name][_row_key(row)] = value
        truth[_row_key(row)] = arrays["gt"]
        # Assessment truth enters diagnostics only after every prediction exists.
        target = associate_targets(batch, arrays["gt"])
        diagnostics.append({"key": list(_row_key(row)),
                            **_prediction_summary(batch, target, predicted)})
    scores = {name: _score(predictions[name], truth) for name in METHODS}
    prediction_path = directory / "predictions.npz"
    saved = {f"{name}__{'__'.join(map(str, key))}": value
             for name, rows in predictions.items() for key, value in rows.items()}
    np.savez_compressed(prediction_path, **saved)
    split_description = {key: value for key, value in split.items()
                         if key not in ("training_records", "assessment_records")}
    result = {"split": split_description, "protocol_sha256": protocol_sha,
              "training_audit": training_audit, "assessment_audit": assessment_audit,
              "offset_fit": offsets, "scores": scores, "diagnostics": diagnostics,
              "fit_diagnostics": model.fit_diagnostics,
              "model_sha256": sha256_file(model_path),
              "predictions_sha256": sha256_file(prediction_path),
              "elapsed_seconds": time.monotonic() - started}
    write_json(directory / "result.json", result)
    status("split_complete", split=split["id"],
           objective={name: score["objective"] for name, score in scores.items()})
    return result


def summarize(results):
    paired = [row for row in results if row["split"]["kind"] == "paired_holdout"]
    guards = [row for row in results if row["split"]["kind"] == "oof_training_guard"]
    if (len(results) != 7 or len(paired) != 6 or len(guards) != 1
            or any(len(row["split"]["assessment"]) != 1 for row in results)
            or len({row["split"]["assessment"][0] for row in results}) != 7):
        raise ValueError("summary requires six unique assessment matches and one distinct guard")
    methods = {}
    for method in METHODS:
        values = [row["scores"][method] for row in paired]
        means = {name: float(np.mean([score["means"][name] for score in values]))
                 for name in WEIGHTS}
        deltas = {row["split"]["assessment"][0]:
                  row["scores"][method]["objective"] - row["scores"]["identity"]["objective"]
                  for row in paired}
        delta_values = np.array(list(deltas.values()))
        rng = np.random.default_rng(20261116)
        draws = rng.choice(delta_values, size=(10000, len(delta_values))).mean(axis=1)
        methods[method] = {
            "objective": float(np.mean([score["objective"] for score in values])),
            "missing_ball": float(np.mean([score["missing_ball"] for score in values])),
            "means": means, "by_match_objective_delta": deltas,
            "objective_delta": float(delta_values.mean()),
            "matches_improved": int(np.sum(delta_values > 1e-12)),
            "descriptive_match_bootstrap_95pct": np.quantile(draws, [.025, .975]).tolist(),
        }
    contextual = methods["contextual"]
    baseline = methods["identity"]
    guard = guards[0]["scores"] if guards else None
    gates = {
        "paired_objective_gain_gt_0_002": contextual["objective_delta"] > .002,
        "contextual_beats_simple_offsets": contextual["objective"] > max(
            methods[name]["objective"] for name in ("global_offset", "family_offsets")),
        "four_of_six_matches_improved": contextual["matches_improved"] >= 4,
        "missing_ball_improved": contextual["missing_ball"] > baseline["missing_ball"],
        "no_scenario_loss_over_0_01": all(contextual["means"][name] - baseline["means"][name]
                                         >= -.01 for name in WEIGHTS),
        "guard_objective_loss_at_most_0_01": guard is not None and
        guard["contextual"]["objective"] - guard["identity"]["objective"] >= -.01,
    }
    return {"paired_holdout": methods, "historical_guard": guard,
            "gates": gates, "advance_to_full_nested_study": all(gates.values()),
            "promotion_authorized_by_this_study": False,
            "interpretation": (
                "The paired comparison fits the calibrator on one excluded match and assesses "
                "the other excluded match, in both directions for each base fold. It is not the "
                "full nested study with four calibration-training matches. The guard fits on "
                "six OOF development matches. All uncertainty summaries are descriptive because "
                "of historical reuse and shared base fits. " + HISTORICAL_REUSE_CAVEAT)}


def run_study(cache_dir, output_dir):
    cache_dir, output = Path(cache_dir).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest = read_record(manifest_path)
    if manifest is None or manifest.get("schema") != "public_dfl_timing_cache_v1":
        raise ValueError("missing or invalid timing cache manifest")
    records = manifest["records"]
    for row in records:
        path = Path(row["path"])
        if not path.is_absolute():
            row["path"] = str(cache_dir / path)
    _grid(records)
    config = TimingConfig()
    splits = _make_splits(records)
    universe = sorted({row["match"] for row in records})
    components = EXPECTED_COMPONENTS
    root = Path(__file__).resolve().parents[1]
    source_files = ["possession/timing.py", "experiments/timing_validation.py",
                    "experiments/timing_cache.py", "experiments/timing_study.py",
                    "evaluation/possession_metrics.py"]
    protocol = {"schema": "timing_paired_study_v1", "cache_manifest_sha256": sha256_file(manifest_path),
                "sources": {name: sha256_file(root / name) for name in source_files},
                "config": asdict(config), "offset_grid": list(OFFSET_GRID),
                "offset_penalty": OFFSET_PENALTY, "family_search_passes": 2,
                "scenario_weights": WEIGHTS, "splits": splits, "components": components,
                "selection": "fixed configurations, baseline parameters fitted on calibration only",
                "target_labels_used": False, "historical_reuse_caveat": HISTORICAL_REUSE_CAVEAT}
    freeze_protocol(output / "protocol.json", protocol)
    protocol_sha = canonical_sha256(protocol)

    def status(phase, **kwargs):
        item = {"phase": phase, "protocol_sha256": protocol_sha, **kwargs}
        write_json(output / "status.json", item)
        print(json.dumps(item), flush=True)

    result_path = output / "result.json"
    if result_path.exists():
        result = read_record(result_path)
        if result is None:
            raise ValueError("completed timing result is invalid")
        if result["protocol_sha256"] != protocol_sha:
            raise ValueError("completed study protocol differs")
        for split_result in result["splits"]:
            directory = output / "splits" / split_result["split"]["id"]
            for filename, checksum in (("model.npz", split_result["model_sha256"]),
                                       ("predictions.npz", split_result["predictions_sha256"])):
                if sha256_file(directory / filename) != checksum:
                    raise ValueError("completed timing artifact checksum differs")
        if summarize(result["splits"]) != result["summary"]:
            raise ValueError("completed timing summary differs from split scores")
        status("already_complete")
        return result
    results = []
    for split in splits:
        results.append(_run_split(split, records, output, config, protocol_sha, universe,
                                  components, status))
        gc.collect()
    result = {"schema": "timing_paired_result_v1", "protocol_sha256": protocol_sha,
              "summary": summarize(results), "splits": results}
    from experiments.campaign_protocol import publish

    publish(result_path, result)
    status("complete", objective={name: row["objective"] for name, row in
                                   result["summary"]["paired_holdout"].items()},
           gates=result["summary"]["gates"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=4):
        run_study(args.cache_dir, args.output_dir)


if __name__ == "__main__":
    main()
