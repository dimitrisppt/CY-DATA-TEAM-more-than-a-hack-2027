"""Verify and score the portable public DFL temporal-calibration comparison.

Run ``python -m scripts.evaluate_timing_reference`` to score the committed labels.
Use ``--export-from STUDY_DIR`` once to export a completed signed-score study.
Only decoded labels and public ground truth enter the portable artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "public_dfl_timing_reference_v1"
METHODS = ("identity", "global_offset", "family_offsets", "contextual")
CODES = {"H": official.H, "A": official.A, "D": official.D, "X": official.X}
METRICS = ("spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells", "fragmentation", "n_graded")
CAVEAT = (
    "Development comparison with six paired assessment matches and one separate historically "
    "reused guard. Shared upstream fits and historical recipe selection preclude an independent "
    "test claim. Hashes verify consistency of the declared ancestry, not the truthfulness of "
    "undeclared upstream training."
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _array_hash(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text())


def _read_wrapped(path):
    value = _read_json(path)
    if set(value) != {"sha256", "payload"} or value["sha256"] != _digest(value["payload"]):
        raise ValueError("source record hash differs")
    return value["payload"], value["sha256"]


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("invalid portable identifier")
    return value


def _hash_value(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid SHA-256")


def _portable(value):
    if isinstance(value, dict):
        for key, item in value.items():
            _portable(key)
            _portable(item)
    elif isinstance(value, list):
        for item in value:
            _portable(item)
    elif isinstance(value, str) and (value.startswith(("/", "file://"))
                                    or re.match(r"[A-Za-z]:[\\/]", value)):
        raise ValueError("portable manifest cannot contain absolute machine paths")


def _validate_codes(value, n=None):
    if (value.ndim != 1 or value.dtype.kind not in "iu"
            or np.any((value < official.H) | (value > official.X))
            or (n is not None and value.shape != (n,))):
        raise ValueError("label array must use integer H/A/D/X codes on the declared grid")


def _check_array_archive(directory, manifest):
    archive = Path(directory) / "labels.npz"
    if _file_hash(archive) != manifest["labels_sha256"]:
        raise ValueError("label archive hash differs")
    with np.load(archive, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    hashes = manifest["label_array_hashes"]
    if set(hashes) != set(arrays) or any(_array_hash(value) != hashes[key]
                                       for key, value in arrays.items()):
        raise ValueError("label array checksum differs")
    if manifest["label_codes"] != CODES:
        raise ValueError("label code convention differs from the official scorer")
    for value in arrays.values():
        _validate_codes(value)
    return arrays


def _split_grid(splits):
    if len(splits) != 7:
        raise ValueError("comparison requires six paired matches and one distinct guard")
    if len({_safe_name(split["id"]) for split in splits}) != 7:
        raise ValueError("duplicate split identifier")
    paired = [split for split in splits if split["kind"] == "paired_holdout"]
    guards = [split for split in splits if split["kind"] == "oof_training_guard"]
    if len(paired) != 6 or len(guards) != 1:
        raise ValueError("paired matches and historical guard must remain separate")
    assessments = []
    for split in splits:
        if len(split["assessment"]) != 1:
            raise ValueError("each split must assess exactly one whole match")
        for role in ("assessment", "calibration", "base_train"):
            if len(split[role]) != len(set(split[role])):
                raise ValueError("duplicate match IDs in split role")
            for match in split[role]:
                _safe_name(match)
        assess, calibrate, base = (set(split[name]) for name in ("assessment", "calibration", "base_train"))
        if not calibrate or not base or assess & (calibrate | base):
            raise ValueError("assessment match occurs in training roles")
        if split["kind"] == "paired_holdout" and (len(calibrate) != 1 or len(base) != 4 or base & calibrate):
            raise ValueError("paired study requires disjoint four-base/one-calibration/one-assessment roles")
        assessments.extend(assess)
    if len(set(assessments)) != 7:
        raise ValueError("historical guard and paired assessment matches overlap")
    development = {split["assessment"][0] for split in paired}
    if set(guards[0]["calibration"]) != development or set(guards[0]["base_train"]) != development:
        raise ValueError("historical guard must train on the six development matches")
    for split in paired:
        if set(split["assessment"] + split["calibration"] + split["base_train"]) != development:
            raise ValueError("paired split uses an unexpected training match")
    return paired, guards


def _weights(weights):
    if (set(weights) != {"clean", "short", "long", "replay"}
            or any(not np.isfinite(value) or value <= 0 for value in weights.values())
            or not np.isclose(sum(weights.values()), 1, rtol=0, atol=1e-12)):
        raise ValueError("invalid four-scenario comparison weights")


def _audit_split(result, batches, weights, universe):
    split = result["split"]
    summaries = {}
    for field, role in (("training_audit", "calibration"), ("assessment_audit", "assessment")):
        rows = result[field]
        expected = dict.fromkeys(split[role], len(batches) * len(weights) * 2)
        if Counter(row["match"] for row in rows) != expected:
            raise ValueError("ancestry audit does not cover every match and scenario")
        training, selection, historical, provenance = set(), set(), set(), set()
        for row in rows:
            fit, select = set(row["training_matches"]), set(row["selection_matches"])
            forbidden = set(split["assessment"]) | {row["match"]}
            if (forbidden & (fit | select) or not fit <= set(split["base_train"])
                    or not select <= set(split["base_train"])):
                raise ValueError("assessment leakage in declared training ancestry")
            if not fit or not set(row["historical_selection_matches"]) <= universe:
                raise ValueError("missing or unknown declared training ancestry")
            if split["kind"] == "paired_holdout" and fit != set(split["base_train"]):
                raise ValueError("paired upstream fit IDs differ from declared base training")
            _hash_value(row["provenance_sha256"])
            training.update(fit)
            selection.update(select)
            historical.update(row["historical_selection_matches"])
            provenance.add(row["provenance_sha256"])
        summaries[field] = {
            "prediction_audit_count": len(rows), "sample_matches": sorted(expected),
            "training_matches": sorted(training), "selection_matches": sorted(selection),
            "historical_selection_matches": sorted(historical),
            "assessment_training_intersection": [], "provenance_sha256": sorted(provenance),
        }
    return summaries


def _check_score(actual, expected, context):
    """Compare the declared score projection, allowing only roundoff in sums."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"score structure differs for {context}")
        for key in expected:
            _check_score(actual[key], expected[key], f"{context}/{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"score rows differ for {context}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _check_score(left, right, f"{context}/{index}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not np.isfinite(expected) or not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise ValueError(f"recomputed score differs for {context}")
    elif actual != expected:
        raise ValueError(f"score metadata differs for {context}")


def _score_projection(score):
    return {"objective": score["objective"], "missing_ball": score["missing_ball"],
            "means": score["means"], "records": [
                {name: row[name] for name in ("batch", "match", "scenario", *METRICS)}
                for row in sorted(score["records"], key=lambda item: (
                    item["batch"], item["match"], item["scenario"]))]}


def _score_split(arrays, rows, method, weights):
    index = {(row["batch"], row["match"], row["scenario"], row["period"]): row for row in rows}
    measured = []
    for batch, match, scenario in sorted({key[:3] for key in index}):
        truth, prediction = [], []
        for period in (1, 2):
            row = index[batch, match, scenario, period]
            truth.append(arrays[row["ground_truth_key"]])
            prediction.append(arrays[row["prediction_keys"][method]])
        # The separator prevents merging possessions between halves.
        gt = np.concatenate((truth[0], [official.X], truth[1])).astype(np.uint8)
        pred = np.concatenate((prediction[0], [official.X], prediction[1])).astype(np.uint8)
        metrics = official.evaluate(gt, pred, bootstrap_n=0)
        measured.append({"batch": batch, "match": match, "scenario": scenario,
                         **{name: metrics[name] for name in METRICS}})
    means = {scenario: float(np.mean([row["spell_f1_mean"] for row in measured
                                      if row["scenario"] == scenario])) for scenario in weights}
    return {"objective": float(sum(weights[name] * means[name] for name in weights)),
            "missing_ball": float(sum(weights[name] * means[name] for name in weights
                                      if name != "clean") / (1 - weights["clean"])),
            "means": means, "records": measured}


def _evaluate_content(arrays, manifest):
    if manifest.get("schema") != SCHEMA or tuple(manifest["methods"]) != METHODS:
        raise ValueError("unsupported timing reference schema or methods")
    if (manifest.get("independent_test") is not False
            or manifest.get("target_labels_used") is not False):
        raise ValueError("timing comparison cannot claim an independent or target-label test")
    if manifest["label_codes"] != CODES:
        raise ValueError("label code convention differs from the official scorer")
    weights, batches, splits = manifest["scenario_weights"], manifest["batches"], manifest["splits"]
    _weights(weights)
    if len(batches) != 2 or len(set(batches)) != 2:
        raise ValueError("comparison requires two distinct outage batches")
    paired, guards = _split_grid(splits)
    rows = manifest["records"]
    grid = {(row["batch"], row["match"], row["scenario"], row["period"]) for row in rows}
    expected_grid = {(batch, split["assessment"][0], scenario, period) for split in splits
                     for batch in batches for scenario in weights for period in (1, 2)}
    if len(rows) != len(grid) or grid != expected_grid:
        raise ValueError("timing comparison grid is incomplete or duplicated")
    gt_keys, used_keys = {}, set()
    by_id = {split["id"]: split for split in splits}
    for row in rows:
        split = by_id[row["split_id"]]
        expected_scope = "paired_holdout" if split["kind"] == "paired_holdout" else "historical_guard"
        if row["match"] not in split["assessment"] or row["scope"] != expected_scope:
            raise ValueError("record mixes paired and historical guard matches")
        if set(row["prediction_keys"]) != set(METHODS):
            raise ValueError("record does not contain all four methods")
        gt_key = row["ground_truth_key"]
        source = arrays[gt_key]
        _validate_codes(source, row["n_frames"])
        identity = arrays[row["prediction_keys"]["identity"]]
        _validate_codes(identity, row["n_frames"])
        unit = row["match"], row["period"]
        if unit in gt_keys and gt_keys[unit] != gt_key:
            raise ValueError("ground truth must be deduplicated across scenarios and batches")
        gt_keys[unit] = gt_key
        used_keys.add(gt_key)
        for key in row["prediction_keys"].values():
            _validate_codes(arrays[key], row["n_frames"])
            if not np.array_equal(arrays[key] == official.X, identity == official.X):
                raise ValueError("prediction changed the frozen X exclusion mask")
            used_keys.add(key)
    if set(arrays) != used_keys or len(gt_keys) != 14:
        raise ValueError("archive contains undeclared fields or incorrect ground-truth grid")
    scores = {}
    for split in splits:
        selected = [row for row in rows if row["split_id"] == split["id"]]
        scores[split["id"]] = {method: _score_split(arrays, selected, method, weights)
                              for method in METHODS}
        _check_score(scores[split["id"]], manifest["expected_scores"][split["id"]], split["id"])
    summary = {"paired_holdout": {}, "historical_guard": scores[guards[0]["id"]]}
    delta_table = []
    for method in METHODS:
        selected = [scores[split["id"]][method] for split in paired]
        objective = float(np.mean([score["objective"] for score in selected]))
        baseline = float(np.mean([scores[split["id"]]["identity"]["objective"] for split in paired]))
        by_match = {split["assessment"][0]: scores[split["id"]][method]["objective"]
                    - scores[split["id"]]["identity"]["objective"] for split in paired}
        summary["paired_holdout"][method] = {
            "objective": objective,
            "missing_ball": float(np.mean([score["missing_ball"] for score in selected])),
            "means": {scenario: float(np.mean([score["means"][scenario] for score in selected]))
                      for scenario in weights}, "objective_delta": objective - baseline,
            "by_match_objective_delta": by_match,
        }
        guard_score = summary["historical_guard"][method]
        guard_delta = guard_score["objective"] - summary["historical_guard"]["identity"]["objective"]
        delta_table.append({"method": method, "paired_objective": objective,
                            "paired_delta": objective - baseline,
                            "guard_objective": guard_score["objective"], "guard_delta": guard_delta})
    expected = manifest["expected_summary"]
    for method in METHODS:
        projection = {key: summary["paired_holdout"][method][key]
                      for key in ("objective", "missing_ball", "means", "objective_delta")}
        declared = {key: expected["paired_holdout"][method][key] for key in projection}
        _check_score(projection, declared, f"paired summary/{method}")
        _check_score(summary["historical_guard"][method],
                     _score_projection(expected["historical_guard"][method]), f"guard/{method}")
    return {"schema": SCHEMA, "independent_test": False, "target_labels_used": False,
            "paired_match_count": 6, "historical_guard_match_count": 1,
            "scores": scores, "summary": summary, "delta_table": delta_table,
            "interpretation": CAVEAT}



def _verify_portable_provenance(manifest):
    provenance = manifest["source_provenance"]
    protocol = provenance["protocol"]
    if (_digest(protocol) != provenance["protocol_sha256"]
            or protocol.get("target_labels_used") is not False
            or protocol["splits"] != manifest["splits"]
            or protocol["scenario_weights"] != manifest["scenario_weights"]
            or protocol["sources"] != provenance["source_sha256"]):
        raise ValueError("portable protocol snapshot differs from its frozen hash or comparison")
    for name in ("protocol_sha256", "result_sha256", "protocol_file_sha256", "result_file_sha256",
                 "public_manifest_sha256", "public_labels_sha256"):
        _hash_value(provenance[name])
    for checksum in provenance["source_sha256"].values():
        _hash_value(checksum)
    split_ids = {split["id"] for split in manifest["splits"]}
    if (set(provenance["split_artifacts"]) != split_ids
            or set(manifest["training_ancestry_id_audit"]) != split_ids):
        raise ValueError("portable ancestry or artifact grid differs from frozen splits")
    universe = {split["assessment"][0] for split in manifest["splits"]}
    for split in manifest["splits"]:
        for checksum in provenance["split_artifacts"][split["id"]].values():
            _hash_value(checksum)
        audits = manifest["training_ancestry_id_audit"][split["id"]]
        for field, role in (("training_audit", "calibration"), ("assessment_audit", "assessment")):
            audit = audits[field]
            fit, select = set(audit["training_matches"]), set(audit["selection_matches"])
            expected_count = len(split[role]) * len(manifest["batches"]) * len(manifest["scenario_weights"]) * 2
            if (audit["sample_matches"] != sorted(split[role])
                    or audit["prediction_audit_count"] != expected_count
                    or not fit or not fit <= set(split["base_train"])
                    or not select <= set(split["base_train"])
                    or set(split["assessment"]) & (fit | select)
                    or audit["assessment_training_intersection"] != []
                    or not set(audit["historical_selection_matches"]) <= universe):
                raise ValueError("portable ancestry summary contains leakage or inconsistent match IDs")
            if split["kind"] == "paired_holdout" and fit != set(split["base_train"]):
                raise ValueError("portable ancestry summary differs from paired base-training IDs")
            if not audit["provenance_sha256"]:
                raise ValueError("portable ancestry summary has no bound prediction hashes")
            for checksum in audit["provenance_sha256"]:
                _hash_value(checksum)


def evaluate_reference(reference_dir):
    """Verify the portable manifest and label hashes, then rerun the official scorer."""
    directory = Path(reference_dir)
    manifest = _read_json(directory / "manifest.json")
    declared = manifest.pop("manifest_sha256", None)
    if declared != _digest(manifest):
        raise ValueError("portable manifest hash differs")
    _portable(manifest)
    _verify_portable_provenance(manifest)
    if _file_hash(ROOT / "evaluation/possession_metrics.py") != manifest["scorer_sha256"]:
        raise ValueError("official scorer source hash differs")
    arrays = _check_array_archive(directory, manifest)
    return _evaluate_content(arrays, manifest)


def export_reference(study_dir, output_dir, public_reference_dir=ROOT / "benchmarks/v6/public-reference"):
    """Export corrected schema-2 study predictions after public-reference and ancestry checks.

    Export requires a completed hash-wrapped result. Frozen source hashes and all
    split models and predictions are verified, and identity predictions must equal
    the public v6 reference exactly. Source features are never opened or copied.
    """
    study, output, public = Path(study_dir), Path(output_dir), Path(public_reference_dir)
    if output.exists():
        raise FileExistsError("timing reference destination already exists")
    protocol, protocol_sha = _read_wrapped(study / "protocol.json")
    result, result_sha = _read_wrapped(study / "result.json")
    if (protocol.get("schema") != "timing_paired_study_v1"
            or result.get("schema") != "timing_paired_result_v1"
            or result["protocol_sha256"] != protocol_sha
            or protocol.get("target_labels_used") is not False):
        raise ValueError("invalid or inconsistent source study protocol")
    _portable(protocol)
    _split_grid(protocol["splits"])
    for name, checksum in protocol["sources"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("source paths must be repository relative")
        if _file_hash(ROOT / name) != checksum:
            raise ValueError("frozen study source hash differs")
    public_manifest = _read_json(public / "manifest.json")
    if public_manifest["schema"] != "public_dfl_decoded_reference_v1":
        raise ValueError("unsupported public reference schema")
    source_arrays = _check_array_archive(public, public_manifest)
    if protocol["scenario_weights"] != public_manifest["scenario_weights"]:
        raise ValueError("study and public reference scenario weights differ")
    public_rows = {(row["batch"], row["match"], row["scenario"], row["period"]): row
                   for row in public_manifest["records"]}
    if len(public_rows) != len(public_manifest["records"]):
        raise ValueError("duplicate public reference record")
    arrays, records, audits, artifacts, expected = {}, [], {}, {}, {}
    declared_splits = {split["id"]: split for split in protocol["splits"]}
    if len(result["splits"]) != 7 or {split["split"]["id"] for split in result["splits"]} != set(declared_splits):
        raise ValueError("completed result does not cover the frozen splits")
    for split_result in result["splits"]:
        split = split_result["split"]
        split_id = split["id"]
        if split != declared_splits[split_id] or split_result["protocol_sha256"] != protocol_sha:
            raise ValueError("split differs from the frozen protocol")
        directory = study / "splits" / split_id
        if _read_json(directory / "result.json") != split_result:
            raise ValueError("split result differs from its completed study record")
        for name, field in (("model.npz", "model_sha256"), ("predictions.npz", "predictions_sha256")):
            if _file_hash(directory / name) != split_result[field]:
                raise ValueError("study artifact checksum differs")
        with np.load(directory / "model.npz", allow_pickle=False) as model:
            metadata = json.loads(str(model["metadata"]))
            if metadata.get("version") != 2 or "typed_boundary_score" not in metadata.get("feature_names", []):
                raise ValueError("export requires the corrected signed-score model schema")
        audits[split_id] = _audit_split(
            split_result, public_manifest["batches"], protocol["scenario_weights"],
            {item["assessment"][0] for item in protocol["splits"]})
        artifacts[split_id] = {"model_sha256": split_result["model_sha256"],
                               "predictions_sha256": split_result["predictions_sha256"],
                               "result_sha256": _file_hash(directory / "result.json")}
        expected[split_id] = {method: _score_projection(split_result["scores"][method]) for method in METHODS}
        match = split["assessment"][0]
        expected_archive_keys = set()
        with np.load(directory / "predictions.npz", allow_pickle=False) as predictions:
            for batch in public_manifest["batches"]:
                for scenario in protocol["scenario_weights"]:
                    for period in (1, 2):
                        row = public_rows[batch, match, scenario, period]
                        expected_scope = "inner" if split["kind"] == "paired_holdout" else "outer"
                        if row["scope"] != expected_scope:
                            raise ValueError("public reference mixes paired and guard scopes")
                        ground_truth = source_arrays[row["ground_truth_key"]]
                        original = source_arrays[row["prediction_keys"]["coarse_v6"]]
                        gt_key = f"gt_{match}_p{period}"
                        if gt_key in arrays and not np.array_equal(arrays[gt_key], ground_truth):
                            raise ValueError("ground truth differs between public scenarios")
                        arrays[gt_key] = ground_truth.astype(np.uint8)
                        keys = {}
                        for method in METHODS:
                            archive_key = f"{method}__{batch}__{match}__{scenario}__{period}"
                            expected_archive_keys.add(archive_key)
                            value = predictions[archive_key]
                            _validate_codes(value, row["n_frames"])
                            if method == "identity" and not np.array_equal(value, original):
                                raise ValueError("identity labels differ from the frozen public v6 reference")
                            if not np.array_equal(value == official.X, original == official.X):
                                raise ValueError("prediction changed the public v6 X exclusion mask")
                            key = f"pred_{method}_{batch}_{match}_{scenario}_p{period}"
                            keys[method] = key
                            arrays[key] = value.astype(np.uint8)
                        records.append({"split_id": split_id, "batch": batch, "match": match,
                                        "scenario": scenario, "period": period,
                                        "scope": "paired_holdout" if expected_scope == "inner" else "historical_guard",
                                        "n_frames": row["n_frames"], "ground_truth_key": gt_key,
                                        "prediction_keys": keys})
            if set(predictions.files) != expected_archive_keys:
                raise ValueError("prediction archive contains missing or undeclared arrays")
    manifest = {
        "schema": SCHEMA, "title": "Public DFL timing-calibration development comparison",
        "methods": list(METHODS), "label_codes": CODES, "batches": public_manifest["batches"],
        "scenario_weights": protocol["scenario_weights"], "splits": protocol["splits"],
        "records": records, "expected_scores": expected, "expected_summary": result["summary"],
        "independent_test": False, "target_labels_used": False, "target_inputs_included": False,
        "interpretation": CAVEAT, "training_ancestry_id_audit": audits,
        "scorer_sha256": _file_hash(ROOT / "evaluation/possession_metrics.py"),
        "source_provenance": {
            "protocol_sha256": protocol_sha, "result_sha256": result_sha,
            "protocol_file_sha256": _file_hash(study / "protocol.json"),
            "result_file_sha256": _file_hash(study / "result.json"),
            "source_sha256": protocol["sources"], "split_artifacts": artifacts,
            "public_manifest_sha256": _file_hash(public / "manifest.json"),
            "public_labels_sha256": public_manifest["labels_sha256"],
            "dataset_attribution": public_manifest["source_provenance"]["dataset_attribution"],
            "protocol": protocol,
        },
        "excluded_fields": ["raw tracking", "input features", "participant data", "model weights", "absolute machine paths"],
        "label_array_hashes": {key: _array_hash(value) for key, value in arrays.items()},
    }
    _portable(manifest)
    _evaluate_content(arrays, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".timing-reference-", dir=output.parent) as temporary:
        staged = Path(temporary) / "reference"
        staged.mkdir()
        np.savez_compressed(staged / "labels.npz", **arrays)
        manifest["labels_sha256"] = _file_hash(staged / "labels.npz")
        manifest["manifest_sha256"] = _digest(manifest)
        _write_json(staged / "manifest.json", manifest)
        os.rename(staged, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", nargs="?", type=Path,
                        default=ROOT / "benchmarks/timing-calibration-v1")
    parser.add_argument("--export-from", type=Path, metavar="STUDY_DIR")
    parser.add_argument("--public-reference", type=Path, default=ROOT / "benchmarks/v6/public-reference")
    parser.add_argument("--output", type=Path, help="save the complete recomputed scores as JSON")
    args = parser.parse_args()
    if args.export_from is not None:
        export_reference(args.export_from, args.reference_dir, args.public_reference)
    result = evaluate_reference(args.reference_dir)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.output, result)
    print("Method             Paired spellF1     Delta    Guard spellF1     Delta")
    for row in result["delta_table"]:
        print(f"{row['method']:<18} {row['paired_objective']:.6f}   {row['paired_delta']:+.6f}"
              f"      {row['guard_objective']:.6f}   {row['guard_delta']:+.6f}")
    print(CAVEAT)


if __name__ == "__main__":
    main()
