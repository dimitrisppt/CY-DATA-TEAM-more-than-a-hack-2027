"""Verify and score the saved public labels from the stopped refinement study.

This command needs neither study storage nor models. It performs no training,
inference, decoding, tuning or promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from experiments.spell_evidence_audit import spell_counts
from possession.storage import sha256_file
from scripts.evaluate_reference import _score_rows

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "public_dfl_partial_refinement_reference_v1"
WEIGHTS = {"clean": .50, "short": .20, "long": .15, "replay": .15}
GROUPS = {
    "completed_two_fine_seeds": {
        "seeds": [8101, 8102], "variants": ["baseline", "control", "spell_balanced"],
    },
    "matched_single_fine_seed": {
        "seeds": [8101],
        "variants": ["baseline", "control", "spell_balanced", "masked_teacher"],
    },
}
CONTEXT = {
    "partial": True, "independent_test": False,
    "target_inputs_included": False, "target_labels_used": False,
    "training_status": "stopped_by_user", "training_restarted": False,
    "inference_run": False, "promotion_evaluated": False,
    "assessment_match_count": 2, "assessment_matches": ["DFL-MAT-J03WOH", "DFL-MAT-J03WOY"],
    "completed_assessment_folds": 1, "planned_assessment_folds": 3,
    "completed_fits": 11, "planned_fits": 36,
}


def _summary(scores, weights):
    batches = sorted(scores)
    matches = sorted(scores[batches[0]]["by_match"])
    by_match = {match: {scenario: float(np.mean([
        scores[batch]["by_match"][match][scenario]["spell_f1_mean"] for batch in batches
    ])) for scenario in weights} for match in matches}
    means = {scenario: float(np.mean([row[scenario] for row in by_match.values()]))
             for scenario in weights}
    return {
        "objective": sum(weights[s] * means[s] for s in weights),
        "missing": sum(weights[s] * means[s] for s in weights if s != "clean") / .5,
        "means": means, "by_match": by_match,
        "match_objectives": {match: sum(weights[s] * row[s] for s in weights)
                             for match, row in by_match.items()},
        "batch_objectives": {batch: scores[batch]["objective"] for batch in batches},
    }


def evaluate_reference(reference_dir=ROOT / "benchmarks/partial-refinement-v1",
                       progress_callback=None):
    """Authenticate public label arrays and reproduce both distinct comparisons."""
    reference_dir = Path(reference_dir)
    manifest = json.loads((reference_dir / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported partial reference schema")
    for name, field in (("labels.npz", "labels_sha256"), ("result.json", "results_sha256")):
        if sha256_file(reference_dir / name) != manifest.get(field):
            raise ValueError(f"partial reference {name} hash differs from manifest")
    if manifest.get("label_codes") != {
            "H": official.H, "A": official.A, "D": official.D, "X": official.X}:
        raise ValueError("partial reference label code convention differs")
    with np.load(reference_dir / "labels.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    declared_keys = {key for rows in manifest["references"].values() for row in rows
                     for key in (row["ground_truth_key"], *row["prediction_keys"].values())}
    if set(arrays) != declared_keys or set(manifest["label_array_hashes"]) != declared_keys:
        raise ValueError("partial reference label keys differ from manifest")
    for key, array in arrays.items():
        if hashlib.sha256(array.tobytes(order="C")).hexdigest() != manifest[
                "label_array_hashes"][key]:
            raise ValueError("partial reference label-array hash differs from manifest")
    expected = json.loads((reference_dir / "result.json").read_text())
    if expected.get("schema") != "public_dfl_partial_refinement_results_v1":
        raise ValueError("unsupported partial result schema")
    for record in (manifest, expected):
        if any(type(record.get(key)) is not type(value) or record[key] != value
               for key, value in CONTEXT.items()):
            raise ValueError("partial reference privacy, stopped status or assessment context differs")
    if (manifest["scenario_weights"] != WEIGHTS
            or manifest["scenario_order"] != list(WEIGHTS)
            or manifest["batches"] != ["batch_1", "batch_2"]
            or manifest["groups"] != GROUPS):
        raise ValueError("partial reference fixed comparison configuration differs")
    if set(manifest["references"]) != set(manifest["groups"]) or set(
            expected["groups"]) != set(manifest["groups"]):
        raise ValueError("partial reference groups differ")
    weights = {name: manifest["scenario_weights"][name] for name in manifest["scenario_order"]}
    results = {}
    for group, config in manifest["groups"].items():
        rows = manifest["references"][group]
        variants = config["variants"]
        if any(set(row["prediction_keys"]) != set(variants) for row in rows):
            raise ValueError("partial reference variants differ")
        if {row["match"] for row in rows} != set(manifest["assessment_matches"]):
            raise ValueError("partial reference assessment matches differ")
        if any(row["scope"] != "inner" for row in rows) or {
                row["batch"] for row in rows} != set(manifest["batches"]):
            raise ValueError("partial reference batches or scopes differ")
        score_manifest = {"records": rows, "scenario_weights": manifest["scenario_weights"]}
        scores, diagnostics = {}, {}
        for variant in variants:
            if progress_callback is not None:
                progress_callback(f"scoring {group}/{variant}")
            scores[variant] = {batch: _score_rows(arrays, score_manifest, variant, batch, "inner")
                               for batch in manifest["batches"]}
            diagnostics[variant] = {scenario: [
                spell_counts(arrays[row["ground_truth_key"]],
                             arrays[row["prediction_keys"][variant]])
                for row in rows if row["batch"] == "batch_1" and row["scenario"] == scenario
            ] for scenario in weights}
        results[group] = {
            "configuration": config, "scores": scores,
            "summary": {variant: _summary(scores[variant], weights) for variant in variants},
            "spell_diagnostics_batch_1": diagnostics,
        }
        if results[group] != expected["groups"][group]:
            raise ValueError(f"recomputed {group} differs from saved expected scores or counts")
    return {"schema": SCHEMA, "partial": True, "independent_test": False,
            "target_inputs_included": False, "target_labels_used": False,
            "recomputed_from_saved_labels": True, "groups": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path, nargs="?",
                        default=ROOT / "benchmarks/partial-refinement-v1")
    parser.add_argument("--output", type=Path, help="write the complete recomputed JSON record")
    args = parser.parse_args()

    def progress(message):
        print(message, file=sys.stderr, flush=True)

    result = evaluate_reference(args.reference_dir, progress_callback=progress)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({group: value["summary"] for group, value in result["groups"].items()},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
