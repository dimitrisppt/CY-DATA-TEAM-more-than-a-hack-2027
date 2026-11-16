"""Recompute official development scores from the compact public DFL label reference."""
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from evaluation import possession_metrics as official
from experiments.augmentation_evaluation import _compact
from experiments.evaluate import _join_with_x
from possession.storage import sha256_file

_SCHEMA = "public_dfl_decoded_reference_v1"


def _score_rows(arrays: dict[str, np.ndarray], manifest: dict[str, Any],
                variant: str, batch: str, scope: str) -> dict[str, Any]:
    rows = [row for row in manifest["records"]
            if row["batch"] == batch and row["scope"] == scope]
    scenarios = tuple(manifest.get("scenario_order", sorted(manifest["scenario_weights"])))
    if set(scenarios) != set(manifest["scenario_weights"]):
        raise ValueError("reference scenario order differs from its weight table")
    matches = sorted({row["match"] for row in rows})
    expected = {(match, scenario, period) for match in matches
                for scenario in scenarios for period in (1, 2)}
    actual = {(row["match"], row["scenario"], row["period"]) for row in rows}
    if not matches or len(rows) != len(actual) or actual != expected:
        raise ValueError(f"reference grid is incomplete for {batch}/{scope}")
    index = {(row["match"], row["scenario"], row["period"]): row for row in rows}
    by_match: dict[str, dict] = {}
    scenario_rows = {scenario: [] for scenario in scenarios}
    for match in matches:
        by_match[match] = {}
        for scenario in scenarios:
            gt_halves, pred_halves = [], []
            for period in (1, 2):
                row = index[(match, scenario, period)]
                gt = arrays[row["ground_truth_key"]]
                pred = arrays[row["prediction_keys"][variant]]
                if gt.shape != (row["n_frames"],) or pred.shape != gt.shape:
                    raise ValueError("reference label array has the wrong frame count")
                if gt.dtype.kind not in "iu" or pred.dtype.kind not in "iu":
                    raise ValueError("reference labels must use integer H/A/D/X codes")
                if (np.any(gt < 0) or np.any(pred < 0)
                        or np.any(gt > official.X) or np.any(pred > official.X)):
                    raise ValueError("reference labels contain an unknown code")
                gt_halves.append(gt)
                pred_halves.append(pred)
            result = _compact(official.evaluate(_join_with_x(gt_halves),
                                                _join_with_x(pred_halves), bootstrap_n=0))
            by_match[match][scenario] = result
            scenario_rows[scenario].append(result)
    means = {}
    for scenario, scores in scenario_rows.items():
        means[scenario] = {
            "spell_f1_mean": float(np.mean([score["spell_f1_mean"] for score in scores])),
            "macro_f1": float(np.mean([score["macro_f1"] for score in scores])),
            "transition_f1_at_12": float(np.mean(
                [score["transitions"]["12"]["f1"] for score in scores])),
        }
    weights = manifest["scenario_weights"]
    return {"objective": float(sum(weights[name] * means[name]["spell_f1_mean"]
                                   for name in scenarios)),
            "scenario_weights": weights, "means": means, "by_match": by_match}


def evaluate_reference(reference_dir: str | Path,
                       progress_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Verify exported hashes and recompute each stored objective from decoded labels."""
    reference_dir = Path(reference_dir)
    manifest_path = reference_dir / "manifest.json"
    arrays_path = reference_dir / "labels.npz"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != _SCHEMA:
        raise ValueError("unsupported public reference schema")
    if sha256_file(arrays_path) != manifest.get("labels_sha256"):
        raise ValueError("reference NPZ hash differs from its manifest")
    expected_codes = {"H": official.H, "A": official.A, "D": official.D, "X": official.X}
    if manifest.get("label_codes") != expected_codes:
        raise ValueError("reference label code convention differs from official scorer")
    with np.load(arrays_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    declared_keys = {key for row in manifest["records"]
                     for key in (row["ground_truth_key"], *row["prediction_keys"].values())}
    if set(arrays) != declared_keys:
        raise ValueError("reference NPZ keys do not match the manifest grid")
    declared_array_hashes = manifest.get("label_array_hashes", {})
    if set(declared_array_hashes) != declared_keys or any(
            hashlib.sha256(arrays[key].tobytes(order="C")).hexdigest()
            != declared_array_hashes[key] for key in declared_keys):
        raise ValueError("reference label-array hash differs from its manifest")
    variants = tuple(manifest["variants"])
    results = {}
    for variant in variants:
        results[variant] = {}
        for batch in manifest["batches"]:
            results[variant][batch] = {}
            for scope in ("inner", "outer"):
                if progress_callback is not None:
                    progress_callback(f"scoring {variant}/{batch}/{scope}")
                score = _score_rows(arrays, manifest, variant, batch, scope)
                expected_score = manifest["expected_scores"][variant][batch][scope]
                if score != expected_score:
                    raise ValueError(f"recomputed {variant}/{batch}/{scope} differs from frozen score")
                results[variant][batch][scope] = score
    summary = {}
    for variant in variants:
        inner = [results[variant][batch]["inner"]["objective"]
                 for batch in manifest["batches"]]
        outer = [results[variant][batch]["outer"]["objective"]
                 for batch in manifest["batches"]]
        missing_names = ("short", "long", "replay")
        missing_aggregate = (float(np.mean([
            sum(manifest["scenario_weights"][name]
                * results[variant][batch]["inner"]["means"][name]["spell_f1_mean"]
                for name in missing_names) / .5
            for batch in manifest["batches"]]))
            if set(missing_names) <= set(manifest["scenario_weights"]) else None)
        summary[variant] = {"mean_inner_objective": float(np.mean(inner)),
                            "mean_outer_objective": float(np.mean(outer)),
                            "mean_inner_missing_aggregate": missing_aggregate}
        summary[variant]["mean_inner_scenario_spell_f1"] = {
            scenario: float(np.mean([results[variant][batch]["inner"]["means"][scenario]
                                     ["spell_f1_mean"] for batch in manifest["batches"]]))
            for scenario in manifest["scenario_weights"]}
    return {"schema": _SCHEMA, "recomputed_from_saved_oof_predictions": True,
            "independent_test": False, "scores": results, "summary": summary}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    default_reference = Path(__file__).resolve().parents[1] / "benchmarks/v6/public-reference"
    parser.add_argument("reference_dir", type=Path, nargs="?", default=default_reference)
    parser.add_argument("--output", type=Path,
                        help="write the complete recomputed score record as JSON")
    args = parser.parse_args()
    def progress(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    result = evaluate_reference(args.reference_dir, progress_callback=progress)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                          allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
