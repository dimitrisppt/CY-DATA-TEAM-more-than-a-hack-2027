"""Extend the authenticated partial study with descriptive temporal-IoU curves.

The fixed threshold sweep does not fit models, change labels, or select a
decoder. Each match has equal weight; outage batches are averaged within it.
An optional prepared-data check connects the reference labels to fresh DFL XML.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from experiments.evaluate import _join_with_x, _target_codes
from experiments.spell_evidence_audit import duration_bin
from possession.prepared import verify_prepared
from possession.storage import load_half, sha256_file, write_json
from scripts.evaluate_partial_reference import ROOT, evaluate_reference

THRESHOLDS = tuple(round(i / 10, 1) for i in range(1, 10))


def score_iou(gt, pred, thresholds=THRESHOLDS):
    """Use the official abstention filling and one-to-one possession matching."""
    runs = official.valid_runs(gt)
    filled = official.fill_abstentions(pred, runs)
    gold = official.possession_spells(gt, runs)
    predicted = official.possession_spells(filled, runs)
    overlaps = official.spell_overlaps(gold, predicted)
    scores = official.spell_scores(gt, filled, runs, ious=thresholds)
    result = {}
    for threshold, metrics in scores["per_iou"].items():
        flags = official.matched_gt_flags(overlaps, len(gold), threshold)
        bins = {}
        for index, (start, stop, _) in enumerate(gold):
            name = duration_bin(stop - start, official.FPS)
            counts = bins.setdefault(name, {"gt": 0, "matched": 0})
            counts["gt"] += 1
            counts["matched"] += int(flags[index])
        result[str(threshold)] = {**metrics, "duration_bins": bins}
    return result


def check_prepared(directory, manifest, arrays):
    """Require verified caches and byte-identical truth on both assessment matches."""
    directory = Path(directory).resolve()
    matches = manifest["assessment_matches"]
    paths = [directory / f"{match}_half{period}.npz"
             for match in matches for period in (1, 2)]
    provenance = verify_prepared(directory, paths)
    if provenance["status"] != "verified_manifest":
        raise ValueError("fresh DFL data must have a verified preparation manifest")
    records = []
    for match in matches:
        for period in (1, 2):
            path = directory / f"{match}_half{period}.npz"
            frames, targets = load_half(path)
            gt = _target_codes(frames, targets)
            key = f"gt_{match}_{period}"
            if not np.array_equal(gt, arrays[key]):
                raise ValueError(f"fresh DFL truth differs from the partial reference: {key}")
            records.append({"match": match, "period": period, "frames": len(gt),
                            "prepared_sha256": sha256_file(path), "exact_truth_match": True})
    return {"provenance": provenance, "checks": records}


def analyze(reference_dir, *, prepared_dir=None):
    reference_dir = Path(reference_dir)
    reproduced = evaluate_reference(reference_dir)
    manifest = json.loads((reference_dir / "manifest.json").read_text())
    with np.load(reference_dir / "labels.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    groups = {}
    for group, config in manifest["groups"].items():
        rows = manifest["references"][group]
        index = {(r["batch"], r["match"], r["scenario"], r["period"]): r for r in rows}
        variants = {}
        for variant in config["variants"]:
            records = []
            for batch in manifest["batches"]:
                for match in manifest["assessment_matches"]:
                    for scenario in manifest["scenario_order"]:
                        halves = [index[batch, match, scenario, period] for period in (1, 2)]
                        gt = _join_with_x([arrays[r["ground_truth_key"]] for r in halves])
                        pred = _join_with_x([arrays[r["prediction_keys"][variant]] for r in halves])
                        records.append({"batch": batch, "match": match, "scenario": scenario,
                                        "iou": score_iou(gt, pred)})
            curves = {}
            for threshold in THRESHOLDS:
                key = str(threshold)
                by_match = {
                    match: {
                        scenario: float(np.mean([r["iou"][key]["f1"] for r in records
                                                if r["match"] == match
                                                and r["scenario"] == scenario]))
                        for scenario in manifest["scenario_order"]}
                    for match in manifest["assessment_matches"]}
                scenarios = {scenario: float(np.mean([m[scenario] for m in by_match.values()]))
                             for scenario in manifest["scenario_order"]}
                curves[key] = {
                    "scenario_f1": scenarios, "by_match": by_match,
                    "weighted_f1": sum(manifest["scenario_weights"][s] * scenarios[s]
                                       for s in manifest["scenario_order"])}
            # The official objective is the mean over IoUs .3, .4, .5, .6, .7.
            objective = float(np.mean([curves[str(t)]["weighted_f1"]
                                       for t in official.SPELL_IOUS]))
            expected = reproduced["groups"][group]["summary"][variant]["objective"]
            if not np.isclose(objective, expected, atol=1e-12, rtol=0):
                raise ValueError("IoU sweep does not reproduce the original official objective")
            variants[variant] = {"official_objective": objective, "curves": curves,
                                 "records": records}
        groups[group] = {"configuration": config, "variants": variants}
    return {
        "schema": "partial_refinement_temporal_iou_analysis_v1",
        "reference_sha256": {name: sha256_file(reference_dir / name)
                              for name in ("manifest.json", "labels.npz", "result.json")},
        "script_sha256": sha256_file(Path(__file__)),
        "thresholds": list(THRESHOLDS), "official_thresholds": list(official.SPELL_IOUS),
        "scenario_weights": manifest["scenario_weights"],
        "assessment_matches": manifest["assessment_matches"],
        "independent_test": False, "training_run": False, "inference_run": False,
        "promotion_evaluated": False, "groups": groups,
        "fresh_dfl_truth_verification": (check_prepared(prepared_dir, manifest, arrays)
                                          if prepared_dir is not None else None),
    }


def write_report(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "result.json", result)
    with (output / "iou-curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "variant", "iou", "weighted_f1", "clean_f1",
                         "short_f1", "long_f1", "replay_f1"])
        for group, data in result["groups"].items():
            for variant, metrics in data["variants"].items():
                for threshold, curve in metrics["curves"].items():
                    writer.writerow([group, variant, threshold, curve["weighted_f1"],
                                     *[curve["scenario_f1"][s]
                                       for s in ("clean", "short", "long", "replay")]])
    lines = ["# Temporal IoU analysis of the partial refinement study", "",
             "This extends the saved WOH/WOY comparison. No models were trained or selected.",
             "Both outage batches are averaged within each match, then matches receive equal weight.",
             "The original official objective averages temporal IoUs 0.3, 0.4, 0.5, 0.6 and 0.7.",
             "The wider 0.1–0.9 sweep is descriptive; it is not a new selection criterion.", ""]
    for group, data in result["groups"].items():
        lines.extend([f"## {group}", "", "| Variant | Official objective | F1 @ .3 | F1 @ .5 | F1 @ .7 |",
                      "|---|---:|---:|---:|---:|"])
        for variant, metrics in data["variants"].items():
            values = [metrics["official_objective"],
                      *[metrics["curves"][str(t)]["weighted_f1"] for t in (.3, .5, .7)]]
            lines.append(f"| {variant} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
        lines.extend(["", "Clean scenario, batch 1: pooled event counts at temporal IoU 0.5.", "",
                      "| Variant | <2s matched / true | <2s recall | All-duration spurious |",
                      "|---|---:|---:|---:|"])
        for variant, metrics in data["variants"].items():
            records = [r["iou"]["0.5"] for r in metrics["records"]
                       if r["batch"] == "batch_1" and r["scenario"] == "clean"]
            bins = [r["duration_bins"].get(name, {"matched": 0, "gt": 0})
                    for r in records for name in ("under_1s", "1_to_2s")]
            matched, true = sum(b["matched"] for b in bins), sum(b["gt"] for b in bins)
            recall = f"{matched / true:.1%}" if true else "n/a"
            lines.append(f"| {variant} | {matched}/{true} | {recall} | "
                         f"{sum(r['n_spurious'] for r in records)} |")
        lines.append("")
    if result["fresh_dfl_truth_verification"] is not None:
        lines.extend(["Freshly downloaded DFL XML reproduces the reference truth exactly on all four assessment halves.", ""])
    lines.extend(["The study still covers one of three planned folds and 11 of 36 completed fits.",
                  "All public matches have historical development use. Seeds, masks and thresholds are not independent matches.",
                  "No complete-study advancement gate can be evaluated from this partial reference.", ""])
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "benchmarks/partial-refinement-v1")
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/possession-refinement/iou-analysis"))
    args = parser.parse_args()
    result = analyze(args.reference_dir, prepared_dir=args.prepared_dir)
    write_report(result, args.output)
    print(json.dumps({"report": str(args.output / "report.md"),
                      "fresh_dfl_truth_verified": args.prepared_dir is not None,
                      "training_run": False}, indent=2))


if __name__ == "__main__":
    main()
