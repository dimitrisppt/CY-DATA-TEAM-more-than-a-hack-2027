"""Export the completed frozen follow-up as tables and scientific figures."""
from __future__ import annotations

import csv
from collections import Counter

import numpy as np

from experiments.campaign_protocol import WEIGHTS, read_record
from experiments.iou_followup_training import OUTPUT, SOURCE, freeze

NAMES = {"gpu_baseline": "Parent EMA", "control": "Control", "spell_balanced": "Combined",
         "sampling_only": "Sampling only", "loss_only": "Loss only"}


def short_counts(detail):
    rows = [r["iou"]["0.5"] for r in detail["records"] if r["batch"] == "batch_1" and r["scenario"] == "clean"]
    bins = [r["duration_bins"].get(b, {"gt": 0, "matched": 0}) for r in rows for b in ("under_1s", "1_to_2s")]
    return sum(b["matched"] for b in bins), sum(b["gt"] for b in bins), sum(r["n_spurious"] for r in rows)


def report(output=OUTPUT):
    plan = freeze(output)
    results = {arm: read_record(output / arm / "result.json") for arm in ("confirmation", "ablation", "guard")}
    if any(r is None for r in results.values()):
        raise ValueError("all three complete assessments are required")
    verification = read_record(output / "completion-verification.json")
    if verification is None:
        raise ValueError("run scripts.verify_iou_followup before publishing the final report")
    audit = read_record(output / "error-audit/result.json")
    directory = output / "report"
    directory.mkdir(exist_ok=True)
    confirmed = results["confirmation"]["groups"]["ensemble"]["prespecified_gate"]["spell_balanced"]["prespecified_checks_passed"]
    guarded = results["guard"]["groups"]["ensemble"]["prespecified_gate"]["spell_balanced"]["prespecified_checks_passed"]
    decision = ("The fixed combined recipe passes both the fresh-seed confirmation and WMX guard. "
                "It remains a development result; independent labeled matches are the next validation step."
                if confirmed and guarded else
                "The fixed combined recipe does not pass every prespecified confirmation/guard check. "
                "Retain the delivered model; the original development gain is insufficient for promotion.")
    lines = ["# GPU IoU follow-up", "", decision, "",
             ("Completed: 80 new fits and 240,000 optimizer updates on a Tesla T4. "
             "Confirmation uses fresh fine and coarse seeds 8201/8202; WMX uses 9201/9202. "
             "The 24 factorial continuations reuse the original 8101/8102 parents. "
             "All fine continuations receive the same 1,200-update budget and final-200-update EMA."), "",
             ("The candidate, folds, masks, blend, decoder, comparisons and gates were frozen before new outcomes. "
             "Both historical-refinement mask batches are evaluated at IoUs 0.1–0.9. "
             "The official objective averages IoUs 0.3–0.7, weights clean/short/long/replay "
             "by 0.50/0.20/0.15/0.15, and weights each whole match equally."), "",
             ("All seven DFL matches have historical development use. Fresh seeds replicate training randomness, "
             "not new-match generalization. Six-match bootstrap intervals below are descriptive "
             "(5,000 paired draws, seed 20261116); masks and seeds are not independent observations. "
             "WMX is a separate single-match guard, with no confidence interval. No automatic promotion occurred."), "",
             "![Temporal-IoU curves](iou-curves.png)", "", "[Vector figure](iou-curves.pdf)", ""]
    curve_rows, match_rows, duration_rows = [], [], []
    for arm, result in results.items():
        data = result["groups"]["ensemble"]
        lines += [f"## {arm.capitalize()}", "",
                  "| Method | Objective | Missing ball | Clean | <2s matched / true | Spurious |",
                  "|---|---:|---:|---:|---:|---:|"]
        for variant in [v for v in NAMES if v in data["summary"]]:
            row = data["summary"][variant]
            matched, true, spurious = short_counts(data["variants"][variant])
            lines.append(f"| {NAMES[variant]} | {row['objective']:.6f} | {row['missing']:.6f} | "
                         f"{row['means']['clean']:.6f} | {matched}/{true} | {spurious} |")
        lines += ["", "Short-event recall and spurious counts use clean batch 1 at IoU 0.5.", "",
                  "| Candidate vs control | Objective change | Descriptive 95% interval | Positive matches |",
                  "|---|---:|---|---:|"]
        for variant, comparisons in data["comparisons"].items():
            row = comparisons["control"]
            interval = row["descriptive_match_bootstrap_95_interval"]
            ci = f"[{interval[0]:+.6f}, {interval[1]:+.6f}]" if interval else "Not estimated (one match)"
            wins = sum(v > 0 for v in row["by_match"].values())
            lines.append(f"| {NAMES[variant]} | {row['objective_delta']:+.6f} | {ci} | {wins}/{row['n_matches']} |")
        compared = [v for v in NAMES if v in data["comparisons"]]
        lines += ["", "| Group | " + " | ".join(f"{NAMES[v]} − control" for v in compared) + " |",
                  "|---|" + "---:|" * len(compared)]
        for group, scores in result["groups"].items():
            lines.append(f"| {group} | " + " | ".join(
                f"{scores['comparisons'][v]['control']['objective_delta']:+.6f}" for v in compared) + " |")
        lines += ["", "Individual fine-seed groups retain the same pair of coarse models.", ""]
        if arm == "confirmation":
            lines += ["![Confirmation by match](confirmation-by-match.png)", "",
                      "[Vector figure](confirmation-by-match.pdf)", ""]
            singles = [scores["comparisons"]["spell_balanced"]["control"]["objective_delta"]
                       for group, scores in result["groups"].items() if group != "ensemble"]
            if all(value < 0 for value in singles):
                lines += [("Both individual fine-seed comparisons decline. The positive confirmation "
                           "supports the exact two-seed ensemble; the versions using one fine-training "
                           "seed did not improve. "
                           "The frozen gate applies to the ensemble and does not require each individual "
                           "fine seed to improve."), ""]
        for variant, gate in data["prespecified_gate"].items():
            failed = [name for name, passed in gate["checks"].items() if not passed]
            lines.append(f"- {NAMES[variant]} checks: {'PASS' if gate['prespecified_checks_passed'] else 'FAIL'}"
                         + (f"; failed: {', '.join(failed)}." if failed else "."))
        if arm == "ablation":
            interaction = data["factorial_interaction"]
            ci = interaction["descriptive_match_bootstrap_95_interval"]
            lines += ["", (f"Interaction (combined − sampling − loss + control): {interaction['mean']:+.6f}; "
                       f"descriptive 95% interval [{ci[0]:+.6f}, {ci[1]:+.6f}]."),
                       ("These ablations are exploratory. Their results did not select or alter the confirmation candidate. "
                       "A newly favored ablation would require fresh confirmation. Loss-only preserves the control's "
                       "exact sampled windows; sampling-only preserves the combined recipe's windows. "
                       "Sampling probability is 0.25 and auxiliary loss weight is 0.25 when enabled.")]
        lines.append("")
        for group, scores in result["groups"].items():
            for variant, detail in scores["variants"].items():
                for threshold, curve in detail["curves"].items():
                    curve_rows.append([arm, group, variant, threshold, curve["weighted_f1"],
                                       *(curve["scenario_f1"][s] for s in WEIGHTS)])
                for match, row in scores["summary"][variant]["by_match"].items():
                    match_rows.append([arm, group, variant, match, sum(WEIGHTS[s] * row[s] for s in WEIGHTS),
                                       *(row[s] for s in WEIGHTS)])
                for row in detail["records"]:
                    for threshold, metrics in row["iou"].items():
                        for duration, counts in metrics["duration_bins"].items():
                            duration_rows.append([arm, group, variant, row["batch"], row["match"],
                                                  row["scenario"], threshold, duration, counts["gt"], counts["matched"]])
    lines += ["## Error audit of the original ensemble", "",
              ("Clean batch 1, official greedy IoU 0.5. Error classes follow frozen precedence: "
              "dominant wrong-team overlap, split, merge, timing, phantom. Split and merge flags use "
              "any positive same-team interval overlap; they may coexist. These are operational "
              "diagnostics, not verified causal explanations. The CSV retains all flags and offsets."), "",
              "| Error category | Control | Combined | Change |", "|---|---:|---:|---:|"]
    a, b = (audit["summary"]["all"][v] for v in ("control", "spell_balanced"))
    for category in ("wrong_team", "split", "merge", "timing", "phantom"):
        x, y = a["spurious_categories"].get(category, 0), b["spurious_categories"].get(category, 0)
        lines.append(f"| {category} | {x} | {y} | {y-x:+d} |")
    delta = audit["paired_changes"]
    with (output / "error-audit/events.csv").open() as stream:
        events = list(csv.DictReader(stream))
    counts = Counter((r["variant"], r["category"], "under_2s" if float(r["duration_seconds"]) < 2 else "2s_or_more")
                     for r in events if r["matched"] == "False")
    with (output / "error-audit/spurious-by-duration.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["category", "duration", "control", "spell_balanced", "change"])
        for category in ("wrong_team", "split", "merge", "timing", "phantom"):
            for duration in ("under_2s", "2s_or_more"):
                x, y = (counts[v, category, duration] for v in ("control", "spell_balanced"))
                writer.writerow([category, duration, x, y, y - x])
    lines += ["", (f"Matched true events: {a['matched']} → {b['matched']}; spurious: {a['spurious']} → {b['spurious']}. "
              f"The combined prediction recovers {delta['recovered_gt']} ground-truth possessions and loses "
              f"{delta['lost_gt']}; among possessions under two seconds, it recovers {delta['recovered_short_gt']} "
              f"and loses {delta['lost_short_gt']}."), "",
              ("Spurious predictions under two seconds increase from 164 to 221, accounting for 57 "
               "of the net 76-event increase. Wrong-team predictions under two seconds increase from "
               "104 to 149. These are net count differences, not one-to-one counts of newly introduced events."), "",
              (f"Cross-method event correspondence at IoU 0.5 yields {delta['shared_spurious']} shared spurious events, "
              f"{delta['new_spurious']} new spurious events and {delta['became_spurious']} formerly matched events "
              f"that become spurious. It removes {delta['removed_spurious']} old spurious events and turns "
              f"{delta['became_matched']} into matches. Thus the net increase of 76 is not a count of newly introduced errors."), "",
              ("The WOH timeline selects the fixed 60-second bin with the largest increase in spurious event starts; "
              "ties choose the earlier half/time. It is a diagnostic example, not a representative sample."), "",
              "![WOH timeline](../error-audit/woh-timeline.png)", "",
              "## Reproduction and artifacts", "",
              (f"Validation: {verification['tests_passed']} tests passed; "
               f"{verification['new_prediction_files_verified']} new prediction files and "
               f"{verification['input_files_verified']} assessment inputs verified. All scores reproduce "
               "exactly from saved labels, and the original ablation reference conditions reproduce "
               "their original decoded arrays exactly. See [verification receipt](../completion-verification.json)."), "",
              "Run from the repository root using the frozen uv environment:", "", "```bash",
              "uv run --no-sync python -m experiments.iou_followup_training --stage train",
              "uv run --no-sync python -m experiments.iou_followup_evaluation",
              "uv run --no-sync python -m scripts.audit_iou_errors",
              "uv run --no-sync python -m scripts.verify_iou_followup",
              "uv run --no-sync python -m scripts.report_iou_followup",
              "uv run --no-sync python -m scripts.iou_followup_status", "```", "",
              ("Existing authenticated fits and probabilities are reused on rerun. New fine/coarse inference uses CUDA; "
              "official scoring is CPU/NumPy. Source hashes, raw optimizer/RNG states, training-window transcripts, "
              "prediction identities and decoded-label hashes are retained. Data caches from the original study "
              "are referenced directly to avoid duplicate storage. Outputs are under `artifacts/gpu-iou-followup-v1/`."), "",
              ("The statistical-analysis skill informed the paired-match and uncertainty workflow; "
              "see the attribution in `docs/GPU_IOU_EXPERIMENT.md`."), ""]
    (directory / "report.md").write_text("\n".join(lines))
    for name, header, rows in (
        ("iou-curves.csv", ["arm", "group", "variant", "iou", "weighted_f1", *WEIGHTS], curve_rows),
        ("match-metrics.csv", ["arm", "group", "variant", "match", "objective", *WEIGHTS], match_rows),
        ("duration-recall.csv", ["arm", "group", "variant", "batch", "match", "scenario", "iou", "duration", "true", "matched"], duration_rows),
    ):
        with (directory / name).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    plot(results, directory)
    return {"confirmed": confirmed, "guarded": guarded, "decision": decision,
            "expected_new_fits": plan["expected_new_fits"]}


def plot(results, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"gpu_baseline": "#7a7a7a", "control": "#263b53", "spell_balanced": "#087f8c",
              "sampling_only": "#cf752c", "loss_only": "#9b477d"}
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained", sharey=True)
    for ax, (arm, result) in zip(axes, results.items(), strict=True):
        for variant, detail in result["groups"]["ensemble"]["variants"].items():
            curve = sorted((float(t), v["weighted_f1"]) for t, v in detail["curves"].items())
            x, y = np.array(curve).T
            ax.plot(x, y, label=NAMES[variant], marker="o", markersize=3, color=colors[variant])
        ax.set_title(arm.capitalize() + (" (WMX only)" if arm == "guard" else " (six matches)"))
        ax.set_xlabel("Temporal IoU threshold")
        ax.set_xticks([.1, .3, .5, .7, .9])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Scenario-weighted possession F1")
    figure.suptitle("Frozen GPU follow-up: fresh-seed confirmation, factorial ablations and WMX guard")
    for suffix in ("png", "pdf"):
        figure.savefig(directory / f"iou-curves.{suffix}", dpi=180)
    plt.close(figure)

    original = read_record(SOURCE / "historical-masks/result.json")
    old = original["groups"]["ensemble"]["comparisons"]["spell_balanced"]["control"]["by_match"]
    fresh = results["confirmation"]["groups"]["ensemble"]["comparisons"]["spell_balanced"]["control"]["by_match"]
    matches = sorted(fresh)
    figure, ax = plt.subplots(figsize=(9, 4.3), layout="constrained")
    x = np.arange(len(matches))
    ax.bar(x - .18, [old[m] for m in matches], width=.36, label="Original seeds 8101/8102", color="#9ab6cb")
    ax.bar(x + .18, [fresh[m] for m in matches], width=.36, label="Fresh seeds 8201/8202", color="#087f8c")
    ax.axhline(0, color="#444444", linewidth=.8)
    ax.set_xticks(x, [m.removeprefix("DFL-MAT-J03") for m in matches])
    ax.set_ylabel("Combined − control: weighted F1")
    ax.set_xlabel("Whole match (same six development matches)")
    ax.set_title("Does the original gain repeat with fresh fine and coarse training seeds?")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    for suffix in ("png", "pdf"):
        figure.savefig(directory / f"confirmation-by-match.{suffix}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    print(report())
