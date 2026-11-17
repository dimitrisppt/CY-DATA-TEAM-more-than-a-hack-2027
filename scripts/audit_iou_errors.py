"""Operational split/merge/team/timing audit of the completed paired predictions."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from experiments.campaign_protocol import publish
from experiments.gpu_iou_evaluation import load_reference
from experiments.iou_followup_training import OUTPUT, SOURCE, freeze
from experiments.robustness import _digest
from possession.storage import sha256_file


def greedy_pairs(overlaps, threshold=.5):
    first, second, pairs = set(), set(), []
    for iou, i, j in overlaps:
        if iou < threshold:
            break
        if i not in first and j not in second:
            first.add(i)
            second.add(j)
            pairs.append((i, j, float(iou)))
    return pairs


def audit_half(gt, pred):
    runs = official.valid_runs(gt)
    filled = official.fill_abstentions(pred, runs)
    gold, predicted = official.possession_spells(gt, runs), official.possession_spells(filled, runs)
    overlaps = official.spell_overlaps(gold, predicted)
    pairs = greedy_pairs(overlaps)
    matched = {j: (i, iou) for i, j, iou in pairs}
    gt_degree = Counter(i for _, i, _ in overlaps)
    pred_degree = Counter(j for _, _, j in overlaps)
    by_prediction = {j: [(iou, i) for iou, i, p in overlaps if p == j] for j in range(len(predicted))}
    rows = []
    for j, (start, stop, team) in enumerate(predicted):
        same = int(np.sum(gt[start:stop] == team))
        opponent = int(np.sum(gt[start:stop] == 1 - team))
        dead = int(np.sum(gt[start:stop] == official.D))
        best = by_prediction[j][0] if by_prediction[j] else (0., None)
        split = any(gt_degree[i] > 1 for _, i in by_prediction[j])
        merge = pred_degree[j] > 1
        wrong = opponent > max(same, dead)
        category = ("matched" if j in matched else "wrong_team" if wrong else "split" if split
                    else "merge" if merge else "timing" if same else "phantom")
        target = matched[j][0] if j in matched else best[1]
        rows.append({"prediction_index": j, "start_frame": int(start), "stop_frame": int(stop),
                     "start_seconds": start / 25., "stop_seconds": stop / 25., "team": "HA"[team],
                     "duration_seconds": (stop - start) / 25., "matched": j in matched,
                     "category": category, "split_overlap_flag": split, "merge_overlap_flag": merge,
                     "dominant_wrong_team_flag": wrong, "same_team_frames": same,
                     "opponent_frames": opponent, "dead_frames": dead, "best_same_team_iou": best[0],
                     "target_index": target,
                     "start_offset_seconds": (start - gold[target][0]) / 25. if target is not None else None,
                     "stop_offset_seconds": (stop - gold[target][1]) / 25. if target is not None else None})
    expected = official.spell_scores(gt, filled, runs, ious=(.5,))["per_iou"][.5]
    if sum(not r["matched"] for r in rows) != expected["n_spurious"] or len(pairs) != expected["n_matched"]:
        raise AssertionError("audit does not reproduce official greedy matching")
    return rows, predicted, gold, pairs


def paired_audit(gt, control, candidate):
    left, a, gold, left_matches = audit_half(gt, control)
    right, b, _, right_matches = audit_half(gt, candidate)
    correspondence = greedy_pairs(official.spell_overlaps(a, b))
    paired_left = {i: j for i, j, _ in correspondence}
    paired_right = {j: i for i, j, _ in correspondence}
    changes = Counter()
    for j, row in enumerate(right):
        i = paired_right.get(j)
        row["paired_control_index"] = i
        row["change_status"] = ("matched" if row["matched"] else "new_spurious" if i is None
                                else "became_spurious" if left[i]["matched"] else "shared_spurious")
        if not row["matched"]:
            changes[row["change_status"]] += 1
    for i, row in enumerate(left):
        j = paired_left.get(i)
        row["paired_candidate_index"] = j
        row["change_status"] = ("matched" if row["matched"] else "removed_spurious" if j is None
                                else "became_matched" if right[j]["matched"] else "shared_spurious")
        if not row["matched"] and row["change_status"] != "shared_spurious":
            changes[row["change_status"]] += 1
    left_fp, right_fp = sum(not r["matched"] for r in left), sum(not r["matched"] for r in right)
    if right_fp - left_fp != (changes["new_spurious"] + changes["became_spurious"]
                              - changes["removed_spurious"] - changes["became_matched"]):
        raise AssertionError("paired spurious accounting differs")
    recovered = {i for i, _, _ in right_matches} - {i for i, _, _ in left_matches}
    lost = {i for i, _, _ in left_matches} - {i for i, _, _ in right_matches}
    changes["recovered_gt"] = len(recovered)
    changes["lost_gt"] = len(lost)
    changes["recovered_short_gt"] = sum(gold[i][1] - gold[i][0] < 50 for i in recovered)
    changes["lost_short_gt"] = sum(gold[i][1] - gold[i][0] < 50 for i in lost)
    return left, right, dict(changes)


def run(output=OUTPUT):
    plan = freeze(output)
    directory = output / "error-audit"
    directory.mkdir(parents=True, exist_ok=True)
    publish(directory / "protocol.json", {
        "followup_protocol_sha256": _digest(plan), "script_sha256": sha256_file(Path(__file__)),
        "rules": plan["prespecified_analysis"]["audit"],
        "note": "overlap flags are descriptive, may coexist, and include any positive overlap; "
                "dominant wrong-team means opponent overlap exceeds both same-team and dead overlap; "
                "cross-method correspondence uses same-team greedy IoU .5; offsets use matched or best same-team GT"})
    labels, manifest = load_reference(SOURCE / "historical-masks/reference")
    rows, changes = [], {}
    for source in manifest["records"]:
        if source["batch"] != "batch_1" or source["scenario"] != "clean":
            continue
        keys = source["group_prediction_keys"]["ensemble"]
        left, right, delta = paired_audit(labels[source["ground_truth_key"]], labels[keys["control"]],
                                         labels[keys["spell_balanced"]])
        changes[f"{source['match']}_half{source['period']}"] = delta
        for variant, events in (("control", left), ("spell_balanced", right)):
            rows.extend({"match": source["match"], "period": source["period"], "variant": variant, **r}
                        for r in events)
    summary = {}
    for match in ["all", *manifest["assessment_matches"]]:
        summary[match] = {}
        for variant in ("control", "spell_balanced"):
            selected = [r for r in rows if r["variant"] == variant and (match == "all" or r["match"] == match)]
            matched = [r for r in selected if r["matched"]]
            summary[match][variant] = {
                "predictions": len(selected), "matched": len(matched),
                "spurious": len(selected) - len(matched),
                "spurious_categories": dict(Counter(r["category"] for r in selected if not r["matched"])),
                "spurious_duration_under_2s": sum(not r["matched"] and r["duration_seconds"] < 2 for r in selected),
                "matched_median_absolute_start_seconds": float(np.median([abs(r["start_offset_seconds"]) for r in matched])),
                "matched_median_absolute_stop_seconds": float(np.median([abs(r["stop_offset_seconds"]) for r in matched])),
                "matched_median_signed_start_seconds": float(np.median([r["start_offset_seconds"] for r in matched])),
                "matched_median_signed_stop_seconds": float(np.median([r["stop_offset_seconds"] for r in matched])),
            }
    if (summary["all"]["control"]["spurious"], summary["all"]["spell_balanced"]["spurious"]) != (457, 533):
        raise AssertionError("audit must reproduce completed study counts exactly")
    totals = dict(sum((Counter(v) for v in changes.values()), Counter()))
    result = {"summary": summary, "paired_changes_by_half": changes, "paired_changes": totals,
              "n_matches": 6, "scenario": "clean", "batch": "batch_1", "iou": .5,
              "independent_test": False, "causal_error_labels": False}
    publish(directory / "result.json", result)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (directory / "events.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plot_woh(labels, manifest, rows, directory)
    return result


def plot_woh(labels, manifest, rows, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    match = "DFL-MAT-J03WOH"
    counts = Counter()
    for r in rows:
        if r["match"] == match and not r["matched"]:
            counts[r["period"], int(r["start_seconds"] // 60)] += 1 if r["variant"] == "spell_balanced" else -1
    period, minute = min(counts, key=lambda k: (-counts[k], *k))
    source = next(r for r in manifest["records"] if r["match"] == match and r["period"] == period
                  and r["batch"] == "batch_1" and r["scenario"] == "clean")
    keys = source["group_prediction_keys"]["ensemble"]
    start, stop = minute * 1500, min((minute + 1) * 1500, source["n_frames"])
    matrix = np.stack([labels[k][start:stop] for k in
                       (source["ground_truth_key"], keys["control"], keys["spell_balanced"])])
    figure, ax = plt.subplots(figsize=(12, 3.5), layout="constrained")
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=ListedColormap(
        ["#1976d2", "#e76f51", "#d9d9d9", "#333333"]), vmin=0, vmax=3,
        extent=(start / 25., stop / 25., 2.5, -.5))
    ax.set_yticks([0, 1, 2], ["Ground truth", "Control", "Spell-balanced"])
    ax.set_xlabel(f"Seconds from start of half {period}")
    ax.set_title(f"WOH: fixed 60-second bin with largest spurious-count increase ({counts[period, minute]:+d})")
    bar = figure.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], pad=.02)
    bar.ax.set_yticklabels(["Home", "Away", "Dead", "Unscored"])
    for suffix in ("png", "pdf"):
        figure.savefig(directory / f"woh-timeline.{suffix}", dpi=180)
    plt.close(figure)
    publish(directory / "woh-window.json", {"period": period, "start_seconds": start / 25.,
                                            "stop_seconds": stop / 25., "spurious_increase": counts[period, minute]})


if __name__ == "__main__":
    run()
