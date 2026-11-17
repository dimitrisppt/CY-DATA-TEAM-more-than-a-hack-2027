"""Comprehensive, paired whole-match IoU evaluation of fresh GPU fits."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluation import possession_metrics as official
from experiments import campaign_evaluation as inputs
from experiments.campaign_decode import decode_arrays
from experiments.campaign_protocol import WEIGHTS, publish, read_record
from experiments.coarse_context import blend_coarse_context, predict_coarse_native
from experiments.ensemble_polish_search import _blend_pairs
from experiments.evaluate import _join_with_x
from experiments.gpu_iou import ROOT, _completed, _identity, status, verify_hash
from experiments.possession_refinement import aggregate_scores, comparison_gate
from experiments.robustness import _digest
from possession.predict import load_model_checkpoint, predict_features
from possession.schema import FeatureBatch
from possession.storage import load_half, sha256_file, write_json
from scripts.analyze_partial_iou import THRESHOLDS, score_iou
from scripts.evaluate_reference import _score_rows

VARIANTS = ("gpu_baseline", "control", "spell_balanced", "masked_teacher")
GROUPS = ("ensemble", "fine_seed_8101", "fine_seed_8102")


def verify_training(plan, output):
    """Verify the full grid, split ancestry, checkpoints, and matched transcripts."""
    record = read_record(output / "training.json")
    if record is None or record["protocol_sha256"] != _digest(plan):
        raise ValueError("complete authenticated training.json is required before assessment")
    expected = {(fold, seed, member["id"], role)
                for fold in range(3) for seed in plan["seeds"]
                for member in plan["selection"]["members"]
                for role in ("parent", *plan["variants"])}
    expected |= {(fold, seed, "coarse_player_only", "coarse")
                 for fold in range(3) for seed in plan["seeds"]}
    index, checks = {}, []
    for row in record["fits"]:
        identity = row["identity"]
        key = (identity["fold"], identity["base_seed"], identity["member"], identity["role"])
        if key in index or key not in expected:
            raise ValueError("training grid has duplicate or unexpected fits")
        directory = Path(row["directory"])
        canonical = output / "fits" / key[3] / f"fold_{key[0]}" / f"seed_{key[1]}" / key[2]
        if directory != canonical:
            raise ValueError("fit directory does not match its declared identity")
        fit = _completed(directory, identity)
        if fit != {k: v for k, v in row.items() if k != "directory"}:
            raise ValueError("training manifest differs from saved fit record")
        base = _identity(plan, *key)
        if any(identity.get(k) != v for k, v in base.items()):
            raise ValueError("fit ancestry differs from protocol")
        expected_steps = 7200 if key[3] in ("parent", "coarse") else 8400
        if row["steps"] != expected_steps:
            raise ValueError("fit did not complete its fixed update schedule")
        state_path = directory / f"state_{expected_steps:06d}.pt"
        payload = torch.load(state_path, map_location="cpu", weights_only=True)
        base_state = payload.get("campaign", payload)
        state = base_state["state"]
        if base_state["signature"]["identity"] != identity or state["steps"] != expected_steps:
            raise ValueError("actual checkpoint signature differs from fit ancestry")
        lines = state["training_transcript"].decode().splitlines()
        actual_matches = {line.split(":", 1)[0].rsplit("_half", 1)[0] for line in lines}
        if len(lines) != expected_steps or not actual_matches <= set(identity["train_matches"]):
            raise ValueError("actual sampled windows include an excluded match or wrong step count")
        if any(not torch.isfinite(t).all() for t in state["model"].values()):
            raise ValueError("nonfinite trained weights")
        model, saved = load_model_checkpoint(directory / "model.pt")
        del model
        if saved["training_config"]["train_matches"] != identity["train_matches"]:
            raise ValueError("exported model training split differs")
        weights = payload["refinement"]["tail_ema"] if "refinement" in payload else state["model"]
        if any(not torch.equal(saved["state_dict"][k], v) for k, v in weights.items()):
            raise ValueError("exported model differs from completed training state")
        if "refinement" in payload and payload["refinement"]["tail_ema_updates"] != 200:
            raise ValueError("refinement EMA has the wrong number of updates")
        index[key] = {**row, "transcript": state["training_transcript"]}
        checks.append({"fold": key[0], "seed": key[1], "member": key[2], "role": key[3],
                       "actual_train_matches": sorted(actual_matches), "steps": expected_steps,
                       "excluded_matches_verified": True, "export_matches_state": True})
    if set(index) != expected or len(index) != 54:
        raise ValueError("all 18 parents and 36 continuations must complete before assessment")
    for fold, seed, member, role in sorted(expected):
        if role in ("parent", "coarse"):
            continue
        row, parent = index[fold, seed, member, role], index[fold, seed, member, "parent"]
        if (row["identity"]["parent_state_sha256"] != parent["files"]["state_007200.pt"]
                or not row["transcript"].startswith(parent["transcript"])):
            raise ValueError("continuation is not bound to its matching raw parent")
        if role == "masked_teacher" and (
                row["identity"]["teacher_sha256"] != parent["files"]["ema.pt"]
                or row["transcript"] != index[fold, seed, member, "control"]["transcript"]):
            raise ValueError("teacher ancestry or matched control sampling differs")
    for row in index.values():
        row.pop("transcript")
    publish(output / "training-audit.json", {"protocol_sha256": _digest(plan), "fits": checks,
                                             "all_54_fits_verified": True,
                                             "control_teacher_sampling_matched": True})
    return index


def _predict(arrays, features, model, source, fit, path, device):
    identity = {"checkpoint_sha256": fit["checkpoint_sha256"], "fit": fit["identity"],
                "input_sha256": source["input_sha256"], "device": device, "center_size": 2048}
    meta = read_record(path.with_suffix(".json"))
    if meta is not None:
        if meta["identity"] != identity:
            raise ValueError("cached GPU inference identity differs")
        verify_hash(path, meta["sha256"])
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    prediction = predict_features(features, model, device=device, center_size=2048)
    result = {"neural_live": prediction["live_probability"],
              "neural_home": prediction["home_probability"],
              "neural_boundary": prediction["boundary"]}
    n = len(arrays["frame_id"])
    if any(not np.isfinite(value).all() or len(value) != n for value in result.values()):
        raise ValueError("GPU inference returned invalid scores")
    inputs._atomic_npz(path, result, compressed=True)
    publish(path.with_suffix(".json"), {"identity": identity, "sha256": sha256_file(path)})
    return result


def _blend(first, second, weight):
    # The caller passes predictions for one already authenticated input row.
    # Reuse the historical arithmetic without inventing cached-record metadata.
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("blend weight must be a probability")
    for name in ("neural_live", "neural_home", "neural_boundary"):
        if first[name].shape != second[name].shape:
            raise ValueError("blend prediction shapes differ")
        if not np.isfinite(first[name]).all() or not np.isfinite(second[name]).all():
            raise ValueError("blend predictions must be finite")
    return _blend_pairs([({"arrays": first}, {"arrays": second})], weight)[0]["arrays"]


def infer(plan, output, fits):
    """Run native-rate CUDA inference with identical masks for every method."""
    device = plan["resources"]["device"]
    matches = sorted({m for fold in plan["folds"] for m in fold["validation"]})
    records = []
    for batch, scenarios in plan["batches"].items():
        status(output, "preparing_assessment_inputs", batch=batch)
        rows = inputs.prepare_inputs(plan["data_dir"], plan["target_dir"],
                                     output / "assessment" / batch, matches, scenarios)
        records.extend({"batch": batch, **r} for r in rows)
    labels, references = {}, []
    members = [m["id"] for m in plan["selection"]["members"]]
    for fold_index, fold in enumerate(plan["folds"]):
        models = {}
        for key, fit in fits.items():
            if key[0] != fold_index:
                continue
            filename = "ema.pt" if key[3] == "parent" else "model.pt"
            path = Path(fit["directory"]) / filename
            verify_hash(path, fit["files"][filename])
            model, payload = load_model_checkpoint(path, device=device)
            if set(payload["training_config"]["train_matches"]) & set(fold["validation"]):
                raise ValueError("assessment match appears in model's actual training provenance")
            models[key[1:]] = (model, {**fit, "checkpoint_sha256": fit["files"][filename]})
        coarse_cache = {}
        for source in [r for r in records if r["match"] in fold["validation"]]:
            batch, match, period, scenario = (source[k] for k in ("batch", "match", "period", "scenario"))
            status(output, "gpu_inference", assessment_fold=fold_index, batch=batch,
                   match=match, period=period, scenario=scenario)
            arrays = inputs.load_arrays(source)
            features = FeatureBatch(*(arrays[k] for k in
                                      ("player_tokens", "player_mask", "global_tokens", "time_valid")))
            if (match, period) not in coarse_cache:
                frames, _ = load_half(Path(plan["data_dir"]) / f"{match}_half{period}.npz")
                coarse = []
                for seed in plan["seeds"]:
                    model, _ = models[seed, "coarse_player_only", "coarse"]
                    pred = predict_coarse_native(frames, model)
                    coarse.append({"neural_live": pred["live_probability"],
                                   "neural_home": pred["home_probability"],
                                   "neural_boundary": np.zeros((len(frames.frame_id), 3, 3))})
                coarse_cache[match, period] = _blend(coarse[0], coarse[1], .5)
                del frames
            coarse = coarse_cache[match, period]
            gt_key = f"gt_{match}_p{period}"
            if gt_key in labels and not np.array_equal(labels[gt_key], arrays["gt"]):
                raise ValueError("public truth differs across assessment scenarios")
            labels[gt_key] = arrays["gt"].astype(np.uint8)
            grouped_keys = {group: {} for group in GROUPS}
            for variant in VARIANTS:
                role = "parent" if variant == "gpu_baseline" else variant
                neural = {}
                for seed in plan["seeds"]:
                    for member in members:
                        model, fit = models[seed, member, role]
                        path = (output / "inference" / batch / variant / f"seed_{seed}" / member
                                / match / f"half{period}" / f"{scenario}.npz")
                        neural[seed, member] = _predict(arrays, features, model, source, fit, path, device)
                for group in GROUPS:
                    chosen = plan["seeds"] if group == "ensemble" else [int(group.rsplit("_", 1)[1])]
                    pair = [(_blend(neural[chosen[0], member], neural[chosen[1], member], .5)
                             if len(chosen) == 2 else neural[chosen[0], member]) for member in members]
                    mixed = _blend(pair[0], pair[1], plan["selection"]["blend_weight"])
                    blended = blend_coarse_context({**arrays, **mixed}, coarse["neural_live"],
                                                    coarse["neural_home"])
                    key = f"{group}_{variant}_{batch}_{match}_{scenario}_p{period}"
                    labels[key] = decode_arrays(blended, plan["selection"]["decoder"]).astype(np.uint8)
                    grouped_keys[group][variant] = key
            references.append({"scope": "inner", "batch": batch, "match": match, "period": period,
                               "scenario": scenario, "fold": fold_index, "ground_truth_key": gt_key,
                               "group_prediction_keys": grouped_keys, "n_frames": len(arrays["gt"]),
                               "input_sha256": source["input_sha256"]})
        del models
        gc.collect()
    directory = output / "reference"
    inputs._atomic_npz(directory / "labels.npz", labels, compressed=True)
    manifest = {"schema": "fresh_gpu_iou_labels_v1", "protocol_sha256": _digest(plan),
                "labels_sha256": sha256_file(directory / "labels.npz"), "records": references,
                "array_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in labels.items()},
                "scenario_weights": WEIGHTS, "scenario_order": list(WEIGHTS),
                "batches": list(plan["batches"]), "groups": list(GROUPS), "variants": list(VARIANTS),
                "assessment_matches": matches, "label_codes": {"H": 0, "A": 1, "D": 2, "X": 3},
                "single_seed_groups": "one fine seed; both fixed coarse seeds retained",
                "target_ground_truth_used": False, "independent_test": False}
    publish(directory / "manifest.json", manifest)
    return labels, manifest


def load_reference(directory):
    manifest = read_record(directory / "manifest.json")
    if manifest is None or manifest["schema"] != "fresh_gpu_iou_labels_v1":
        raise ValueError("unsupported GPU reference")
    verify_hash(directory / "labels.npz", manifest["labels_sha256"])
    with np.load(directory / "labels.npz", allow_pickle=False) as archive:
        labels = {k: archive[k] for k in archive.files}
    if {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in labels.items()} != manifest["array_sha256"]:
        raise ValueError("decoded label array hashes differ")
    if len(manifest["assessment_matches"]) != 6 or set(manifest["groups"]) != set(GROUPS):
        raise ValueError("incomplete assessment groups or matches")
    return labels, manifest


def score(labels, manifest):
    result = {"schema": "fresh_gpu_comprehensive_iou_v1", "protocol_sha256": manifest["protocol_sha256"],
              "reference_manifest_sha256": _digest(manifest),
              "independent_test": False, "thresholds": list(THRESHOLDS), "groups": {}}
    for group in GROUPS:
        grouped = {**manifest, "records": [{**r, "prediction_keys": r["group_prediction_keys"][group]}
                                           for r in manifest["records"]]}
        summaries, variants = {}, {}
        for variant in VARIANTS:
            raw = {batch: _score_rows(labels, grouped, variant, batch, "inner")
                   for batch in manifest["batches"]}
            summaries[variant] = aggregate_scores(raw)
            index = {(r["batch"], r["match"], r["scenario"], r["period"]): r for r in grouped["records"]}
            rows = []
            for batch in manifest["batches"]:
                for match in manifest["assessment_matches"]:
                    for scenario in WEIGHTS:
                        halves = [index[batch, match, scenario, period] for period in (1, 2)]
                        gt = _join_with_x([labels[r["ground_truth_key"]] for r in halves])
                        pred = _join_with_x([labels[r["prediction_keys"][variant]] for r in halves])
                        rows.append({"batch": batch, "match": match, "scenario": scenario,
                                     "iou": score_iou(gt, pred)})
            curves = {}
            for threshold in THRESHOLDS:
                key = str(threshold)
                by_match = {match: {s: float(np.mean([r["iou"][key]["f1"] for r in rows
                                                      if r["match"] == match and r["scenario"] == s]))
                                    for s in WEIGHTS} for match in manifest["assessment_matches"]}
                means = {s: float(np.mean([m[s] for m in by_match.values()])) for s in WEIGHTS}
                curves[key] = {"by_match": by_match, "scenario_f1": means,
                               "weighted_f1": sum(WEIGHTS[s] * means[s] for s in WEIGHTS)}
            objective = np.mean([curves[str(t)]["weighted_f1"] for t in official.SPELL_IOUS])
            if not np.isclose(objective, summaries[variant]["objective"], atol=1e-12, rtol=0):
                raise AssertionError("IoU sweep does not reproduce official scoring")
            variants[variant] = {"curves": curves, "records": rows, "official_scores": raw}
        comparisons = {}
        for variant in ("spell_balanced", "masked_teacher"):
            comparisons[variant] = {}
            for reference in ("gpu_baseline", "control"):
                candidate, control = summaries[variant], summaries[reference]
                deltas = {m: sum(WEIGHTS[s] * (candidate["by_match"][m][s]
                                             - control["by_match"][m][s]) for s in WEIGHTS)
                          for m in manifest["assessment_matches"]}
                rng = np.random.default_rng(20261116)
                draws = rng.choice(list(deltas.values()), size=(5000, 6), replace=True).mean(axis=1)
                comparisons[variant][reference] = {
                    "objective_delta": candidate["objective"] - control["objective"],
                    "by_match": deltas, "descriptive_match_bootstrap_95_interval":
                    np.quantile(draws, [.025, .975]).tolist(), "n_matches": 6}
        gates = {}
        if group == "ensemble":
            for variant in ("spell_balanced", "masked_teacher"):
                gate = comparison_gate(summaries["gpu_baseline"], summaries["control"], summaries[variant])
                # The reused screen's implementation calls its baseline v6.
                # Explicitly relabel it: these are newly rebuilt GPU weights.
                gate = json.loads(json.dumps(gate).replace("v6", "gpu_baseline"))
                gates[variant] = gate
        result["groups"][group] = {"summary": summaries, "variants": variants,
                                   "comparisons": comparisons, "development_screen": gates}
    result["interpretation"] = ("Six historically reused development matches; equal match weighting. "
                                 "Mask batches and fine seeds are paired checks, not independent samples. "
                                 "Bootstrap intervals are descriptive. Fresh CUDA parents; no historical "
                                 "CPU checkpoint parity claim. No automatic model promotion.")
    return result


def write_report(result, output):
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / "result.json", result)
    with (report / "iou-curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "variant", "iou", "weighted_f1", *WEIGHTS])
        for group, data in result["groups"].items():
            for variant, detail in data["variants"].items():
                for threshold, curve in detail["curves"].items():
                    writer.writerow([group, variant, threshold, curve["weighted_f1"],
                                     *(curve["scenario_f1"][s] for s in WEIGHTS)])
    with (report / "match-metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "variant", "match", "objective", *WEIGHTS])
        for group, data in result["groups"].items():
            for variant, detail in data["summary"].items():
                for match, scores in detail["by_match"].items():
                    writer.writerow([group, variant, match,
                                     sum(WEIGHTS[s] * scores[s] for s in WEIGHTS),
                                     *(scores[s] for s in WEIGHTS)])
    with (report / "duration-recall.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "variant", "batch", "match", "scenario", "iou",
                         "duration", "true_events", "matched", "recall"])
        for group, data in result["groups"].items():
            for variant, detail in data["variants"].items():
                for row in detail["records"]:
                    for threshold, metrics in row["iou"].items():
                        for duration, count in metrics["duration_bins"].items():
                            writer.writerow([group, variant, row["batch"], row["match"], row["scenario"],
                                             threshold, duration, count["gt"], count["matched"],
                                             count["matched"] / count["gt"] if count["gt"] else 0.])
    lines = ["# Fresh GPU IoU experiment", "", result["interpretation"], "",
             ("Completed: 18 parent fits and 36 matched continuation fits; three folds, two fine "
              "architectures, two seeds, four scenarios, two mask batches, and nine IoU thresholds."),
             "", ("The official objective averages IoU 0.3–0.7 with scenario weights "
                  "0.50 clean / 0.20 short / 0.15 long / 0.15 replay."), ""]
    for group, data in result["groups"].items():
        lines += [f"## {group}", "", "| Method | Objective | Missing ball | Clean |",
                  "|---|---:|---:|---:|"]
        for variant, row in data["summary"].items():
            lines.append(f"| {variant} | {row['objective']:.6f} | {row['missing']:.6f} "
                         f"| {row['means']['clean']:.6f} |")
        lines += ["", "| Candidate vs control | Objective delta | Descriptive 95% interval |",
                  "|---|---:|---|"]
        for variant, rows in data["comparisons"].items():
            row = rows["control"]
            lo, hi = row["descriptive_match_bootstrap_95_interval"]
            lines.append(f"| {variant} | {row['objective_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
        lines.append("")
    lines += ["## Development screen", "", ("Passing this screen calls for confirmation and the "
              "separate WMX guard; the delivered model is not replaced."), ""]
    for variant, gate in result["groups"]["ensemble"]["development_screen"].items():
        lines.append(f"- {variant}: confirmation eligible = {gate['confirmation_eligible']}")
    lines += ["", ("All per-match, per-scenario, per-seed and per-IoU scores are in `result.json`. "
              "Frame macro F1 and transition F1 are included in `official_scores`. "
              "`duration-recall.csv` includes the event counts at every IoU threshold."), "",
              "![Weighted temporal-IoU curves](iou-curves.png)", ""]
    lines += ["## Workflow attribution", "", ("The matched-unit and uncertainty reporting workflow "
              "used the statistical-analysis skill from Kassis, T., Agarwal, V., He, Y., Patel, D., "
              "and Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural "
              "Knowledge for Research Agents*, [arXiv:2609.00065]"
              "(https://doi.org/10.48550/arXiv.2609.00065)."), ""]
    (report / "report.md").write_text("\n".join(lines))
    plot(result, report)


def plot(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    data = result["groups"]["ensemble"]
    for variant, detail in data["variants"].items():
        axes[0].plot(THRESHOLDS, [detail["curves"][str(t)]["weighted_f1"] for t in THRESHOLDS],
                     marker="o", markersize=4, label=variant.replace("_", " "))
    axes[0].set(xlabel="Temporal IoU threshold", ylabel="Weighted possession F1", ylim=(0, 1),
                title="Six development matches · two fine seeds")
    axes[0].legend(fontsize=8)
    variants = ["spell_balanced", "masked_teacher"]
    matches = sorted(data["summary"]["control"]["by_match"])
    x = np.arange(len(matches))
    for i, variant in enumerate(variants):
        values = data["comparisons"][variant]["control"]["by_match"]
        axes[1].bar(x + (i - .5) * .35, [values[m] for m in matches], width=.35,
                    label=variant.replace("_", " "))
    axes[1].axhline(0, color="black", linewidth=.7)
    axes[1].set(xticks=x, xticklabels=[m[-3:] for m in matches], ylabel="Objective delta vs control",
                title="Paired match differences")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.savefig(directory / "iou-curves.png", dpi=180)
    fig.savefig(directory / "iou-curves.pdf")
    plt.close(fig)


def evaluate(plan, output):
    output = Path(output)
    evaluation_sources = [Path(__file__), ROOT / "scripts/analyze_partial_iou.py",
                          ROOT / "scripts/evaluate_reference.py"]
    publish(output / "evaluation-protocol.json", {
        "training_protocol_sha256": _digest(plan), "groups": list(GROUPS), "variants": list(VARIANTS),
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in evaluation_sources},
        "thresholds": list(THRESHOLDS), "paired_match_bootstrap": {"draws": 5000, "seed": 20261116},
        "all_fits_complete_before_assessment": True})
    status(output, "verifying_training")
    fits = verify_training(plan, output)
    if (output / "reference/manifest.json").exists():
        labels, manifest = load_reference(output / "reference")
        if manifest["protocol_sha256"] != _digest(plan):
            raise ValueError("reference belongs to a different training protocol")
    else:
        labels, manifest = infer(plan, output, fits)
    status(output, "scoring")
    result = score(labels, manifest)
    write_report(result, output)
    publish(output / "result.json", result)
    status(output, "complete", status="complete", completed_fits=54,
           report=str(output / "report/report.md"))
    return result


def main():
    parser = argparse.ArgumentParser(description="Recompute GPU-study IoUs from authenticated labels")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/gpu-iou-v1")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    labels, manifest = load_reference(output / "reference")
    result = score(labels, manifest)
    saved = read_record(output / "result.json")
    if saved is not None and saved != result:
        raise ValueError("independently rescored labels differ from the saved GPU result")
    write_report(result, output)
    print(json.dumps({"label_only_reproduction": True, "saved_result_matches": saved == result}))


if __name__ == "__main__":
    main()
