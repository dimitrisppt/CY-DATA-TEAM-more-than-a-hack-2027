"""Authenticated, parallel inference and paired scoring for the frozen follow-up."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from evaluation import possession_metrics as official
from experiments import campaign_evaluation as inputs
from experiments.campaign_decode import decode_arrays
from experiments.campaign_protocol import WEIGHTS, publish, read_record
from experiments.coarse_context import blend_coarse_context, predict_coarse_native
from experiments.evaluate import _join_with_x
from experiments.gpu_iou import ROOT, _completed, _identity, status, verify_hash
from experiments.gpu_iou_evaluation import _blend, _predict
from experiments.gpu_runtime import configure_device
from experiments.iou_followup_training import MODES, OUTPUT, SOURCE, freeze
from experiments.possession_refinement import comparison_gate
from experiments.robustness import _digest
from possession.predict import load_model_checkpoint
from possession.schema import FeatureBatch
from possession.storage import load_half, sha256_file
from scripts.analyze_partial_iou import THRESHOLDS, score_iou
from scripts.evaluate_reference import _score_rows


def verify_training(plan, output):
    record = read_record(output / "training.json")
    if record is None or record["protocol_sha256"] != _digest(plan):
        raise ValueError("complete authenticated training manifest required")
    rows = list(record["fits"])
    if plan["arm"] == "ablation":
        rows += [dict(r, prediction_root=str(SOURCE / "historical-masks"))
                 for r in read_record(SOURCE / "training.json")["fits"]
                 if r["identity"]["role"] != "masked_teacher"]
    expected = {(f, s, m["id"], role) for f in range(len(plan["folds"]))
                for s in plan["seeds"] for m in plan["selection"]["members"]
                for role in ("parent", *plan["variants"])}
    expected |= {(f, s, "coarse_player_only", "coarse")
                 for f in range(len(plan["folds"])) for s in plan["seeds"]}
    index, transcripts, checks = {}, {}, []
    original = read_record(SOURCE / "protocol.json")
    for row in rows:
        identity = row["identity"]
        key = tuple(identity[k] for k in ("fold", "base_seed", "member", "role"))
        if key not in expected or key in index:
            raise ValueError("unexpected or duplicate training fit")
        directory = Path(row["directory"])
        is_old = plan["arm"] == "ablation" and key[3] not in MODES
        root = SOURCE if is_old else output
        if directory != root / "fits" / key[3] / f"fold_{key[0]}" / f"seed_{key[1]}" / key[2]:
            raise ValueError("fit path differs from identity")
        saved = _completed(directory, identity)
        if saved != {k: v for k, v in row.items() if k not in ("directory", "prediction_root")}:
            raise ValueError("fit record differs from training manifest")
        base = _identity(original if plan["arm"] == "ablation" else plan, *key)
        if any(identity.get(k) != v for k, v in base.items()):
            raise ValueError("fit ancestry differs from frozen protocol")
        if key[3] in MODES and identity.get("followup_protocol_sha256") != _digest(plan):
            raise ValueError("factorial adapter protocol differs")
        steps = 7200 if key[3] in ("parent", "coarse") else 8400
        payload = torch.load(directory / f"state_{steps:06d}.pt", map_location="cpu", weights_only=True)
        base_state = payload.get("campaign", payload)
        state = base_state["state"]
        if base_state["signature"]["identity"] != identity or state["steps"] != steps or row["steps"] != steps:
            raise ValueError("actual saved training identity or step count differs")
        lines = state["training_transcript"].decode().splitlines()
        used = {line.split(":", 1)[0].rsplit("_half", 1)[0] for line in lines}
        if len(lines) != steps or not used <= set(plan["folds"][key[0]]["train"]):
            raise ValueError("actual training transcript violates assessment exclusions")
        if used & set(plan["folds"][key[0]]["validation"]):
            raise ValueError("validation leakage")
        model, exported = load_model_checkpoint(directory / "model.pt")
        del model
        weights = payload["refinement"]["tail_ema"] if "refinement" in payload else state["model"]
        if any(not torch.equal(exported["state_dict"][k], v) or not torch.isfinite(v).all()
               for k, v in weights.items()):
            raise ValueError("exported weights differ from completed state")
        if exported["training_config"]["train_matches"] != identity["train_matches"]:
            raise ValueError("exported training provenance differs")
        if "refinement" in payload:
            refinement = payload["refinement"]
            if refinement["tail_ema_updates"] != 200 or refinement["counters"]["updates"] != 1200:
                raise ValueError("continuation or EMA budget differs")
            if key[3] in MODES and refinement["config"].get("factorial_mode") != key[3]:
                raise ValueError("factorial adapter identity differs")
        transcripts[key] = state["training_transcript"]
        index[key] = row
        checks.append({"key": key, "steps": steps, "actual_train_matches": sorted(used)})
    if set(index) != expected:
        raise ValueError("incomplete training grid")
    for key, row in index.items():
        if key[3] in ("parent", "coarse"):
            continue
        parent_key = (*key[:3], "parent")
        if (row["identity"]["parent_state_sha256"] != index[parent_key]["files"]["state_007200.pt"]
                or not transcripts[key].startswith(transcripts[parent_key])):
            raise ValueError("continuation does not descend from matching raw parent")
        if key[3] in MODES:
            paired = "control" if key[3] == "loss_only" else "spell_balanced"
            if transcripts[key] != transcripts[(*key[:3], paired)]:
                raise ValueError("factorial training windows differ from paired condition")
    publish(output / "training-audit.json", {"protocol_sha256": _digest(plan), "fits": checks,
                                            "all_fits_verified": True, "paired_windows_verified": True})
    return index


def assessment_inputs(plan):
    matches = sorted({m for fold in plan["folds"] for m in fold["validation"]})
    records = []
    for batch, scenarios in plan["batches"].items():
        directory = Path(plan["input_root"]) / batch
        if plan["arm"] == "guard":
            rows = inputs.prepare_inputs(plan["data_dir"], plan["target_dir"], directory, matches, scenarios)
        else:
            # Read the existing complete cache without rewriting its manifest.
            path = directory / "input-manifest.json"
            verify_hash(path, json.loads(path.with_suffix(".sha256.json").read_text())["sha256"])
            manifest = json.loads(path.read_text())
            signature = manifest["signature"]
            if (manifest["status"] != "complete" or signature["matches"] != matches
                    or signature["scenarios"] != scenarios
                    or manifest["signature_sha256"] != _digest(signature)):
                raise ValueError("upstream assessment signature differs")
            for name, digest in signature["prepared_sha256"].items():
                verify_hash(Path(plan["data_dir"]) / name, digest)
            rows = [{**r, "input_path": str(inputs._input_path(directory, r["match"], r["period"], r["scenario"]))}
                    for r in manifest["records"]]
            expected = {(m, p, s) for m in matches for p in (1, 2) for s in scenarios}
            if len(rows) != len(expected) or {(r["match"], r["period"], r["scenario"]) for r in rows} != expected:
                raise ValueError("incomplete input grid")
            for row in rows:
                if row["input_identity"] != manifest["signature_sha256"]:
                    raise ValueError("input identity differs")
                verify_hash(row["input_path"], row["input_sha256"])
        records.extend({"batch": batch, **r} for r in rows)
    return records

def infer_fold(plan, output, fits, records, selected_fold):
    """Run native-rate CUDA inference with identical masks for every method."""
    output = Path(output)
    device = configure_device(plan["resources"]["device"])
    variants = ("gpu_baseline", *plan["variants"])
    groups = ("ensemble", *(f"fine_seed_{seed}" for seed in plan["seeds"]))
    matches = plan["folds"][selected_fold]["validation"]
    labels, references = {}, []
    members = [m["id"] for m in plan["selection"]["members"]]
    for fold_index, fold in enumerate(plan["folds"]):
        if fold_index != selected_fold:
            continue
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
            grouped_keys = {group: {} for group in groups}
            for variant in variants:
                role = "parent" if variant == "gpu_baseline" else variant
                neural = {}
                for seed in plan["seeds"]:
                    for member in members:
                        model, fit = models[seed, member, role]
                        path = (Path(fit.get("prediction_root", output)) / "inference" / batch / variant / f"seed_{seed}" / member
                                / match / f"half{period}" / f"{scenario}.npz")
                        neural[seed, member] = _predict(arrays, features, model, source, fit, path, device)
                for group in groups:
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
    directory = output / f"reference-fold{selected_fold}"
    inputs._atomic_npz(directory / "labels.npz", labels, compressed=True)
    manifest = {"schema": "followup_iou_labels_v1", "protocol_sha256": _digest(plan),
                "labels_sha256": sha256_file(directory / "labels.npz"), "records": references,
                "array_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in labels.items()},
                "scenario_weights": WEIGHTS, "scenario_order": list(WEIGHTS),
                "batches": list(plan["batches"]), "groups": list(groups), "variants": list(variants),
                "assessment_matches": matches, "label_codes": {"H": 0, "A": 1, "D": 2, "X": 3},
                "single_seed_groups": "one fine seed; both fixed coarse seeds retained",
                "target_ground_truth_used": False, "independent_test": False}
    publish(directory / "manifest.json", manifest)
    return str(directory)



def validate_grid(manifest):
    expected = {(b, m, s, p) for b in manifest["batches"] for m in manifest["assessment_matches"]
                for s in WEIGHTS for p in (1, 2)}
    actual = [(r["batch"], r["match"], r["scenario"], r["period"]) for r in manifest["records"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("incomplete reference grid")
    for row in manifest["records"]:
        if set(row["group_prediction_keys"]) != set(manifest["groups"]):
            raise ValueError("incomplete prediction groups")
        if any(set(v) != set(manifest["variants"]) for v in row["group_prediction_keys"].values()):
            raise ValueError("incomplete prediction variants")


def aggregate_scores(scores, matches):
    if set(scores) != {"batch_1", "batch_2"} or len(matches) not in (1, 6):
        raise ValueError("two mask batches and one guard or six inner matches required")
    if any(set(row["by_match"]) != set(matches) for row in scores.values()):
        raise ValueError("score match sets differ")
    by_match = {m: {s: float(np.mean([scores[b]["by_match"][m][s]["spell_f1_mean"] for b in scores]))
                    for s in WEIGHTS} for m in matches}
    means = {s: float(np.mean([r[s] for r in by_match.values()])) for s in WEIGHTS}
    return {"objective": sum(WEIGHTS[s] * means[s] for s in WEIGHTS),
            "missing": sum(WEIGHTS[s] * means[s] for s in WEIGHTS if s != "clean") / .5,
            "means": means, "by_match": by_match, "by_batch": {b: r["objective"] for b, r in scores.items()}}


def score(labels, manifest):
    validate_grid(manifest)
    result = {"schema": "followup_comprehensive_iou_v1", "protocol_sha256": manifest["protocol_sha256"],
              "reference_manifest_sha256": _digest(manifest),
              "independent_test": False, "thresholds": list(THRESHOLDS), "groups": {}}
    for group in manifest["groups"]:
        grouped = {**manifest, "records": [{**r, "prediction_keys": r["group_prediction_keys"][group]}
                                           for r in manifest["records"]]}
        summaries, variants = {}, {}
        for variant in manifest["variants"]:
            raw = {batch: _score_rows(labels, grouped, variant, batch, "inner")
                   for batch in manifest["batches"]}
            summaries[variant] = aggregate_scores(raw, manifest["assessment_matches"])
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
        for variant in [v for v in manifest["variants"] if v not in ("gpu_baseline", "control")]:
            comparisons[variant] = {}
            for reference in ("gpu_baseline", "control"):
                candidate, control = summaries[variant], summaries[reference]
                deltas = {m: sum(WEIGHTS[s] * (candidate["by_match"][m][s]
                                             - control["by_match"][m][s]) for s in WEIGHTS)
                          for m in manifest["assessment_matches"]}
                rng = np.random.default_rng(20261116)
                draws = rng.choice(list(deltas.values()), size=(5000, len(deltas)), replace=True).mean(axis=1)
                comparisons[variant][reference] = {
                    "objective_delta": candidate["objective"] - control["objective"],
                    "by_match": deltas, "descriptive_match_bootstrap_95_interval":
                    (np.quantile(draws, [.025, .975]).tolist() if len(deltas) > 1 else None), "n_matches": len(deltas)}
        gates = {}
        if group == "ensemble":
            for variant in [v for v in manifest["variants"] if v not in ("gpu_baseline", "control")]:
                if len(manifest["assessment_matches"]) == 6:
                    gate = comparison_gate(summaries["gpu_baseline"], summaries["control"], summaries[variant])
                    gate = json.loads(json.dumps(gate).replace("v6", "gpu_baseline"))
                    gate["prespecified_checks_passed"] = gate.pop("confirmation_eligible")
                else:
                    checks = {}
                    for reference in ("control", "gpu_baseline"):
                        checks[f"objective_vs_{reference}"] = (
                            summaries[variant]["objective"] >= summaries[reference]["objective"] - .01)
                        for scenario in WEIGHTS:
                            checks[f"{scenario}_vs_{reference}"] = (
                                summaries[variant]["means"][scenario] >= summaries[reference]["means"][scenario] - .01)
                    gate = {"checks": checks, "prespecified_checks_passed": all(checks.values()),
                            "automatic_promotion": False}
                gates[variant] = gate
        result["groups"][group] = {"summary": summaries, "variants": variants,
                                   "comparisons": comparisons, "prespecified_gate": gates}
    result["interpretation"] = (
        "Historically reused development matches, including WMX. Equal whole-match weighting; "
        "masks and training seeds are paired checks, not new matches. Match bootstrap intervals "
        "are descriptive. WMX is a single-match guard with no interval. No automatic promotion.")
    if "sampling_only" in manifest["variants"]:
        for group, data in result["groups"].items():
            delta = {}
            for match in manifest["assessment_matches"]:
                values = {v: sum(WEIGHTS[s] * data["summary"][v]["by_match"][match][s] for s in WEIGHTS)
                          for v in ("control", "sampling_only", "loss_only", "spell_balanced")}
                delta[match] = values["spell_balanced"] - values["sampling_only"] - values["loss_only"] + values["control"]
            draws = np.random.default_rng(20261116).choice(list(delta.values()), size=(5000, len(delta))).mean(axis=1)
            data["factorial_interaction"] = {"by_match": delta, "mean": float(np.mean(list(delta.values()))),
                                             "descriptive_match_bootstrap_95_interval": np.quantile(draws, [.025, .975]).tolist()}
    return result


def load_reference(directory):
    manifest = read_record(directory / "manifest.json")
    if manifest is None or manifest["schema"] != "followup_iou_labels_v1":
        raise ValueError("missing or unsupported follow-up reference")
    verify_hash(directory / "labels.npz", manifest["labels_sha256"])
    with np.load(directory / "labels.npz", allow_pickle=False) as archive:
        labels = {k: archive[k] for k in archive.files}
    if {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in labels.items()} != manifest["array_sha256"]:
        raise ValueError("decoded label hash differs")
    validate_grid(manifest)
    return labels, manifest


def evaluate_arm(plan, output):
    fits = verify_training(plan, output)
    if (output / "reference/manifest.json").exists():
        labels, manifest = load_reference(output / "reference")
    else:
        records = assessment_inputs(plan)
        with ProcessPoolExecutor(max_workers=min(3, len(plan["folds"])),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(infer_fold, plan, str(output), fits, records, i)
                       for i in range(len(plan["folds"]))]
            directories = [Path(f.result()) for f in futures]
        labels, references = {}, []
        for directory in directories:
            part, manifest = load_reference(directory)
            if set(part) & set(labels):
                raise ValueError("overlapping validation folds")
            labels.update(part)
            references.extend(manifest["records"])
        manifest["assessment_matches"] = sorted({m for f in plan["folds"] for m in f["validation"]})
        manifest["records"] = references
        directory = output / "reference"
        inputs._atomic_npz(directory / "labels.npz", labels, compressed=True)
        manifest["labels_sha256"] = sha256_file(directory / "labels.npz")
        manifest["array_sha256"] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in labels.items()}
        publish(directory / "manifest.json", manifest)
    if (manifest["protocol_sha256"] != _digest(plan)
            or manifest["assessment_matches"] != sorted({m for f in plan["folds"] for m in f["validation"]})
            or manifest["groups"] != ["ensemble", *(f"fine_seed_{s}" for s in plan["seeds"])]
            or manifest["variants"] != ["gpu_baseline", *plan["variants"]]):
        raise ValueError("reference differs from frozen arm")
    status(output, "scoring")
    result = score(labels, manifest)
    result["arm"] = plan["arm"]
    publish(output / "result.json", result)
    status(output, "complete", status="complete")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    plan = freeze(output)
    # The analysis rules are already frozen in protocol.json. Pin this adapter
    # before any newly trained checkpoints are scored.
    source = {str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__))}
    publish(output / "evaluation-protocol.json", {
        "protocol_sha256": _digest(plan), "source_sha256": source,
        "prespecified_analysis": plan["prespecified_analysis"]})
    if args.plan_only:
        return
    if args.wait:
        while not all((output / arm / "training.json").exists() for arm in plan["arms"]):
            time.sleep(30)
    configure_device("cuda")
    for arm in ("confirmation", "ablation", "guard"):
        evaluate_arm(plan["arms"][arm], output / arm)
    status(output, "complete", status="complete", completed_fits=80)


if __name__ == "__main__":
    main()
