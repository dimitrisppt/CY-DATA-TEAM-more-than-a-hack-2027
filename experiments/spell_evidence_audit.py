"""Descriptive short-possession evidence audit on authenticated public caches.

No fitting or candidate selection occurs. Per-frame argmax paths and fixed
sequence-term ablations are diagnostics, not oracle segmentations or deployable
candidates. Keep this module separate from the source-pinned v6 implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from experiments.campaign_decode import (
    _switch_matrix,
    _validate_config,
    decode_arrays,
    reference_from_basis,
)
from experiments.campaign_protocol import read_record
from experiments.timing_validation import validate_prediction_provenance
from possession.calibrate import CalibrationConfig, apply_temperature_calibration
from possession.decode import decode_half, emissions_from_probabilities
from possession.storage import sha256_file, write_json

DURATION_BINS = ("under_1s", "1_to_2s", "2_to_5s", "5s_and_above")


def duration_bin(frames, fps):
    seconds = frames / fps
    return DURATION_BINS[0 if seconds < 1 else 1 if seconds < 2 else 2 if seconds < 5 else 3]


def reconstruct_evidence(arrays, config):
    """Mirror decode_arrays exactly, exposing the final calibrated emissions.

    Deliberately independent copy of its blend/gate construction, checked against
    both the original decoder and saved labels on every audited half.
    """
    _validate_config(config)
    live = np.asarray(arrays["neural_live"], dtype=np.float64)
    home = np.asarray(arrays["neural_home"], dtype=np.float64)
    boundary = np.asarray(arrays["neural_boundary"], dtype=np.float64)
    seconds = np.asarray(arrays["nearest_ball_seconds"], dtype=np.float64)
    reference = reference_from_basis(
        {k: v for k, v in arrays.items() if k.startswith("ref_")},
        config["reference"] if config["reference_mode"] == "tuned" else None,
    )
    gate = np.minimum(seconds, config["transition_seconds"]) / config["transition_seconds"]
    gate[arrays["ball_valid"]] = 0.0
    if config["blend_mode"] == "static":
        weight_live = weight_owner = np.full(len(live), config["neural_weight"])
    elif config["blend_mode"] == "quality_shared":
        weight_live = weight_owner = config["visible_neural_weight"] + (
            config["missing_neural_weight"] - config["visible_neural_weight"]) * gate
    else:
        weight_live = config["visible_live_weight"] + (
            config["missing_live_weight"] - config["visible_live_weight"]) * gate
        weight_owner = config["visible_owner_weight"] + (
            config["missing_owner_weight"] - config["visible_owner_weight"]) * gate

    def blend(neural, ref, weight):
        p = np.clip(neural, 1e-6, 1 - 1e-6)
        r = np.clip(ref, 1e-6, 1 - 1e-6)
        logit = weight * (np.log(p) - np.log1p(-p)) + (1 - weight) * (
            np.log(r) - np.log1p(-r))
        output = 1 / (1 + np.exp(-np.clip(logit, -40., 40.)))
        output[weight == 0] = ref[weight == 0]
        output[weight == 1] = neural[weight == 1]
        return output

    blended_live = blend(live, reference["live_probability"], weight_live)
    blended_home = blend(home, reference["home_probability"], weight_owner)
    boundary_scores = boundary * ((weight_live + weight_owner) / 2)[:, None, None]
    boundary_scores *= config["boundary_scale"]
    final_live, final_home, final_boundary = apply_temperature_calibration(
        blended_live, blended_home, boundary_scores,
        CalibrationConfig(config["live_temperature"], config["owner_temperature"],
                          config["boundary_temperature"]),
    )
    return {
        "live": final_live, "home": final_home, "boundary": final_boundary,
        "emissions": emissions_from_probabilities(final_live, final_home),
        "blended_emissions": emissions_from_probabilities(blended_live, blended_home),
        "weight_live": weight_live, "weight_owner": weight_owner, "gate": gate,
    }


def insertion_delta(emissions, boundary, switch_cost, path, start, stop, state):
    """Score replacing exactly one GT interval, keeping decoded context fixed.

    Compare the decoded path with a gold-state insertion at known endpoints.
    This uses labels for diagnosis only. It is not a candidate-generation method.
    Signed boundary scores apply only to state changes, as in viterbi_python.
    """
    n = len(path)
    if not 0 <= start < stop <= n or state not in (0, 1, 2):
        raise ValueError("invalid insertion interval or state")
    cost = np.asarray(switch_cost, dtype=np.float64)
    if cost.ndim == 0:
        cost = np.full((3, 3), float(cost))
    idx = np.arange(start, stop)
    emission_delta = float(np.sum(emissions[idx, state] - emissions[idx, path[idx]]))
    boundary_delta = cost_delta = 0.
    for t in range(max(1, start), min(n, stop + 1)):
        old_prev, old_dest = path[t - 1], path[t]
        new_prev = state if start <= t - 1 < stop else old_prev
        new_dest = state if start <= t < stop else old_dest
        if old_prev != old_dest:
            boundary_delta -= boundary[t, old_prev, old_dest]
            cost_delta += cost[old_prev, old_dest]
        if new_prev != new_dest:
            boundary_delta += boundary[t, new_prev, new_dest]
            cost_delta -= cost[new_prev, new_dest]
    return {"emission": emission_delta, "signed_boundary": float(boundary_delta),
            "switch_cost": float(cost_delta),
            "total": float(emission_delta + boundary_delta + cost_delta)}


def _spells(gt, path):
    runs = official.valid_runs(gt)
    gold = official.possession_spells(gt, runs)
    predicted = official.possession_spells(official.fill_abstentions(path, runs), runs)
    overlaps = official.spell_overlaps(gold, predicted)
    matched = official.matched_gt_flags(overlaps, len(gold), .5)
    overlap_count = np.bincount([i for _, i, _ in overlaps], minlength=len(gold))
    return gold, predicted, matched, overlap_count



def spell_counts(gt, pred, *, fps=25.):
    """Official IoU .5 counts and gold duration bins on one native timeline.

    Call per half, or join halves with an X separator in both arrays. Never
    concatenate halves without a separator. Unknown GT frames split valid runs.
    Predicted X frames use official forward/back filling inside each valid run.
    The returned counts may be summed over halves or matches, but F1 and recall
    must be recomputed from their integer numerators and denominators.
    """
    gt, pred = np.asarray(gt), np.asarray(pred)
    if (gt.ndim != 1 or pred.shape != gt.shape or gt.dtype.kind not in "iu"
            or pred.dtype.kind not in "iu" or not np.isin(gt, (0, 1, 2, 3)).all()
            or not np.isin(pred, (0, 1, 2, 3)).all()):
        raise ValueError("gt and pred must be equal integer H/A/D/X timelines")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    gold, predicted, matched, overlap_count = _spells(gt, pred)
    rows = [{"matched": bool(matched[i]), "overlaps": int(overlap_count[i]),
             "duration_bin": duration_bin(b - a, fps)}
            for i, (a, b, _) in enumerate(gold)]
    tp = int(matched.sum())
    return {"counts": {"gt_spells": len(gold), "pred_spells": len(predicted),
                       "matched_at_0_5": tp, "missed_at_0_5": len(gold) - tp,
                       "spurious_at_0_5": len(predicted) - tp},
            "duration_bins": {name: aggregate_stage([r for r in rows
                                                     if r["duration_bin"] == name])
                              for name in DURATION_BINS}}


def _summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return {"mean": float(values.mean()), "p10": float(np.quantile(values, .1)),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9))}


def aggregate_stage(rows):
    gold = len(rows)
    matched = sum(r["matched"] for r in rows)
    return {"gt": gold, "matched_at_0_5": matched, "missed_at_0_5": gold - matched,
            "recall_at_0_5": matched / gold if gold else None,
            "no_same_team_overlap": sum(r["overlaps"] == 0 for r in rows),
            "gt_with_multiple_same_team_overlaps": sum(r["overlaps"] > 1 for r in rows)}


def aggregate_evidence(rows):
    counts = Counter()
    for row in rows:
        for stage, evidence in row["evidence"].items():
            counts[f"{stage}_any_correct_argmax"] += evidence["correct_argmax_fraction"] > 0
            counts[f"{stage}_majority_correct_argmax"] += evidence["correct_argmax_fraction"] > .5
            counts[f"{stage}_mean_correct_state_margin_positive"] += evidence["state_margin_mean"] > 0
            counts[f"{stage}_mean_owner_margin_positive"] += evidence["owner_margin_mean"] > 0
            counts[f"{stage}_mean_live_margin_positive"] += evidence["live_margin_mean"] > 0
        raw_any = row["evidence"]["neural"]["correct_argmax_fraction"] > 0
        final_any = row["evidence"]["final"]["correct_argmax_fraction"] > 0
        counts[f"raw_final_any_argmax_{int(raw_any)}_{int(final_any)}"] += 1
        delta = row["insertion_delta"]
        if delta["emission"] > 1e-8:
            counts["positive_emission_gain_with_boundary_penalty"] += delta["signed_boundary"] < -1e-8
            counts["positive_emission_gain_with_switch_penalty"] += delta["switch_cost"] < -1e-8
        counts["negative_emission_gain_over_decoded_interval"] += delta["emission"] < -1e-8
        counts["zero_emission_gain_over_decoded_interval"] += abs(delta["emission"]) <= 1e-8
        counts["positive_emission_gain_over_decoded_interval"] += delta["emission"] > 1e-8
        counts["positive_emission_gain_outweighed_by_sequence_terms"] += (
            delta["emission"] > 1e-8 and delta["total"] < -1e-8)
        counts["collapsed_constant_other_state"] += row["collapsed_constant_other_state"]
        counts["collapsed_gt_sandwich"] += row["collapsed_gt_sandwich"]
        if row["collapsed_constant_other_state"]:
            counts["collapsed_to_dead" if row["collapsed_state"] == 2 else "collapsed_to_other_owner"] += 1
    return {"n": len(rows), "counts": dict(counts),
            "mean_ball_visible_fraction": _summary([r["ball_visible_fraction"] for r in rows]),
            "stages": {stage: {field: _summary([r["evidence"][stage][field] for r in rows])
                               for field in ("correct_argmax_fraction", "state_margin_mean",
                                             "owner_margin_mean", "live_margin_mean")}
                       for stage in ("fine", "coarse", "neural", "final")},
            "insertion_delta": {key: _summary([r["insertion_delta"][key] for r in rows])
                                for key in ("emission", "signed_boundary", "switch_cost", "total")}}


def audit(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = read_record(manifest_path)
    if manifest is None or manifest.get("exact_v6_parity") is not True:
        raise ValueError("an authenticated exact-v6 cache manifest is required")
    selected = [r for r in manifest["records"] if r["batch"] == "batch_1"
                and r["scope"] == "inner" and r["scenario"] == "clean"]
    matches = sorted({r["match"] for r in selected})
    if len(matches) != 6 or {(r["match"], r["half"]) for r in selected} != {
            (m, h) for m in matches for h in (1, 2)} or len(selected) != 12:
        raise ValueError("expected exactly six inner public matches with two clean halves")
    config = manifest["decoder"]
    cost = _switch_matrix(config)
    rows_by_stage, pred_counts, details, parity = {}, Counter(), [], []
    for record in sorted(selected, key=lambda r: (r["match"], r["half"])):
        path = Path(record["path"]).resolve()
        path.relative_to(manifest_path.parent.resolve())
        if not record["match"].startswith("DFL-MAT-") or sha256_file(path) != record["sha256"]:
            raise ValueError("public cache identity or checksum differs")
        provenance = record["provenance"]
        if (provenance["match"] != record["match"] or provenance["half"] != record["half"]
                or provenance["scenario"] != "clean"
                or provenance["feature_sha256"] != record["sha256"]):
            raise ValueError("cache provenance identity differs")
        validate_prediction_provenance(
            provenance,
            outer_assessment_matches=manifest["inner_folds"][record["fold"]]["validation"],
        )
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        gt = arrays["gt"]
        if len(gt) != record["n_frames"] or arrays["v6_labels"].shape != gt.shape:
            raise ValueError("cache timeline metadata differs")
        if not np.isin(gt, (0, 1, 2, 3)).all():
            raise ValueError("unsupported gold-state encoding")
        evidence = reconstruct_evidence(arrays, config)
        decoded = decode_half(evidence["emissions"], evidence["boundary"], cost)
        expected = decode_arrays(arrays, config)
        if not np.array_equal(decoded, expected) or not np.array_equal(decoded, arrays["v6_labels"]):
            raise ValueError("reconstruction/original/saved decoder parity failed")
        parity.append({"match": record["match"], "half": record["half"],
                       "frames": len(gt), "sha256": record["sha256"], "equal": True})
        probs = {stage: (arrays[f"{stage}_live"], arrays[f"{stage}_home"])
                 for stage in ("fine", "coarse", "neural")}
        probs["final"] = evidence["live"], evidence["home"]
        emissions = {stage: emissions_from_probabilities(*pair) for stage, pair in probs.items()}
        paths = {f"{stage}_argmax": np.argmax(emit, axis=1) for stage, emit in emissions.items()}
        paths["blended_uncalibrated_argmax"] = np.argmax(evidence["blended_emissions"], axis=1)
        paths["v6"] = decoded
        paths["final_without_boundary"] = decode_half(evidence["emissions"], switch_cost=cost)
        paths["final_without_switch_cost"] = decode_half(evidence["emissions"], evidence["boundary"], 0.)
        stage_data = {}
        for stage, pred in paths.items():
            gold, pred_spells, matched, overlap_count = _spells(gt, pred)
            pred_counts[stage] += len(pred_spells)
            stage_data[stage] = matched, overlap_count
            for i, (a, b, state) in enumerate(gold):
                rows_by_stage.setdefault(stage, []).append({
                    "match": record["match"], "duration_bin": duration_bin(b - a, float(arrays["ref_fps"])),
                    "matched": bool(matched[i]), "overlaps": int(overlap_count[i]),
                })
        for i, (a, b, state) in enumerate(gold):
            if (b - a) / float(arrays["ref_fps"]) >= 2:
                continue
            item = {"match": record["match"], "half": record["half"], "start": a, "stop": b,
                    "state": int(state), "duration_bin": duration_bin(b - a, float(arrays["ref_fps"])),
                    "decoded_matched": bool(stage_data["v6"][0][i]),
                    "decoded_no_overlap": bool(stage_data["v6"][1][i] == 0),
                    "ball_visible_fraction": float(arrays["ball_valid"][a:b].mean()), "evidence": {}}
            for stage, (live, home) in probs.items():
                p = np.exp(emissions[stage][a:b])
                correct_owner = home[a:b] if state == 0 else 1 - home[a:b]
                item["evidence"][stage] = {
                    "correct_argmax_fraction": float(np.mean(np.argmax(p, axis=1) == state)),
                    "state_margin_mean": float(np.mean(p[:, state] - np.max(np.delete(p, state, axis=1), axis=1))),
                    "owner_margin_mean": float(np.mean(2 * correct_owner - 1)),
                    "live_margin_mean": float(np.mean(2 * live[a:b] - 1)),
                }
            item["insertion_delta"] = insertion_delta(
                evidence["emissions"], evidence["boundary"], cost, decoded, a, b, state)
            if item["insertion_delta"]["total"] > 1e-6:
                raise ValueError("positive replacement score contradicts Viterbi optimality")
            collapsed = bool(np.all(decoded[a:b] == decoded[a]) and decoded[a] != state)
            item["collapsed_constant_other_state"] = collapsed
            item["collapsed_state"] = int(decoded[a]) if collapsed else None
            item["collapsed_gt_sandwich"] = bool(
                collapsed and a > 0 and b < len(gt) and gt[a - 1] == gt[b]
                and gt[a - 1] in (0, 1, 2) and gt[a - 1] == decoded[a])
            item["stage_matched"] = {s: bool(v[0][i]) for s, v in stage_data.items()}
            details.append(item)
    stages = {}
    for stage, rows in rows_by_stage.items():
        summary = aggregate_stage(rows)
        summary.update(pred_spells=pred_counts[stage],
                       spurious_at_0_5=pred_counts[stage] - summary["matched_at_0_5"])
        tp = summary["matched_at_0_5"]
        summary["precision_at_0_5"] = tp / pred_counts[stage] if pred_counts[stage] else None
        summary["f1_at_0_5"] = 2 * tp / (len(rows) + pred_counts[stage])
        summary["duration_bins"] = {name: aggregate_stage([r for r in rows if r["duration_bin"] == name])
                                    for name in DURATION_BINS}
        stages[stage] = summary
    groups = {"all_under_2s": details,
              "decoded_missed_under_2s": [r for r in details if not r["decoded_matched"]],
              "decoded_no_overlap_under_2s": [r for r in details if r["decoded_no_overlap"]]}
    groups["decoded_missed_under_2s_fully_visible"] = [
        r for r in details if not r["decoded_matched"] and r["ball_visible_fraction"] == 1]
    groups["decoded_missed_under_2s_partly_visible"] = [
        r for r in details if not r["decoded_matched"] and 0 < r["ball_visible_fraction"] < 1]
    groups["decoded_missed_under_2s_fully_missing"] = [
        r for r in details if not r["decoded_matched"] and r["ball_visible_fraction"] == 0]
    for name in DURATION_BINS[:2]:
        groups[f"decoded_missed_{name}"] = [r for r in details if not r["decoded_matched"]
                                             and r["duration_bin"] == name]
    return {
        "schema": "predecoder_spell_evidence_audit_v1", "independent_test": False,
        "scope": "Six inner public DFL matches, batch_1, clean, two native halves each",
        "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "decoder": config, "parity": parity, "stages": stages,
        "short_evidence": {name: aggregate_evidence(rows) for name, rows in groups.items()},
        "short_matched_cross_table": {stage: {
            "argmax_or_ablation_matched_decoded_missed": sum(r["stage_matched"][stage] and not r["decoded_matched"] for r in details),
            "argmax_or_ablation_missed_decoded_matched": sum(not r["stage_matched"][stage] and r["decoded_matched"] for r in details),
        } for stage in stages if stage != "v6"},
        "by_match": {m: {s: aggregate_stage([r for r in rows if r["match"] == m])
                          for s, rows in rows_by_stage.items()} for m in matches},
        "definitions": {
            "state_margin": "Correct H/A probability minus largest competing H/A/D probability",
            "owner_margin": "Conditional correct-team probability minus opposite-team probability",
            "live_margin": "Live probability minus dead probability",
            "positive_emission_gain": "Gold-state insertion beats decoded emissions summed over the exact GT interval",
            "insertion_delta": "Replace the exact GT interval with its true state, hold all decoded context fixed, include changed transitions at both edges and inside the interval",
            "collapsed_constant_other_state": "Entire GT interval decoded as one constant incorrect state",
            "collapsed_gt_sandwich": "Collapsed interval whose immediate GT neighbors share the decoded incorrect state, including dead state",
            "overfragmentation": "Spurious predictions and GT spells with multiple same-team overlapping predictions, reported for every fixed diagnostic path",
        },
        "limitations": [
            "Development evidence on historically reused matches, no independent generalization claim.",
            "Argmax paths ignore sequence terms and can fragment, so coverage is not oracle recoverability.",
            "Fixed ablations remove one sequence term without tuning and are not promotion candidates.",
            "Gold interval replacement uses known endpoints and labels for explanation only, not candidate generation.",
            "A positive replacement emission gain with negative total demonstrates suppression for that exact local insertion, not that every missed spell can be recovered by changing the decoder.",
            "Signed boundary scores and switch costs interact with global path choices, counts are not a causal partition of model errors.",
            "No participant data, raw tracking, video, fitting, or pinned source modifications are used.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest)
    write_json(args.output, result)
    print({"v6": result["stages"]["v6"], "parity_halves": len(result["parity"])})


if __name__ == "__main__":
    main()
