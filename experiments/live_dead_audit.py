"""Bounded live/dead stability audit over authenticated v4 OOF caches.

This module reads public source-match labels only. Target handling is restricted
to the manifest-authenticated tracking and metadata files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from experiments import campaign_decode, campaign_protocol
from experiments.campaign_evaluation import _digest as _input_digest
from experiments.campaign_evaluation import load_arrays
from experiments.campaign_submission import N_FRAMES, TARGET_SPANS, _verified_inputs
from games import GAME, GAMES
from possession.calibrate import CalibrationConfig, apply_temperature_calibration
from possession.features import build_features
from possession.io_atd import read_atd
from possession.storage import sha256_file, write_json

REPO = Path(__file__).resolve().parents[1]
T7 = Path("/Volumes/T7/Data/more-than-a-hack-2027/artifacts/ensemble-bagging-v1")
OUT = REPO / "artifacts/post-v4-stability-audit"
ARCHES = ("mean_007", "ball_attention_002")
SEEDS = (8101, 8102)
GAP_EDGES = (0., 1., 10., 30., 60., 120., float("inf"))
GAP_NAMES = ("observed", "0-1s", "1-10s", "10-30s", "30-60s", "60-120s", ">120s")


def full_gap_seconds(ball_valid: np.ndarray, fps: float = 25.) -> np.ndarray:
    """Assign each frame its full contiguous outage length, or zero if observed."""
    valid = np.asarray(ball_valid, dtype=bool)
    if valid.ndim != 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("ball_valid must be one-dimensional and fps positive")
    missing = ~valid
    starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
    stops = np.flatnonzero(missing & ~np.r_[missing[1:], False]) + 1
    output = np.zeros(len(valid), dtype=np.float64)
    for start, stop in zip(starts, stops, strict=True):
        output[start:stop] = (stop - start) / fps
    return output


def gap_strata(ball_valid: np.ndarray, fps: float = 25.) -> np.ndarray:
    seconds = full_gap_seconds(ball_valid, fps)
    indices = np.searchsorted(GAP_EDGES[1:-1], seconds, side="right") + 1
    indices[seconds == 0] = 0
    return indices.astype(np.int8)


def calibration_summary(probability: np.ndarray, truth_live: np.ndarray,
                        bins: int = 10) -> dict:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(truth_live, dtype=np.float64)
    if p.ndim != 1 or y.shape != p.shape or bins < 1:
        raise ValueError("probability and truth must align and bins be positive")
    finite_probability = np.isfinite(p)
    if np.any(finite_probability & ((p < 0) | (p > 1))):
        raise ValueError("finite probabilities must lie in [0,1]")
    keep = finite_probability & np.isfinite(y) & ((y == 0) | (y == 1))
    n_input = len(keep)
    p, y = p[keep], y[keep]
    if not len(p):
        return {"n_input": n_input, "n": 0, "n_invalid": n_input,
                "brier": None, "log_loss": None, "reliability": []}
    eps = 1e-12
    edges = np.linspace(0., 1., bins + 1)
    idx = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    reliability = []
    for i in range(bins):
        selected = idx == i
        reliability.append({"bin": i, "lower": float(edges[i]), "upper": float(edges[i + 1]),
                            "n": int(selected.sum()),
                            "mean_probability": float(p[selected].mean()) if selected.any() else None,
                            "live_rate": float(y[selected].mean()) if selected.any() else None})
    return {"n_input": n_input, "n": len(p),
            "n_invalid": n_input - len(p),
            "n_live": int(y.sum()), "n_dead": len(y) - int(y.sum()),
            "live_rate": float(y.mean()), "mean_live_probability": float(p.mean()),
            "predicted_live_rate_at_0_5": float(np.mean(p >= .5)),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(np.mean(-(y * np.log(np.clip(p, eps, 1-eps))
                                        + (1-y) * np.log(np.clip(1-p, eps, 1-eps))))),
            "reliability": reliability}


def _nested_record(path: Path) -> dict:
    wrapped = json.loads(path.read_text())
    if set(wrapped) != {"sha256", "payload"}:
        raise ValueError(f"unwrapped record: {path}")
    from experiments.robustness import _digest
    if wrapped["sha256"] != _digest(wrapped["payload"]):
        raise ValueError(f"record digest differs: {path}")
    return wrapped["payload"]


def _verify_input_manifest(batch: str, records: list[list[dict]]) -> dict:
    path = T7 / batch / "inner/inputs/input-manifest.json"
    sidecar = path.with_suffix(".sha256.json")
    if not sidecar.is_file() or json.loads(sidecar.read_text()).get("sha256") != sha256_file(path):
        raise ValueError(f"input manifest sidecar differs: {path}")
    manifest = json.loads(path.read_text())
    signature = manifest.get("signature")
    if manifest.get("signature_sha256") != _input_digest(signature):
        raise ValueError(f"input manifest signature differs: {path}")
    indexed = {(r["match"], r["period"], r["scenario"]): r
               for r in manifest.get("records", [])}
    flattened = [record for member in records for record in member]
    if len(indexed) != 48 or len(flattened) != 96:
        raise ValueError(f"unexpected input manifest or prediction record count: {path}")
    for record in flattened:
        key = (record["match"], record["period"], record["scenario"])
        manifest_record = indexed.get(key)
        if (manifest_record is None
                or record["input_identity"] != manifest["signature_sha256"]
                or manifest_record.get("input_identity") != record["input_identity"]
                or manifest_record.get("input_sha256") != record["input_sha256"]):
            raise ValueError(f"input record is not authenticated by its manifest: {key}")
    return {"manifest_sha256": sha256_file(path), "input_signature_sha256": manifest["signature_sha256"],
            "unique_input_records": len(indexed), "architecture_record_links_verified": len(flattened)}


def _load_cache_records(batch: str, seed: int) -> dict[tuple, dict]:
    path = T7 / batch / f"inner_seed_{seed}.json"
    payload = _nested_record(path)
    protocol = _nested_record(T7 / "protocol.json")
    if payload.get("scope") != "inner" or payload.get("seed") != seed:
        raise ValueError(f"unexpected inner cache record identity: {path}")
    if payload.get("selection_sha256") != protocol.get("candidate_selection_sha256"):
        raise ValueError(f"OOF cache selection differs from frozen candidate: {path}")
    members = payload.get("records")
    if not isinstance(members, list) or len(members) != 2:
        raise ValueError(f"expected two architecture records: {path}")
    seven = sorted({"DFL-MAT-J03WMX", "DFL-MAT-J03WN1", "DFL-MAT-J03WOH",
                    "DFL-MAT-J03WOY", "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ",
                    "DFL-MAT-J03WR9"})
    outer = next(x for x in campaign_protocol.nested_splits(seven, inner_folds=3, seed=0)
                 if x["outer_test"] == ["DFL-MAT-J03WMX"])
    expected_fold = {i: fold for i, fold in enumerate(outer["inner"])}
    indexed = {}
    for architecture, records in zip(ARCHES, members, strict=True):
        if len(records) != 48:
            raise ValueError(f"{architecture} cache does not contain 48 half-scenarios")
        for record in records:
            identity = record.get("model_identity", {})
            key = (record["match"], record["period"], record["scenario"])
            if key in indexed.setdefault(architecture, {}):
                raise ValueError(f"duplicate cache key {key}")
            fold = identity.get("fold")
            if (identity.get("scope") != "inner" or identity.get("seed") != seed
                    or record["match"] not in identity.get("validation_matches", [])
                    or record["match"] in identity.get("train_matches", [])):
                raise ValueError(f"whole-match exclusion or seed provenance failed: {key}")
            expected = expected_fold.get(fold)
            if (expected is None or set(expected["train"]) != set(identity["train_matches"])
                    or set(expected["validation"]) != set(identity["validation_matches"])):
                raise ValueError(f"fold train membership differs: {key}")
            if not record.get("prediction_path") or sha256_file(record["prediction_path"]) != record["prediction_sha256"]:
                raise ValueError(f"prediction hash differs: {key}")
            indexed[architecture][key] = record
    if set(indexed[ARCHES[0]]) != set(indexed[ARCHES[1]]):
        raise ValueError("architecture cache key grids differ")
    return indexed


def _logit_blend(p, q, weight):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1-1e-7)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-7, 1-1e-7)
    logits = weight * (np.log(p) - np.log1p(-p)) + (1-weight) * (np.log(q) - np.log1p(-q))
    result = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    if weight == 1:
        return np.asarray(p, dtype=np.float64).copy()
    if weight == 0:
        return np.asarray(q, dtype=np.float64).copy()
    return result


def _blend_model_arrays(seed_arch_arrays, architecture_weight, bagged):
    """Average seeds within architecture, then apply the frozen architecture blend."""
    arch_outputs = {}
    for arch in ARCHES:
        seed_arrays = seed_arch_arrays[arch]
        if bagged:
            base = dict(seed_arrays[0])
            for name in ("neural_live", "neural_home"):
                base[name] = _logit_blend(seed_arrays[0][name], seed_arrays[1][name], .5)
            base["neural_boundary"] = (np.asarray(seed_arrays[0]["neural_boundary"])
                                       + np.asarray(seed_arrays[1]["neural_boundary"])) / 2
            arch_outputs[arch] = base
        else:
            arch_outputs[arch] = dict(seed_arrays[0])
    left, right = (arch_outputs[a] for a in ARCHES)
    output = dict(left)
    for name in ("neural_live", "neural_home"):
        output[name] = _logit_blend(left[name], right[name], architecture_weight)
    output["neural_boundary"] = (architecture_weight * left["neural_boundary"]
                                 + (1-architecture_weight) * right["neural_boundary"])
    for name in ("gt", "frame_id", "ball_valid", "nearest_ball_seconds"):
        if not np.array_equal(left[name], right[name]):
            raise ValueError(f"OOF architectures have different cached input: {name}")
    return output


def live_probability(arrays, decoder):
    """Reproduce the decoder's calibrated live marginal for calibration metrics."""
    basis = {key: value for key, value in arrays.items() if key.startswith("ref_")}
    reference = campaign_decode.reference_from_basis(
        basis, decoder["reference"] if decoder["reference_mode"] == "tuned" else None)
    seconds = np.asarray(arrays["nearest_ball_seconds"], dtype=np.float64)
    valid = np.asarray(arrays["ball_valid"], dtype=bool)
    gate = np.minimum(seconds, decoder["transition_seconds"]) / decoder["transition_seconds"]
    gate[valid] = 0
    if decoder["blend_mode"] == "static":
        weight = np.full(len(seconds), decoder["neural_weight"])
    elif decoder["blend_mode"] == "quality_shared":
        weight = decoder["visible_neural_weight"] + (decoder["missing_neural_weight"]
                                                     - decoder["visible_neural_weight"]) * gate
    else:
        weight = decoder["visible_live_weight"] + (decoder["missing_live_weight"]
                                                    - decoder["visible_live_weight"]) * gate
    p = np.clip(arrays["neural_live"], 1e-6, 1-1e-6)
    r = np.clip(reference["live_probability"], 1e-6, 1-1e-6)
    logits = weight * (np.log(p) - np.log1p(-p)) + (1-weight) * (np.log(r)-np.log1p(-r))
    blended = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    blended[weight == 0] = reference["live_probability"][weight == 0]
    blended[weight == 1] = arrays["neural_live"][weight == 1]
    live, _, _ = apply_temperature_calibration(
        blended, np.full(len(blended), .5), config=CalibrationConfig(
            live_temperature=decoder["live_temperature"]))
    return live


def _binary_metric_rows(p, y, mask):
    summary = calibration_summary(p[mask], y[mask])
    hard = p[mask] >= .5
    truth = y[mask].astype(bool)
    tp = int(np.count_nonzero(hard & truth)); fp = int(np.count_nonzero(hard & ~truth))
    fn = int(np.count_nonzero(~hard & truth))
    summary.update({"threshold": .5, "accuracy": float(np.mean(hard == truth)) if len(truth) else None,
                    "live_precision": tp / (tp + fp) if tp + fp else 0.,
                    "live_recall": tp / (tp + fn) if tp + fn else 0.})
    return summary


def _audit_model(batch, seed_records, selection, model_name):
    combos = {}
    for arch in ARCHES:
        combos[arch] = {seed: seed_records[seed][arch] for seed in SEEDS}
    keys = sorted(seed_records[8101][ARCHES[0]])
    groups = {}
    inputs_for_features = []
    for key in keys:
        match, period, scenario = key
        loaded = {}
        # Only this half is in memory at once. Existing loader verifies input and
        # prediction hashes plus the prediction sidecar identity.
        for seed in SEEDS:
            loaded[seed] = {}
            for arch in ARCHES:
                rec = combos[arch][seed][key]
                loaded[seed][arch] = load_arrays(rec)
        seed_arch = {}
        for arch in ARCHES:
            seed_arch[arch] = [loaded[s][arch] for s in SEEDS]
        seeds_to_score = [(s, False) for s in SEEDS] + [(None, True)]
        for seed, is_bag in seeds_to_score:
            use = {arch: (seed_arch[arch] if is_bag else [loaded[seed][arch]]) for arch in ARCHES}
            arr = _blend_model_arrays(use, selection["blend_weight"], is_bag)
            pred = campaign_decode.decode_arrays(arr, selection["decoder"])
            live = live_probability(arr, selection["decoder"])
            gt = arr["gt"].astype(np.int16)
            graded = gt != official.X
            actual_live = (gt != official.D).astype(np.float64)
            gaps = gap_strata(arr["ball_valid"], official.FPS)
            half_score = official.evaluate(gt, pred, bootstrap_n=0)
            token = "bagged" if is_bag else f"seed_{seed}"
            bucket = groups.setdefault((token, scenario), {"live_p": [], "live_y": [],
                "decoded_gt": [], "decoded_pred": [], "strata": [], "valid_run_metrics": [],
                "match_frames": {}})
            bucket["live_p"].append(live[graded]); bucket["live_y"].append(actual_live[graded])
            bucket["decoded_gt"].append(gt[graded]); bucket["decoded_pred"].append(pred[graded])
            bucket["strata"].append(gaps[graded]);
            bucket["match_frames"].setdefault(match, {})[period] = (gt.copy(), pred.copy())
            bucket["valid_run_metrics"].append({"match": match, "period": period,
                "n_graded": int(graded.sum()), "spell_f1": half_score["spell_f1_mean"],
                "macro_f1": half_score["macro_f1"],
                "transition_f1_at_12": half_score["transitions"][12]["f1"],
                "n_gt_spells": half_score["n_gt_spells"],
                "n_gt_transitions": half_score["transitions"][12]["n_gt"]})
            if scenario == "clean" and period == 1:
                inputs_for_features.append((match, loaded[8101][ARCHES[0]]))
    reports = {}
    for (token, scenario), bucket in groups.items():
        p = np.concatenate(bucket["live_p"]); y = np.concatenate(bucket["live_y"])
        gt = np.concatenate(bucket["decoded_gt"]); pred = np.concatenate(bucket["decoded_pred"])
        strata = np.concatenate(bucket["strata"])
        by_gap = {}
        for i, name in enumerate(GAP_NAMES):
            mask = strata == i
            cm = official.scored_confusion(official.confusion(gt[mask], pred[mask]))
            by_gap[name] = {"frames": int(mask.sum()),
                            "accuracy": float(np.trace(cm)/cm.sum()) if cm.sum() else None,
                            "macro_f1": float(official.macro_f1(cm)) if cm.sum() else None,
                            "live_dead_f1": float(official.alive_dead_f1(cm)) if cm.sum() else None,
                            "live_dead_calibration": calibration_summary(p[mask], y[mask])}
        match_metrics = []
        for match, halves in sorted(bucket["match_frames"].items()):
            if set(halves) != {1, 2}:
                raise ValueError(f"official whole-match grid is incomplete: {match} {scenario}")
            full_gt = np.concatenate((halves[1][0], np.asarray([official.X]), halves[2][0]))
            full_pred = np.concatenate((halves[1][1], np.asarray([official.X]), halves[2][1]))
            result = official.evaluate(full_gt, full_pred, bootstrap_n=0)
            match_metrics.append({"match": match, "n_graded": result["n_graded"],
                                  "spell_f1": result["spell_f1_mean"],
                                  "macro_f1": result["macro_f1"],
                                  "transition_f1_at_12": result["transitions"][12]["f1"]})
        reports.setdefault(token, {})[scenario] = {
            "n_graded": len(gt), "official_half_metrics": bucket["valid_run_metrics"],
            "official_match_metrics": match_metrics,
            "mean_spell_f1_across_matches": float(np.mean([x["spell_f1"] for x in match_metrics])),
            "mean_transition_f1_at_12_across_matches": float(np.mean([x["transition_f1_at_12"] for x in match_metrics])),
            "live_dead": calibration_summary(p, y), "decoded_by_full_gap": by_gap}
    return reports


def _source_target_features(batch_records):
    source_rows = {}
    for batch, recs in batch_records.items():
        for record in recs[8101][ARCHES[0]].values():
            scenario = record["scenario"]
            a = load_arrays(record)
            source_rows.setdefault((batch, scenario), []).append(
                (a["ball_valid"], a["global_tokens"], a["player_mask"], a["player_tokens"]))
    target_inputs = _verified_inputs(REPO / "data/target/2027")
    frames = read_atd(target_inputs["raw"]["path"], target_inputs["metadata"]["path"],
                      n_frames=N_FRAMES, pitch=GAMES[GAME]["pitch"], fps=GAMES[GAME]["fps"],
                      graded_spans=TARGET_SPANS)
    def summarize(parts):
        vals = {"ball_observed_fraction": [], "player_count": [], "valid_player_speed_mps": [],
                "gap_age_capped_fraction": []}
        for ball_valid, globals_, player_mask, player_tokens in parts:
            valid = np.asarray(ball_valid, bool)
            vals["ball_observed_fraction"].append(valid.astype(float))
            vals["player_count"].append(np.asarray(player_mask, bool).sum(axis=(1, 2)).astype(float))
            frame_speed = np.asarray(globals_)[:, 8] * 15
            has_motion = (np.asarray(player_tokens)[..., 4] > 0).any(axis=(1, 2))
            vals["valid_player_speed_mps"].append(frame_speed[has_motion])
            vals["gap_age_capped_fraction"].append((np.asarray(globals_)[:, 6:8] >= 1).any(axis=1).astype(float))
        joined = {}
        for key, arrays in vals.items():
            if key == "valid_player_speed_mps":
                flat = np.concatenate([x.ravel() for x in arrays if x.size]) if any(x.size for x in arrays) else np.array([])
                joined[key] = {"n": int(flat.size), "median": float(np.median(flat)) if flat.size else None,
                               "p90": float(np.percentile(flat, 90)) if flat.size else None}
            else:
                flat = np.concatenate([x.ravel() for x in arrays])
                joined[key] = {"n": int(flat.size), "mean": float(flat.mean()),
                               "p50": float(np.percentile(flat, 50)),
                               "p90": float(np.percentile(flat, 90))}
        return joined
    target_source = []
    for period, (start, stop) in enumerate(TARGET_SPANS, 1):
        half = frames.subset(start, stop)
        feat = build_features(half)
        target_source.append((half.ball_valid, feat.global_tokens, feat.player_mask, feat.player_tokens))
    return {"source_scenarios": {f"{batch}/{scenario}": summarize(rows)
                                 for (batch, scenario), rows in sorted(source_rows.items())},
            "target_inputs_only": summarize(target_source),
            "target_input_manifest_sha256": target_inputs["manifest"]["sha256"],
            "target_frames_scored_for_features": int(sum(stop-start for start, stop in TARGET_SPANS)),
            "target_possession_labels_read": False}


def run(output: Path = OUT):
    protocol = _nested_record(T7 / "protocol.json")
    selection = protocol["candidate_selection"]
    if selection.get("members", [{}])[0].get("id") != ARCHES[0] or selection.get("members", [{}])[1].get("id") != ARCHES[1]:
        raise ValueError("frozen candidate architecture order differs")
    all_reports = {"schema": "post_v4_live_dead_audit_v1", "cache_root": str(T7),
        "candidate_selection_sha256": protocol["candidate_selection_sha256"],
        "protocol_file_sha256": sha256_file(T7 / "protocol.json"),
        "fold_exclusion": "verified from each prediction model identity against nested whole-match fold training sets",
        "target_possession_labels_read": False, "model_fits": 0, "scenarios": {}}
    feature_records = {}
    input_integrity = {}
    for batch in ("batch_1", "batch_2"):
        seed_records = {seed: _load_cache_records(batch, seed) for seed in SEEDS}
        input_integrity[batch] = _verify_input_manifest(
            batch, [seed_records[8101][ARCHES[0]].values(), seed_records[8101][ARCHES[1]].values()])
        feature_records[batch] = seed_records
        # Current candidate uses one pair per seed and 50/50 seed logit averages
        # within architecture before the frozen architecture blend.
        # One invocation scores individual seed pairs plus the current bagged model.
        reports = _audit_model(batch, seed_records, selection, "all")
        for scenario_rows in reports.values():
            objective = float(sum(campaign_protocol.WEIGHTS[name]
                                  * scenario_rows[name]["mean_spell_f1_across_matches"]
                                  for name in campaign_protocol.WEIGHTS))
            scenario_rows["weighted_objective"] = objective
        saved = _nested_record(T7 / batch / "result.json")["candidate"]
        reproduced = reports["bagged"]
        comparisons = {name: {"audit": reproduced[name]["mean_spell_f1_across_matches"],
                              "saved": saved["means"][name]["spell_f1_mean"],
                              "absolute_delta": abs(reproduced[name]["mean_spell_f1_across_matches"]
                                                    - saved["means"][name]["spell_f1_mean"])}
                       for name in campaign_protocol.WEIGHTS}
        comparisons["weighted_objective"] = {"audit": reproduced["weighted_objective"],
            "saved": saved["objective"], "absolute_delta": abs(reproduced["weighted_objective"]
                                                                  - saved["objective"])}
        if any(row["absolute_delta"] > 1e-9 for row in comparisons.values()):
            raise ValueError(f"cached candidate score reconstruction differs for {batch}: {comparisons}")
        reports["saved_candidate_score_reconstruction"] = comparisons
        all_reports["scenarios"][batch] = reports
    all_reports["input_manifest_integrity"] = input_integrity
    all_reports["input_feature_comparison"] = _source_target_features(feature_records)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "audit.json", all_reports)
    (output / "README.md").write_text(
        "# Post-v4 live/dead stability audit\n\n"
        "This bounded audit uses only authenticated inner-fold OOF predictions and public source labels. "
        "Target comparison uses tracking inputs and metadata only. No final all-seven model is evaluated on training matches. "
        "Official spell and transition scores are computed per native half, with each half as one contiguous timeline. "
        "Gap-duration rows are framewise metrics keyed by the full contiguous missing-ball run length. "
        "Disjoint gaps are never stitched together for spell scoring. See `audit.json`.\n")
    return all_reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({"output": str(args.output), "batches": list(report["scenarios"]),
                      "target_possession_labels_read": False}, sort_keys=True))


if __name__ == "__main__":
    main()
