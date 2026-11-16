"""OOF calibration audit adapter for frozen tail-EMA prediction caches.

The adapter is intentionally separate from EMA fitting. It verifies persisted
EMA fit and prediction ancestry, then delegates metrics to live_dead_audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments import campaign_protocol, ema_stability, live_dead_audit
from experiments.robustness import _execution_environment, _source_hashes
from possession.storage import sha256_file, write_json

EMA_DIR = Path("/Volumes/T7/Data/more-than-a-hack-2027/artifacts/ema-stability-v1")
OUT = Path(__file__).resolve().parents[1] / "artifacts/post-v4-ema-audit"
ARCHES = ("mean_007", "ball_attention_002")
SEEDS = (8101, 8102)


def _wrapped(path: Path) -> dict:
    payload = campaign_protocol.read_record(path)
    if payload is None:
        raise FileNotFoundError(path)
    return payload


def _verify_frozen_result():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    protocol = _wrapped(EMA_DIR / "protocol.json")
    result = _wrapped(EMA_DIR / "result.json")
    protocol_sha = ema_stability._digest(protocol)
    if (result.get("protocol_sha256") != protocol_sha
            or protocol.get("candidate_selection_sha256")
            != ema_stability._digest(protocol.get("candidate_selection"))
            or result.get("candidate_selection") != protocol.get("candidate_selection")
            or len(result.get("batches", [])) != 2):
        raise ValueError("completed EMA result differs from its frozen protocol")
    # The adapter itself is added after the frozen training protocol. Compare every
    # pre-existing source entry and ignore only this adapter's new source path.
    current = _source_hashes()
    adapter_rel = str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[1]))
    current.pop(adapter_rel, None)
    frozen_sources = protocol.get("source_sha256", {})
    changed = [name for name, digest in frozen_sources.items() if current.get(name) != digest]
    if changed:
        raise ValueError("EMA source snapshot differs from the completed frozen protocol")
    added = sorted(set(current) - set(frozen_sources))
    if _execution_environment() != protocol.get("runtime"):
        raise ValueError("EMA runtime differs from the completed frozen protocol")
    bagging_dir = Path(protocol["bagging_dir"]).resolve()
    if sha256_file(bagging_dir / "result.json") != protocol.get("bagging_result_sha256"):
        raise ValueError("the v4 bagging result used by EMA has changed")
    return protocol, result, protocol_sha, bagging_dir, added


def _validate_fit(protocol, batch_result, batch_name, seed, fold_index, member_id):
    task = next((row for row in protocol["tail_tasks"] if row["scope"] == "inner"
                 and row["seed"] == seed and row["fold"] == fold_index
                 and row["member_id"] == member_id), None)
    if task is None:
        raise ValueError("EMA protocol lacks this inner tail task")
    fit_path = EMA_DIR / "fits" / "inner" / member_id / f"seed_{seed}" / f"fold_{fold_index:02d}" / "fit.json"
    model_path = fit_path.parent / "model.pt"
    fit = _wrapped(fit_path)
    folds = protocol["inner_folds"]
    fold = folds[fold_index]
    if (fit.get("steps") != protocol["tail"]["end_steps"]
            or fit.get("validation_matches") != fold["validation"]
            or fit.get("inputs", {}).get("baseline_checkpoint_sha256") != task["original_fit_sha256"]
            or fit.get("inputs", {}).get("start_state_sha256") != task["start_state_sha256"]
            or fit.get("tail_ema", {}).get("decay") != protocol["tail"]["ema_decay"]
            or fit.get("tail_ema", {}).get("updates") != protocol["tail"]["updates"]
            or fit.get("checkpoint_sha256") != sha256_file(model_path)):
        raise ValueError(f"EMA fit signature differs: {batch_name}/{seed}/{member_id}/{fold_index}")
    parity = {"raw_weights_match_endpoint_state": True,
              "raw_weights_match_baseline_checkpoint": True,
              "raw_training_progress_matches_endpoint": True,
              "compared_optimizer_rng_transcript_and_counters": True,
              "start_steps": protocol["tail"]["start_steps"],
              "end_steps": protocol["tail"]["end_steps"]}
    if fit.get("raw_parity") != parity or fit.get("tail_ema", {}).get("raw_parity") != parity:
        raise ValueError("EMA raw-tail replay parity evidence differs")
    if fit.get("identity", {}).get("train_matches") != fold["train"]:
        raise ValueError("EMA fit train matches differ from frozen whole-match fold")
    return fit_path, fit, model_path


def _input_records(bagging_dir: Path, batch_name: str):
    root = bagging_dir / batch_name / "inner"
    manifest = root / "inputs" / "input-manifest.json"
    sidecar = manifest.with_suffix(".sha256.json")
    if json.loads(sidecar.read_text()).get("sha256") != sha256_file(manifest):
        raise ValueError("reused v4 input manifest sidecar differs")
    body = json.loads(manifest.read_text())
    from experiments.campaign_evaluation import _digest, _input_path

    if body.get("signature_sha256") != _digest(body.get("signature")):
        raise ValueError("reused v4 input manifest signature differs")
    records = []
    for row in body.get("records", []):
        path = _input_path(root / "inputs", row["match"], row["period"], row["scenario"])
        if not path.is_file() or sha256_file(path) != row.get("input_sha256"):
            raise ValueError("reused v4 OOF input array hash differs")
        records.append({**row, "input_path": str(path)})
    if len(records) != 48:
        raise ValueError("reused EMA input grid must contain 48 inner records")
    return records, body


def load_ema_records(batch_name: str, seed: int, protocol: dict,
                     bagging_dir: Path, batch_result: dict) -> dict:
    """Return the existing audit's seed -> architecture -> OOF record mapping."""
    if batch_name not in {"batch_1", "batch_2"} or seed not in SEEDS:
        raise ValueError("unsupported fixed EMA batch or training seed")
    inputs, input_manifest = _input_records(bagging_dir, batch_name)
    input_reuse = protocol["input_reuse"][batch_name]["inner"]
    if (input_reuse["manifest_sha256"] != sha256_file(
            bagging_dir / batch_name / "inner/inputs/input-manifest.json")
            or input_reuse["signature_sha256"] != input_manifest["signature_sha256"]):
        raise ValueError("EMA frozen input reuse identity differs")
    expected_keys = {(r["match"], r["period"], r["scenario"]) for r in inputs}
    mapped = {}
    artifact_hashes = batch_result.get("artifact_sha256", {})
    for member_id in ARCHES:
        all_records = []
        seen = set()
        for fold_index, fold in enumerate(protocol["inner_folds"]):
            fit_path, fit, _model_path = _validate_fit(
                protocol, batch_result, batch_name, seed, fold_index, member_id)
            for path in (fit_path,):
                rel = path.relative_to(EMA_DIR).as_posix()
                if artifact_hashes.get(rel) != sha256_file(path):
                    raise ValueError("EMA result does not authenticate its fit record")
            fit_hash = sha256_file(fit_path)
            identity = {"protocol_sha256": ema_stability._digest(protocol),
                        "ema_fit_sha256": fit_hash, "ema_sha256": fit["ema_sha256"],
                        "ema_checkpoint_sha256": fit["checkpoint_sha256"],
                        "raw_checkpoint_sha256": fit["inputs"]["baseline_checkpoint_sha256"],
                        "scope": "inner", "seed": seed, "fold": fold_index,
                        "train_matches": fold["train"], "validation_matches": fold["validation"],
                        "input_manifest_sha256": input_manifest["signature_sha256"]}
            manifest_path = (EMA_DIR / "predictions" / batch_name / "inner" / f"seed_{seed}"
                             / member_id / f"fold_{fold_index:02d}.json")
            rel_manifest = manifest_path.relative_to(EMA_DIR).as_posix()
            if artifact_hashes.get(rel_manifest) != sha256_file(manifest_path):
                raise ValueError("EMA result does not authenticate its prediction manifest")
            saved = campaign_protocol.read_record(manifest_path)
            if saved is None or saved.get("identity") != identity:
                raise ValueError("EMA prediction manifest identity differs from fit ancestry")
            records = saved.get("records", [])
            ema_stability._validate_prediction_grid(records, fold, inputs, identity)
            for record in records:
                key = (record["match"], record["period"], record["scenario"])
                model_identity = record.get("model_identity", {})
                if (key in seen or record["match"] not in fold["validation"]
                        or record["match"] in model_identity.get("train_matches", [])
                        or model_identity != identity):
                    raise ValueError("EMA OOF record violates whole-match or identity rules")
                seen.add(key)
            all_records.extend(records)
        if seen != expected_keys or len(all_records) != 48:
            raise ValueError("merged EMA fold predictions do not cover the frozen OOF grid")
        mapped[member_id] = { (r["match"], r["period"], r["scenario"]): r for r in all_records }
    if set(mapped[ARCHES[0]]) != set(mapped[ARCHES[1]]):
        raise ValueError("EMA architecture grids differ")
    return mapped


def _compare_saved(audit_report: dict, saved: dict, label: str) -> dict:
    comparisons = {}
    for scenario, weight in campaign_protocol.WEIGHTS.items():
        observed = audit_report[scenario]["mean_spell_f1_across_matches"]
        reference = saved["means"][scenario]["spell_f1_mean"]
        comparisons[scenario] = {"audit": observed, "saved": reference,
                                  "absolute_delta": abs(observed-reference), "weight": weight}
    observed_objective = audit_report["weighted_objective"]
    comparisons["weighted_objective"] = {"audit": observed_objective,
        "saved": saved["objective"], "absolute_delta": abs(observed_objective-saved["objective"])}
    if any(row["absolute_delta"] > 1e-9 for row in comparisons.values()):
        raise ValueError(f"EMA score reconstruction differs for {label}: {comparisons}")
    return comparisons


def _paired_seed_disagreement(records_by_seed: dict, selection: dict) -> dict:
    """Compare post-decoder live probabilities on identical OOF frames."""
    keys = sorted(records_by_seed[8101][ARCHES[0]])
    rows = {}
    for key in keys:
        arrays_by_seed = {}
        for seed in SEEDS:
            arch_arrays = {}
            for arch in ARCHES:
                record = records_by_seed[seed][arch][key]
                arch_arrays[arch] = live_dead_audit.load_arrays(record)
            arrays_by_seed[seed] = live_dead_audit._blend_model_arrays(
                {arch: [arch_arrays[arch]] for arch in ARCHES},
                selection["blend_weight"], False)
        one = arrays_by_seed[8101]
        two = arrays_by_seed[8102]
        for field in ("gt", "frame_id", "ball_valid", "nearest_ball_seconds"):
            if not (one[field] == two[field]).all():
                raise ValueError(f"seed OOF inputs do not align on {key}: {field}")
        valid = one["gt"] != live_dead_audit.official.X
        p1 = live_dead_audit.live_probability(one, selection["decoder"])[valid]
        p2 = live_dead_audit.live_probability(two, selection["decoder"])[valid]
        rows.setdefault(key[2], []).append((p1, p2))
    result = {}
    for scenario, pairs in rows.items():
        p1, p2 = (np.concatenate([pair[i] for pair in pairs]) for i in (0, 1))
        diff = p2 - p1
        result[scenario] = {"n_paired_graded_frames": len(p1),
            "mean_absolute_live_probability_difference": float(np.mean(np.abs(diff))),
            "rms_live_probability_difference": float(np.sqrt(np.mean(diff ** 2))),
            "mean_signed_difference_seed8102_minus_8101": float(np.mean(diff)),
            "threshold_disagreement_rate_at_0_5": float(np.mean((p1 >= .5) != (p2 >= .5))),
            "pearson_correlation": float(np.corrcoef(p1, p2)[0, 1]) if len(p1) > 1 else None}
    return result


def add_paired_calibration_comparison(output: Path = OUT):
    """Add paired EMA versus v4 Brier and seed-disagreement summaries to audit.json."""
    protocol, result, protocol_sha, bagging_dir, _later_sources = _verify_frozen_result()
    audit_path = output / "audit.json"
    report = json.loads(audit_path.read_text())
    if report.get("ema_protocol_sha256") != protocol_sha:
        raise ValueError("existing EMA audit was produced from another protocol")
    baseline_path = OUT.parent / "post-v4-stability-audit/audit.json"
    baseline = json.loads(baseline_path.read_text())
    comparisons = {}
    disagreements = {}
    selection = protocol["candidate_selection"]
    for batch_row in result["batches"]:
        batch = batch_row["batch"]
        ema_records = {seed: load_ema_records(batch, seed, protocol, bagging_dir, batch_row)
                       for seed in SEEDS}
        v4_records = {seed: live_dead_audit._load_cache_records(batch, seed)
                      for seed in SEEDS}
        disagreements[batch] = {
            "v4": _paired_seed_disagreement(v4_records, selection),
            "ema": _paired_seed_disagreement(ema_records, selection)}
    for model in ("seed_8101", "seed_8102", "bagged"):
        per_scenario = {}
        for scenario in campaign_protocol.WEIGHTS:
            base_rows, ema_rows = [], []
            for batch in ("batch_1", "batch_2"):
                base_rows.append(baseline["scenarios"][batch][model][scenario]["live_dead"]["brier"])
                ema_rows.append(report["batches"][batch][model][scenario]["live_dead"]["brier"])
            per_scenario[scenario] = {"v4_batch_mean_brier": float(np.mean(base_rows)),
                "ema_batch_mean_brier": float(np.mean(ema_rows)),
                "ema_minus_v4": float(np.mean(ema_rows) - np.mean(base_rows))}
        base = [baseline["scenarios"][batch][model][scenario]["live_dead"]["brier"]
                for batch in ("batch_1", "batch_2") for scenario in campaign_protocol.WEIGHTS]
        ema = [report["batches"][batch][model][scenario]["live_dead"]["brier"]
               for batch in ("batch_1", "batch_2") for scenario in campaign_protocol.WEIGHTS]
        comparisons[model] = {"batch_scenario_equal_weight_mean_v4_brier": float(np.mean(base)),
                              "batch_scenario_equal_weight_mean_ema_brier": float(np.mean(ema)),
                              "ema_minus_v4": float(np.mean(ema) - np.mean(base)),
                              "per_scenario": per_scenario}
    report["paired_public_brier_comparison"] = comparisons
    report["paired_seed_disagreement"] = disagreements
    report["paired_comparison_note"] = (
        "The equal-weight mean averages two batches by four scenarios. Clean and replay masks are "
        "reused across batches. Seed disagreement is measured on paired OOF live probabilities, "
        "not as independent model-seed replications.")
    write_json(audit_path, report)
    return report


def run(output: Path = OUT):
    protocol, result, protocol_sha, bagging_dir, later_sources = _verify_frozen_result()
    all_reports = {"schema": "post_v4_ema_calibration_audit_v1",
                   "ema_protocol_sha256": protocol_sha,
                   "ema_result_sha256": sha256_file(EMA_DIR / "result.json"),
                   "post_freeze_unreferenced_sources": later_sources,
                   "development_only": True, "target_possession_labels_read": False,
                   "final_all_seven_models_evaluated_on_training_matches": False,
                   "batches": {}}
    for batch_row in result["batches"]:
        batch = batch_row["batch"]
        saved_seed_scores = batch_row["ema_inner_by_seed"]
        seeds = {seed: load_ema_records(batch, seed, protocol, bagging_dir, batch_row)
                 for seed in SEEDS}
        report = live_dead_audit._audit_model(
            batch, seeds, protocol["candidate_selection"], "all")
        for rows in report.values():
            rows["weighted_objective"] = float(sum(
                campaign_protocol.WEIGHTS[scenario]
                * rows[scenario]["mean_spell_f1_across_matches"]
                for scenario in campaign_protocol.WEIGHTS))
        comparisons = {"bagged": _compare_saved(report["bagged"], batch_row["ema_inner"],
                                                   f"{batch}/bagged")}
        for seed in SEEDS:
            comparisons[f"seed_{seed}"] = _compare_saved(
                report[f"seed_{seed}"], saved_seed_scores[str(seed)], f"{batch}/seed_{seed}")
        report["saved_score_reconstruction"] = comparisons
        report["frozen_ema_inner_gate"] = batch_row["inner_checks"]
        report["frozen_outer_guard"] = batch_row["outer_checks"]
        all_reports["batches"][batch] = report
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "audit.json", all_reports)
    (output / "README.md").write_text(
        "# Post-v4 tail-EMA calibration audit\n\n"
        "This read-only audit consumes authenticated six-match inner-fold EMA predictions. "
        "It does not train models, read target possession labels, or evaluate final all-seven models "
        "on training matches. The EMA model uses the same 25 Hz prediction path as v4. "
        "The training tail covers updates 7000 to 7200 with decay 0.99. See `audit.json` for "
        "per-seed and bagged Brier score, log loss, reliability bins, full-gap frame strata, "
        "official half and match metrics, and exact reconstruction checks.\n")
    return all_reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--paired-comparison-only", action="store_true")
    args = parser.parse_args()
    report = (add_paired_calibration_comparison(args.output) if args.paired_comparison_only
              else run(args.output))
    batches = sorted(report.get("batches", {}))
    print(json.dumps({"output": str(args.output), "batches": batches,
                      "target_possession_labels_read": False}, sort_keys=True))


if __name__ == "__main__":
    main()
