"""Export compact decoded-label references from frozen public DFL OOF caches."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation import possession_metrics as official
from experiments import campaign_evaluation, campaign_protocol
from experiments.campaign_decode import decode_arrays
from experiments.coarse_stability import (
    _INNER_SEEDS,
    _OUTER_SEEDS,
    _average_logits,
    _bagged_loaded,
    _mix_coarse,
    _verify_ema_result,
)
from experiments.ema_stability import (
    _ema_model_records,
    _read_original_seed_records,
    _verify_input_cache,
)
from experiments.ensemble_polish_confirm import _folds
from experiments.robustness import _execution_environment, _source_hashes
from possession.prepared import verify_prepared
from possession.storage import sha256_file

_SCHEMA = "public_dfl_decoded_reference_v1"
_VARIANTS = ("v4", "ema_v5", "coarse_v6")
_SCENARIOS = ("clean", "short", "long", "replay")
_DEFAULT_ARTIFACTS = Path("artifacts")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode()).hexdigest()


def _required(path: Path) -> dict:
    value = campaign_protocol.read_record(path)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _hash_study(directory: Path) -> dict:
    return {"protocol_sha256": sha256_file(directory / "protocol.json"),
            "result_sha256": sha256_file(directory / "result.json")}


def _portable_source_fingerprints(sources: dict) -> dict:
    return {match: {
        kind: {"file_id": Path(record["path"]).name, "sha256": record["sha256"]}
        for kind, record in rows.items()}
        for match, rows in sources.items()}


def _dataset_attribution() -> dict:
    return {
        "authors": "Manuel Bassek, Robert Rein, Hendrik Weber, Daniel Memmert",
        "year": 2025,
        "title": "An integrated dataset of spatiotemporal and event data in elite soccer",
        "journal": "Scientific Data 12, 195",
        "paper_doi": "https://doi.org/10.1038/s41597-025-04505-y",
        "dataset_version": 1,
        "dataset_doi": "https://doi.org/10.6084/m9.figshare.28196177.v1",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "changes": "This reference contains only ground-truth and decoded prediction label codes. It excludes source position data and model inputs.",
    }


def _verify_relative_artifacts(directory: Path, hashes: dict) -> None:
    root = directory.resolve()
    for relative, expected in hashes.items():
        rel = Path(relative)
        path = (directory / rel).resolve()
        if rel.is_absolute() or ".." in rel.parts or not path.is_relative_to(root):
            raise ValueError(f"unsafe frozen artifact path: {relative}")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen artifact checksum differs: {relative}")


def _score_exported(records: list[dict], decoder: dict) -> dict:
    from experiments.campaign_evaluation import score_records

    weights = campaign_protocol.WEIGHTS
    return score_records(records, decoder, weights)


def _public_records(loaded: list[dict], variant: str, batch: str, scope: str,
                    decoder: dict, arrays_out: dict[str, np.ndarray], rows_out: list[dict],
                    truth_by_half: dict[tuple[str, int], np.ndarray]) -> None:
    for record in loaded:
        arrays = record["arrays"]
        match, period, scenario = record["match"], int(record["period"]), record["scenario"]
        truth = np.asarray(arrays["gt"], dtype=np.uint8)
        key = (match, period)
        previous = truth_by_half.setdefault(key, truth.copy())
        if not np.array_equal(previous, truth):
            raise ValueError("public ground-truth timelines differ across scenarios or batches")
        arrays_out.setdefault(f"gt_{match}_p{period}", truth)
        pred = np.asarray(decode_arrays(arrays, decoder), dtype=np.uint8)
        pred_key = f"pred_{variant}_{batch}_{match}_{scenario}_p{period}"
        arrays_out[pred_key] = pred
        identity = record.get("model_identity", {})
        fold = identity.get("fold", identity.get("base", {}).get("fold"))
        rows_out.append({"batch": batch, "scope": scope, "match": match,
                         "fold": fold,
                         "scenario": scenario, "period": period,
                         "n_frames": len(truth), "ground_truth_key": f"gt_{match}_p{period}",
                         "prediction_keys": {variant: pred_key}})


def finalize_existing_reference(artifacts_root: str | Path, data_dir: str | Path,
                                output_dir: str | Path) -> dict:
    """Attach frozen score/provenance metadata to an already exported label archive."""
    artifacts_root, data_dir, output_dir = (Path(p).resolve()
                                             for p in (artifacts_root, data_dir, output_dir))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    campaign_dir = artifacts_root / "overnight-joint-v1"
    bag_dir = artifacts_root / "ensemble-bagging-v1"
    ema_dir = artifacts_root / "ema-stability-v1"
    coarse_dir = artifacts_root / "coarse-stability-v1"
    labels_path = output_dir / "labels.npz"
    if not labels_path.is_file() or (output_dir / "manifest.json").exists():
        raise ValueError("recovery requires labels.npz and no existing manifest")
    campaign = _required(campaign_dir / "protocol.json")
    bag_protocol, bag_result = _required(bag_dir / "protocol.json"), _required(bag_dir / "result.json")
    ema_protocol, ema_result = _required(ema_dir / "protocol.json"), _required(ema_dir / "result.json")
    coarse_protocol, coarse_result = (_required(coarse_dir / "protocol.json"),
                                      _required(coarse_dir / "result.json"))
    if not (bag_result.get("eligible") and ema_result.get("eligible")
            and coarse_result.get("eligible")):
        raise ValueError("all frozen development studies must be eligible")
    if (bag_result.get("protocol_sha256") != _digest(bag_protocol)
            or ema_result.get("protocol_sha256") != _digest(ema_protocol)
            or coarse_result.get("protocol_sha256") != _digest(coarse_protocol)
            or coarse_protocol.get("baseline_kind") != "ema"
            or coarse_protocol.get("ema_protocol_sha256") != _digest(ema_protocol)
            or coarse_protocol.get("bagging_protocol_sha256") != _digest(bag_protocol)
            or coarse_protocol.get("campaign_protocol_sha256") != _digest(campaign)):
        raise ValueError("frozen study ancestry differs")
    if coarse_protocol.get("runtime") != _execution_environment():
        raise ValueError("current runtime differs from frozen source study")
    sources = _source_hashes()
    pinned = coarse_protocol.get("source_sha256", {})
    if not pinned or any(sources.get(name) != checksum for name, checksum in pinned.items()):
        raise ValueError("source hashes differ from the frozen coarse study")
    prepared = verify_prepared(data_dir, sorted(data_dir.glob("DFL-MAT-*_half*.npz")))
    if prepared != campaign.get("prepared"):
        raise ValueError("prepared public DFL provenance differs")
    _verify_ema_result(ema_dir, bag_dir, ema_result, _digest(ema_protocol),
                       coarse_protocol["candidate_selection"], bag_result)
    _verify_relative_artifacts(coarse_dir, {
        name: checksum for batch in coarse_result["batches"]
        for name, checksum in batch["artifact_sha256"].items()})
    folds = _folds(campaign)
    expected_matches = {match: ("inner", index) for index, fold in enumerate(folds)
                        for match in fold["validation"]}
    expected_matches["DFL-MAT-J03WMX"] = ("outer", 0)
    expected_scores = {variant: {} for variant in _VARIANTS}
    ema_batches = {row["batch"]: row for row in ema_result["batches"]}
    coarse_batches = {row["batch"]: row for row in coarse_result["batches"]}
    batch_names = ("batch_1", "batch_2")
    for batch in batch_names:
        expected_scores["v4"][batch] = {}
        expected_scores["ema_v5"][batch] = {}
        expected_scores["coarse_v6"][batch] = {}
        for scope in ("inner", "outer"):
            scope_key = "inner" if scope == "inner" else "outer"
            ema_row, coarse_row = ema_batches[batch], coarse_batches[batch]
            expected_scores["v4"][batch][scope] = (
                ema_row["raw_v4_inner"] if scope == "inner" else ema_row["raw_v4_outer"])
            expected_scores["ema_v5"][batch][scope] = (
                ema_row["ema_inner"] if scope == "inner" else ema_row["ema_outer"])
            expected_scores["coarse_v6"][batch][scope] = coarse_row["candidate_scores"][scope_key]
    records = []
    expected_keys = set()
    with np.load(labels_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for batch in batch_names:
        for match, (scope, fold) in sorted(expected_matches.items()):
            for scenario in _SCENARIOS:
                for period in (1, 2):
                    gt_key = f"gt_{match}_p{period}"
                    expected_keys.add(gt_key)
                    if gt_key not in arrays:
                        raise ValueError(f"label archive is missing {gt_key}")
                    truth = arrays[gt_key]
                    if truth.ndim != 1 or truth.dtype.kind not in "iu" or np.any(truth < 0) or np.any(truth > official.X):
                        raise ValueError("public reference contains invalid ground-truth codes")
                    prediction_keys = {}
                    for variant in _VARIANTS:
                        pred_key = f"pred_{variant}_{batch}_{match}_{scenario}_p{period}"
                        expected_keys.add(pred_key)
                        prediction_keys[variant] = pred_key
                        pred = arrays.get(pred_key)
                        if (pred is None or pred.shape != truth.shape or pred.dtype.kind not in "iu"
                                or np.any(pred < 0) or np.any(pred > official.X)):
                            raise ValueError(f"public reference prediction array is invalid: {pred_key}")
                    records.append({"batch": batch, "scope": scope, "match": match,
                                    "fold": fold, "scenario": scenario, "period": period,
                                    "n_frames": len(truth), "ground_truth_key": gt_key,
                                    "prediction_keys": prediction_keys})
    if set(arrays) != expected_keys or len(records) != 2 * 7 * 4 * 2:
        raise ValueError("decoded-label archive does not match the frozen seven-match grid")
    manifest = {
        "schema": _SCHEMA,
        "title": "Public DFL decoded-label development reference",
        "interpretation": "Recomputation from saved public OOF predictions. Not retraining or an independent test.",
        "target_ground_truth_used": False, "target_inputs_included": False,
        "raw_participant_assets_included": False,
        "included_fields": ["H/A/D/X ground-truth class codes", "decoded H/A/D/X prediction class codes"],
        "excluded_fields": ["player coordinates", "ball coordinates", "input features", "outage masks", "target arrays", "model weights"],
        "label_codes": {"H": official.H, "A": official.A, "D": official.D, "X": official.X},
        "fps": official.FPS,
        "half_join_rule": "Score each match and scenario after concatenating period 1, one official X sentinel, then period 2. This keeps spells separated at halftime.",
        "scenario_weights": campaign_protocol.WEIGHTS,
        "scenario_order": sorted(campaign_protocol.WEIGHTS), "variants": list(_VARIANTS),
        "variant_meaning": {"v4": "Frozen four-network seed-bagged development candidate, before EMA.",
                            "ema_v5": "Same frozen candidate with its saved tail-EMA checkpoint predictions.",
                            "coarse_v6": "EMA v5 plus the frozen gated one-hertz player-context specialist."},
        "batches": list(batch_names),
        "cv": {"inner_folds": folds, "inner_seeds": list(_INNER_SEEDS),
               "outer_seeds": list(_OUTER_SEEDS), "outer_validation_match": "DFL-MAT-J03WMX",
               "outer_train_matches": campaign["fold"]["outer_train"]},
        "source_provenance": {
            "dataset": "public DFL prepared data", "prepared_manifest_sha256": _digest(prepared),
            "public_source_fingerprints": _portable_source_fingerprints(prepared.get("sources", {})),
            "dataset_attribution": _dataset_attribution(),
            "campaign_protocol_sha256": _digest(campaign), "coarse_pinned_source_sha256": pinned,
            "studies": {"v4_seed_bagging": _hash_study(bag_dir), "ema_v5": _hash_study(ema_dir),
                        "coarse_v6": _hash_study(coarse_dir)},
            "mask_batch_specs": bag_protocol["mask_batches"],
            "input_cache_signatures": {
                batch: {"inner": ema_batches[batch]["inner_inputs_sha256"],
                        "outer": ema_batches[batch]["outer_inputs_sha256"]}
                for batch in batch_names},
            "batch_result_sha256": {
                "v4_seed_bagging": {batch: sha256_file(bag_dir / batch / "result.json") for batch in batch_names},
                "ema_v5": {batch: sha256_file(ema_dir / "batches" / batch / "result.json") for batch in batch_names},
                "coarse_v6": {batch: sha256_file(coarse_dir / "batches" / batch / "result.json") for batch in batch_names}},
        },
        "decoder": coarse_protocol["candidate_selection"]["decoder"],
        "expected_scores": expected_scores, "records": records,
        "labels_file": "labels.npz", "labels_sha256": sha256_file(labels_path),
        "labels_bytes": labels_path.stat().st_size, "label_array_count": len(arrays),
        "record_count": len(records),
        "label_array_hashes": {key: hashlib.sha256(value.tobytes(order="C")).hexdigest()
                               for key, value in sorted(arrays.items())},
        "recovery_note": "Manifest finalized from the label archive written after all exporter score-equality checks succeeded.",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {"manifest_path": str(manifest_path), "labels_path": str(labels_path),
            "labels_bytes": labels_path.stat().st_size, "manifest_bytes": manifest_path.stat().st_size,
            "record_count": len(records), "label_array_count": len(arrays),
            "recovery": True}


def export_reference(artifacts_root: str | Path, data_dir: str | Path,
                     output_dir: str | Path) -> dict:
    """Authenticate cached studies and export labels only, without fitting or target access."""
    artifacts_root, data_dir, output_dir = (Path(p).resolve()
                                             for p in (artifacts_root, data_dir, output_dir))
    campaign_dir = artifacts_root / "overnight-joint-v1"
    bag_dir = artifacts_root / "ensemble-bagging-v1"
    ema_dir = artifacts_root / "ema-stability-v1"
    coarse_dir = artifacts_root / "coarse-stability-v1"
    if any(output_dir == path or output_dir.is_relative_to(path) or path.is_relative_to(output_dir)
           for path in (artifacts_root, data_dir, campaign_dir, bag_dir, ema_dir, coarse_dir)):
        raise ValueError("reference output must be separate from all source artifacts")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    campaign = _required(campaign_dir / "protocol.json")
    bag_protocol, bag_result = _required(bag_dir / "protocol.json"), _required(bag_dir / "result.json")
    ema_protocol, ema_result = _required(ema_dir / "protocol.json"), _required(ema_dir / "result.json")
    coarse_protocol, coarse_result = (_required(coarse_dir / "protocol.json"),
                                      _required(coarse_dir / "result.json"))
    if not (bag_result.get("eligible") and ema_result.get("eligible")
            and coarse_result.get("eligible")):
        raise ValueError("all frozen development studies must be eligible")
    if (bag_result.get("protocol_sha256") != _digest(bag_protocol)
            or ema_result.get("protocol_sha256") != _digest(ema_protocol)
            or coarse_result.get("protocol_sha256") != _digest(coarse_protocol)):
        raise ValueError("a saved study result is detached from its protocol")
    if (coarse_protocol.get("baseline_kind") != "ema"
            or coarse_protocol.get("ema_protocol_sha256") != _digest(ema_protocol)
            or coarse_protocol.get("bagging_protocol_sha256") != _digest(bag_protocol)
            or coarse_protocol.get("campaign_protocol_sha256") != _digest(campaign)):
        raise ValueError("coarse comparison baseline ancestry differs")
    if coarse_protocol.get("runtime") != _execution_environment():
        raise ValueError("current runtime differs from the frozen coarse study")
    sources = _source_hashes()
    pinned = coarse_protocol.get("source_sha256", {})
    if not pinned or any(sources.get(name) != digest for name, digest in pinned.items()):
        raise ValueError("current source differs from the frozen coarse study source pins")
    if verify_prepared(data_dir, sorted(data_dir.glob("DFL-MAT-*_half*.npz"))) != campaign.get("prepared"):
        raise ValueError("prepared public DFL provenance differs from the campaign")
    folds = _folds(campaign)
    if len(folds) != 3 or {m for fold in folds for m in fold["validation"]} != {
            "DFL-MAT-J03WN1", "DFL-MAT-J03WOH", "DFL-MAT-J03WOY",
            "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9"}:
        raise ValueError("unexpected public inner OOF grid")
    decoder = coarse_protocol["candidate_selection"]["decoder"]
    selection = coarse_protocol["candidate_selection"]
    _verify_ema_result(ema_dir, bag_dir, ema_result, _digest(ema_protocol),
                       selection, bag_result)
    _verify_relative_artifacts(coarse_dir, {
        name: checksum for batch in coarse_result["batches"]
        for name, checksum in batch["artifact_sha256"].items()})

    fit_rows = {}
    outer_folds = [{"train": campaign["fold"]["outer_train"],
                    "validation": ["DFL-MAT-J03WMX"]}]
    for scope, seeds, scope_folds in (("inner", _INNER_SEEDS, folds),
                                      ("outer", _OUTER_SEEDS, outer_folds)):
        for seed in seeds:
            for fold_index, fold in enumerate(scope_folds):
                for member in selection["members"]:
                    fit_path = (ema_dir / "fits" / scope / member["id"] / f"seed_{seed}"
                                / f"fold_{fold_index:02d}" / "fit.json")
                    fit = _required(fit_path)
                    model_path = fit_path.parent / "model.pt"
                    if sha256_file(model_path) != fit.get("checkpoint_sha256"):
                        raise ValueError("EMA fit checkpoint hash differs")
                    fit_rows[(scope, seed, member["id"], fold_index)] = (fit, fit_path, model_path)

    batch_rows = {row["batch"]: row for row in coarse_result["batches"]}
    ema_batch_rows = {row["batch"]: row for row in ema_result["batches"]}
    arrays_out: dict[str, np.ndarray] = {}
    records_out: list[dict] = []
    expected_scores = {variant: {} for variant in _VARIANTS}
    truth_by_half: dict[tuple[str, int], np.ndarray] = {}
    for batch_index, batch in enumerate(("batch_1", "batch_2")):
        bag_result_path = bag_dir / batch / "result.json"
        raw_batch = _required(bag_result_path)
        if raw_batch.get("protocol_sha256") != _digest(bag_protocol):
            raise ValueError("bagging batch result protocol differs")
        coarse_batch = batch_rows[batch]
        ema_batch = ema_batch_rows[batch]
        expected_scores["v4"][batch] = {}
        expected_scores["ema_v5"][batch] = {}
        expected_scores["coarse_v6"][batch] = {}
        for scope, seeds, scope_folds in (("inner", _INNER_SEEDS, folds),
                                          ("outer", _OUTER_SEEDS, outer_folds)):
            input_sha = raw_batch[f"{scope}_inputs_sha256"]
            expected_matches = {match for fold in scope_folds for match in fold["validation"]}
            input_records, _manifest = _verify_input_cache(
                bag_dir / batch / scope, input_sha, expected_matches)
            baseline_by_seed = {}
            for seed in seeds:
                bundles = _read_original_seed_records(
                    bag_dir, batch, scope, seed, selection, campaign, campaign_dir,
                    scope_folds, input_sha)
                baseline_by_seed[seed] = [campaign_evaluation.load_records(bundle)
                                          for bundle in bundles]
            v4_records = _bagged_loaded(baseline_by_seed, selection)
            v4_score = _score_exported(v4_records, decoder)
            saved_v4 = (ema_batch["raw_v4_inner"] if scope == "inner"
                        else ema_batch["raw_v4_outer"])
            if v4_score != saved_v4:
                raise ValueError(f"v4 {batch}/{scope} public score differs")
            expected_scores["v4"][batch][scope] = v4_score
            _public_records(v4_records, "v4", batch, scope, decoder, arrays_out, records_out, truth_by_half)
            del v4_records, baseline_by_seed
            gc.collect()

            ema_by_seed = {}
            for seed in seeds:
                bundles = _ema_model_records(
                    ema_dir, batch, scope, seed, selection, scope_folds, input_records,
                    input_sha, fit_rows, _digest(ema_protocol), lambda: None)
                ema_by_seed[seed] = [campaign_evaluation.load_records(bundle)
                                     for bundle in bundles]
            ema_records = _bagged_loaded(ema_by_seed, selection)
            ema_score = _score_exported(ema_records, decoder)
            saved_ema = (ema_batch["ema_inner"] if scope == "inner"
                         else ema_batch["ema_outer"])
            if ema_score != saved_ema:
                raise ValueError(f"EMA v5 {batch}/{scope} public score differs")
            expected_scores["ema_v5"][batch][scope] = ema_score
            _public_records(ema_records, "ema_v5", batch, scope, decoder, arrays_out, records_out, truth_by_half)

            fold_for_match = {match: index for index, fold in enumerate(scope_folds)
                              for match in fold["validation"]}
            per_seed_coarse = {}
            for seed in seeds:
                seed_predictions = {}
                for match, fold_index in fold_for_match.items():
                    for period in (1, 2):
                        base = (coarse_dir / "coarse_predictions" / scope / f"seed_{seed}"
                                / f"fold_{fold_index:02d}" / f"{match}_half{period}")
                        metadata_path, prediction_path = base.with_suffix(".json"), base.with_suffix(".npz")
                        metadata = json.loads(metadata_path.read_text())
                        with np.load(prediction_path, allow_pickle=False) as saved:
                            prediction = {key: saved[key] for key in saved.files}
                        identity = metadata.get("identity", {})
                        if (identity.get("scope") != scope or identity.get("seed") != seed
                                or identity.get("fold") != fold_index or identity.get("match") != match
                                or identity.get("period") != period
                                or metadata.get("prediction_sha256") != sha256_file(prediction_path)):
                            raise ValueError("coarse prediction cache identity differs")
                        seed_predictions[(match, period)] = prediction
                per_seed_coarse[seed] = seed_predictions
            coarse_by_key = {}
            for key in per_seed_coarse[seeds[0]]:
                coarse_by_key[key] = {
                    "live_probability": _average_logits(
                        per_seed_coarse[seeds[0]][key]["live_probability"],
                        per_seed_coarse[seeds[1]][key]["live_probability"]),
                    "home_probability": _average_logits(
                        per_seed_coarse[seeds[0]][key]["home_probability"],
                        per_seed_coarse[seeds[1]][key]["home_probability"]),
                }
            coarse_records = _mix_coarse(ema_records, coarse_by_key, batch_name=batch)
            coarse_score = _score_exported(coarse_records, decoder)
            saved_coarse = (coarse_batch["candidate_scores"]["inner"] if scope == "inner"
                            else coarse_batch["candidate_scores"]["outer"])
            if coarse_score != saved_coarse:
                raise ValueError(f"coarse v6 {batch}/{scope} public score differs")
            expected_scores["coarse_v6"][batch][scope] = coarse_score
            _public_records(coarse_records, "coarse_v6", batch, scope, decoder, arrays_out, records_out,
                            truth_by_half)
            del ema_records, ema_by_seed, coarse_records, per_seed_coarse
            gc.collect()

    # The shared record grid is kept once per variant. Each variant key is recorded by variant.
    merged = {}
    for row in records_out:
        key = (row["batch"], row["scope"], row["match"], row["scenario"], row["period"])
        entry = merged.setdefault(key, {k: row[k] for k in
                                        ("batch", "scope", "match", "fold", "scenario", "period",
                                         "n_frames", "ground_truth_key")})
        variant = next(iter(row["prediction_keys"]))
        entry.setdefault("prediction_keys", {})[variant] = row["prediction_keys"][variant]
    compact_records = [merged[key] for key in sorted(merged)]
    required_variants = set(_VARIANTS)
    if any(set(row["prediction_keys"]) != required_variants for row in compact_records):
        raise ValueError("export did not produce all three variants for each reference row")
    if len(compact_records) != 2 * 7 * len(_SCENARIOS) * 2:
        raise ValueError("export grid must contain two batches, seven matches, four scenarios, two halves")
    gt_keys = {row["ground_truth_key"] for row in compact_records}
    if len(gt_keys) != 14:
        raise ValueError("ground-truth arrays must be shared across scenarios and batches")

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "labels.npz"
    with arrays_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays_out)
    prepared = campaign["prepared"]
    manifest = {
        "schema": _SCHEMA,
        "title": "Public DFL decoded-label development reference",
        "interpretation": "Recomputation from saved public OOF predictions. Not retraining or an independent test.",
        "target_ground_truth_used": False,
        "target_inputs_included": False,
        "raw_participant_assets_included": False,
        "included_fields": ["H/A/D/X ground-truth class codes", "decoded H/A/D/X prediction class codes"],
        "excluded_fields": ["player coordinates", "ball coordinates", "input features", "outage masks", "target arrays", "model weights"],
        "label_codes": {"H": official.H, "A": official.A, "D": official.D, "X": official.X},
        "fps": official.FPS,
        "half_join_rule": "Score each match and scenario after concatenating period 1, one official X sentinel, then period 2. This keeps spells separated at halftime.",
        "scenario_weights": campaign_protocol.WEIGHTS,
        "scenario_order": sorted(campaign_protocol.WEIGHTS),
        "variants": list(_VARIANTS),
        "variant_meaning": {
            "v4": "Frozen four-network seed-bagged development candidate, before EMA.",
            "ema_v5": "Same frozen candidate with its saved tail-EMA checkpoint predictions.",
            "coarse_v6": "EMA v5 plus the frozen gated one-hertz player-context specialist.",
        },
        "batches": ["batch_1", "batch_2"],
        "cv": {"inner_folds": folds,
               "inner_seeds": list(_INNER_SEEDS), "outer_seeds": list(_OUTER_SEEDS),
               "outer_validation_match": "DFL-MAT-J03WMX",
               "outer_train_matches": campaign["fold"]["outer_train"]},
        "source_provenance": {
            "dataset": "public DFL prepared data",
            "prepared_manifest_sha256": _digest(prepared),
            "public_source_fingerprints": _portable_source_fingerprints(prepared.get("sources", {})),
            "dataset_attribution": _dataset_attribution(),
            "campaign_protocol_sha256": _digest(campaign),
            "coarse_pinned_source_sha256": pinned,
            "studies": {"v4_seed_bagging": _hash_study(bag_dir),
                        "ema_v5": _hash_study(ema_dir),
                        "coarse_v6": _hash_study(coarse_dir)},
            "mask_batch_specs": bag_protocol["mask_batches"],
            "input_cache_signatures": {
                batch: {"inner": ema_batch_rows[batch]["inner_inputs_sha256"],
                        "outer": ema_batch_rows[batch]["outer_inputs_sha256"]}
                for batch in ("batch_1", "batch_2")},
        "batch_result_sha256": {
            "v4_seed_bagging": {batch: sha256_file(bag_dir / batch / "result.json")
                                for batch in ("batch_1", "batch_2")},
            "ema_v5": {batch: sha256_file(ema_dir / "batches" / batch / "result.json")
                       for batch in ("batch_1", "batch_2")},
            "coarse_v6": {batch: sha256_file(coarse_dir / "batches" / batch / "result.json")
                           for batch in ("batch_1", "batch_2")}},
        },
        "decoder": decoder,
        "expected_scores": expected_scores,
        "records": compact_records,
        "labels_file": "labels.npz",
        "labels_sha256": sha256_file(arrays_path),
        "labels_bytes": arrays_path.stat().st_size,
        "label_array_count": len(arrays_out),
        "record_count": len(compact_records),
        "label_array_hashes": {key: hashlib.sha256(value.tobytes(order="C")).hexdigest()
                               for key, value in sorted(arrays_out.items())},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
    return {"manifest_path": str(manifest_path), "labels_path": str(arrays_path),
            "labels_bytes": arrays_path.stat().st_size,
            "manifest_bytes": manifest_path.stat().st_size,
            "record_count": len(compact_records),
            "label_array_count": len(arrays_out),
            "scores": {variant: {batch: {scope: expected_scores[variant][batch][scope]["objective"]
                                          for scope in ("inner", "outer")}
                                 for batch in ("batch_1", "batch_2")}
                       for variant in _VARIANTS}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=_DEFAULT_ARTIFACTS)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/dfl"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/v6/public-reference"))
    parser.add_argument("--finalize-existing", action="store_true",
                        help="validate and finalize labels.npz from an interrupted completed export")
    args = parser.parse_args()
    result = (finalize_existing_reference(args.artifacts_root, args.data_dir, args.output_dir)
              if args.finalize_existing else
              export_reference(args.artifacts_root, args.data_dir, args.output_dir))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
