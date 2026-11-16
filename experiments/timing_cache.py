"""Stream verified public DFL v6 evidence into a separate compact timing cache.

Only saved public prediction bundles, prepared public cache hashes and public
reference labels are accessed. This module never opens participant inputs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments import campaign_protocol
from experiments.campaign_decode import decode_arrays
from experiments.coarse_context import blend_coarse_context
from experiments.coarse_stability import _average_logits, _bagged_loaded
from experiments.ema_stability import _verify_input_cache
from experiments.robustness import _source_hashes
from experiments.timing_validation import (
    BasePredictionProvenance,
    ComponentProvenance,
    ConfigLineage,
    canonical_sha256,
    validate_prediction_provenance,
)
from experiments.tune import _seed_for_fold
from possession.prepared import verify_prepared
from possession.storage import sha256_file

_DEFAULT_ARTIFACTS = Path("/Volumes/T7/Data/more-than-a-hack-2027/artifacts")
_REPO = Path(__file__).resolve().parents[1]
_NEURAL = ("neural_live", "neural_home", "neural_boundary")


def _required(path):
    record = campaign_protocol.read_record(path)
    if record is None:
        raise FileNotFoundError(path)
    return record


def _check_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError(f"artifact checksum differs: {path}")


def _public_path(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("public prediction path is outside its declared artifact directory")
    return path


def _fit_component(root, scope, seed, index, fold, spec, name, protocol_sha, *, fine):
    """Bind declared split to saved fit, checkpoint config and EMA raw ancestry."""
    import torch

    if fine:
        directory = root / "ema-stability-v1" / "fits" / scope / name / f"seed_{seed}"
    else:
        directory = root / "coarse-stability-v1" / "fits" / scope / f"seed_{seed}"
    directory /= f"fold_{index:02d}"
    fit_path, model_path = directory / "fit.json", directory / "model.pt"
    fit = _required(fit_path)
    expected = {
        "protocol_sha256": protocol_sha,
        "spec_sha256": canonical_sha256(spec),
        "train_matches": fold["train"],
        "seed": _seed_for_fold(seed, "DFL-MAT-J03WMX", index),
        "name": name,
    }
    if fit.get("identity") != expected or fit.get("steps") != 7200:
        raise ValueError("saved base fit identity differs from declared fold")
    _check_hash(model_path, fit.get("checkpoint_sha256"))
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    config = payload.get("training_config", {})
    if (
        config.get("train_matches") != fold["train"]
        or config.get("seed") != expected["seed"]
        or config.get("steps") != 7200
        or config.get("campaign_spec") != spec
        or config.get("validation_matches") not in ([], fold["validation"])
    ):
        raise ValueError("base checkpoint training config differs from fit ancestry")
    ancestors = []
    if fine:
        raw_dir = (
            root / "overnight-joint-v1" / "fits" / scope / name / f"seed_{seed}" / f"fold_{index}"
        )
        raw_fit_path = raw_dir / "rung_7200" / "fit.json"
        raw_fit = _required(raw_fit_path)
        if raw_fit.get("identity") != expected or raw_fit.get("steps") != 7200:
            raise ValueError("EMA raw fit ancestry differs")
        inputs = fit.get("inputs", {})
        _check_hash(raw_dir / "rung_7200" / "model.pt", raw_fit.get("checkpoint_sha256"))
        if raw_fit["checkpoint_sha256"] != inputs.get("baseline_checkpoint_sha256"):
            raise ValueError("EMA baseline checkpoint differs from raw ancestor")
        for step, key in ((7000, "start_state_sha256"), (7200, "end_state_sha256")):
            _check_hash(raw_dir / f"state_{step:06d}.pt", inputs.get(key))
        if fit.get("tail_ema", {}).get("decay") != 0.99 or not all(
            fit.get("raw_parity", {}).get(k)
            for k in (
                "raw_weights_match_endpoint_state",
                "raw_weights_match_baseline_checkpoint",
                "raw_training_progress_matches_endpoint",
            )
        ):
            raise ValueError("EMA tail recipe or replay parity proof differs")
        ancestors.append(
            ComponentProvenance(
                f"{name}_raw_seed_{seed}",
                "fine_model",
                fold["train"],
                [],
                sha256_file(raw_fit_path),
                raw_fit["checkpoint_sha256"],
            )
        )
    component = ComponentProvenance(
        f"{name}_seed_{seed}",
        "fine_model" if fine else "coarse_model",
        fold["train"],
        [],
        sha256_file(fit_path),
        fit["checkpoint_sha256"],
        ancestors,
    )
    return fit, fit_path, model_path, component


def _input_arrays(record):
    """Read compact fields only, leaving large player token members compressed."""
    with np.load(record["input_path"], allow_pickle=False) as source:
        keep = {"frame_id", "gt", "ball_valid", "nearest_ball_seconds"}
        keep.update(name for name in source.files if name.startswith("ref_"))
        arrays = {name: source[name] for name in keep}
    n = len(arrays["frame_id"])
    if (
        n == 0
        or arrays["frame_id"].dtype.kind not in "iu"
        or np.any(np.diff(arrays["frame_id"]) != 1)
        or arrays["gt"].shape != (n,)
        or arrays["ball_valid"].shape != (n,)
        or arrays["ball_valid"].dtype.kind != "b"
        or arrays["nearest_ball_seconds"].shape != (n,)
    ):
        raise ValueError("invalid compact public input timeline")
    return arrays


def _fine_arrays(record, source, expected_identity, root):
    if (
        record.get("model_identity") != expected_identity
        or record.get("input_sha256") != source["input_sha256"]
        or record.get("input_identity") != source["input_identity"]
        or Path(record["input_path"]).resolve() != Path(source["input_path"]).resolve()
    ):
        raise ValueError("fine prediction input or model ancestry differs")
    path = _public_path(record["prediction_path"], root / "ema-stability-v1" / "predictions")
    _check_hash(path, record.get("prediction_sha256"))
    metadata = json.loads(path.with_suffix(".json").read_text())
    expected = {
        "schema": "campaign_neural_prediction_v1",
        "input_sha256": source["input_sha256"],
        "model_identity": expected_identity,
        "center_size": record["center_size"],
        "match": source["match"],
        "period": source["period"],
        "scenario": source["scenario"],
    }
    # The original prediction schema is imported to avoid silently accepting a different format.
    from experiments.campaign_evaluation import PREDICTION_SCHEMA

    expected["schema"] = PREDICTION_SCHEMA
    if (
        metadata.get("identity") != expected
        or metadata.get("prediction_sha256") != record["prediction_sha256"]
    ):
        raise ValueError("fine prediction sidecar differs")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(_NEURAL):
            raise ValueError("unexpected fine prediction fields")
        return {name: saved[name] for name in _NEURAL}


def export_timing_cache(
    output_dir, artifacts_root=_DEFAULT_ARTIFACTS, batches=("batch_1", "batch_2")
) -> dict:
    """Export compact evidence with exact frozen v6 parity and complete fit ancestry.

    Existing records may be reproduced, but a differing output is rejected.
    The returned ordinary JSON manifest contains ``records`` with local NPZ paths
    and provenance accepted by ``timing_validation``. Both halves and all four
    scenarios are retained for six inner matches and the historically reused WMX
    guard. This is a development cache, not independent generalization evidence.
    """
    root, output_dir = Path(artifacts_root).resolve(), Path(output_dir).resolve()
    batches = tuple(batches)
    if not batches or len(set(batches)) != len(batches) or set(batches) - {"batch_1", "batch_2"}:
        raise ValueError("batches must be a nonempty subset of the frozen pair")
    if output_dir == root or root.is_relative_to(output_dir):
        raise ValueError("timing output may not contain its source artifact root")
    for study in (
        "ema-stability-v1",
        "coarse-stability-v1",
        "ensemble-bagging-v1",
        "overnight-joint-v1",
    ):
        if output_dir.is_relative_to(root / study):
            raise ValueError("timing output must be separate from frozen studies")
    campaign = _required(root / "overnight-joint-v1/protocol.json")
    protocols, results = {}, {}
    for name in ("ensemble-bagging-v1", "ema-stability-v1", "coarse-stability-v1"):
        protocols[name] = _required(root / name / "protocol.json")
        results[name] = _required(root / name / "result.json")
        if results[name].get("protocol_sha256") != canonical_sha256(protocols[name]) or not results[
            name
        ].get("eligible"):
            raise ValueError("frozen study result is detached from eligible protocol")
    ema, coarse, bag = (
        protocols[x] for x in ("ema-stability-v1", "coarse-stability-v1", "ensemble-bagging-v1")
    )
    if (
        coarse.get("baseline_kind") != "ema"
        or coarse.get("ema_protocol_sha256") != canonical_sha256(ema)
        or coarse.get("bagging_protocol_sha256") != canonical_sha256(bag)
        or coarse.get("campaign_protocol_sha256") != canonical_sha256(campaign)
    ):
        raise ValueError("coarse recipe ancestry differs")
    source_hashes = _source_hashes()
    pinned = coarse.get("source_sha256", {})
    if not pinned or any(source_hashes.get(name) != digest for name, digest in pinned.items()):
        raise ValueError("frozen v6 implementation source pins differ")
    selection = coarse["candidate_selection"]
    if (
        selection != ema["candidate_selection"]
        or selection["decoder"]["reference_mode"] != "default"
    ):
        raise ValueError("unsupported changed recipe or learned reference ancestry")
    data_dir = Path(bag["data_dir"]).resolve()
    public_paths = sorted(data_dir.glob("DFL-MAT-*_half*.npz"))
    if verify_prepared(data_dir, public_paths) != campaign["prepared"]:
        raise ValueError("prepared public DFL provenance differs")
    coarse_data = root / "coarse-stability-v1/derived_data"
    derivation = _required(coarse_data / "coarse-derivation.json")
    if canonical_sha256(derivation) != coarse["coarse_derivation_sha256"]:
        raise ValueError("coarse data derivation differs")
    verify_prepared(coarse_data, sorted(coarse_data.glob("DFL-MAT-*_half*.npz")))
    ref_dir = _REPO / "benchmarks/v6/public-reference"
    reference = json.loads((ref_dir / "manifest.json").read_text())
    _check_hash(ref_dir / "labels.npz", reference["labels_sha256"])
    if reference["decoder"] != selection["decoder"]:
        raise ValueError("public parity reference uses a different decoder")
    folds = campaign["fold"]["inner"]
    if folds != ema["inner_folds"] or folds != coarse["inner_folds"]:
        raise ValueError("frozen fit split declarations differ")
    universe = sorted(set(campaign["fold"]["outer_train"]) | {"DFL-MAT-J03WMX"})
    historical = ConfigLineage("v6_recipe", canonical_sha256(selection), [], universe, True)
    deterministic_sha = canonical_sha256(
        {
            "reference": "default",
            "features": "fixed physical scales",
            "blend": selection["blend_weight"],
        }
    )
    deterministic = ComponentProvenance(
        "fixed_features_reference_blend",
        "deterministic",
        [],
        [],
        deterministic_sha,
        deterministic_sha,
    )
    rows, fit_cache = [], {}
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(ref_dir / "labels.npz", allow_pickle=False) as labels:
        for batch in batches:
            batch_record = _required(root / "ensemble-bagging-v1" / batch / "result.json")
            for scope, seeds, scope_folds in (
                ("inner", (8101, 8102), folds),
                (
                    "outer",
                    (9101, 9102),
                    [{"train": campaign["fold"]["outer_train"], "validation": ["DFL-MAT-J03WMX"]}],
                ),
            ):
                input_sha = batch_record[f"{scope}_inputs_sha256"]
                source_records, _ = _verify_input_cache(
                    root / "ensemble-bagging-v1" / batch / scope,
                    input_sha,
                    {m for f in scope_folds for m in f["validation"]},
                )
                source_index = {(r["match"], r["period"], r["scenario"]): r for r in source_records}
                for index, fold in enumerate(scope_folds):
                    components, fine_index, coarse_fits = [], {}, {}
                    for seed in seeds:
                        for member in selection["members"]:
                            key = (scope, seed, index, member["id"])
                            if key not in fit_cache:
                                fit_cache[key] = _fit_component(
                                    root,
                                    scope,
                                    seed,
                                    index,
                                    fold,
                                    member["spec"],
                                    member["id"],
                                    canonical_sha256(campaign),
                                    fine=True,
                                )
                            fit, fit_path, _, component = fit_cache[key]
                            components.append(component)
                            manifest_path = (
                                root
                                / "ema-stability-v1/predictions"
                                / batch
                                / scope
                                / f"seed_{seed}"
                                / member["id"]
                                / f"fold_{index:02d}.json"
                            )
                            manifest = _required(manifest_path)
                            identity = {
                                "protocol_sha256": canonical_sha256(ema),
                                "ema_fit_sha256": sha256_file(fit_path),
                                "ema_sha256": fit["ema_sha256"],
                                "ema_checkpoint_sha256": fit["checkpoint_sha256"],
                                "raw_checkpoint_sha256": fit["inputs"][
                                    "baseline_checkpoint_sha256"
                                ],
                                "scope": scope,
                                "seed": seed,
                                "fold": index,
                                "train_matches": fold["train"],
                                "validation_matches": fold["validation"],
                                "input_manifest_sha256": input_sha,
                            }
                            if manifest["identity"] != identity:
                                raise ValueError("fine prediction manifest ancestry differs")
                            indexed = {
                                (r["match"], r["period"], r["scenario"]): r
                                for r in manifest["records"]
                            }
                            expected_keys = {k for k in source_index if k[0] in fold["validation"]}
                            if (
                                len(indexed) != len(manifest["records"])
                                or set(indexed) != expected_keys
                            ):
                                raise ValueError("fine prediction manifest grid differs")
                            fine_index[(seed, member["id"])] = (identity, indexed)
                        key = (scope, seed, index, "coarse")
                        if key not in fit_cache:
                            fit_cache[key] = _fit_component(
                                root,
                                scope,
                                seed,
                                index,
                                fold,
                                coarse["model_spec"],
                                "coarse_player_only",
                                canonical_sha256(coarse),
                                fine=False,
                            )
                        coarse_fits[seed] = fit_cache[key]
                        components.append(fit_cache[key][3])
                    for match in fold["validation"]:
                        for period in (1, 2):
                            per_seed_coarse = {}
                            for seed in seeds:
                                fit, fit_path, _, _ = coarse_fits[seed]
                                if (
                                    fit.get("derivation_sha256")
                                    != coarse["coarse_derivation_sha256"]
                                ):
                                    raise ValueError("coarse fit derivation differs")
                                base = (
                                    root
                                    / "coarse-stability-v1/coarse_predictions"
                                    / scope
                                    / f"seed_{seed}"
                                    / f"fold_{index:02d}"
                                    / f"{match}_half{period}"
                                )
                                meta = json.loads(base.with_suffix(".json").read_text())
                                expected = {
                                    "fit_sha256": sha256_file(fit_path),
                                    "checkpoint_sha256": fit["checkpoint_sha256"],
                                    "source_sha256": sha256_file(
                                        data_dir / f"{match}_half{period}.npz"
                                    ),
                                    "match": match,
                                    "period": period,
                                    "scope": scope,
                                    "seed": seed,
                                    "fold": index,
                                }
                                if meta.get("identity") != expected:
                                    raise ValueError("coarse prediction fit/input ancestry differs")
                                _check_hash(base.with_suffix(".npz"), meta.get("prediction_sha256"))
                                with np.load(base.with_suffix(".npz"), allow_pickle=False) as saved:
                                    per_seed_coarse[seed] = {
                                        name: saved[name]
                                        for name in ("live_probability", "home_probability")
                                    }
                            averaged = {
                                name: _average_logits(
                                    per_seed_coarse[seeds[0]][name], per_seed_coarse[seeds[1]][name]
                                )
                                for name in ("live_probability", "home_probability")
                            }
                            for scenario in campaign_protocol.WEIGHTS:
                                key = (match, period, scenario)
                                source = source_index[key]
                                _public_path(
                                    source["input_path"],
                                    root / "ensemble-bagging-v1" / batch / scope,
                                )
                                arrays = _input_arrays(source)
                                by_seed, raw_predictions = {}, []
                                for seed in seeds:
                                    bundles = []
                                    for member in selection["members"]:
                                        identity, indexed = fine_index[(seed, member["id"])]
                                        prediction = _fine_arrays(
                                            indexed[key], source, identity, root
                                        )
                                        raw_predictions.append(prediction)
                                        bundles.append(
                                            [
                                                {
                                                    **source,
                                                    "arrays": {**arrays, **prediction},
                                                    "model_identity": identity,
                                                }
                                            ]
                                        )
                                    by_seed[seed] = bundles
                                fine = _bagged_loaded(by_seed, selection)[0]["arrays"]
                                mixed = blend_coarse_context(
                                    fine, averaged["live_probability"], averaged["home_probability"]
                                )
                                decoded = np.asarray(
                                    decode_arrays(mixed, selection["decoder"]), dtype=np.uint8
                                )
                                reference_key = (
                                    f"pred_coarse_v6_{batch}_{match}_{scenario}_p{period}"
                                )
                                if not np.array_equal(
                                    decoded, labels[reference_key]
                                ) or not np.array_equal(
                                    mixed["gt"], labels[f"gt_{match}_p{period}"]
                                ):
                                    raise ValueError(
                                        f"frozen public v6 parity failed: {reference_key}"
                                    )
                                compact = {
                                    **mixed,
                                    "v6_labels": decoded,
                                    "fine_live": fine["neural_live"],
                                    "fine_home": fine["neural_home"],
                                    "coarse_live": averaged["live_probability"],
                                    "coarse_home": averaged["home_probability"],
                                    "member_live_std": np.std(
                                        [p["neural_live"] for p in raw_predictions], axis=0
                                    ),
                                    "member_home_std": np.std(
                                        [p["neural_home"] for p in raw_predictions], axis=0
                                    ),
                                }
                                path = (
                                    output_dir
                                    / batch
                                    / scope
                                    / match
                                    / f"half{period}"
                                    / f"{scenario}.npz"
                                )
                                path.parent.mkdir(parents=True, exist_ok=True)
                                if path.exists():
                                    with np.load(path, allow_pickle=False) as old:
                                        if set(old.files) != set(compact) or any(
                                            not np.array_equal(old[n], compact[n], equal_nan=True)
                                            for n in compact
                                        ):
                                            raise ValueError(
                                                "existing timing cache differs from verified reconstruction"
                                            )
                                else:
                                    temporary = path.with_suffix(".tmp.npz")
                                    np.savez_compressed(temporary, **compact)
                                    temporary.replace(path)
                                checksum = sha256_file(path)
                                provenance = asdict(
                                    BasePredictionProvenance(
                                        match,
                                        period,
                                        scenario,
                                        f"{scope}_fold_{index}",
                                        checksum,
                                        components + [deterministic],
                                        [historical],
                                    )
                                )
                                validate_prediction_provenance(
                                    provenance,
                                    outer_assessment_matches=[],
                                    allowed_matches=universe,
                                )
                                rows.append(
                                    {
                                        "match": match,
                                        "half": period,
                                        "period": period,
                                        "scenario": scenario,
                                        "batch": batch,
                                        "fold": index,
                                        "scope": scope,
                                        "path": str(path),
                                        "sha256": checksum,
                                        "n_frames": len(decoded),
                                        "provenance": provenance,
                                        "source_input_sha256": source["input_sha256"],
                                    }
                                )
    manifest = {
        "schema": "public_dfl_timing_cache_v1",
        "records": rows,
        "record_count": len(rows),
        "batches": list(batches),
        "decoder": selection["decoder"],
        "inner_folds": folds,
        "public_reference_sha256": reference["labels_sha256"],
        "exact_v6_parity": True,
        "target_inputs_accessed": False,
        "target_ground_truth_used": False,
        "historical_recipe_reuse": True,
        "source_protocol_sha256": {k: canonical_sha256(v) for k, v in protocols.items()},
    }
    campaign_protocol.publish(output_dir / "manifest.json", manifest)
    return manifest
