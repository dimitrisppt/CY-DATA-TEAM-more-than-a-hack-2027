"""Fixed one-hertz player-only context specialist for the frozen bagged model."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from experiments import (
    campaign_evaluation,
    campaign_protocol,
    campaign_training,
    ema_stability,
    ensemble_bagging,
    ensemble_polish_search,
)
from experiments.ensemble_polish_confirm import _folds
from experiments.robustness import _execution_environment, _source_hashes
from experiments.tune import _seed_for_fold
from possession.prepared import verify_prepared
from possession.storage import load_half, sha256_file
from possession.training_session import HalfCache

_SCHEMA = "coarse_stability_v1"
_MEMBER_ID = "coarse_player_only"
_INNER_SEEDS = (8101, 8102)
_OUTER_SEEDS = (9101, 9102)
_STATE_INTERVAL = 200
_CACHE_LIMIT = 1_000_000_000


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode()).hexdigest()


def _required(path: Path) -> dict:
    value = campaign_protocol.read_record(path)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _spec(base_selection: dict) -> dict:
    from experiments.coarse_context import COARSE_MODEL_KWARGS

    kwargs = dict(COARSE_MODEL_KWARGS)
    if kwargs.get("dilations") != [1, 2, 4, 8, 16, 32, 64]:
        raise ValueError("coarse specialist dilations differ from the fixed RF255 recipe")
    return {
        "model_kwargs": kwargs,
        "training_args": {"learning_rate": 3e-4, "weight_decay": 1e-4,
                          "boundary_weight": 0., "boundary_pos_weight": 1.,
                          "augment": True, "center_size": 32},
        "outage_config": {**base_selection["members"][0]["spec"]["outage_config"],
                          "mode": "legacy"},
        "augmentation_config": {"mode": "none"},
        "schedule": {"kind": "constant", "total_steps": 7200,
                     "warmup_steps": 0, "min_lr_ratio": 1.},
    }


def _gate(raw: dict, candidate: dict) -> dict:
    return ema_stability._gate(raw, candidate)


def _outer_gate(raw: dict, candidate: dict) -> dict:
    return ema_stability._outer_gate(raw, candidate)


def _verify_ema_result(ema_dir: Path, bagging_dir: Path, result: dict, protocol_sha: str,
                      selection: dict, bag_result: dict) -> None:
    summaries = result.get("batches", [])
    bag_summaries = bag_result.get("batches", [])
    if len(summaries) != 2 or len(bag_summaries) != 2:
        raise ValueError("EMA and v4 results must each contain two batches")
    for summary, bag_summary in zip(summaries, bag_summaries, strict=True):
        name = summary.get("batch")
        saved = _required(ema_dir / "batches" / name / "result.json")
        raw_batch = _required(bagging_dir / name / "result.json")
        if (saved != summary or saved.get("protocol_sha256") != protocol_sha
                or summary.get("inner_inputs_sha256") != raw_batch.get("inner_inputs_sha256")
                or summary.get("outer_inputs_sha256") != raw_batch.get("outer_inputs_sha256")
                or summary.get("raw_v4_inner") != bag_summary.get("candidate")
                or summary.get("raw_v4_outer") != bag_summary.get("outer", {}).get("candidate")):
            raise ValueError("EMA batch result or input ancestry differs")
        inner = _gate(saved["raw_v4_inner"], saved["ema_inner"])
        outer = _outer_gate(saved["raw_v4_outer"], saved["ema_outer"])
        if inner != saved.get("inner_checks") or outer != saved.get("outer_checks"):
            raise ValueError("EMA stored gates do not match authenticated scores")
        for relative, digest in saved.get("artifact_sha256", {}).items():
            relative_path = Path(relative)
            path = (ema_dir / relative_path).resolve()
            if (relative_path.is_absolute() or ".." in relative_path.parts
                    or not path.is_relative_to(ema_dir.resolve())):
                raise ValueError(f"EMA batch artifact path is unsafe: {relative}")
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"EMA batch artifact hash differs: {relative}")
    if (result.get("candidate_selection") != selection
            or result.get("baseline_selection") != bag_result.get("baseline_selection")):
        raise ValueError("EMA result selections differ from frozen pair")
    verified_eligible = all(row["inner_checks"]["passed"] and row["outer_checks"]["passed"]
                            for row in summaries)
    if result.get("eligible") is not verified_eligible:
        raise ValueError("EMA eligibility flag differs from the recomputed batch gates")


def _model_fit(data_dir: Path, output_dir: Path, spec: dict, protocol_sha: str,
               derivation_sha: str, fold: dict, seed: int, fold_index: int,
               scope: str, cache: HalfCache, check: Callable[[], None],
               status: Callable[..., None]) -> tuple[dict, Path, Path]:
    actual_seed = _seed_for_fold(seed, "DFL-MAT-J03WMX", fold_index)
    identity = {"protocol_sha256": protocol_sha, "spec_sha256": _digest(spec),
                "train_matches": list(fold["train"]), "seed": actual_seed,
                "name": _MEMBER_ID}
    directory = output_dir / "fits" / scope / f"seed_{seed}" / f"fold_{fold_index:02d}"
    fit_path, model_path = directory / "fit.json", directory / "model.pt"
    fit = campaign_protocol.read_record(fit_path)
    if fit is not None:
        if (fit.get("identity") != identity or fit.get("steps") != 7200
                or fit.get("derivation_sha256") != derivation_sha
                or not model_path.is_file()
                or sha256_file(model_path) != fit.get("checkpoint_sha256")):
            raise ValueError("completed coarse fit identity or checkpoint differs")
        import torch

        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        config = payload.get("training_config", {})
        train_paths = [data_dir / f"{match}_half{period}.npz"
                       for match in fold["train"] for period in (1, 2)]
        validation_paths = [data_dir / f"{match}_half{period}.npz"
                            for match in fold["validation"] for period in (1, 2)]
        selected_paths = sorted(train_paths + validation_paths)
        actual_hashes = {path.name: sha256_file(path) for path in selected_paths}
        actual_provenance = verify_prepared(data_dir, selected_paths)
        if (payload.get("model_kwargs") != spec["model_kwargs"]
                or config.get("campaign_spec") != spec
                or config.get("steps") != 7200
                or config.get("train_matches") != fold["train"]
                or config.get("seed") != actual_seed
                or config.get("validation_matches") != fold["validation"]
                or payload.get("prepared_sha256") != actual_hashes
                or payload.get("dataset_provenance") != actual_provenance
                or config.get("coarse_derivation_sha256") != derivation_sha
                or config.get("coarse_protocol_sha256") != protocol_sha
                or config.get("coarse_scope") != scope
                or config.get("coarse_fold") != fold_index
                or config.get("coarse_seed") != seed):
            raise ValueError("completed coarse checkpoint training config differs")
        return fit, fit_path, model_path

    session = campaign_training.build_session(data_dir, spec, fold["train"], actual_seed, cache)
    state_paths = sorted(directory.glob("state_*.pt"),
                         key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    committed = [path for path in state_paths
                 if path.with_suffix(path.suffix + ".sha256.json").is_file()]
    if committed:
        campaign_training.load_training_state(committed[-1], session, identity)
    if session.steps > 7200:
        raise ValueError("coarse checkpoint is beyond the frozen step count")
    started = time.monotonic()
    status("training_coarse", scope=scope, seed=seed, fold=fold_index,
           step=session.steps, target_steps=7200)
    while session.steps < 7200:
        check()
        session.advance_to(min(7200, session.steps + _STATE_INTERVAL))
        state_path = directory / f"state_{session.steps:06d}.pt"
        campaign_training.save_training_state(state_path, session, identity)
        states = sorted(directory.glob("state_*.pt"),
                        key=lambda path: int(path.stem.rsplit("_", 1)[1]))
        for obsolete in states[:-2]:
            obsolete.unlink()
            obsolete.with_suffix(obsolete.suffix + ".sha256.json").unlink(missing_ok=True)
        status("training_coarse", scope=scope, seed=seed, fold=fold_index,
               step=session.steps, target_steps=7200)
    directory.mkdir(parents=True, exist_ok=True)
    session.save_checkpoint(model_path, validation_matches=fold["validation"],
                            extra_training_config={"coarse_derivation_sha256": derivation_sha,
                                                   "coarse_protocol_sha256": protocol_sha,
                                                   "coarse_scope": scope, "coarse_fold": fold_index,
                                                   "coarse_seed": seed})
    payload = {"schema": "coarse_member_fit_v1", "identity": identity, "steps": 7200,
               "derivation_sha256": derivation_sha, "checkpoint_sha256": sha256_file(model_path),
               "parameters": sum(p.numel() for p in session.model.parameters()),
               "progress": session._progress(), "invocation_seconds": time.monotonic() - started}
    check()
    fit = campaign_protocol.publish(fit_path, payload)
    status("coarse_fit_complete", scope=scope, seed=seed, fold=fold_index,
           step=7200, target_steps=7200)
    del session
    return fit, fit_path, model_path


def _load_fit_rows(data_dir: Path, output_dir: Path, spec: dict, protocol_sha: str,
                   derivation_sha: str, folds: list[dict], outer_train: list[str],
                   check: Callable[[], None], status: Callable[..., None]) -> dict:
    cache = HalfCache(data_dir, max_bytes=_CACHE_LIMIT)
    rows = {}
    outer_folds = [{"train": outer_train,
                    "validation": ["DFL-MAT-J03WMX"]}]
    for scope, seeds, scope_folds in (("inner", _INNER_SEEDS, folds),
                                      ("outer", _OUTER_SEEDS, outer_folds)):
        for seed in seeds:
            for index, fold in enumerate(scope_folds):
                fit, fit_path, model_path = _model_fit(
                    data_dir, output_dir, spec, protocol_sha, derivation_sha,
                    fold, seed, index, scope, cache, check, status)
                rows[(scope, seed, index)] = (fit, fit_path, model_path)
    if cache.peak_bytes > _CACHE_LIMIT:
        raise RuntimeError("coarse specialist raw-half cache exceeded one GB")
    return rows


def _coarse_prediction(output_dir: Path, data_dir: Path, fit_row: tuple,
                       scope: str, seed: int, fold_index: int, match: str, period: int,
                       check: Callable[[], None]) -> dict:
    from experiments.coarse_context import predict_coarse_native
    from possession.predict import load_model_checkpoint
    from possession.storage import write_json

    fit, fit_path, model_path = fit_row
    source_path = data_dir / f"{match}_half{period}.npz"
    cache_path = (output_dir / "coarse_predictions" / scope / f"seed_{seed}"
                  / f"fold_{fold_index:02d}" / f"{match}_half{period}.npz")
    metadata_path = cache_path.with_suffix(".json")
    identity = {"fit_sha256": sha256_file(fit_path),
                "checkpoint_sha256": fit["checkpoint_sha256"],
                "source_sha256": sha256_file(source_path),
                "match": match, "period": period, "scope": scope,
                "seed": seed, "fold": fold_index}
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("identity") != identity
                or metadata.get("prediction_sha256") != sha256_file(cache_path)):
            raise ValueError("cached coarse prediction identity/hash differs")
        with np.load(cache_path, allow_pickle=False) as stored:
            return {name: stored[name] for name in stored.files}
    check()
    frames, _ = load_half(source_path)
    model, _ = load_model_checkpoint(model_path, device="cpu")
    prediction = predict_coarse_native(frames, model, center_size=64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        np.savez_compressed(handle, **prediction)
    write_json(metadata_path, {"identity": identity,
                               "prediction_sha256": sha256_file(cache_path)})
    del model
    return prediction


def _seed_member_records(bagging_dir: Path, ema_dir: Path, baseline_kind: str,
                         batch_name: str, scope: str, seed: int, selection: dict,
                         campaign: dict, campaign_dir: Path, folds: list[dict],
                         input_records: list[dict], input_sha: str,
                         ema_fit_rows: dict, ema_protocol_sha: str,
                         check: Callable[[], None]) -> list[list[dict]]:
    if baseline_kind == "v4":
        bundles = ema_stability._read_original_seed_records(
            bagging_dir, batch_name, scope, seed, selection, campaign,
            campaign_dir, folds, input_sha)
    else:
        bundles = ema_stability._ema_model_records(
            ema_dir, batch_name, scope, seed, selection, folds, input_records,
            input_sha, ema_fit_rows, ema_protocol_sha, check)
    return [campaign_evaluation.load_records(bundle) for bundle in bundles]


def _bagged_loaded(seed_bundles: dict[int, list[list[dict]]], selection: dict) -> list[dict]:
    seeds = sorted(seed_bundles)
    if len(seeds) != 2:
        raise ValueError("coarse comparison requires the two frozen training seeds")
    averaged = [ensemble_polish_search.blend_loaded(
        seed_bundles[seeds[0]][index], seed_bundles[seeds[1]][index], .5)
        for index in range(2)]
    return ensemble_polish_search.blend_loaded(
        averaged[0], averaged[1], float(selection["blend_weight"]))


def _average_logits(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(first, dtype=np.float64), 1e-7, 1. - 1e-7)
    q = np.clip(np.asarray(second, dtype=np.float64), 1e-7, 1. - 1e-7)
    if p.shape != q.shape:
        raise ValueError("coarse seed predictions use different native grids")
    logits = .5 * ((np.log(p) - np.log1p(-p)) + (np.log(q) - np.log1p(-q)))
    return 1. / (1. + np.exp(-np.clip(logits, -40., 40.)))


def _mix_coarse(base_records: list[dict], coarse_by_key: dict,
                *, batch_name: str) -> list[dict]:
    from experiments.coarse_context import blend_coarse_context

    output = []
    for record in base_records:
        arrays = record["arrays"]
        key = (record["match"], record["period"])
        coarse = coarse_by_key[key]
        if record["scenario"] == "clean" and not np.asarray(arrays["ball_valid"]).all():
            raise ValueError("clean inputs contain naturally missing ball observations")
        mixed = blend_coarse_context(arrays, coarse["live_probability"],
                                     coarse["home_probability"])
        if record["scenario"] == "clean" and any(
                not np.array_equal(mixed[name], arrays[name])
                for name in ("neural_live", "neural_home", "neural_boundary")):
            raise ValueError("clean coarse prediction differs from frozen baseline")
        if (not np.array_equal(mixed["neural_boundary"], arrays["neural_boundary"])
                or any(not np.array_equal(mixed[name], arrays[name])
                       for name in arrays if name.startswith("ref_"))):
            raise ValueError("coarse context changed boundary or reference arrays")
        output.append({**record, "arrays": mixed,
                       "model_identity": {"kind": "frozen_plus_coarse_context",
                                          "batch": batch_name,
                                          "base": record.get("model_identity")}})
    return output


def _artifact_hashes(output_dir: Path, fit_rows: dict,
                     prediction_keys: set[tuple[str, int, int, str]]) -> dict:
    hashes = {}
    for fit, fit_path, model_path in fit_rows.values():
        for path in (fit_path, model_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            hashes[path.relative_to(output_dir).as_posix()] = sha256_file(path)
    for scope, seed, fold_index, match in sorted(prediction_keys):
        for period in (1, 2):
            base = (output_dir / "coarse_predictions" / scope / f"seed_{seed}"
                    / f"fold_{fold_index:02d}" / f"{match}_half{period}")
            for path in (base.with_suffix(".npz"), base.with_suffix(".json")):
                if not path.is_file():
                    raise FileNotFoundError(path)
                hashes[path.relative_to(output_dir).as_posix()] = sha256_file(path)
    return dict(sorted(hashes.items()))


def run_coarse_stability(campaign_dir, polish_dir, bagging_dir, ema_dir, data_dir,
                         target_dir, output_dir, *, baseline_kind: str,
                         max_seconds: int = 14_400,
                         check: Callable[[], None] | None = None) -> dict:
    """Fit fixed 1 Hz specialists and compare their gated context on frozen masks."""
    if baseline_kind not in ("v4", "ema"):
        raise ValueError("baseline_kind must be explicitly frozen as v4 or ema")
    campaign_dir, polish_dir, bagging_dir, ema_dir, data_dir, target_dir, output_dir = (
        Path(path).resolve() for path in
        (campaign_dir, polish_dir, bagging_dir, ema_dir, data_dir, target_dir, output_dir))
    for source in (campaign_dir, polish_dir, bagging_dir, ema_dir, data_dir, target_dir):
        if output_dir.is_relative_to(source) or source.is_relative_to(output_dir):
            raise ValueError("coarse output must be separate from all source inputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(max_seconds, int) or isinstance(max_seconds, bool) or not 1 <= max_seconds <= 14_400:
        raise ValueError("coarse budget must be between 1 and 14400 seconds")
    budget = campaign_protocol.Budget(output_dir, max_seconds / 3600.)

    def tick():
        budget.check()
        if check is not None:
            check()

    def status(stage: str, **detail):
        budget.status(stage, **detail)

    tick()
    (campaign_dir, polish_dir, campaign, _original_selection, polish_protocol,
     selection, _diagnostic) = ensemble_bagging._authenticated_inputs(
        campaign_dir, polish_dir, data_dir, target_dir)
    folds = _folds(campaign)
    ema_protocol = _required(ema_dir / "protocol.json")
    ema_result = _required(ema_dir / "result.json")
    ema_protocol_sha = _digest(ema_protocol)
    if (ema_result.get("protocol_sha256") != ema_protocol_sha
            or ema_protocol.get("campaign_protocol_sha256") != _digest(campaign)
            or ema_protocol.get("candidate_selection") != selection
            or len(ema_result.get("batches", [])) != 2):
        raise ValueError("frozen EMA study is incomplete or uses a different candidate")
    expected_baseline = "ema" if ema_result.get("eligible") is True else "v4"
    if baseline_kind != expected_baseline:
        raise ValueError("baseline_kind differs from the completed EMA eligibility result")
    current_sources = _source_hashes()
    if (ema_protocol.get("runtime") != _execution_environment()
            or any(current_sources.get(name) != digest
                   for name, digest in ema_protocol.get("source_sha256", {}).items())):
        raise ValueError("EMA source/runtime differs from current coarse runner")
    bag_protocol = _required(bagging_dir / "protocol.json")
    bag_result = _required(bagging_dir / "result.json")
    if (bag_result.get("eligible") is not True
            or bag_protocol.get("candidate_selection") != selection
            or bag_result.get("protocol_sha256") != _digest(bag_protocol)):
        raise ValueError("bagged v4 candidate proof differs from frozen selection")
    _verify_ema_result(ema_dir, bagging_dir, ema_result, ema_protocol_sha,
                       selection, bag_result)
    if (bag_protocol.get("runtime") != _execution_environment()
            or any(current_sources.get(name) != digest
                   for name, digest in bag_protocol.get("source_sha256", {}).items())):
        raise ValueError("bagging source/runtime differs from current coarse runner")
    spec = _spec(selection)

    batch_config = bag_protocol.get("mask_batches", [])
    if len(batch_config) != 2 or tuple(row.get("name") for row in batch_config) != ("batch_1", "batch_2"):
        raise ValueError("coarse study requires the two frozen bagging batches")
    prepared_manifest = verify_prepared(data_dir, sorted(data_dir.glob("DFL-MAT-*_half*.npz")))
    if prepared_manifest != campaign.get("prepared"):
        raise ValueError("coarse source prepared dataset differs from frozen campaign")
    inputs = {}
    bag_protocol_sha = _digest(bag_protocol)
    for batch in batch_config:
        name = batch["name"]
        batch_dir = bagging_dir / name
        raw = _required(batch_dir / "result.json")
        inner_records, inner_manifest = ema_stability._verify_input_cache(
            batch_dir / "inner", raw["inner_inputs_sha256"],
            {match for fold in folds for match in fold["validation"]})
        outer_records, outer_manifest = ema_stability._verify_input_cache(
            batch_dir / "outer", raw["outer_inputs_sha256"], {"DFL-MAT-J03WMX"})
        ensemble_bagging._verify_batch_result(
            raw, bag_protocol_sha, _digest(_original_selection), _digest(selection),
            raw["inner_inputs_sha256"], raw["outer_inputs_sha256"], campaign_dir,
            campaign, folds, batch)
        inputs[name] = {"raw": raw,
                        "inner": (inner_records, raw["inner_inputs_sha256"], inner_manifest),
                        "outer": (outer_records, raw["outer_inputs_sha256"], outer_manifest)}
    ensemble_bagging._validate_completed_result(
        bag_result, [inputs[row["name"]]["raw"] for row in batch_config],
        _original_selection, selection, bag_protocol_sha)
    outer_folds = [{"train": campaign["fold"]["outer_train"],
                    "validation": ["DFL-MAT-J03WMX"]}]
    for scope, seeds, scope_folds in (("inner", _INNER_SEEDS, folds),
                                      ("outer", _OUTER_SEEDS, outer_folds)):
        for seed in seeds:
            for fold in scope_folds:
                if set(fold["train"]) & set(fold["validation"]):
                    raise ValueError("coarse training and evaluation folds overlap")
                if scope == "inner" and len(fold["validation"]) != 2:
                    raise ValueError("coarse specialist requires three two-match validation folds")

    from experiments.coarse_context import prepare_coarse_data

    coarse_data = output_dir / "derived_data"
    derivation = prepare_coarse_data(data_dir, coarse_data)
    derivation_sha = _digest(derivation)
    derivation_file = coarse_data / "coarse-derivation.json"
    if not derivation_file.is_file():
        raise ValueError("coarse derivation record is missing")
    protocol = {"schema": _SCHEMA, "development_only": True,
                "baseline_kind": baseline_kind, "baseline_selection": selection,
                "candidate_selection": selection, "campaign_protocol_sha256": _digest(campaign),
                "polish_protocol_sha256": _digest(polish_protocol),
                "bagging_protocol_sha256": _digest(bag_protocol),
                "bagging_result_sha256": sha256_file(bagging_dir / "result.json"),
                "ema_protocol_sha256": ema_protocol_sha,
                "ema_result_sha256": sha256_file(ema_dir / "result.json"),
                "inner_folds": folds, "inner_seeds": list(_INNER_SEEDS),
                "outer_seeds": list(_OUTER_SEEDS), "steps": 7200,
                "checkpoint_interval": _STATE_INTERVAL, "model_id": _MEMBER_ID,
                "model_spec": spec, "model_spec_sha256": _digest(spec),
                "coarse_derivation_sha256": derivation_sha,
                "source_prepared_sha256": _digest(prepared_manifest),
                "input_caches": {
                    name: {
                        scope: {"signature_sha256": row[1],
                                "manifest_sha256": sha256_file(
                                    bagging_dir / name / scope / "inputs" / "input-manifest.json")}
                        for scope, row in values.items() if scope in ("inner", "outer")}
                    for name, values in inputs.items()},
                "gate": {"inner_objective_gain_gt": .002, "missing_aggregate_gain_gt": 0.,
                         "positive_matches_at_least": 4, "scenario_loss_at_most": .01,
                         "missing_transition_loss_at_most": .01,
                         "outer_objective_loss_at_most": .01},
                "source_sha256": current_sources, "runtime": _execution_environment(),
                "budget_seconds": max_seconds,
                "interpretation": "Ball-free 1 Hz player context is added only to missing-ball inputs. Clean neural arrays and all fine boundary/reference arrays are preserved."}
    campaign_protocol.publish(output_dir / "protocol.json", protocol)
    protocol_sha = _digest(protocol)
    campaign_protocol.publish(output_dir / "preflight.json", {
        "protocol_sha256": protocol_sha, "derivation_sha256": derivation_sha,
        "input_cache_signatures": {
            name: {scope: row[1] for scope, row in values.items()
                   if scope in ("inner", "outer")} for name, values in inputs.items()},
        "target_ground_truth_used": False, "target_inputs_read_for_training": False,
        "validation_grid_verified": True})

    fit_rows = _load_fit_rows(coarse_data, output_dir, spec, protocol_sha,
                              derivation_sha, folds,
                              campaign["fold"]["outer_train"], tick, status)
    ema_fit_rows = {}
    if baseline_kind == "ema":
        for scope, seeds, scope_folds in (("inner", _INNER_SEEDS, folds),
                                          ("outer", _OUTER_SEEDS, outer_folds)):
            for seed in seeds:
                for index, _fold in enumerate(scope_folds):
                    for member in selection["members"]:
                        path = (ema_dir / "fits" / scope / member["id"] / f"seed_{seed}"
                                / f"fold_{index:02d}" / "fit.json")
                        fit = _required(path)
                        model_path = path.parent / "model.pt"
                        if sha256_file(model_path) != fit.get("checkpoint_sha256"):
                            raise ValueError("EMA baseline fit checkpoint hash differs")
                        ema_fit_rows[(scope, seed, member["id"], index)] = (fit, path, model_path)

    completed_batches = []
    ema_batch_rows = {row["batch"]: row for row in ema_result["batches"]}
    for batch in batch_config:
        tick()
        name = batch["name"]
        result_path = output_dir / "batches" / name / "result.json"
        baseline_scores, candidate_scores, score_pairs = {}, {}, {}
        for scope, seeds, scope_folds in (("inner", _INNER_SEEDS, folds),
                                          ("outer", _OUTER_SEEDS, outer_folds)):
            input_records, input_sha, _manifest = inputs[name][scope]
            baseline_by_seed = {}
            for seed in seeds:
                baseline_by_seed[seed] = _seed_member_records(
                    bagging_dir, ema_dir, baseline_kind, name, scope, seed, selection,
                    campaign, campaign_dir, scope_folds, input_records, input_sha,
                    ema_fit_rows, ema_protocol_sha, tick)
            base_loaded = _bagged_loaded(baseline_by_seed, selection)
            baseline_score = campaign_evaluation.score_records(
                base_loaded, selection["decoder"], campaign_protocol.WEIGHTS)
            raw_batch = inputs[name]["raw"]
            frozen_score = (raw_batch["candidate"] if scope == "inner"
                            else raw_batch["outer"]["candidate"])
            expected_baseline = (frozen_score if baseline_kind == "v4"
                                 else ema_batch_rows[name][
                                     "ema_inner" if scope == "inner" else "ema_outer"])
            if _digest(baseline_score) != _digest(expected_baseline):
                raise ValueError("reconstructed baseline score differs from frozen study result")
            fold_for_match = {match: index for index, fold in enumerate(scope_folds)
                              for match in fold["validation"]}
            per_seed_coarse = {}
            for seed in seeds:
                seed_coarse = {}
                for match, fold_index in sorted(fold_for_match.items()):
                    fit_row = fit_rows[(scope, seed, fold_index)]
                    for period in (1, 2):
                        seed_coarse[(match, period)] = _coarse_prediction(
                            output_dir, data_dir, fit_row, scope, seed, fold_index,
                            match, period, tick)
                per_seed_coarse[seed] = seed_coarse
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
            mixed = _mix_coarse(base_loaded, coarse_by_key, batch_name=name)
            candidate_score = campaign_evaluation.score_records(
                mixed, selection["decoder"], campaign_protocol.WEIGHTS)
            checks = (_gate(baseline_score, candidate_score) if scope == "inner"
                      else _outer_gate(baseline_score, candidate_score))
            baseline_scores[scope], candidate_scores[scope] = baseline_score, candidate_score
            score_pairs[scope] = checks
        prediction_keys = {
            (scope, seed, fold_index, match)
            for scope, seeds, scope_folds in (("inner", _INNER_SEEDS, folds),
                                              ("outer", _OUTER_SEEDS, outer_folds))
            for seed in seeds
            for fold_index, fold in enumerate(scope_folds)
            for match in fold["validation"]
        }
        row = {"schema": _SCHEMA, "protocol_sha256": protocol_sha, "batch": name,
               "baseline_kind": baseline_kind, "baseline_scores": baseline_scores,
               "candidate_scores": candidate_scores, "inner_checks": score_pairs["inner"],
               "outer_checks": score_pairs["outer"], "target_ground_truth_used": False,
               "development_only": True,
               "artifact_sha256": _artifact_hashes(output_dir, fit_rows, prediction_keys)}
        tick()
        published = campaign_protocol.publish(result_path, row)
        completed_batches.append(published)
        budget.status(f"batch_{name}_complete",
                      inner_passed=published["inner_checks"]["passed"],
                      outer_passed=published["outer_checks"]["passed"])
    result = {"schema": _SCHEMA, "protocol_sha256": protocol_sha,
              "baseline_kind": baseline_kind, "baseline_selection": selection,
              "candidate_selection": selection, "batches": completed_batches,
              "eligible": len(completed_batches) == 2 and all(
                  row["inner_checks"]["passed"] and row["outer_checks"]["passed"]
                  for row in completed_batches),
              "summary": {"inner_gates_passed": all(
                  row["inner_checks"]["passed"] for row in completed_batches),
                          "outer_gates_passed": all(
                              row["outer_checks"]["passed"] for row in completed_batches)},
              "notes": ["Fixed RF255 one-hertz player-only specialist, 7200 updates, no HPO.",
                        "The coarse branch is mixed only on missing-ball inputs.",
                        "Clean arrays, fine boundaries, and reference arrays are preserved.",
                        "Inputs are reused from the frozen two-batch bagging study.",
                        "No target ground truth or final all-seven validation was used."]}
    tick()
    campaign_protocol.publish(output_dir / "result.json", result)
    budget.status("complete", eligible=result["eligible"], batches=2)
    return result
