"""Portable final ensemble export and inference for verified tail-EMA weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments import (
    campaign_protocol,
    campaign_refit,
    campaign_submission,
    ema_stability,
    ensemble_bagging_submission,
)
from experiments.campaign_training import build_session, load_training_state
from experiments.ensemble_bagging_submission import (
    CENTER_SIZE,
    MEMBERS,
    N_FRAMES,
    SEEDS,
    TARGET_SPANS,
    combine_four_predictions,
)
from experiments.robustness import _digest, _execution_environment, _source_hashes
from games import GAME, GAMES
from possession.features import build_features
from possession.io_atd import read_atd
from possession.predict import load_model_checkpoint, predict_features
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache

_SCHEMA = "ema_ensemble_submission_v1"
_FIT_MATCHES = campaign_refit.FIT_MATCHES
_START_STEPS = 7000
_END_STEPS = 7200
_DECAY = .99
_CACHE_LIMIT = 1_000_000_000


def _record(path: Path) -> dict:
    value = campaign_protocol.read_record(path)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _inside(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def _safe_path(root: Path, relative: str) -> Path:
    path_part = PurePosixPath(relative)
    if path_part.is_absolute() or ".." in path_part.parts or not path_part.parts:
        raise ValueError(f"bundle path is not a safe relative path: {relative}")
    resolved = (root / Path(*path_part.parts)).resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"bundle path escapes bundle root: {relative}")
    return resolved


def _tensor_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tree_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(left, right, equal_nan=True))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(_tree_equal(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True)))
    if isinstance(left, float) and isinstance(right, float):
        return (math.isnan(left) and math.isnan(right)) or left == right
    return left == right


def _ema_update(average: dict, raw: dict) -> None:
    if average.keys() != raw.keys():
        raise ValueError("EMA and raw checkpoint tensors differ")
    for name, value in raw.items():
        source = value.detach().cpu()
        target = average[name]
        if target.shape != source.shape or target.dtype != source.dtype:
            raise ValueError(f"EMA tensor layout differs: {name}")
        if target.is_floating_point() or target.is_complex():
            target.mul_(_DECAY).add_(source, alpha=1. - _DECAY)
        else:
            target.copy_(source)


def _assert_raw_tail_parity(replayed_state: dict, endpoint_state: dict,
                            baseline_weights: dict) -> dict:
    if (not _tree_equal(replayed_state, endpoint_state)
            or not _tree_equal(replayed_state.get("model"), baseline_weights)):
        raise ValueError("raw 7200 replay failed full training-state or checkpoint parity")
    return {"raw_weights_match_endpoint_state": True,
            "raw_weights_match_baseline_checkpoint": True,
            "raw_training_progress_matches_endpoint": True,
            "compared_optimizer_rng_transcript_and_counters": True,
            "start_steps": _START_STEPS, "end_steps": _END_STEPS}


def _state_hash(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"final tail state or SHA256 sidecar missing: {path}")
    actual = sha256_file(path)
    if json.loads(sidecar.read_text()).get("sha256") != actual:
        raise ValueError(f"final tail state SHA256 differs: {path}")
    return actual


def _ema_metadata(fit: dict) -> dict:
    return {
        "kind": "tail_ema",
        "decay": _DECAY,
        "start_steps": _START_STEPS,
        "end_steps": _END_STEPS,
        "updates": _END_STEPS - _START_STEPS,
        "initialization": "raw model state at step 7000",
        "interpretation": "200-update tail EMA, not full-run EMA",
        "raw_parity": fit["raw_parity"],
    }


def _final_identity(plan: dict, member_id: str, seed: int, spec: dict) -> dict:
    if (plan.get("seed") != seed or plan.get("fit_matches") != list(_FIT_MATCHES)
            or plan.get("selection_matches") != list(campaign_refit.SELECTION_MATCHES)):
        raise ValueError("final fit plan seed or seven-match split differs")
    selection = plan.get("selection", {})
    row = next((item for item in selection.get("members", [])
                if item.get("id") == member_id), None)
    if row is None or row.get("spec") != spec or row.get("steps") != _END_STEPS:
        raise ValueError(f"final fit spec or step count differs: {member_id}")
    return plan


def _verify_prepared_data(data_dir: Path, plan: dict) -> None:
    expected = plan.get("prepared_sha256")
    if not isinstance(expected, dict) or set(expected) != {
            f"{match}_half{period}.npz" for match in _FIT_MATCHES for period in (1, 2)}:
        raise ValueError("final fit plan does not authenticate all fourteen prepared caches")
    paths = [data_dir / name for name in sorted(expected)]
    if any(not path.is_file() or sha256_file(path) != expected[path.name] for path in paths):
        raise ValueError("final prepared cache hashes differ from authenticated plan")
    from possession.prepared import verify_prepared

    if verify_prepared(data_dir, paths) != plan.get("prepared_provenance"):
        raise ValueError("final prepared provenance differs from authenticated plan")


def _verify_study(study_dir: Path, source_selection: dict) -> tuple[dict, dict, dict]:
    protocol = _record(study_dir / "protocol.json")
    result = _record(study_dir / "result.json")
    protocol_sha = _digest(protocol)
    if (protocol.get("schema") != "ema_stability_v1"
            or protocol.get("tail") != {"start_steps": _START_STEPS,
                                        "end_steps": _END_STEPS,
                                        "updates": 200, "ema_decay": _DECAY}
            or result.get("schema") != "ema_stability_v1"
            or result.get("protocol_sha256") != protocol_sha
            or result.get("candidate_selection") != protocol.get("candidate_selection")
            or result.get("baseline_selection") != protocol.get("baseline_selection")
            or result.get("eligible") is not True
            or protocol.get("baseline_selection") != source_selection):
        raise ValueError("completed EMA stability protocol or candidate differs")
    candidate = protocol["candidate_selection"]
    ensemble_bagging_submission._selection_valid(candidate, source_selection)
    current_sources = _source_hashes()
    if (protocol.get("runtime") != _execution_environment()
            or any(current_sources.get(name) != digest
                   for name, digest in protocol.get("source_sha256", {}).items())):
        raise ValueError("EMA stability source or runtime ancestry differs")
    batches = result.get("batches")
    if not isinstance(batches, list) or [row.get("batch") for row in batches] != ["batch_1", "batch_2"]:
        raise ValueError("both named EMA stability batches are required")
    for row in batches:
        raw = _record(study_dir / "batches" / row["batch"] / "result.json")
        if raw != row or raw.get("protocol_sha256") != protocol_sha:
            raise ValueError(f"EMA batch record differs: {row['batch']}")
        if (ema_stability._gate(raw["raw_v4_inner"], raw["ema_inner"])
                != raw.get("inner_checks")
                or ema_stability._outer_gate(raw["raw_v4_outer"], raw["ema_outer"])
                != raw.get("outer_checks")
                or raw["inner_checks"].get("passed") is not True
                or raw["outer_checks"].get("passed") is not True):
            raise ValueError(f"EMA gate did not pass: {row['batch']}")
        for relative, digest in raw.get("artifact_sha256", {}).items():
            path = _safe_path(study_dir, relative)
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"EMA study artifact hash differs: {relative}")
    return protocol, result, {"protocol_sha256": protocol_sha,
                              "protocol_file_sha256": sha256_file(study_dir / "protocol.json"),
                              "result_file_sha256": sha256_file(study_dir / "result.json")}


def _copy_verified(source: Path, target: Path, digest: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"refusing to replace different bundle file: {target}")
        return
    shutil.copyfile(source, target)
    if sha256_file(target) != digest:
        target.unlink(missing_ok=True)
        raise ValueError(f"copied bundle file hash differs: {target}")


def _replay_final_tail(*, data_dir: Path, plan: dict, selection: dict, seed: int,
                       member: dict, source_root: Path, source_manifest: dict,
                       output_dir: Path, cache=None) -> dict:
    member_id, spec = member["id"], member["spec"]
    plan_identity = _final_identity(plan, member_id, seed, spec)
    identity_sha = _digest(plan_identity)
    state_identity = {"member_id": member_id, **plan_identity}
    source_row = source_manifest["members"].get(member_id)
    if not isinstance(source_row, dict):
        raise TypeError(f"final source bundle lacks fit row: {member_id}")
    source_fit_path = source_root / source_row["fit_record"]
    fit = _record(source_fit_path)
    baseline_path = source_root / source_row["checkpoint"]
    if (fit.get("identity_sha256") != identity_sha or fit.get("member_id") != member_id
            or fit.get("recipe_sha256") != _digest(spec) or fit.get("seed") != seed
            or fit.get("steps") != _END_STEPS or fit.get("fit_matches") != list(_FIT_MATCHES)
            or fit.get("selection_matches") != list(campaign_refit.SELECTION_MATCHES)
            or fit.get("checkpoint_sha256") != source_row.get("sha256")
            or sha256_file(baseline_path) != fit.get("checkpoint_sha256")):
        raise ValueError(f"final source fit ancestry differs: {member_id}/{seed}")
    payload = torch.load(baseline_path, map_location="cpu", weights_only=True)
    expected_kwargs = {"player_dim": 8, "global_dim": 9,
                       "dilations": [1, 2, 4, 8, 16, 32, 64, 128], **spec["model_kwargs"]}
    config = payload.get("training_config", {})
    if (payload.get("model_kwargs") != expected_kwargs
            or config.get("campaign_spec") != spec or config.get("seed") != seed
            or config.get("steps") != _END_STEPS or config.get("train_matches") != list(_FIT_MATCHES)
            or config.get("validation_matches") != []
            or config.get("campaign_identity_sha256") != identity_sha
            or config.get("campaign_recipe_sha256") != _digest(spec)
            or config.get("campaign_member_id") != member_id):
        raise ValueError(f"final raw checkpoint spec or full-data split differs: {member_id}/{seed}")

    source_fit_dir = source_fit_path.parent
    if source_fit_dir != (source_root / "members" / member_id):
        raise ValueError("final fit record path is outside its member ancestry")
    state_paths = {7000: source_fit_dir / "resume-1.pt",
                   7200: source_fit_dir / "resume-0.pt"}
    start_hash = _state_hash(state_paths[_START_STEPS])
    end_hash = _state_hash(state_paths[_END_STEPS])
    checkpoint_hash = sha256_file(baseline_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_path = output_dir / "fit.json"
    model_path = output_dir / "model.pt"
    expected_inputs = {"source_bundle_manifest_sha256":
                       sha256_file(source_root / "manifest.json"),
                       "source_fit_sha256": sha256_file(source_fit_path),
                       "baseline_checkpoint_sha256": checkpoint_hash,
                       "start_state_sha256": start_hash, "end_state_sha256": end_hash}
    previous = campaign_protocol.read_record(fit_path)
    if previous is not None:
        if (previous.get("identity_sha256") != identity_sha
                or previous.get("member_id") != member_id or previous.get("seed") != seed
                or previous.get("spec_sha256") != _digest(spec)
                or previous.get("inputs") != expected_inputs
                or previous.get("steps") != _END_STEPS
                or previous.get("train_matches") != list(_FIT_MATCHES)
                or previous.get("validation_matches") != []):
            raise ValueError("resumed final EMA fit identity differs")
        parity_saved = {"raw_weights_match_endpoint_state": True,
                        "raw_weights_match_baseline_checkpoint": True,
                        "raw_training_progress_matches_endpoint": True,
                        "compared_optimizer_rng_transcript_and_counters": True,
                        "start_steps": _START_STEPS, "end_steps": _END_STEPS}
        if (previous.get("raw_parity") != parity_saved
                or not model_path.is_file()
                or previous.get("checkpoint_sha256") != sha256_file(model_path)):
            raise ValueError("resumed final EMA fit parity or checkpoint differs")
        _, existing_payload = load_model_checkpoint(model_path)
        if (existing_payload.get("ema_metadata") != _ema_metadata(previous)
                or _tensor_digest(existing_payload["state_dict"]) != previous.get("ema_sha256")
                or existing_payload.get("training_config") != payload.get("training_config")):
            raise ValueError("resumed final EMA checkpoint metadata differs")
        return {"fit": previous, "checkpoint": model_path, "fit_record": fit_path}
    model_path.unlink(missing_ok=True)

    from experiments.campaign_training import _session_state

    session = build_session(data_dir, spec, list(_FIT_MATCHES), seed, cache=cache)
    loaded = load_training_state(state_paths[_START_STEPS], session, state_identity)
    if loaded["steps"] != _START_STEPS:
        raise ValueError("final tail start state is not step 7000")
    average = {name: tensor.detach().cpu().clone()
               for name, tensor in session.model.state_dict().items()}
    while session.steps < _END_STEPS:
        session.advance_to(session.steps + 1)
        _ema_update(average, session.model.state_dict())
    if session.steps != _END_STEPS:
        raise ValueError("final tail replay ended at the wrong step")
    replayed_state = _session_state(session)
    del session
    endpoint_session = build_session(data_dir, spec, list(_FIT_MATCHES), seed, cache=cache)
    endpoint_progress = load_training_state(state_paths[_END_STEPS], endpoint_session, state_identity)
    if endpoint_progress.get("steps") != _END_STEPS:
        raise ValueError("authenticated final endpoint state is not step 7200")
    parity = _assert_raw_tail_parity(
        replayed_state, _session_state(endpoint_session), payload["state_dict"])
    fit_record = {
        "schema": "final_tail_ema_fit_v1", "member_id": member_id, "seed": seed,
        "identity_sha256": identity_sha, "spec_sha256": _digest(spec),
        "train_matches": list(_FIT_MATCHES), "validation_matches": [],
        "steps": _END_STEPS,
        "inputs": expected_inputs,
        "raw_parity": parity,
        "tail_ema": {"kind": "tail_ema", "decay": _DECAY,
                     "start_steps": _START_STEPS, "end_steps": _END_STEPS,
                     "updates": _END_STEPS - _START_STEPS,
                     "initialization": "raw model state at step 7000",
                     "interpretation": "200-update tail EMA, not full-run EMA",
                     "raw_parity": parity},
        "ema_sha256": _tensor_digest(average),
        "training_config_sha256": _digest(payload["training_config"]),
    }
    ema_payload = dict(payload)
    ema_payload["state_dict"] = average
    ema_payload["ema_metadata"] = _ema_metadata(fit_record)
    checkpoint = model_path
    torch.save(ema_payload, checkpoint)
    fit_record["checkpoint_sha256"] = sha256_file(checkpoint)
    fit_record["model_path"] = "model.pt"
    campaign_protocol.publish(output_dir / "fit.json", fit_record)
    return {"fit": fit_record, "checkpoint": checkpoint,
            "fit_record": output_dir / "fit.json"}


def export_ema_bundle(ema_study_dir, first_bundle, second_bundle, data_dir, output_dir) -> dict:
    """Replay final fit tails after both stability gates pass, then publish portable weights."""
    study_dir, first_bundle, second_bundle, data_dir, output_dir = (
        Path(value).resolve() for value in
        (ema_study_dir, first_bundle, second_bundle, data_dir, output_dir))
    if any(_inside(output_dir, path) for path in
           (study_dir, first_bundle, second_bundle, data_dir)):
        raise ValueError("EMA output must be separate from study, source bundles, and data")
    # Authenticate both full-data source bundles before writing output.
    first = campaign_refit.verify_bundle(first_bundle)
    second = campaign_refit.verify_bundle(second_bundle)
    by_seed = {first["manifest"]["seed"]: first, second["manifest"]["seed"]: second}
    if tuple(sorted(by_seed)) != tuple(sorted(SEEDS)):
        raise ValueError("final source bundles must use the two declared refit seeds")
    manifests = {seed: value["manifest"] for seed, value in by_seed.items()}
    baseline_selection = manifests[SEEDS[0]]["selection"]
    if any(manifests[seed]["selection"] != baseline_selection
           or manifests[seed]["steps"] != _END_STEPS
           or manifests[seed]["fit_matches"] != list(_FIT_MATCHES)
           or manifests[seed]["selection_matches"] != list(campaign_refit.SELECTION_MATCHES)
           for seed in SEEDS):
        raise ValueError("final source bundle selections, steps, or splits differ")
    study_protocol, study_result, study_refs = _verify_study(study_dir, baseline_selection)
    if (study_result.get("baseline_selection") != baseline_selection
            or study_protocol.get("baseline_selection") != baseline_selection):
        raise ValueError("EMA stability baseline differs from final source selection")
    plans = {seed: _record(Path(by_seed[seed]["bundle_dir"]) / "plan.json") for seed in SEEDS}
    if any(plans[seed].get("prepared_sha256") != plans[SEEDS[0]].get("prepared_sha256")
           or plans[seed].get("prepared_provenance")
           != plans[SEEDS[0]].get("prepared_provenance") for seed in SEEDS):
        raise ValueError("final source bundle prepared-data ancestry differs between seeds")
    _verify_prepared_data(data_dir, plans[SEEDS[0]])

    # Copy only authenticated proof and the four final outputs into a fresh directory.
    if (output_dir / "manifest.json").is_file():
        existing = verify_ema_bundle(output_dir)
        expected_sources = {str(seed): sha256_file(
            Path(by_seed[seed]["bundle_dir"]) / "manifest.json") for seed in SEEDS}
        if (existing["manifest"].get("source_bundle_manifest_sha256") != expected_sources
                or existing["manifest"].get("ema_protocol_sha256")
                != study_refs["protocol_sha256"]
                or existing["manifest"].get("ema_protocol_file_sha256")
                != study_refs["protocol_file_sha256"]
                or existing["manifest"].get("ema_result_file_sha256")
                != study_refs["result_file_sha256"]):
            raise ValueError("completed EMA bundle belongs to different source artifacts")
        return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_files = {}
    for seed in SEEDS:
        source_root = Path(by_seed[seed]["bundle_dir"])
        source_manifest = manifests[seed]
        for relative in ("manifest.json", "plan.json", "recipe.json"):
            source = source_root / relative
            target_relative = f"proof/source_seed_{seed}/{relative}"
            digest = sha256_file(source)
            _copy_verified(source, output_dir / target_relative, digest)
            copied_files[target_relative] = digest
        for member_id, row in source_manifest["members"].items():
            relative_paths = (row["checkpoint"], row["fit_record"])
            for relative in relative_paths:
                source = source_root / relative
                target_relative = f"proof/source_seed_{seed}/{relative}"
                digest = sha256_file(source)
                _copy_verified(source, output_dir / target_relative, digest)
                copied_files[target_relative] = digest
    study_proof = [("protocol.json", "proof/study/protocol.json"),
                    ("result.json", "proof/study/result.json"),
                    ("batches/batch_1/result.json", "proof/study/batch_1.json"),
                    ("batches/batch_2/result.json", "proof/study/batch_2.json")]
    for source_rel, target_rel in study_proof:
        source = study_dir / source_rel
        digest = sha256_file(source)
        _copy_verified(source, output_dir / target_rel, digest)
        copied_files[target_rel] = digest

    members = {}
    cache = HalfCache(data_dir, max_bytes=_CACHE_LIMIT)
    for seed in SEEDS:
        members[str(seed)] = {}
        source_root = Path(by_seed[seed]["bundle_dir"])
        for member in baseline_selection["members"]:
            member_id = member["id"]
            target = output_dir / "members" / str(seed) / member_id
            replayed = _replay_final_tail(
                data_dir=data_dir, plan=_record(source_root / "plan.json"),
                selection=baseline_selection, seed=seed, member=member,
                source_root=source_root, source_manifest=manifests[seed],
                output_dir=target, cache=cache)
            fit, checkpoint, fit_path = (replayed["fit"], replayed["checkpoint"],
                                         replayed["fit_record"])
            rel_checkpoint = checkpoint.relative_to(output_dir).as_posix()
            rel_fit = fit_path.relative_to(output_dir).as_posix()
            members[str(seed)][member_id] = {
                "checkpoint": rel_checkpoint, "sha256": sha256_file(checkpoint),
                "fit_record": rel_fit, "fit_record_sha256": sha256_file(fit_path),
                "model_kwargs": {"player_dim": 8, "global_dim": 9,
                                 "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                                 **member["spec"]["model_kwargs"]},
                "spec": member["spec"], "seed": seed, "steps": _END_STEPS,
                "ema_sha256": fit["ema_sha256"],
            }
            copied_files[rel_checkpoint] = sha256_file(checkpoint)
            copied_files[rel_fit] = sha256_file(fit_path)
    if cache.peak_bytes > _CACHE_LIMIT:
        raise RuntimeError("final EMA raw-half cache exceeded its one GB cap")
    del cache
    manifest = {
        "schema": _SCHEMA, "portable": True, "development_only": True,
        "seeds": list(SEEDS), "member_ids": list(MEMBERS),
        "seed_weights": {str(seed): .5 for seed in SEEDS},
        "members": members, "baseline_selection": baseline_selection,
        "candidate_selection": study_result["candidate_selection"],
        "ema_protocol_sha256": study_refs["protocol_sha256"],
        "ema_protocol_file_sha256": study_refs["protocol_file_sha256"],
        "ema_result_file_sha256": study_refs["result_file_sha256"],
        "study_source_sha256": study_protocol.get("source_sha256", {}),
        "study_runtime": study_protocol.get("runtime", {}),
        "source_bundle_manifest_sha256": {
            str(seed): sha256_file(Path(by_seed[seed]["bundle_dir"]) / "manifest.json")
            for seed in SEEDS},
        "files": copied_files,
        "steps": _END_STEPS,
        "ema": {"decay": _DECAY, "start_steps": _START_STEPS,
                "end_steps": _END_STEPS, "updates": 200,
                "interpretation": "200-update tail EMA, not full-run EMA"},
        "source_sha256": _source_hashes(), "runtime": _execution_environment(),
        "target_ground_truth_used": False,
    }
    campaign_protocol.publish(output_dir / "manifest.json", manifest)
    return verify_ema_bundle(output_dir)


def verify_ema_bundle(bundle_dir) -> dict:
    """Verify a self-contained EMA bundle without touching source artifact paths."""
    root = Path(bundle_dir).resolve()
    manifest = _record(root / "manifest.json")
    if (manifest.get("schema") != _SCHEMA or manifest.get("portable") is not True
            or manifest.get("seeds") != list(SEEDS) or manifest.get("member_ids") != list(MEMBERS)
            or manifest.get("steps") != _END_STEPS or manifest.get("seed_weights") != {
                str(seed): .5 for seed in SEEDS}):
        raise ValueError("EMA bundle manifest schema or ensemble configuration differs")
    for relative, digest in manifest.get("files", {}).items():
        path = _safe_path(root, relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"EMA bundle file hash differs: {relative}")
    protocol = _record(root / "proof/study/protocol.json")
    result = _record(root / "proof/study/result.json")
    if (_digest(protocol) != manifest.get("ema_protocol_sha256")
            or sha256_file(root / "proof/study/protocol.json")
            != manifest.get("ema_protocol_file_sha256")
            or sha256_file(root / "proof/study/result.json")
            != manifest.get("ema_result_file_sha256")
            or result.get("protocol_sha256") != manifest.get("ema_protocol_sha256")
            or result.get("eligible") is not True
            or result.get("candidate_selection") != manifest.get("candidate_selection")
            or protocol.get("candidate_selection") != manifest.get("candidate_selection")
            or result.get("baseline_selection") != manifest.get("baseline_selection")
            or protocol.get("baseline_selection") != manifest.get("baseline_selection")
            or protocol.get("source_sha256") != manifest.get("study_source_sha256")
            or protocol.get("runtime") != manifest.get("study_runtime")
            or manifest.get("candidate_selection") is None):
        raise ValueError("contained EMA proof or selections differ")
    ensemble_bagging_submission._selection_valid(
        manifest["candidate_selection"], manifest["baseline_selection"])
    batches = result.get("batches", [])
    if len(batches) != 2 or [row.get("batch") for row in batches] != ["batch_1", "batch_2"]:
        raise ValueError("contained EMA proof does not cover both batches")
    for index, batch in enumerate(batches, 1):
        contained = _record(root / "proof/study" / f"batch_{index}.json")
        if contained != batch:
            raise ValueError("contained EMA batch differs from result")
        if (ema_stability._gate(batch["raw_v4_inner"], batch["ema_inner"])
                != batch.get("inner_checks")
                or ema_stability._outer_gate(batch["raw_v4_outer"], batch["ema_outer"])
                != batch.get("outer_checks")
                or batch["inner_checks"].get("passed") is not True
                or batch["outer_checks"].get("passed") is not True):
            raise ValueError("contained EMA batch gate failed")
    for seed in SEEDS:
        source_root = root / "proof" / f"source_seed_{seed}"
        source_manifest = _record(source_root / "manifest.json")
        source_plan = _record(source_root / "plan.json")
        source_recipe = _record(source_root / "recipe.json")
        if (sha256_file(source_root / "manifest.json")
                != manifest.get("source_bundle_manifest_sha256", {}).get(str(seed))
                or source_manifest.get("seed") != seed
                or source_manifest.get("selection") != manifest.get("baseline_selection")
                or set(source_manifest.get("members", {})) != set(MEMBERS)
                or source_manifest.get("steps") != _END_STEPS
                or source_manifest.get("fit_matches") != list(_FIT_MATCHES)
                or source_plan.get("selection") != manifest.get("baseline_selection")
                or source_manifest.get("provenance_sha256") != _digest(source_plan)
                or source_recipe.get("selection") != manifest.get("baseline_selection")):
            raise ValueError(f"contained source bundle proof differs: {seed}")
        for relative, digest in source_manifest.get("files", {}).items():
            source_path = _safe_path(source_root, relative)
            if not source_path.is_file() or sha256_file(source_path) != digest:
                raise ValueError(f"contained original bundle artifact differs: {seed}/{relative}")
        seed_rows = manifest.get("members", {}).get(str(seed), {})
        if set(seed_rows) != set(MEMBERS):
            raise ValueError(f"EMA seed member order differs: {seed}")
        for member_id in MEMBERS:
            row = seed_rows[member_id]
            checkpoint = _safe_path(root, row["checkpoint"])
            fit_path = _safe_path(root, row["fit_record"])
            if (sha256_file(checkpoint) != row.get("sha256")
                    or sha256_file(fit_path) != row.get("fit_record_sha256")):
                raise ValueError(f"EMA member artifact hash differs: {seed}/{member_id}")
            fit = _record(fit_path)
            source_member = source_manifest.get("members", {}).get(member_id, {})
            original_fit_path = _safe_path(source_root, source_member.get("fit_record", ""))
            original_checkpoint = _safe_path(source_root, source_member.get("checkpoint", ""))
            original_fit = _record(original_fit_path)
            original_identity = source_plan
            _, original_payload = load_model_checkpoint(original_checkpoint, device="cpu")
            model, payload = load_model_checkpoint(checkpoint, device="cpu")
            config = payload["training_config"]
            parity = {"raw_weights_match_endpoint_state": True,
                      "raw_weights_match_baseline_checkpoint": True,
                      "raw_training_progress_matches_endpoint": True,
                      "compared_optimizer_rng_transcript_and_counters": True,
                      "start_steps": _START_STEPS, "end_steps": _END_STEPS}
            metadata = _ema_metadata({"raw_parity": parity})
            if (fit.get("schema") != "final_tail_ema_fit_v1"
                    or fit.get("member_id") != member_id or fit.get("seed") != seed
                    or fit.get("steps") != _END_STEPS
                    or fit.get("train_matches") != list(_FIT_MATCHES)
                    or fit.get("validation_matches") != []
                    or fit.get("raw_parity") != parity or fit.get("tail_ema") != metadata
                    or fit.get("ema_sha256") != row.get("ema_sha256")
                    or fit.get("checkpoint_sha256") != row.get("sha256")
                    or fit.get("training_config_sha256") != _digest(payload["training_config"])
                    or fit.get("identity_sha256") != _digest(original_identity)
                    or fit.get("inputs", {}).get("source_bundle_manifest_sha256")
                    != manifest["source_bundle_manifest_sha256"][str(seed)]
                    or fit.get("inputs", {}).get("source_fit_sha256")
                    != sha256_file(original_fit_path)
                    or fit.get("inputs", {}).get("baseline_checkpoint_sha256")
                    != sha256_file(original_checkpoint)
                    or original_fit.get("checkpoint_sha256") != sha256_file(original_checkpoint)
                    or original_fit.get("checkpoint_sha256") != source_member.get("sha256")
                    or original_fit.get("identity_sha256") != _digest(source_plan)
                    or original_fit.get("member_id") != member_id
                    or original_fit.get("recipe_sha256") != _digest(row.get("spec"))
                    or original_fit.get("steps") != _END_STEPS
                    or original_fit.get("training_config_sha256")
                    != _digest(original_payload["training_config"])
                    or original_payload["training_config"].get("campaign_identity_sha256")
                    != _digest(source_plan)
                    or original_payload["training_config"].get("campaign_member_id") != member_id
                    or original_payload["training_config"].get("campaign_spec") != row.get("spec")
                    or original_payload["training_config"].get("train_matches") != list(_FIT_MATCHES)
                    or original_payload["training_config"].get("validation_matches") != []
                    or original_payload["training_config"].get("seed") != seed
                    or original_payload["training_config"].get("steps") != _END_STEPS
                    or original_payload.get("model_kwargs") != source_member.get("model_kwargs")
                    or _tensor_digest(payload["state_dict"]) != fit.get("ema_sha256")
                    or payload.get("ema_metadata") != metadata
                    or config.get("seed") != seed or config.get("steps") != _END_STEPS
                    or config.get("campaign_spec") != row.get("spec")
                    or config.get("train_matches") != list(_FIT_MATCHES)
                    or config.get("validation_matches") != []
                    or payload.get("model_kwargs") != row.get("model_kwargs")):
                raise ValueError(f"EMA checkpoint provenance differs: {seed}/{member_id}")
            del model
    return {"bundle_dir": str(root), "manifest": manifest,
            "protocol": protocol, "result": result}


def _predict_half(frames, models: dict[int, dict], selection: dict) -> np.ndarray:
    features = build_features(frames)
    predictions = {seed: {} for seed in SEEDS}
    for member in selection["members"]:
        member_id = member["id"]
        for seed in SEEDS:
            net = models[seed][member_id]
            predictions[seed][member_id] = predict_features(
                features, net, device="cpu", center_size=CENTER_SIZE, halo=net.radius)
    scores = combine_four_predictions(predictions, selection)
    from experiments.campaign_decode import decode_arrays, reference_basis
    from experiments.campaign_evaluation import _gap_seconds

    arrays = {"neural_live": scores["live_probability"],
              "neural_home": scores["home_probability"],
              "neural_boundary": scores["boundary"], "ball_valid": frames.ball_valid,
              "nearest_ball_seconds": _gap_seconds(frames.ball_valid, frames.fps),
              **reference_basis(frames, features)}
    return decode_arrays(arrays, selection["decoder"])


def infer_ema_submission(bundle_dir, target_dir, output_path) -> dict:
    """Create a portable target submission from only the EMA bundle and target inputs."""
    started = time.perf_counter()
    bundle_dir, target_dir, output = (Path(value).resolve()
                                      for value in (bundle_dir, target_dir, output_path))
    sidecar = output.with_suffix(".manifest.json")
    if output == sidecar or any(_inside(base, path) for base in (bundle_dir, target_dir)
                                for path in (output, sidecar)):
        raise ValueError("submission and sidecar output must be outside bundle and target inputs")
    bundle = verify_ema_bundle(bundle_dir)
    inputs = campaign_submission._verified_inputs(target_dir)
    signature = {"bundle_manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
                 "target_inputs": inputs,
                 "selection": bundle["manifest"]["candidate_selection"],
                 "rows": N_FRAMES, "spans": [list(span) for span in TARGET_SPANS],
                 "center_size": CENTER_SIZE, "device": "cpu",
                 "seed_weights": {str(seed): .5 for seed in SEEDS}}
    fingerprint = _digest(signature)
    if output.exists() or sidecar.exists():
        if not (output.is_file() and sidecar.is_file()):
            raise ValueError("existing EMA submission is incomplete, use a new output path")
        previous = json.loads(sidecar.read_text())
        if (previous.get("request_sha256") != fingerprint
                or previous.get("output", {}).get("sha256") != sha256_file(output)):
            raise ValueError("existing EMA output belongs to a different inference request")
        campaign_submission._validate_output(output)
        return previous
    models = {}
    root = bundle_dir
    for seed in SEEDS:
        models[seed] = {}
        for member_id in MEMBERS:
            row = bundle["manifest"]["members"][str(seed)][member_id]
            models[seed][member_id] = load_model_checkpoint(
                _safe_path(root, row["checkpoint"]), device="cpu")[0]
    frames = read_atd(inputs["raw"]["path"], inputs["metadata"]["path"],
                      n_frames=N_FRAMES, pitch=GAMES[GAME]["pitch"],
                      fps=GAMES[GAME]["fps"], graded_spans=TARGET_SPANS)
    codes = np.full(len(frames.frame_id), 3, dtype=np.int8)
    halves = []
    for period, (start, stop) in enumerate(TARGET_SPANS, 1):
        half = frames.subset(start, stop)
        half.period = period
        path = _predict_half(half, models, bundle["manifest"]["candidate_selection"])
        if path.shape != (stop - start,) or not np.isin(path, [0, 1, 2]).all():
            raise ValueError("EMA decoder returned invalid half labels")
        codes[start:stop] = path
        halves.append({"period": period, "span": [start, stop],
                       "switches": int(np.count_nonzero(np.diff(path))),
                       "ball_observed_fraction": float(half.ball_valid.mean())})
    from possession.cli import write_submission

    write_submission(output, codes, n_expected=N_FRAMES)
    campaign_submission._validate_output(output)
    report = {"schema": _SCHEMA, "ground_truth_used": False,
              "request_sha256": fingerprint, "request": signature,
              "output": {"path": str(output), "sha256": sha256_file(output),
                         "rows": N_FRAMES,
                         "label_counts": dict(zip("HADX", np.bincount(codes, minlength=4).tolist()))},
              "halves": halves, "elapsed_seconds": time.perf_counter() - started,
              "format_validated": True}
    write_json(sidecar, report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    report = infer_ema_submission(args.bundle, args.target, args.output)
    print(json.dumps({"output": report["output"], "halves": report["halves"],
                      "format_validated": report["format_validated"]}, indent=2))


if __name__ == "__main__":
    main()
