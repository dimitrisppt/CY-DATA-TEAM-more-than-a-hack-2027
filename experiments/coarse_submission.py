"""Portable two-branch submission bundle for the fixed coarse-context specialist."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments import (
    campaign_protocol,
    campaign_refit,
    campaign_submission,
    coarse_context,
    coarse_stability,
    ema_submission,
)
from experiments.ensemble_bagging_submission import (
    CENTER_SIZE,
    MEMBERS,
    N_FRAMES,
    SEEDS,
    TARGET_SPANS,
    combine_four_predictions,
)
from experiments.robustness import _execution_environment, _source_hashes
from games import GAME, GAMES
from possession.features import build_features
from possession.io_atd import read_atd
from possession.predict import load_model_checkpoint, predict_features
from possession.prepared import verify_prepared
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache

_SCHEMA = "coarse_context_submission_v1"
_COARSE_MEMBER = "coarse_player_only"
_FINAL_FIT_MATCHES = campaign_refit.FIT_MATCHES
_CACHE_LIMIT = 1_000_000_000


def _record(path: Path) -> dict:
    value = campaign_protocol.read_record(path)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _inside(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _safe_path(root: Path, relative: str) -> Path:
    part = PurePosixPath(relative)
    if not part.parts or part.is_absolute() or ".." in part.parts:
        raise ValueError(f"unsafe relative bundle path: {relative}")
    resolved = (root / Path(*part.parts)).resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"bundle path escapes root: {relative}")
    return resolved


def _copy_verified(source: Path, target: Path, digest: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"refusing to replace different bundle file: {target}")
        return
    shutil.copyfile(source, target)
    if sha256_file(target) != digest:
        target.unlink(missing_ok=True)
        raise ValueError(f"copied file hash differs: {target}")


def _verify_file_map(root: Path, files: dict[str, str]) -> None:
    for relative, digest in files.items():
        path = _safe_path(root, relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"coarse bundle file hash differs: {relative}")


def _verify_source_runtime(manifest: dict) -> None:
    current = _source_hashes()
    changed = sorted(path for path, digest in manifest.get("source_sha256", {}).items()
                     if current.get(path) != digest)
    if changed:
        raise ValueError(f"coarse bundle source code differs: {changed}")
    if _execution_environment() != manifest.get("runtime"):
        raise ValueError("coarse bundle execution environment differs")


def _verify_derived_data(coarse_data_dir: Path, expected_derivation_sha: str,
                         expected_source_prepared_sha: str) -> tuple[dict, dict, dict]:
    derivation_path = coarse_data_dir / "coarse-derivation.json"
    derivation = coarse_context._read_derivation(derivation_path)
    if derivation is None or coarse_context._digest(derivation) != expected_derivation_sha:
        raise ValueError("coarse derivation record differs from frozen stability protocol")
    if (derivation.get("schema") != "coarse_context_derivation_v1"
            or derivation.get("processor") != coarse_context.PROCESSOR
            or coarse_context._digest(derivation.get("source_prepared"))
            != expected_source_prepared_sha):
        raise ValueError("coarse source data or processor differs from frozen protocol")
    expected_names = {f"{match}_half{period}.npz" for match in _FINAL_FIT_MATCHES
                      for period in (1, 2)}
    paths = [coarse_data_dir / name for name in sorted(expected_names)]
    actual_hashes = {path.name: sha256_file(path) for path in paths if path.is_file()}
    if actual_hashes != derivation.get("derived_cache_sha256"):
        raise ValueError("coarse prepared cache hashes differ from derivation")
    manifest_path = coarse_data_dir / "manifest.json"
    if sha256_file(manifest_path) != derivation.get("derived_manifest_sha256"):
        raise ValueError("coarse prepared manifest checksum differs")
    provenance = verify_prepared(coarse_data_dir, paths)
    if (provenance.get("status") != "verified_manifest"
            or provenance.get("sources") != derivation["source_prepared"].get("sources")
            or {row["cache"] for row in provenance.get("caches", [])} != expected_names):
        raise ValueError("coarse prepared-data provenance or cache grid differs")
    return derivation, provenance, actual_hashes


def _verify_study(study_dir: Path, ema_manifest: dict) -> tuple[dict, dict, dict]:
    protocol = _record(study_dir / "protocol.json")
    result = _record(study_dir / "result.json")
    protocol_sha = coarse_stability._digest(protocol)
    if (protocol.get("schema") != "coarse_stability_v1"
            or protocol.get("baseline_kind") != "ema"
            or result.get("schema") != "coarse_stability_v1"
            or result.get("baseline_kind") != "ema"
            or result.get("protocol_sha256") != protocol_sha
            or result.get("eligible") is not True
            or protocol.get("candidate_selection")
            != ema_manifest.get("candidate_selection")
            or protocol.get("baseline_selection")
            != ema_manifest.get("candidate_selection")
            or result.get("candidate_selection") != protocol.get("candidate_selection")
            or result.get("baseline_selection") != protocol.get("baseline_selection")
            or protocol.get("ema_protocol_sha256")
            != ema_manifest.get("ema_protocol_sha256")
            or protocol.get("ema_result_sha256")
            != ema_manifest.get("ema_result_file_sha256")):
        raise ValueError("coarse study is incomplete or differs from verified EMA candidate")
    current_sources = _source_hashes()
    if (protocol.get("runtime") != _execution_environment()
            or any(current_sources.get(name) != digest
                   for name, digest in protocol.get("source_sha256", {}).items())):
        raise ValueError("coarse study source or runtime ancestry differs")
    batches = result.get("batches")
    if not isinstance(batches, list) or [row.get("batch") for row in batches] != [
            "batch_1", "batch_2"]:
        raise ValueError("coarse study must contain both frozen batches")
    artifact_map = {}
    for index, row in enumerate(batches, 1):
        saved = _record(study_dir / "batches" / row["batch"] / "result.json")
        if saved != row or saved.get("protocol_sha256") != protocol_sha:
            raise ValueError(f"coarse batch result differs: {row['batch']}")
        inner = coarse_stability._gate(saved["baseline_scores"]["inner"],
                                       saved["candidate_scores"]["inner"])
        outer = coarse_stability._outer_gate(saved["baseline_scores"]["outer"],
                                             saved["candidate_scores"]["outer"])
        if (inner != saved.get("inner_checks") or outer != saved.get("outer_checks")
                or not inner.get("passed") or not outer.get("passed")):
            raise ValueError(f"coarse promotion checks failed: {row['batch']}")
        for relative, digest in saved.get("artifact_sha256", {}).items():
            source = _safe_path(study_dir, relative)
            if not source.is_file() or sha256_file(source) != digest:
                raise ValueError(f"coarse study artifact hash differs: {relative}")
            artifact_map[relative] = digest
    return protocol, result, {"protocol_sha256": protocol_sha,
                              "protocol_file_sha256": sha256_file(study_dir / "protocol.json"),
                              "result_file_sha256": sha256_file(study_dir / "result.json"),
                              "artifacts": artifact_map}


def _final_fit_protocol(study_protocol: dict, study_sha: str, ema_manifest: dict,
                        derivation: dict, derived_provenance: dict) -> dict:
    spec = coarse_stability._spec(ema_manifest["candidate_selection"])
    if spec != study_protocol.get("model_spec"):
        raise ValueError("frozen coarse final fit spec differs from gated study")
    return {
        "schema": "coarse_context_final_fit_v1", "development_only": True,
        "baseline_kind": "ema", "coarse_study_protocol_sha256": study_sha,
        "ema_manifest_sha256": ema_manifest["_manifest_sha256"],
        "coarse_derivation_sha256": coarse_context._digest(derivation),
        "derived_prepared_provenance_sha256": coarse_context._digest(derived_provenance),
        "candidate_selection": ema_manifest["candidate_selection"],
        "model_id": _COARSE_MEMBER, "model_spec": spec,
        "model_spec_sha256": coarse_stability._digest(spec),
        "train_matches": list(_FINAL_FIT_MATCHES), "validation_matches": [],
        "steps": 7200, "scope": "final", "fold": 0,
        "base_seeds": list(SEEDS), "seed_mapping_match": "DFL-MAT-J03WMX",
        "source_sha256": _source_hashes(), "runtime": _execution_environment(),
        "target_ground_truth_used": False,
    }


def _verify_final_fit(checkpoint: Path, fit_path: Path, *, spec: dict,
                      protocol_sha: str, derivation_sha: str,
                      base_seed: int, prepared_hashes: dict | None = None,
                      prepared_provenance_sha256: str | None = None) -> dict:
    from experiments.tune import _seed_for_fold

    actual_seed = _seed_for_fold(base_seed, "DFL-MAT-J03WMX", 0)
    fit = _record(fit_path)
    expected_identity = {"protocol_sha256": protocol_sha,
                         "spec_sha256": coarse_stability._digest(spec),
                         "train_matches": list(_FINAL_FIT_MATCHES),
                         "seed": actual_seed, "name": _COARSE_MEMBER}
    if (fit.get("schema") != "coarse_member_fit_v1"
            or fit.get("identity") != expected_identity or fit.get("steps") != 7200
            or fit.get("derivation_sha256") != derivation_sha
            or fit.get("checkpoint_sha256") != sha256_file(checkpoint)):
        raise ValueError(f"final coarse fit ancestry differs: seed {base_seed}")
    _, payload = load_model_checkpoint(checkpoint, device="cpu")
    config = payload.get("training_config", {})
    if (payload.get("model_kwargs") != spec["model_kwargs"]
            or config.get("campaign_spec") != spec or config.get("steps") != 7200
            or config.get("train_matches") != list(_FINAL_FIT_MATCHES)
            or config.get("validation_matches") != [] or config.get("seed") != actual_seed
            or config.get("coarse_derivation_sha256") != derivation_sha
            or config.get("coarse_protocol_sha256") != protocol_sha
            or config.get("coarse_scope") != "final" or config.get("coarse_fold") != 0
            or config.get("coarse_seed") != base_seed
            or (prepared_hashes is not None and payload.get("prepared_sha256") != prepared_hashes)
            or (prepared_provenance_sha256 is not None
                and coarse_stability._digest(payload.get("dataset_provenance"))
                != prepared_provenance_sha256)):
        raise ValueError(f"final coarse checkpoint split or config differs: seed {base_seed}")
    return {"base_seed": base_seed, "actual_seed": actual_seed,
            "checkpoint_sha256": sha256_file(checkpoint),
            "fit_record_sha256": sha256_file(fit_path),
            "model_kwargs": payload["model_kwargs"]}


def export_coarse_bundle(coarse_study_dir, ema_bundle_dir, coarse_data_dir, output_dir) -> dict:
    """Authenticate both promotion gates and prepare two resumable all-seven final fits."""
    study_dir, ema_dir, data_dir, output_dir = (Path(value).resolve() for value in
                                                 (coarse_study_dir, ema_bundle_dir,
                                                  coarse_data_dir, output_dir))
    if any(_inside(output_dir, source) for source in (study_dir, ema_dir, data_dir)):
        raise ValueError("coarse bundle output must be separate from every source")
    ema_bundle = ema_submission.verify_ema_bundle(ema_dir)
    ema_manifest = dict(ema_bundle["manifest"])
    ema_manifest["_manifest_sha256"] = sha256_file(ema_dir / "manifest.json")
    study_protocol, _study_result, study_refs = _verify_study(study_dir, ema_manifest)
    derivation, derived_provenance, derived_hashes = _verify_derived_data(
        data_dir, study_protocol["coarse_derivation_sha256"],
        study_protocol["source_prepared_sha256"])
    final_protocol = _final_fit_protocol(study_protocol, study_refs["protocol_sha256"],
                                         ema_manifest, derivation, derived_provenance)
    spec = final_protocol["model_spec"]
    output_dir.mkdir(parents=True, exist_ok=True)
    final_protocol_path = output_dir / "proof" / "final-fit-protocol.json"
    campaign_protocol.publish(final_protocol_path, final_protocol)
    final_protocol_sha = coarse_stability._digest(final_protocol)
    final_protocol_rel = final_protocol_path.relative_to(output_dir).as_posix()
    file_hashes = {final_protocol_rel: sha256_file(final_protocol_path)}
    if (output_dir / "manifest.json").is_file():
        existing = verify_coarse_bundle(output_dir)
        old = existing["manifest"]
        if (old.get("ema_manifest_sha256") != ema_manifest["_manifest_sha256"]
                or old.get("coarse_study_protocol_sha256") != study_refs["protocol_sha256"]
                or old.get("coarse_study_protocol_file_sha256")
                != study_refs["protocol_file_sha256"]
                or old.get("coarse_study_result_file_sha256") != study_refs["result_file_sha256"]
                or old.get("coarse_derivation_sha256") != final_protocol["coarse_derivation_sha256"]
                or old.get("fit_protocol_sha256") != final_protocol_sha
                or old.get("files", {}).get(final_protocol_rel) != file_hashes[final_protocol_rel]):
            raise ValueError("completed coarse bundle belongs to different source ancestry")
        return existing

    source_ema_files = {"manifest.json": sha256_file(ema_dir / "manifest.json"),
                        **ema_manifest["files"]}
    for relative, digest in source_ema_files.items():
        source = _safe_path(ema_dir, relative)
        target_rel = f"ema_bundle/{relative}"
        _copy_verified(source, output_dir / target_rel, digest)
        file_hashes[target_rel] = digest
    study_targets = (("protocol.json", "proof/coarse-study/protocol.json"),
                     ("result.json", "proof/coarse-study/result.json"),
                     ("batches/batch_1/result.json", "proof/coarse-study/batch_1.json"),
                     ("batches/batch_2/result.json", "proof/coarse-study/batch_2.json"))
    for source_rel, target_rel in study_targets:
        source = study_dir / source_rel
        digest = sha256_file(source)
        _copy_verified(source, output_dir / target_rel, digest)
        file_hashes[target_rel] = digest
    study_artifacts = {}
    for source_rel, digest in study_refs["artifacts"].items():
        target_rel = f"proof/coarse-study/artifacts/{source_rel}"
        _copy_verified(_safe_path(study_dir, source_rel), output_dir / target_rel, digest)
        file_hashes[target_rel] = digest
        study_artifacts[source_rel] = target_rel

    derivation_source = data_dir / "coarse-derivation.json"
    manifest_source = data_dir / "manifest.json"
    for source, rel in ((derivation_source, "proof/coarse-data/coarse-derivation.json"),
                        (manifest_source, "proof/coarse-data/manifest.json")):
        digest = sha256_file(source)
        _copy_verified(source, output_dir / rel, digest)
        file_hashes[rel] = digest

    staging = output_dir / ".fit_staging"
    fit_rows = {}
    cache = HalfCache(data_dir, max_bytes=_CACHE_LIMIT)
    final_fold = {"train": list(_FINAL_FIT_MATCHES), "validation": []}
    for base_seed in SEEDS:
        _fit, fit_path, checkpoint = coarse_stability._model_fit(
            data_dir, staging, spec, final_protocol_sha, final_protocol["coarse_derivation_sha256"],
            final_fold, base_seed, 0, "final", cache, lambda: None,
            lambda *_args, **_kwargs: None)
        source_fit_path = fit_path
        source_model_path = checkpoint
        _verify_final_fit(source_model_path, source_fit_path, spec=spec,
                          protocol_sha=final_protocol_sha,
                          derivation_sha=final_protocol["coarse_derivation_sha256"],
                          base_seed=base_seed, prepared_hashes=derived_hashes,
                          prepared_provenance_sha256=final_protocol[
                              "derived_prepared_provenance_sha256"])
        base = output_dir / "members" / str(base_seed) / _COARSE_MEMBER
        target_fit, target_model = base / "fit.json", base / "model.pt"
        model_hash = sha256_file(source_model_path)
        fit_hash = sha256_file(source_fit_path)
        _copy_verified(source_model_path, target_model, model_hash)
        _copy_verified(source_fit_path, target_fit, fit_hash)
        model_rel, fit_rel = (target_model.relative_to(output_dir).as_posix(),
                              target_fit.relative_to(output_dir).as_posix())
        file_hashes[model_rel], file_hashes[fit_rel] = model_hash, fit_hash
        fit_rows[str(base_seed)] = _verify_final_fit(
            target_model, target_fit, spec=spec, protocol_sha=final_protocol_sha,
            derivation_sha=final_protocol["coarse_derivation_sha256"], base_seed=base_seed,
            prepared_hashes=derived_hashes,
            prepared_provenance_sha256=final_protocol["derived_prepared_provenance_sha256"])
        fit_rows[str(base_seed)].update({"checkpoint": model_rel, "fit_record": fit_rel})
    if cache.peak_bytes > _CACHE_LIMIT:
        raise RuntimeError("coarse final raw-half cache exceeded one GB")
    del cache
    manifest = {
        "schema": _SCHEMA, "portable": True, "development_only": True,
        "seeds": list(SEEDS), "base_seeds": list(SEEDS), "actual_seeds": {
            seed: fit_rows[str(seed)]["actual_seed"] for seed in SEEDS},
        "member_id": _COARSE_MEMBER, "steps": 7200,
        "train_matches": list(_FINAL_FIT_MATCHES), "validation_matches": [],
        "coarse_spec": spec, "candidate_selection": ema_manifest["candidate_selection"],
        "baseline_selection": study_protocol["baseline_selection"],
        "original_baseline_selection": ema_manifest["baseline_selection"],
        "ema_manifest_sha256": ema_manifest["_manifest_sha256"],
        "ema_protocol_sha256": ema_manifest["ema_protocol_sha256"],
        "coarse_study_protocol_sha256": study_refs["protocol_sha256"],
        "coarse_study_protocol_file_sha256": study_refs["protocol_file_sha256"],
        "coarse_study_result_file_sha256": study_refs["result_file_sha256"],
        "coarse_derivation_sha256": final_protocol["coarse_derivation_sha256"],
        "derived_cache_sha256": derived_hashes,
        "fit_protocol_sha256": final_protocol_sha,
        "study_artifacts": study_artifacts, "final_fits": fit_rows,
        "source_sha256": _source_hashes(), "runtime": _execution_environment(),
        "files": file_hashes, "target_ground_truth_used": False,
    }
    campaign_protocol.publish(output_dir / "manifest.json", manifest)
    return verify_coarse_bundle(output_dir)


def verify_coarse_bundle(bundle_dir) -> dict:
    """Verify a portable coarse-plus-EMA bundle without source data directories."""
    root = Path(bundle_dir).resolve()
    manifest = _record(root / "manifest.json")
    if (manifest.get("schema") != _SCHEMA or manifest.get("portable") is not True
            or manifest.get("base_seeds") != list(SEEDS)
            or manifest.get("seeds") != list(SEEDS)
            or manifest.get("member_id") != _COARSE_MEMBER or manifest.get("steps") != 7200
            or manifest.get("train_matches") != list(_FINAL_FIT_MATCHES)
            or manifest.get("validation_matches") != []):
        raise ValueError("coarse bundle manifest config differs")
    _verify_source_runtime(manifest)
    _verify_file_map(root, manifest.get("files", {}))
    nested_ema = ema_submission.verify_ema_bundle(root / "ema_bundle")
    if (sha256_file(root / "ema_bundle" / "manifest.json")
            != manifest.get("ema_manifest_sha256")
            or nested_ema["manifest"].get("ema_protocol_sha256")
            != manifest.get("ema_protocol_sha256")
            or nested_ema["manifest"].get("candidate_selection")
            != manifest.get("candidate_selection")
            or nested_ema["manifest"].get("baseline_selection")
            != manifest.get("original_baseline_selection")):
        raise ValueError("nested EMA bundle ancestry differs")
    protocol_path = root / "proof/coarse-study/protocol.json"
    result_path = root / "proof/coarse-study/result.json"
    protocol, result = _record(protocol_path), _record(result_path)
    protocol_sha = coarse_stability._digest(protocol)
    if (protocol_sha != manifest.get("coarse_study_protocol_sha256")
            or sha256_file(protocol_path) != manifest.get("coarse_study_protocol_file_sha256")
            or sha256_file(result_path) != manifest.get("coarse_study_result_file_sha256")
            or result.get("protocol_sha256") != protocol_sha
            or result.get("eligible") is not True or result.get("baseline_kind") != "ema"
            or result.get("candidate_selection") != manifest.get("candidate_selection")
            or result.get("baseline_selection") != manifest.get("baseline_selection")
            or protocol.get("baseline_selection") != manifest.get("baseline_selection")
            or protocol.get("candidate_selection") != manifest.get("candidate_selection")):
        raise ValueError("contained coarse study protocol/result differs")
    batches = result.get("batches", [])
    if len(batches) != 2 or [row.get("batch") for row in batches] != ["batch_1", "batch_2"]:
        raise ValueError("contained coarse study does not include both batches")
    for index, row in enumerate(batches, 1):
        contained = _record(root / "proof/coarse-study" / f"batch_{index}.json")
        if contained != row:
            raise ValueError("contained coarse batch record differs from result")
        if (coarse_stability._gate(row["baseline_scores"]["inner"],
                                   row["candidate_scores"]["inner"])
                != row.get("inner_checks")
                or coarse_stability._outer_gate(row["baseline_scores"]["outer"],
                                                row["candidate_scores"]["outer"])
                != row.get("outer_checks")
                or row["inner_checks"].get("passed") is not True
                or row["outer_checks"].get("passed") is not True):
            raise ValueError("contained coarse promotion gate failed")
    mapping = manifest.get("study_artifacts", {})
    expected_artifacts = {name for row in batches for name in row.get("artifact_sha256", {})}
    if set(mapping) != expected_artifacts:
        raise ValueError("coarse study artifact map differs from batch manifests")
    for original, relative in mapping.items():
        wanted = next(row["artifact_sha256"][original] for row in batches
                      if original in row["artifact_sha256"])
        if sha256_file(_safe_path(root, relative)) != wanted:
            raise ValueError(f"copied coarse study artifact differs: {original}")
    derivation_path = root / "proof/coarse-data/coarse-derivation.json"
    derivation = coarse_context._read_derivation(derivation_path)
    if (derivation is None
            or coarse_context._digest(derivation) != manifest.get("coarse_derivation_sha256")
            or derivation.get("derived_cache_sha256") != manifest.get("derived_cache_sha256")
            or sha256_file(root / "proof/coarse-data/manifest.json")
            != derivation.get("derived_manifest_sha256")):
        raise ValueError("contained coarse derivation proof differs")
    final_protocol = _record(root / "proof/final-fit-protocol.json")
    final_protocol_sha = coarse_stability._digest(final_protocol)
    if (final_protocol_sha != manifest.get("fit_protocol_sha256")
            or final_protocol.get("coarse_study_protocol_sha256") != protocol_sha
            or final_protocol.get("ema_manifest_sha256") != manifest.get("ema_manifest_sha256")
            or final_protocol.get("coarse_derivation_sha256")
            != manifest.get("coarse_derivation_sha256")
            or final_protocol.get("model_spec") != manifest.get("coarse_spec")):
        raise ValueError("contained final fit protocol differs")
    fits = {}
    for base_seed in SEEDS:
        row = manifest.get("final_fits", {}).get(str(base_seed))
        if not isinstance(row, dict):
            raise TypeError(f"final coarse seed fit missing: {base_seed}")
        checkpoint, fit_path = _safe_path(root, row["checkpoint"]), _safe_path(root, row["fit_record"])
        fit_info = _verify_final_fit(checkpoint, fit_path, spec=manifest["coarse_spec"],
                                     protocol_sha=final_protocol_sha,
                                     derivation_sha=manifest["coarse_derivation_sha256"],
                                     base_seed=base_seed,
                                     prepared_hashes=manifest["derived_cache_sha256"],
                                     prepared_provenance_sha256=final_protocol[
                                         "derived_prepared_provenance_sha256"])
        if any(row.get(key) != value for key, value in fit_info.items()):
            raise ValueError(f"final coarse fit record differs: {base_seed}")
        fits[base_seed] = str(checkpoint)
    return {"bundle_dir": str(root), "manifest": manifest, "protocol": protocol,
            "result": result, "ema_bundle": nested_ema, "coarse_model_paths": fits}


def _predict_half(frames, fine_models: dict[int, dict], coarse_models: dict[int, torch.nn.Module],
                  selection: dict) -> np.ndarray:
    features = build_features(frames)
    predictions = {seed: {} for seed in SEEDS}
    for member in selection["members"]:
        member_id = member["id"]
        for seed in SEEDS:
            model = fine_models[seed][member_id]
            predictions[seed][member_id] = predict_features(
                features, model, device="cpu", center_size=CENTER_SIZE, halo=model.radius)
    fine = combine_four_predictions(predictions, selection)
    from experiments.campaign_decode import decode_arrays, reference_basis
    from experiments.campaign_evaluation import _gap_seconds

    fine_arrays = {"neural_live": fine["live_probability"],
                   "neural_home": fine["home_probability"],
                   "neural_boundary": fine["boundary"], "ball_valid": frames.ball_valid,
                   "nearest_ball_seconds": _gap_seconds(frames.ball_valid, frames.fps),
                   **reference_basis(frames, features)}
    coarse_predictions = [coarse_context.predict_coarse_native(frames, coarse_models[seed])
                          for seed in SEEDS]
    coarse_live = coarse_stability._average_logits(
        coarse_predictions[0]["live_probability"], coarse_predictions[1]["live_probability"])
    coarse_home = coarse_stability._average_logits(
        coarse_predictions[0]["home_probability"], coarse_predictions[1]["home_probability"])
    mixed = coarse_context.blend_coarse_context(fine_arrays, coarse_live, coarse_home)
    return decode_arrays({"neural_live": mixed["neural_live"],
                          "neural_home": mixed["neural_home"],
                          "neural_boundary": mixed["neural_boundary"],
                          "ball_valid": frames.ball_valid,
                          "nearest_ball_seconds": fine_arrays["nearest_ball_seconds"],
                          **reference_basis(frames, features)}, selection["decoder"])


def infer_coarse_submission(bundle_dir, target_dir, output_path) -> dict:
    """Infer from only the portable EMA/coarse bundle and automatic target inputs."""
    started = time.perf_counter()
    bundle_dir, target_dir, output = (Path(value).resolve() for value in
                                      (bundle_dir, target_dir, output_path))
    sidecar = output.with_suffix(".manifest.json")
    if output == sidecar or any(_inside(base, path) for base in (bundle_dir, target_dir)
                                for path in (output, sidecar)):
        raise ValueError("submission output must be separate from bundle and target inputs")
    bundle = verify_coarse_bundle(bundle_dir)
    inputs = campaign_submission._verified_inputs(target_dir)
    signature = {"bundle_manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
                 "target_inputs": inputs,
                 "selection": bundle["manifest"]["candidate_selection"],
                 "rows": N_FRAMES, "spans": [list(span) for span in TARGET_SPANS],
                 "device": "cpu", "center_size": CENTER_SIZE,
                 "coarse_rate_hz": coarse_context.COARSE_RATE_HZ,
                 "coarse_blend_max": .25, "base_seeds": list(SEEDS)}
    fingerprint = ema_submission._digest(signature)
    if output.exists() or sidecar.exists():
        if not (output.is_file() and sidecar.is_file()):
            raise ValueError("existing coarse submission is incomplete, use a new path")
        previous = json.loads(sidecar.read_text())
        if (previous.get("request_sha256") != fingerprint
                or previous.get("output", {}).get("sha256") != sha256_file(output)):
            raise ValueError("existing coarse output belongs to another inference request")
        campaign_submission._validate_output(output)
        return previous
    ema_verified = bundle["ema_bundle"]
    ema_root = bundle_dir / "ema_bundle"
    fine_models = {}
    for seed in SEEDS:
        fine_models[seed] = {}
        for member_id in MEMBERS:
            row = ema_verified["manifest"]["members"][str(seed)][member_id]
            fine_models[seed][member_id] = load_model_checkpoint(
                _safe_path(ema_root, row["checkpoint"]), device="cpu")[0]
    coarse_models = {seed: load_model_checkpoint(bundle["coarse_model_paths"][seed],
                                                  device="cpu")[0] for seed in SEEDS}
    frames = read_atd(inputs["raw"]["path"], inputs["metadata"]["path"],
                      n_frames=N_FRAMES, pitch=GAMES[GAME]["pitch"],
                      fps=GAMES[GAME]["fps"], graded_spans=TARGET_SPANS)
    codes = np.full(len(frames.frame_id), 3, dtype=np.int8)
    halves = []
    for period, (start, stop) in enumerate(TARGET_SPANS, 1):
        half = frames.subset(start, stop)
        half.period = period
        path = _predict_half(half, fine_models, coarse_models,
                             bundle["manifest"]["candidate_selection"])
        if path.shape != (stop - start,) or not np.isin(path, [0, 1, 2]).all():
            raise ValueError("coarse decoder returned invalid half labels")
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
    result = infer_coarse_submission(args.bundle, args.target, args.output)
    print(json.dumps({"output": result["output"], "halves": result["halves"],
                      "format_validated": result["format_validated"]}, indent=2))


if __name__ == "__main__":
    main()
