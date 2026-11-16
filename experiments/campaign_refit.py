"""Refit the frozen overnight campaign ensemble on all seven public matches."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import pickle
from contextlib import contextmanager
from pathlib import Path

import torch

from experiments import campaign_protocol
from experiments.campaign_training import build_session, load_training_state, save_training_state
from experiments.robustness import _execution_environment, _source_hashes
from possession.prepared import verify_prepared
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache

SELECTION_MATCHES = (
    "DFL-MAT-J03WN1", "DFL-MAT-J03WOH", "DFL-MAT-J03WOY",
    "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9",
)
FIT_MATCHES = tuple(sorted((*SELECTION_MATCHES, "DFL-MAT-J03WMX")))
MEMBER_IDS = ("mean_007", "ball_attention_002")
CHECKPOINT_INTERVAL = 200


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _inside(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _write_record(path: Path, payload: dict) -> dict:
    return campaign_protocol.publish(path, payload)


def _record(path: Path) -> dict:
    value = campaign_protocol.read_record(path)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _preflight(campaign_dir: Path, data_dir: Path, output_dir: Path,
               cache_bytes: int, seed: int = 20261116) -> tuple[dict, dict]:
    """Verify the frozen promotion ancestry and all prepared inputs before writes."""
    if not 1 <= cache_bytes <= 2_000_000_000:
        raise ValueError("cache_bytes must be positive and at most 2 GB")
    plan = _record(campaign_dir / "protocol.json")
    summary = _record(campaign_dir / "summary.json")
    selection = _record(campaign_dir / "selection.json")
    confirmation = _record(campaign_dir / "confirmation.json")
    outer = _record(campaign_dir / "outer.json")
    middle = _record(campaign_dir / "middle.json")
    final = _record(campaign_dir / "final.json")
    protocol = plan
    protocol_hash = _digest(protocol)
    if (summary.get("status") != "complete"
            or summary.get("protocol_sha256") != protocol_hash
            or summary.get("selection") != selection
            or summary.get("candidate_eligible_for_review") is not True
            or not summary.get("middle_completed")
            or not summary.get("final_completed")):
        raise ValueError("campaign summary or protocol is not complete and eligible")
    if (selection.get("kind") != "convex_pair"
            or selection.get("frozen_before_confirmation") is not True
            or selection.get("automatic_deployment") is not False
            or selection.get("blend_weight") != .5
            or selection.get("decoder") is None
            or [member.get("id") for member in selection.get("members", [])] != list(MEMBER_IDS)):
        raise ValueError("selection is not the frozen overnight convex pair")
    if (confirmation.get("selection_sha256") != _digest(selection)
            or confirmation.get("gate", {}).get("passed") is not True
            or outer.get("gate_passed") is not True
            or outer.get("used_for_tuning") is not False):
        raise ValueError("confirmation or outer promotion gate failed")

    selected = {member["id"]: member for member in selection["members"]}
    middle_rows = {row.get("id"): row for row in middle.get("rows", [])}
    final_rows = {row.get("id"): row for row in final.get("rows", [])}
    if not middle.get("planned_stage_completed") or not final.get("planned_stage_completed"):
        raise ValueError("middle/final stage is not complete")
    for member_id in MEMBER_IDS:
        if (selected[member_id].get("steps") != 7200
                or selected[member_id].get("spec", {}).get("schedule", {}).get(
                    "total_steps") != 7200):
            raise ValueError(f"selection fit budget differs from frozen rung: {member_id}")
        spec = selected[member_id].get("spec")
        row = final_rows.get(member_id)
        mid = middle_rows.get(member_id)
        if (not isinstance(spec, dict) or not row or not mid
                or row.get("spec") != spec or mid.get("spec") != spec
                or row.get("steps") != 7200 or mid.get("steps") != 3600
                or not isinstance(row.get("trials_complete"), int)
                or row["trials_complete"] < 1
                or not isinstance(mid.get("trials_complete"), int)
                or mid["trials_complete"] < 1):
            raise ValueError(f"selected recipe differs from complete stage rows: {member_id}")

    expected_source = protocol.get("source_sha256")
    current_source = _source_hashes()
    changed = sorted(path for path, digest in expected_source.items()
                     if current_source.get(path) != digest)
    additions = set(current_source) - set(expected_source)
    forbidden = sorted(path for path in additions
                       if not (path.startswith("experiments/") and path.endswith(".py")))
    if changed or forbidden:
        raise ValueError(f"campaign source ancestry differs: changed={changed}, added={forbidden}")
    runtime = _execution_environment()
    if runtime != protocol.get("environment"):
        differences = sorted(key for key in set(runtime) | set(protocol["environment"])
                             if runtime.get(key) != protocol["environment"].get(key))
        raise ValueError(f"campaign execution environment differs: {differences}")

    paths = [data_dir / f"{match}_half{period}.npz"
             for match in FIT_MATCHES for period in (1, 2)]
    provenance = verify_prepared(data_dir, paths)
    if provenance != protocol.get("prepared"):
        raise ValueError("all-seven prepared provenance differs from campaign protocol")
    prepared_hashes = {path.name: sha256_file(path) for path in paths}
    expected_prepared_hashes = {
        cache["cache"]: cache["sha256"] for cache in protocol["prepared"]["caches"]
    }
    if prepared_hashes != expected_prepared_hashes:
        raise ValueError("prepared cache hashes differ from campaign protocol")
    artifact_hashes = {
        f"{name}.json": sha256_file(campaign_dir / f"{name}.json")
        for name in ("protocol", "summary", "selection", "confirmation", "outer", "middle", "final")
    }
    source_hashes = dict(current_source)
    source_hashes["experiments/campaign_refit.py"] = sha256_file(Path(__file__))
    provenance_payload = {
        "protocol": "overnight joint ensemble all-seven refit v1",
        "campaign_dir": str(campaign_dir), "data_dir": str(data_dir),
        "selection_matches": list(SELECTION_MATCHES), "fit_matches": list(FIT_MATCHES),
        "selection": selection,
        "campaign_artifacts_sha256": artifact_hashes,
        "prepared_sha256": prepared_hashes, "prepared_provenance": provenance,
        "source_sha256": source_hashes, "runtime": runtime,
        "seed": seed,
        "cache_bytes": cache_bytes,
    }
    recipe_payload = {
        "schema": "campaign_ensemble_recipe_v1", "selection": selection,
        "protocol_sha256": protocol_hash, "campaign_artifacts_sha256": artifact_hashes,
        "selection_matches": list(SELECTION_MATCHES), "fit_matches": list(FIT_MATCHES),
        "seed": provenance_payload["seed"], "prepared_sha256": prepared_hashes,
        "source_sha256": source_hashes, "runtime": runtime,
    }
    return provenance_payload, recipe_payload


@contextmanager
def _lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".refit.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("campaign refit already has a writer") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _fit_member(member: dict, output_dir: Path, data_dir: Path, identity: dict,
                cache: HalfCache, seed: int) -> dict:
    member_id = member["id"]
    directory = output_dir / "members" / member_id
    directory.mkdir(parents=True, exist_ok=True)
    spec, steps = member["spec"], int(member["steps"])
    state_paths = [directory / f"resume-{index}.pt" for index in (0, 1)]
    checkpoint_path = directory / "model.pt"
    fit_path = directory / "fit.json"
    if fit_path.exists():
        fit = _record(fit_path)
        if (fit.get("member_id") != member_id or fit.get("identity_sha256") != _digest(identity)
                or fit.get("checkpoint_sha256") != sha256_file(checkpoint_path)):
            raise ValueError(f"completed member provenance differs: {member_id}")
        payload = _verify_checkpoint(checkpoint_path, spec, seed, identity)
        if fit.get("training_config_sha256") != _digest(payload["training_config"]):
            raise ValueError(f"completed member training signature differs: {member_id}")
        return fit

    session = build_session(data_dir, spec, list(FIT_MATCHES), seed, cache=cache)
    existing_states = [path for path in state_paths if path.exists()]
    if existing_states:
        valid = False
        last_error = None
        for state_path in sorted(existing_states, key=lambda path: path.stat().st_mtime,
                                 reverse=True):
            try:
                load_training_state(state_path, session, {"member_id": member_id, **identity})
                valid = True
                break
            except (ValueError, TypeError, OSError, EOFError, RuntimeError,
                    KeyError, pickle.UnpicklingError) as exc:
                last_error = exc
                continue
        if not valid:
            raise ValueError(f"no valid resumable checkpoint remains for {member_id}") from last_error
    while session.steps < steps:
        next_step = min(steps, ((session.steps // CHECKPOINT_INTERVAL) + 1) * CHECKPOINT_INTERVAL)
        session.advance_to(next_step)
        state_path = state_paths[(session.steps // CHECKPOINT_INTERVAL) % 2]
        save_training_state(state_path, session, {"member_id": member_id, **identity})
        write_json(output_dir / "status.json", {
            "status": "running", "member_id": member_id, "steps": session.steps,
            "target_steps": steps, "fit_matches": list(FIT_MATCHES),
        })
    session.save_checkpoint(checkpoint_path, extra_training_config={
        "campaign_member_id": member_id, "campaign_identity_sha256": _digest(identity),
        "campaign_recipe_sha256": _digest(spec), "campaign_seed": seed,
        "fit_matches": list(FIT_MATCHES), "selection_matches": list(SELECTION_MATCHES),
        "selection_decoder": identity["selection"]["decoder"],
        "selection_blend_kind": identity["selection"]["kind"],
        "selection_blend_weight": identity["selection"]["blend_weight"],
    })
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    fit = {
        "schema": "campaign_member_fit_v1", "member_id": member_id,
        "identity_sha256": _digest(identity), "recipe_sha256": _digest(spec),
        "seed": seed, "steps": session.steps,
        "selection_matches": list(SELECTION_MATCHES), "fit_matches": list(FIT_MATCHES),
        "prepared_sha256": identity["prepared_sha256"],
        "training_config_sha256": _digest(payload["training_config"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    _write_record(fit_path, fit)
    return fit


def _verify_checkpoint(path: Path, spec: dict, seed: int, identity: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload.get("training_config", {})
    expected_model = {"player_dim": 8, "global_dim": 9,
                      "dilations": [1, 2, 4, 8, 16, 32, 64, 128], **spec["model_kwargs"]}
    expected_args = spec["training_args"]
    for key, expected in expected_args.items():
        if config.get(key) != expected:
            raise ValueError(f"checkpoint training config differs for {key}")
    if (payload.get("model_kwargs") != expected_model
            or config.get("steps") != spec["schedule"]["total_steps"]
            or config.get("train_matches") != list(FIT_MATCHES)
            or config.get("validation_matches") != [] or config.get("seed") != seed
            or config.get("campaign_spec") != spec
            or config.get("campaign_recipe_sha256") != _digest(spec)
            or config.get("campaign_identity_sha256") != _digest(identity)
            or config.get("campaign_member_id") not in MEMBER_IDS
            or payload.get("prepared_sha256") != identity["prepared_sha256"]):
        raise ValueError("campaign member checkpoint provenance differs")
    return payload


def _manifest_payload(output_dir: Path, identity: dict, recipe: dict, fits: dict) -> dict:
    members = {}
    for member_id, fit in fits.items():
        base = Path("members") / member_id
        checkpoint = output_dir / base / "model.pt"
        fit_path = output_dir / base / "fit.json"
        payload = _verify_checkpoint(checkpoint, next(
            member["spec"] for member in recipe["selection"]["members"]
            if member["id"] == member_id), identity["seed"], identity)
        members[member_id] = {
            "checkpoint": str(base / "model.pt"), "sha256": sha256_file(checkpoint),
            "fit_record": str(base / "fit.json"), "fit_record_sha256": sha256_file(fit_path),
            "model_kwargs": payload["model_kwargs"],
            "training_config_sha256": _digest(payload["training_config"]),
            "seed": identity["seed"], "steps": fit["steps"],
        }
    portable_files = ["plan.json", "recipe.json"]
    portable_files += [f"members/{mid}/{name}" for mid in MEMBER_IDS
                       for name in ("model.pt", "fit.json")]
    return {
        "schema": "campaign_ensemble_bundle_v1", "files": {
            name: sha256_file(output_dir / name) for name in portable_files
        }, "members": members, "selection_matches": list(SELECTION_MATCHES),
        "fit_matches": list(FIT_MATCHES), "selection": recipe["selection"],
        "decoder": recipe["selection"]["decoder"],
        "blend_kind": recipe["selection"]["kind"],
        "blend_weight": recipe["selection"]["blend_weight"],
        "seed": identity["seed"], "steps": next(iter(fits.values()))["steps"],
        "source_sha256": identity["source_sha256"], "runtime": identity["runtime"],
        "provenance_sha256": _digest(identity),
    }


def verify_bundle(bundle_dir) -> dict:
    """Verify a portable refit bundle without access to campaign or prepared data."""
    bundle_dir = Path(bundle_dir).resolve()
    manifest_record = _record(bundle_dir / "manifest.json")
    recipe_record = _record(bundle_dir / "recipe.json")
    plan_record = _record(bundle_dir / "plan.json")
    manifest, recipe, plan = manifest_record, recipe_record, plan_record
    if (manifest.get("schema") != "campaign_ensemble_bundle_v1"
            or recipe.get("schema") != "campaign_ensemble_recipe_v1"
            or manifest.get("selection") != recipe.get("selection")
            or manifest.get("provenance_sha256") != _digest(plan)
            or recipe.get("source_sha256") != manifest.get("source_sha256")
            or recipe.get("runtime") != manifest.get("runtime")
            or manifest.get("decoder") != recipe["selection"].get("decoder")
            or manifest.get("blend_kind") != recipe["selection"].get("kind")
            or manifest.get("blend_weight") != recipe["selection"].get("blend_weight")):
        raise ValueError("bundle recipe, plan, and manifest ancestry differs")
    source_now = _source_hashes()
    mismatched = sorted(path for path, digest in manifest["source_sha256"].items()
                        if source_now.get(path) != digest)
    if mismatched:
        raise ValueError(f"bundle source code differs: {mismatched}")
    if _execution_environment() != manifest.get("runtime"):
        raise ValueError("bundle execution environment differs")
    if (manifest.get("selection_matches") != list(SELECTION_MATCHES)
            or manifest.get("fit_matches") != list(FIT_MATCHES)):
        raise ValueError("bundle match ancestry differs")
    for name, digest in manifest.get("files", {}).items():
        path = (bundle_dir / name).resolve()
        if bundle_dir not in path.parents:
            raise ValueError("bundle file path escapes bundle directory")
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"bundle file hash differs: {name}")
    member_paths = {}
    for member_id in MEMBER_IDS:
        row = manifest["members"].get(member_id)
        if not isinstance(row, dict):
            raise TypeError(f"bundle member missing: {member_id}")
        checkpoint = (bundle_dir / row["checkpoint"]).resolve()
        fit_path = (bundle_dir / row["fit_record"]).resolve()
        if bundle_dir not in checkpoint.parents or bundle_dir not in fit_path.parents:
            raise ValueError("bundle member path escapes bundle directory")
        if sha256_file(checkpoint) != row["sha256"] or sha256_file(fit_path) != row["fit_record_sha256"]:
            raise ValueError(f"bundle member hash differs: {member_id}")
        fit = _record(fit_path)
        if (fit.get("member_id") != member_id or fit.get("checkpoint_sha256") != row["sha256"]
                or fit.get("training_config_sha256") != row["training_config_sha256"]
                or fit.get("identity_sha256") != _digest(plan)
                or fit.get("prepared_sha256") != plan.get("prepared_sha256")):
            raise ValueError(f"bundle fit record differs: {member_id}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        selected_member = next((member for member in recipe["selection"]["members"]
                                if member.get("id") == member_id), None)
        if selected_member is None:
            raise ValueError(f"bundle recipe member missing: {member_id}")
        expected_kwargs = {"player_dim": 8, "global_dim": 9,
                           "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                           **selected_member["spec"]["model_kwargs"]}
        config = payload.get("training_config", {})
        if (payload.get("model_kwargs") != row["model_kwargs"]
                or payload.get("model_kwargs") != expected_kwargs
                or _digest(config) != row["training_config_sha256"]
                or config.get("train_matches") != list(FIT_MATCHES)
                or config.get("selection_matches") != list(SELECTION_MATCHES)
                or config.get("campaign_member_id") != member_id
                or config.get("campaign_spec") != selected_member["spec"]
                or config.get("steps") != manifest.get("steps")
                or config.get("seed") != manifest.get("seed")
                or config.get("campaign_recipe_sha256") != _digest(selected_member["spec"])
                or config.get("campaign_identity_sha256") != _digest(plan)
                or payload.get("prepared_sha256") != plan.get("prepared_sha256")
                or config.get("selection_decoder") != manifest.get("decoder")
                or config.get("selection_blend_kind") != manifest.get("blend_kind")
                or config.get("selection_blend_weight") != manifest.get("blend_weight")):
            raise ValueError(f"bundle checkpoint metadata differs: {member_id}")
        member_paths[member_id] = str(checkpoint)
    return {"manifest": manifest, "recipe": recipe, "member_paths": member_paths,
            "bundle_dir": str(bundle_dir)}


def run_refit(campaign_dir, data_dir, output_dir, *, seed=20261116,
              cache_bytes=1_000_000_000) -> dict:
    """Resumably refit both frozen ensemble members and publish the bundle."""
    campaign_dir, data_dir, output_dir = (Path(path).resolve()
                                          for path in (campaign_dir, data_dir, output_dir))
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if (_inside(campaign_dir, output_dir) or _inside(output_dir, campaign_dir)
            or _inside(data_dir, output_dir) or _inside(output_dir, data_dir)):
        raise ValueError("output_dir must be separate from campaign and prepared data")
    identity, recipe = _preflight(campaign_dir, data_dir, output_dir, cache_bytes, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    with _lock(output_dir):
        _write_record(output_dir / "plan.json", identity)
        _write_record(output_dir / "recipe.json", recipe)
        completed = output_dir / "manifest.json"
        if completed.exists():
            verified = verify_bundle(output_dir)
            return {"bundle_dir": str(output_dir), "manifest_path": str(completed),
                    "status": "bundle_reused", "manifest": verified["manifest"]}
        cache = HalfCache(data_dir, max_bytes=cache_bytes)
        fits = {}
        for member in recipe["selection"]["members"]:
            fits[member["id"]] = _fit_member(
                member, output_dir, data_dir, identity, cache, seed)
        manifest = _manifest_payload(output_dir, identity, recipe, fits)
        _write_record(completed, manifest)
        write_json(output_dir / "status.json", {
            "status": "complete", "bundle_dir": str(output_dir),
            "members": list(MEMBER_IDS), "fit_matches": list(FIT_MATCHES),
        })
        verify_bundle(output_dir)
        return {"bundle_dir": str(output_dir), "manifest_path": str(completed),
                "status": "bundle_written", "manifest": manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--cache-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    result = run_refit(args.campaign_dir, args.data_dir, args.output_dir,
                       seed=args.seed, cache_bytes=args.cache_bytes)
    print(json.dumps({"bundle_dir": result["bundle_dir"],
                      "manifest_path": result["manifest_path"],
                      "status": result["status"], "members": list(MEMBER_IDS),
                      "selection_matches": list(SELECTION_MATCHES),
                      "fit_matches": list(FIT_MATCHES)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
