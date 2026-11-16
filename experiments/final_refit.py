"""Refit the frozen focused-study baseline on all seven DFL matches."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from experiments import frozen_transfer
from possession.deployment import export_bundle, verify_selected_checkpoint
from possession.outages import OutageConfig
from possession.prepared import verify_prepared
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache, TrainingSession

FINAL_ID = "__final__"
SEED = 30
BASELINE_STEPS = 3600
EXPECTED_SELECTION_MATCHES = (
    "DFL-MAT-J03WN1", "DFL-MAT-J03WOH", "DFL-MAT-J03WOY",
    "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9",
)
EXPECTED_REFIT_MATCHES = ("DFL-MAT-J03WMX", *EXPECTED_SELECTION_MATCHES)
FOCUSED_ARTIFACTS = tuple(
    f"{name}.{suffix}"
    for name in ("protocol", "summary", "selection", "confirmation", "baseline-recipe",
                 "selected-recipe")
    for suffix in ("json", "sha256.json")
) + ("trials/trial_000/trial.json", "trials/trial_000/trial.sha256.json")


def _digest(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _deployment_recipe(baseline: dict, *, selection_matches: list[str],
                       refit_matches: list[str], seed: int, fingerprint: str) -> dict:
    """Wrap the six-match frozen selection in the deployment verifier's schema."""
    args = dict(baseline["training_args"])
    args.update({
        "steps": baseline["steps"],
        "outage_config": OutageConfig(**baseline["outage_config"]).to_dict(),
        "augmentation_config": baseline["augmentation_config"],
    })
    model_recipe = {"model_kwargs": baseline["model_kwargs"], "training_args": args}
    return {
        "protocol": "frozen focused baseline refit on all seven public DFL matches v1",
        "scope": "all_training_matches",
        "evaluation_performed": False,
        "selection_score_role": "six-match focused development OOF selection only",
        "selection_score_type": "clean plus two short-gap OOF objective",
        "selection_scope_note": (
            "No new selection. source_selection_matches records the six development "
            "matches used for the frozen recipe. outer_train and the checkpoint's legacy "
            "selection_matches field identify the seven-match deployment fit cohort."
        ),
        "selection_fingerprint": fingerprint,
        "source_selection_matches": list(selection_matches),
        "refit_matches": list(refit_matches),
        "outer_train": list(refit_matches),
        "outer_test": FINAL_ID,
        "selected_model_recipe": model_recipe,
        "selected_pooling": baseline["model_kwargs"]["pooling"],
        "decoder": baseline["decoder"],
        "calibration": baseline["calibration"],
        "ensemble": baseline["ensemble"],
        "resources": {"full_budget_steps": baseline["steps"]},
        "refit_seed": seed,
    }


@contextmanager
def _locked(study_root: Path):
    study_root.mkdir(parents=True, exist_ok=True)
    with (study_root / "final_refit.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("final refit already has a writer") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _preflight(focused_dir: Path, data_dir: Path, output_dir: Path) -> tuple[dict, dict]:
    plan, _summary, _selection, _confirmation, baseline, _selected = frozen_transfer._records(
        focused_dir)
    source_hashes, environment = frozen_transfer._runtime_ancestry(plan)
    fold = plan["payload"]["fold"]
    selection_matches = fold["outer_train"]
    if (selection_matches != list(EXPECTED_SELECTION_MATCHES)
            or fold.get("outer_test") != ["DFL-MAT-J03WMX"]
            or list(EXPECTED_REFIT_MATCHES) != sorted(EXPECTED_REFIT_MATCHES)):
        raise ValueError("focused baseline match ancestry differs")
    if (baseline.get("development_only") is not True
            or baseline.get("steps") != plan["payload"].get("selected_rung")
            or baseline.get("steps") != BASELINE_STEPS
            or baseline.get("model_kwargs", {}).get("pooling") != "mean"
            or baseline.get("augmentation_config", {}).get("mode") != "none"):
        raise ValueError("frozen baseline is not the verified 3,600-step control")

    paths = [data_dir / f"{match}_half{period}.npz"
             for match in EXPECTED_REFIT_MATCHES for period in (1, 2)]
    prepared_provenance = verify_prepared(data_dir, paths)
    prepared_hashes = {path.name: sha256_file(path) for path in paths}
    expected_hashes = plan["payload"].get("prepared_sha256", {})
    if any(expected_hashes.get(name) != digest for name, digest in prepared_hashes.items()):
        raise ValueError("one or more all-seven prepared caches differ from focused provenance")

    artifact_hashes = {}
    for name in FOCUSED_ARTIFACTS:
        path = focused_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact_hashes[name] = sha256_file(path)
    source_hashes = dict(source_hashes)
    source_hashes["experiments/final_refit.py"] = sha256_file(Path(__file__))
    payload = {
        "protocol": "frozen focused baseline all-match refit v1",
        "focused_protocol_sha256": plan["sha256"],
        "focused_artifacts": artifact_hashes,
        "source_sha256": source_hashes,
        "runtime": environment,
        "prepared_sha256": prepared_hashes,
        "prepared_provenance": prepared_provenance,
        "selection_matches": list(selection_matches),
        "refit_matches": list(EXPECTED_REFIT_MATCHES),
        "seed": SEED,
        "steps": baseline["steps"],
        "baseline_recipe": baseline,
        "data_dir": str(data_dir.resolve()),
        "focused_dir": str(focused_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    final_plan = {"schema_version": 1, "payload": payload, "sha256": _digest(payload)}
    frozen = _deployment_recipe(
        baseline, selection_matches=selection_matches,
        refit_matches=list(EXPECTED_REFIT_MATCHES), seed=SEED,
        fingerprint=final_plan["sha256"],
    )
    return final_plan, frozen


def _check_completed(study_dir: Path, output_dir: Path, plan: dict, frozen: dict) -> dict | None:
    plan_path = study_dir / "finalization_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("final refit plan, focused source, or prepared cache changed")
    checkpoint = study_dir / "selected.pt"
    progress_path = study_dir / "refit_progress.json"
    recipe_path = study_dir / "frozen_recipe.json"
    present = [path.exists() for path in (checkpoint, progress_path, recipe_path)]
    if any(present) and not all(present):
        raise ValueError("partial final refit artifacts require manual inspection")
    if not all(present):
        if output_dir.exists():
            raise ValueError("deployment bundle exists without complete refit artifacts")
        return None
    if json.loads(recipe_path.read_text()) != frozen:
        raise ValueError("completed frozen recipe differs from final refit plan")
    verified = verify_selected_checkpoint(recipe_path, frozen, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = payload.get("training_config", {})
    if (payload.get("prepared_sha256") != plan["payload"]["prepared_sha256"]
            or config.get("source_selection_matches") != list(EXPECTED_SELECTION_MATCHES)
            or config.get("refit_matches") != list(EXPECTED_REFIT_MATCHES)):
        raise ValueError("checkpoint data or selection ancestry differs from final refit plan")
    progress = json.loads(progress_path.read_text())
    if (progress.get("steps") != plan["payload"]["steps"]
            or progress.get("train_matches") != list(EXPECTED_REFIT_MATCHES)
            or progress.get("source_selection_matches") != list(EXPECTED_SELECTION_MATCHES)
            or progress.get("checkpoint_sha256") != sha256_file(checkpoint)):
        raise ValueError("completed refit progress differs from plan or checkpoint")
    manifest = export_bundle(study_dir, output_dir)
    return {"bundle_dir": str(manifest["bundle_dir"]),
            "manifest_path": str(manifest["manifest_path"]),
            "bundle_reused": manifest["reused"],
            "selection_verified": verified["selection_verified"],
            "steps": progress["steps"]}


def run_final_refit(focused_dir, data_dir, output_dir) -> dict:
    """Fit the locked baseline once on all seven matches, without model selection."""
    focused_dir, data_dir, output_dir = (Path(path).resolve()
                                         for path in (focused_dir, data_dir, output_dir))
    study_root = output_dir.parent / f"{output_dir.name}.study"
    study_dir = study_root / FINAL_ID
    roots = (focused_dir, data_dir, output_dir, study_root)
    if any(left == right or left in right.parents or right in left.parents
           for index, left in enumerate(roots) for right in roots[index + 1:]):
        raise ValueError("focused data, prepared data, bundle, and refit study must be separate")

    # Fail source/cache/selection checks before creating output directories or a lock.
    plan, frozen = _preflight(focused_dir, data_dir, output_dir)
    with _locked(study_root):
        plan_path = study_dir / "finalization_plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("final refit plan, focused source, or prepared cache changed")
        if not plan_path.exists() and study_dir.exists() and any(study_dir.iterdir()):
            raise ValueError("existing refit study lacks immutable plan")
        if not plan_path.exists():
            write_json(plan_path, plan)
        completed = _check_completed(study_dir, output_dir, plan, frozen)
        if completed is not None:
            return completed

        baseline = plan["payload"]["baseline_recipe"]
        args = frozen["selected_model_recipe"]["training_args"]
        cache = HalfCache(data_dir, max_bytes=2_000_000_000)
        session = TrainingSession(
            data_dir, train_matches=list(EXPECTED_REFIT_MATCHES), seed=SEED,
            device="cpu", model_kwargs=baseline["model_kwargs"],
            learning_rate=args["learning_rate"], weight_decay=args["weight_decay"],
            boundary_weight=args["boundary_weight"],
            boundary_pos_weight=args["boundary_pos_weight"],
            augment=args["augment"], center_size=args["center_size"], cache=cache,
            outage_config=baseline["outage_config"],
            augmentation_config=baseline["augmentation_config"],
        )
        progress = session.advance_to(baseline["steps"])
        session.save_checkpoint(
            study_dir / "selected.pt",
            extra_training_config={
                "outer_test": FINAL_ID,
                # The existing bundle verifier uses this legacy key for the fit cohort.
                "selection_matches": list(EXPECTED_REFIT_MATCHES),
                "source_selection_matches": list(EXPECTED_SELECTION_MATCHES),
                "refit_matches": list(EXPECTED_REFIT_MATCHES),
                "selection_fingerprint": frozen["selection_fingerprint"],
                "decoder": frozen["decoder"], "calibration": frozen["calibration"],
                "ensemble": frozen["ensemble"],
            },
        )
        checkpoint = study_dir / "selected.pt"
        progress = {**progress, "predictor": "selected_neural",
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "train_matches": list(EXPECTED_REFIT_MATCHES),
                    "source_selection_matches": list(EXPECTED_SELECTION_MATCHES)}
        write_json(study_dir / "frozen_recipe.json", frozen)
        write_json(study_dir / "refit_progress.json", progress)
        verify_selected_checkpoint(study_dir / "frozen_recipe.json", frozen, checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = payload.get("training_config", {})
        if (payload.get("prepared_sha256") != plan["payload"]["prepared_sha256"]
                or config.get("source_selection_matches") != list(EXPECTED_SELECTION_MATCHES)
                or config.get("refit_matches") != list(EXPECTED_REFIT_MATCHES)):
            raise ValueError("checkpoint data or selection ancestry differs from final refit plan")
        exported = export_bundle(study_dir, output_dir)
        return {"protocol": frozen["protocol"], "evaluation_performed": False,
                "selection_matches": list(EXPECTED_SELECTION_MATCHES),
                "refit_matches": list(EXPECTED_REFIT_MATCHES),
                "seed": SEED, "steps": baseline["steps"],
                "bundle_dir": str(exported["bundle_dir"]),
                "manifest_path": str(exported["manifest_path"]),
                "bundle_reused": exported["reused"],
                "selection_verified": True,
                "finalization_plan_sha256": plan["sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focused-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="deployment bundle directory")
    args = parser.parse_args(argv)
    report = run_final_refit(args.focused_dir, args.data_dir, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
