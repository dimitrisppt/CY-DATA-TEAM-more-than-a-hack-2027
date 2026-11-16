"""Verify frozen selection artifacts and export a self-contained inference bundle."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from possession.provenance import runtime_provenance
from possession.storage import sha256_file, write_json

_SOURCE_ROOT = Path(__file__).resolve().parent.parent
_COMMON_INFERENCE_SOURCES = (
    "games.py", "possession/_viterbi.py", "possession/calibrate.py", "possession/cli.py",
    "possession/decode.py", "possession/deployment.py", "possession/ensemble.py",
    "possession/features.py", "possession/io_atd.py", "possession/recipe.py",
    "possession/reference.py", "possession/schema.py", "possession/storage.py",
)
_NEURAL_INFERENCE_SOURCES = ("possession/model.py", "possession/predict.py")
_FEATURE_SCHEMA = {"schema_version": 1, "frame": "FrameTable", "features": "FeatureBatch",
                   "player_dim": 8, "global_dim": 9}


def _inference_source_hashes(*, neural: bool) -> dict[str, str]:
    names = _COMMON_INFERENCE_SOURCES + (_NEURAL_INFERENCE_SOURCES if neural else ())
    return {name: sha256_file(_SOURCE_ROOT / name) for name in names}


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _expected_model_kwargs(recipe: dict) -> dict:
    kwargs = {"player_dim": 8, "global_dim": 9, "width": 64,
              "dilations": [1, 2, 4, 8, 16, 32, 64, 128]}
    kwargs.update(recipe["model_kwargs"])
    kwargs["dilations"] = list(kwargs["dilations"])
    return kwargs


def _check_manifest(directory: Path, *, required: set[str]) -> dict | None:
    path = directory / "deployment_manifest.json"
    if not path.exists():
        return None
    manifest = _json(path)
    hashes = manifest.get("artifacts")
    if manifest.get("schema_version") != 1 or not isinstance(hashes, dict) \
            or set(hashes) != required:
        raise ValueError("deployment manifest artifact set differs")
    for name, expected in hashes.items():
        if not isinstance(expected, str) or sha256_file(directory / name) != expected:
            raise ValueError(f"deployment artifact hash differs: {name}")
    neural = "selected.pt" in required
    if manifest.get("feature_schema") != _FEATURE_SCHEMA:
        raise ValueError("deployment feature schema differs")
    if manifest.get("inference_source_sha256") != _inference_source_hashes(neural=neural):
        raise ValueError("deployment inference source hash differs")
    return manifest


def _integrity(recipe_path: Path, checkpoint_sha: str | None, predictor: str,
               manifest: dict | None) -> dict:
    return {
        "selection_verified": True,
        "predictor": predictor,
        "checkpoint_sha256": checkpoint_sha,
        "frozen_recipe_sha256": sha256_file(recipe_path),
        "refit_progress_sha256": sha256_file(recipe_path.parent / "refit_progress.json"),
        "deployment_manifest_sha256": (
            sha256_file(recipe_path.parent / "deployment_manifest.json")
            if manifest is not None else None),
    }


def verify_selected_checkpoint(recipe_path, frozen, checkpoint, payload=None) -> dict:
    """Validate refit artifact hashes and ancestry before using checkpoint weights.

    The progress file or deployment manifest supplies the trusted checkpoint hash.
    Reference-only recipes still require a matching progress file. A manifest, if
    present, is always checked against every artifact in its bundle.
    """
    recipe_path = Path(recipe_path)
    if _json(recipe_path) != frozen:
        raise ValueError("frozen recipe content differs from recipe file")
    recipe = frozen.get("selected_model_recipe")
    neural = recipe is not None
    weight = frozen.get("ensemble", {}).get("neural_weight")
    if not isinstance(weight, (int, float)) or isinstance(weight, bool) \
            or not math.isfinite(weight) or not 0 <= weight <= 1 \
            or (neural and weight == 0) or (not neural and weight != 0):
        raise ValueError("frozen neural_weight differs from selected model recipe")
    required = {"frozen_recipe.json", "refit_progress.json"}
    if neural:
        required.add("selected.pt")
    manifest = _check_manifest(recipe_path.parent, required=required)
    if manifest is not None and sha256_file(recipe_path) != manifest["artifacts"]["frozen_recipe.json"]:
        raise ValueError("selected recipe hash differs from deployment manifest")
    progress = _json(recipe_path.parent / "refit_progress.json")
    if manifest is not None and (
        manifest.get("selection_fingerprint") != frozen.get("selection_fingerprint")
        or manifest.get("predictor") != ("selected_neural" if neural else "reference")
    ):
        raise ValueError("deployment manifest selection differs from frozen recipe")
    if progress.get("train_matches") != frozen.get("outer_train"):
        raise ValueError("refit progress train matches differ from frozen recipe")
    if not neural:
        if checkpoint is not None or (recipe_path.parent / "selected.pt").exists():
            raise ValueError("reference selection unexpectedly has a neural checkpoint")
        if progress != {"predictor": "reference", "steps": 0,
                        "train_matches": frozen["outer_train"]}:
            raise ValueError("reference refit progress differs from frozen recipe")
        return _integrity(recipe_path, None, "reference", manifest)
    if checkpoint is None:
        raise ValueError("selected neural recipe requires a checkpoint")
    checkpoint = Path(checkpoint)
    if checkpoint.resolve() != (recipe_path.parent / "selected.pt").resolve():
        raise ValueError("selected checkpoint must accompany frozen recipe")
    checkpoint_sha = sha256_file(checkpoint)
    if progress.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("selected checkpoint hash differs from refit progress")
    if progress.get("predictor") != "selected_neural" \
            or progress.get("steps") != frozen["resources"]["full_budget_steps"]:
        raise ValueError("refit progress differs from frozen recipe")
    if payload is None:
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("selected checkpoint payload is invalid")
    config = payload.get("training_config")
    if not isinstance(config, dict):
        raise TypeError("selected checkpoint training configuration is invalid")
    expected = {
        "train_matches": frozen["outer_train"],
        "validation_matches": [],
        "outer_test": frozen["outer_test"],
        "selection_fingerprint": frozen["selection_fingerprint"],
        "selection_matches": frozen["outer_train"],
        "decoder": frozen["decoder"],
        "calibration": frozen["calibration"],
        "ensemble": frozen["ensemble"],
        "steps": frozen["resources"]["full_budget_steps"],
        "seed": frozen["refit_seed"],
    }
    args = recipe["training_args"]
    for key in ("center_size", "learning_rate", "weight_decay", "boundary_weight",
                "boundary_pos_weight", "augment"):
        if key in args:
            expected[key] = args[key]
    if "outage_config" in args:
        from possession.outages import OutageConfig
        expected["outage_config"] = OutageConfig(**args["outage_config"]).to_dict()
    if "augmentation_config" in args:
        from possession.augmentation import normalize_augmentation
        expected["augmentation_config"] = normalize_augmentation(args["augmentation_config"]).to_dict()
    if any(config.get(key) != value for key, value in expected.items()) \
            or payload.get("model_kwargs") != _expected_model_kwargs(recipe):
        raise ValueError("selected checkpoint ancestry differs from frozen recipe")
    model_kwargs = payload.get("model_kwargs")
    if (model_kwargs["player_dim"] != _FEATURE_SCHEMA["player_dim"]
            or model_kwargs["global_dim"] != _FEATURE_SCHEMA["global_dim"]):
        raise ValueError("selected checkpoint feature dimensions differ from inference schema")
    return _integrity(recipe_path, checkpoint_sha, "selected_neural", manifest)


def export_bundle(study_dir, output_dir) -> dict:
    """Atomically publish a verified, portable frozen inference bundle."""
    study_dir, output_dir = Path(study_dir).resolve(), Path(output_dir).resolve()
    if study_dir == output_dir or study_dir.is_relative_to(output_dir):
        raise ValueError("bundle output must not overwrite study inputs")
    recipe_path = study_dir / "frozen_recipe.json"
    frozen = _json(recipe_path)
    neural = frozen.get("selected_model_recipe") is not None
    checkpoint = study_dir / "selected.pt" if neural else None
    verify_selected_checkpoint(recipe_path, frozen, checkpoint)
    names = ["frozen_recipe.json", "refit_progress.json"]
    if neural:
        names.append("selected.pt")
    hashes = {name: sha256_file(study_dir / name) for name in names}
    manifest = {
        "schema_version": 1,
        "predictor": "selected_neural" if neural else "reference",
        "selection_fingerprint": frozen["selection_fingerprint"],
        "selection_score": frozen.get("selection_score"),
        "selection_score_type": frozen.get("selection_score_type"),
        "evaluation_performed": False,
        "artifacts": hashes,
        "feature_schema": _FEATURE_SCHEMA,
        "inference_source_sha256": _inference_source_hashes(neural=neural),
        "runtime_at_export": runtime_provenance(),
    }
    if output_dir.exists():
        existing = _json(output_dir / "deployment_manifest.json")
        existing_identity = {key: value for key, value in existing.items()
                             if key != "runtime_at_export"}
        current_identity = {key: value for key, value in manifest.items()
                            if key != "runtime_at_export"}
        if existing_identity != current_identity \
                or {path.name for path in output_dir.iterdir()} \
                != set(names) | {"deployment_manifest.json"}:
            raise ValueError("existing bundle differs from verified source")
        verify_selected_checkpoint(output_dir / "frozen_recipe.json", frozen,
                                   output_dir / "selected.pt" if neural else None)
        return {"bundle_dir": output_dir, "manifest_path": output_dir / "deployment_manifest.json",
                "manifest": existing, "reused": True}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for name in names:
            shutil.copyfile(study_dir / name, temporary / name)
            if sha256_file(temporary / name) != hashes[name]:
                raise ValueError(f"copied deployment artifact changed: {name}")
        write_json(temporary / "deployment_manifest.json", manifest)
        verify_selected_checkpoint(temporary / "frozen_recipe.json", frozen,
                                   temporary / "selected.pt" if neural else None)
        if output_dir.exists():
            raise FileExistsError(output_dir)
        os.rename(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"bundle_dir": output_dir, "manifest_path": output_dir / "deployment_manifest.json",
            "manifest": manifest, "reused": False}
