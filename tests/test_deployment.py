import json

import pytest
import torch

from possession.deployment import export_bundle, verify_selected_checkpoint
from possession.storage import sha256_file, write_json


def _artifacts(directory, *, neural=True, outage=True, augmentation=None):
    directory.mkdir()
    recipe = {
        "selection_fingerprint": "selected-fingerprint",
        "outer_train": ["a", "b"], "outer_test": "__final__",
        "refit_seed": 53, "resources": {"full_budget_steps": 4},
        "decoder": {"switch_cost": 2.5, "boundary_scale": 0.0},
        "calibration": {"live_temperature": 1.2, "owner_temperature": 0.9},
        "ensemble": {"neural_weight": 1.0 if neural else 0.0},
        "selection_score": 0.7,
        "selection_score_type": "inner OOF official unweighted match mean",
        "selected_model_recipe": None,
    }
    if neural:
        args = {"center_size": 4}
        if outage:
            args["outage_config"] = {"mode": "correlated"}
        if augmentation is not None:
            args["augmentation_config"] = augmentation
        recipe["selected_model_recipe"] = {
            "model_kwargs": {"width": 16, "dilations": [1, 2]},
            "training_args": args,
        }
    write_json(directory / "frozen_recipe.json", recipe)
    if neural:
        config = {
            "train_matches": recipe["outer_train"], "validation_matches": [],
            "outer_test": recipe["outer_test"],
            "selection_fingerprint": recipe["selection_fingerprint"],
            "selection_matches": recipe["outer_train"],
            "decoder": recipe["decoder"], "calibration": recipe["calibration"],
            "ensemble": recipe["ensemble"], "steps": 4, "seed": 53,
            "center_size": 4,
        }
        if outage:
            from possession.outages import OutageConfig
            config["outage_config"] = OutageConfig(mode="correlated").to_dict()
        if augmentation is not None:
            from possession.augmentation import normalize_augmentation
            config["augmentation_config"] = normalize_augmentation(augmentation).to_dict()
        payload = {
            "state_dict": {"weight": torch.tensor([1.0])},
            "training_config": config,
            "model_kwargs": {"player_dim": 8, "global_dim": 9,
                             "width": 16, "dilations": [1, 2]},
        }
        torch.save(payload, directory / "selected.pt")
        progress = {"predictor": "selected_neural", "steps": 4,
                    "train_matches": recipe["outer_train"],
                    "checkpoint_sha256": sha256_file(directory / "selected.pt")}
    else:
        progress = {"predictor": "reference", "steps": 0,
                    "train_matches": recipe["outer_train"]}
    write_json(directory / "refit_progress.json", progress)
    return recipe


def _verify(directory, recipe):
    checkpoint = directory / "selected.pt" if recipe["selected_model_recipe"] else None
    return verify_selected_checkpoint(directory / "frozen_recipe.json", recipe, checkpoint)


def test_mutated_weights_with_original_metadata_rejected(tmp_path):
    directory = tmp_path / "study"
    recipe = _artifacts(directory)
    assert _verify(directory, recipe)["checkpoint_sha256"] == sha256_file(directory / "selected.pt")
    payload = torch.load(directory / "selected.pt", weights_only=True)
    payload["state_dict"]["weight"] += 1
    torch.save(payload, directory / "selected.pt")
    with pytest.raises(ValueError, match="checkpoint hash"):
        _verify(directory, recipe)


@pytest.mark.parametrize("change", ["matches", "steps", "seed", "model", "decoder", "outage", "center_size"])
def test_changed_ancestry_rejected_even_if_progress_hash_updated(tmp_path, change):
    directory = tmp_path / "study"
    recipe = _artifacts(directory)
    payload = torch.load(directory / "selected.pt", weights_only=True)
    config = payload["training_config"]
    if change == "matches":
        config["train_matches"] = ["a"]
    elif change == "steps":
        config["steps"] = 5
    elif change == "seed":
        config["seed"] = 54
    elif change == "model":
        payload["model_kwargs"]["player_dim"] = 7
    elif change == "decoder":
        config["decoder"]["switch_cost"] = 9
    elif change == "center_size":
        config["center_size"] = 5
    else:
        config["outage_config"]["mode"] = "legacy"
    torch.save(payload, directory / "selected.pt")
    progress_path = directory / "refit_progress.json"
    progress = json.loads(progress_path.read_text())
    progress["checkpoint_sha256"] = sha256_file(directory / "selected.pt")
    write_json(progress_path, progress)
    with pytest.raises(ValueError, match="ancestry"):
        _verify(directory, recipe)


def test_legacy_recipe_without_outage_field_verifies(tmp_path):
    directory = tmp_path / "study"
    recipe = _artifacts(directory, outage=False)
    _verify(directory, recipe)


def test_augmentation_ancestry_rejects_changed_policy_with_updated_hash(tmp_path):
    directory = tmp_path / "study"
    recipe = _artifacts(directory, augmentation={"mode": "combined", "strength": .5})
    assert _verify(directory, recipe)["selection_verified"] is True
    checkpoint = directory / "selected.pt"
    payload = torch.load(checkpoint, weights_only=True)
    payload["training_config"]["augmentation_config"]["time_warp_max_rate"] = .2
    torch.save(payload, checkpoint)
    progress_path = directory / "refit_progress.json"
    progress = json.loads(progress_path.read_text())
    progress["checkpoint_sha256"] = sha256_file(checkpoint)
    write_json(progress_path, progress)
    with pytest.raises(ValueError, match="ancestry"):
        _verify(directory, recipe)


def test_reference_bundle_and_idempotent_export(tmp_path):
    study, bundle = tmp_path / "study", tmp_path / "bundle"
    recipe = _artifacts(study, neural=False)
    first = export_bundle(study, bundle)
    assert not first["reused"]
    assert first["manifest"]["evaluation_performed"] is False
    assert first["manifest"]["selection_score_type"] == recipe["selection_score_type"]
    assert set(first["manifest"]["artifacts"]) == {"frozen_recipe.json", "refit_progress.json"}
    assert not (bundle / "selected.pt").exists()
    assert _verify(bundle, recipe)["predictor"] == "reference"
    second = export_bundle(study, bundle)
    assert second["reused"] and second["manifest"] == first["manifest"]
    progress = json.loads((bundle / "refit_progress.json").read_text())
    progress["steps"] = 1
    write_json(bundle / "refit_progress.json", progress)
    with pytest.raises(ValueError, match="artifact hash"):
        export_bundle(study, bundle)


def test_neural_bundle_hashes_and_idempotence(tmp_path):
    study, bundle = tmp_path / "study", tmp_path / "bundle"
    recipe = _artifacts(study)
    exported = export_bundle(study, bundle)
    assert set(exported["manifest"]["artifacts"]) == {
        "frozen_recipe.json", "refit_progress.json", "selected.pt"}
    assert _verify(bundle, recipe)["checkpoint_sha256"] == sha256_file(bundle / "selected.pt")
    assert export_bundle(study, bundle)["reused"]
    payload = torch.load(bundle / "selected.pt", weights_only=True)
    payload["state_dict"]["weight"] += 1
    torch.save(payload, bundle / "selected.pt")
    with pytest.raises(ValueError, match="artifact hash"):
        _verify(bundle, recipe)


def test_reference_progress_and_source_changes_rejected(tmp_path):
    study, bundle = tmp_path / "study", tmp_path / "bundle"
    recipe = _artifacts(study, neural=False)
    progress_path = study / "refit_progress.json"
    write_json(progress_path, {"predictor": "reference", "steps": 1,
                               "train_matches": recipe["outer_train"]})
    with pytest.raises(ValueError, match="reference refit progress"):
        export_bundle(study, bundle)
    write_json(progress_path, {"predictor": "reference", "steps": 0,
                               "train_matches": recipe["outer_train"]})
    export_bundle(study, bundle)
    recipe["decoder"]["switch_cost"] = 5
    write_json(study / "frozen_recipe.json", recipe)
    with pytest.raises(ValueError, match="existing bundle differs"):
        export_bundle(study, bundle)


def test_bundle_rejects_changed_inference_source(tmp_path, monkeypatch):
    from possession import deployment

    study, bundle = tmp_path / "study", tmp_path / "bundle"
    recipe = _artifacts(study)
    exported = export_bundle(study, bundle)
    hashes = exported["manifest"]["inference_source_sha256"].copy()
    hashes["possession/features.py"] = "changed"
    monkeypatch.setattr(deployment, "_inference_source_hashes", lambda **kwargs: hashes)
    with pytest.raises(ValueError, match="inference source hash"):
        _verify(bundle, recipe)


def test_training_only_runtime_changes_do_not_invalidate_bundle(tmp_path, monkeypatch):
    from possession import deployment

    study, bundle = tmp_path / "study", tmp_path / "bundle"
    recipe = _artifacts(study)
    first = export_bundle(study, bundle)
    monkeypatch.setattr(deployment, "runtime_provenance", lambda: {
        "source_sha256": {"possession/train.py": "changed"}, "git_revision": "new"})
    assert _verify(bundle, recipe)["selection_verified"] is True
    reused = export_bundle(study, bundle)
    assert reused["reused"] is True
    assert reused["manifest"] == first["manifest"]


def test_reference_export_and_verification_do_not_import_torch(tmp_path, monkeypatch):
    import builtins
    import importlib

    from possession import deployment

    study, bundle = tmp_path / "study", tmp_path / "bundle"
    recipe = _artifacts(study, neural=False)
    original_import = builtins.__import__

    def block_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("PyTorch deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_torch)
    importlib.reload(deployment)
    deployment.export_bundle(study, bundle)
    assert _verify(bundle, recipe)["selection_verified"] is True


def test_alternative_recipe_file_must_match_bundle_hash(tmp_path):
    study, bundle = tmp_path / "study", tmp_path / "bundle"
    recipe = _artifacts(study, neural=False)
    export_bundle(study, bundle)
    recipe["decoder"]["switch_cost"] = 9
    alternative = bundle / "alternative.json"
    write_json(alternative, recipe)
    with pytest.raises(ValueError, match="recipe hash"):
        verify_selected_checkpoint(alternative, recipe, None)
