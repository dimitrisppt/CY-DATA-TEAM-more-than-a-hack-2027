import json

import numpy as np
import pytest
import torch

from experiments import campaign_refit
from experiments.campaign_protocol import publish
from experiments.campaign_training import build_session, save_training_state
from experiments.robustness import _execution_environment, _source_hashes
from possession.augmentation import normalize_augmentation
from possession.outages import OutageConfig
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half, sha256_file, write_json
from possession.training_session import HalfCache


def _bundle(directory):
    selection = {
        "kind": "convex_pair", "blend_weight": .5, "decoder": {"version": 1},
        "members": [{"id": member_id, "spec": {"model_kwargs": {
            "player_dim": 8, "global_dim": 9,
            "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
            "width": 32, "pooling": pooling}}}
                    for member_id, pooling in (("mean_007", "mean"),
                                               ("ball_attention_002", "ball_attention"))],
    }
    identity = {"seed": 20261116, "prepared_sha256": {}, "source_sha256": _source_hashes(),
                "runtime": _execution_environment(), "selection": selection}
    recipe = {"schema": "campaign_ensemble_recipe_v1", "selection": selection,
              "source_sha256": identity["source_sha256"], "runtime": identity["runtime"]}
    publish(directory / "plan.json", identity)
    publish(directory / "recipe.json", recipe)
    members = {}
    files = {"plan.json": sha256_file(directory / "plan.json"),
             "recipe.json": sha256_file(directory / "recipe.json")}
    for member in selection["members"]:
        member_id = member["id"]
        base = directory / "members" / member_id
        base.mkdir(parents=True)
        model_kwargs = dict(member["spec"]["model_kwargs"])
        config = {"train_matches": list(campaign_refit.FIT_MATCHES),
                  "selection_matches": list(campaign_refit.SELECTION_MATCHES),
                  "campaign_member_id": member_id, "campaign_spec": member["spec"],
                  "steps": 7200, "seed": identity["seed"],
                  "campaign_identity_sha256": campaign_refit._digest(identity),
                  "campaign_recipe_sha256": campaign_refit._digest(member["spec"]),
                  "selection_decoder": selection["decoder"],
                  "selection_blend_kind": selection["kind"],
                  "selection_blend_weight": selection["blend_weight"]}
        checkpoint = base / "model.pt"
        torch.save({"state_dict": {}, "model_kwargs": model_kwargs,
                    "prepared_sha256": identity["prepared_sha256"],
                    "training_config": config}, checkpoint)
        fit = {"member_id": member_id, "checkpoint_sha256": sha256_file(checkpoint),
               "training_config_sha256": campaign_refit._digest(config),
               "identity_sha256": campaign_refit._digest(identity),
               "prepared_sha256": identity["prepared_sha256"]}
        publish(base / "fit.json", fit)
        checkpoint_rel = f"members/{member_id}/model.pt"
        fit_rel = f"members/{member_id}/fit.json"
        members[member_id] = {"checkpoint": checkpoint_rel,
                              "sha256": sha256_file(checkpoint),
                              "fit_record": fit_rel,
                              "fit_record_sha256": sha256_file(base / "fit.json"),
                              "model_kwargs": model_kwargs,
                              "training_config_sha256": campaign_refit._digest(config)}
        files[checkpoint_rel] = sha256_file(checkpoint)
        files[fit_rel] = sha256_file(base / "fit.json")
    manifest = {"schema": "campaign_ensemble_bundle_v1", "files": files,
                "members": members, "selection": selection,
                "seed": identity["seed"], "steps": 7200,
                "decoder": selection["decoder"], "blend_kind": selection["kind"],
                "blend_weight": selection["blend_weight"],
                "provenance_sha256": campaign_refit._digest(identity),
                "source_sha256": identity["source_sha256"], "runtime": identity["runtime"],
                "selection_matches": list(campaign_refit.SELECTION_MATCHES),
                "fit_matches": list(campaign_refit.FIT_MATCHES)}
    publish(directory / "manifest.json", manifest)
    return identity, recipe, manifest


def test_verify_bundle_returns_portable_members(tmp_path):
    _bundle(tmp_path)

    result = campaign_refit.verify_bundle(tmp_path)

    assert result["manifest"]["schema"] == "campaign_ensemble_bundle_v1"
    assert result["recipe"]["selection"]["kind"] == "convex_pair"
    assert set(result["member_paths"]) == set(campaign_refit.MEMBER_IDS)
    assert result["bundle_dir"] == str(tmp_path.resolve())


def test_verify_bundle_rejects_corrupt_checkpoint_even_with_valid_manifest_hash(tmp_path):
    _bundle(tmp_path)
    checkpoint = tmp_path / "members" / "mean_007" / "model.pt"
    checkpoint.write_bytes(b"damaged")

    with pytest.raises(ValueError, match="bundle file hash differs"):
        campaign_refit.verify_bundle(tmp_path)


def test_verify_bundle_rejects_recipe_drift(tmp_path):
    _, recipe, _ = _bundle(tmp_path)
    recipe["selection"]["blend_weight"] = .75
    write_json(tmp_path / "recipe.json", {"sha256": campaign_refit._digest(recipe),
                                           "payload": recipe})

    with pytest.raises(ValueError, match="bundle recipe"):
        campaign_refit.verify_bundle(tmp_path)


def test_completed_run_reuses_bundle_without_fitting(tmp_path, monkeypatch):
    _bundle(tmp_path)
    identity = json.loads((tmp_path / "plan.json").read_text())["payload"]
    recipe = json.loads((tmp_path / "recipe.json").read_text())["payload"]
    monkeypatch.setattr(campaign_refit, "_preflight", lambda *args: (identity, recipe))
    monkeypatch.setattr(campaign_refit, "_fit_member",
                        lambda *args: pytest.fail("completed bundle attempted another fit"))

    result = campaign_refit.run_refit("campaign", "data", tmp_path)

    assert result["status"] == "bundle_reused"


def _synthetic_half(match, period):
    n = 80
    t = np.arange(n)
    players = np.zeros((n, 2, 2, 2), np.float32)
    players[:, 0, :, 0] = 20 + t[:, None] * .05
    players[:, 1, :, 0] = 70 - t[:, None] * .05
    players[..., 1] = 30
    state = np.resize(np.repeat([0, 2, 1], [9, 3, 8]), n).astype(np.int8)
    live = (state != 2).astype(np.int8)
    frames = FrameTable(t, players, np.ones((n, 2, 2), bool),
                        np.column_stack((25 + t * .1, np.full(n, 31))),
                        np.ones(n, bool), np.ones(n, bool),
                        match_id=match, period=period)
    targets = TrainingTargets(live, np.ones(n, bool), (state == 0).astype(np.int8),
                              live.astype(bool), state, np.ones(n, bool))
    return frames, targets


def test_two_step_refit_resumes_from_real_training_state(tmp_path, monkeypatch):
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    for match in campaign_refit.FIT_MATCHES:
        for period in (1, 2):
            save_half(data_dir / f"{match}_half{period}.npz",
                      *_synthetic_half(match, period))
    prepared = {path.name: sha256_file(path) for path in data_dir.glob("*.npz")}
    selection = {"kind": "convex_pair", "blend_weight": .5,
                 "decoder": {"version": 1}, "members": []}
    identity = {"seed": 20261116, "prepared_sha256": prepared,
                "source_sha256": _source_hashes(), "runtime": _execution_environment(),
                "selection": selection}
    members = []
    for member_id, pooling in (("mean_007", "mean"),
                               ("ball_attention_002", "ball_attention")):
        spec = {"model_kwargs": {"player_dim": 8, "global_dim": 9,
                                  "width": 32, "dilations": [1, 2, 4, 8],
                                  "pooling": pooling},
                "training_args": {"learning_rate": .001, "weight_decay": .0001,
                                  "boundary_weight": .25, "boundary_pos_weight": 1.,
                                  "augment": False, "center_size": 512},
                "outage_config": OutageConfig(mode="legacy").to_dict(),
                "augmentation_config": normalize_augmentation(None).to_dict(),
                "schedule": {"kind": "constant", "total_steps": 2,
                             "warmup_steps": 0, "min_lr_ratio": 1.}}
        members.append({"id": member_id, "spec": spec, "steps": 2})
    selection["members"] = members
    recipe = {"schema": "campaign_ensemble_recipe_v1", "selection": selection,
              "source_sha256": identity["source_sha256"], "runtime": identity["runtime"]}
    monkeypatch.setattr(campaign_refit, "_preflight", lambda *args: (identity, recipe))

    member = members[0]
    session = build_session(data_dir, member["spec"], list(campaign_refit.FIT_MATCHES), 20261116,
                            cache=HalfCache(data_dir, max_bytes=100_000_000))
    session.advance_to(1)
    state_dir = tmp_path / "bundle" / "members" / member["id"]
    state_dir.mkdir(parents=True)
    save_training_state(state_dir / "resume-0.pt", session,
                        {"member_id": member["id"], **identity})
    (state_dir / "resume-1.pt").write_bytes(b"interrupted checkpoint write")
    write_json(state_dir / "resume-1.pt.sha256.json", {"sha256": "invalid"})

    result = campaign_refit.run_refit("unused-campaign", data_dir, tmp_path / "bundle")

    assert result["status"] == "bundle_written"
    verified = campaign_refit.verify_bundle(tmp_path / "bundle")
    assert set(verified["member_paths"]) == set(campaign_refit.MEMBER_IDS)
    uninterrupted = build_session(data_dir, member["spec"],
                                   list(campaign_refit.FIT_MATCHES), 20261116)
    uninterrupted.advance_to(2)
    restored = torch.load(verified["member_paths"][member["id"]], weights_only=True)
    for key, expected in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(restored["state_dict"][key], expected, rtol=0, atol=0)
