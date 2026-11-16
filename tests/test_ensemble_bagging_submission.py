import copy

import numpy as np
import pytest
import torch

from experiments import campaign_protocol, campaign_refit, campaign_submission
from experiments import ensemble_bagging_submission as bagging
from experiments.robustness import _execution_environment, _source_hashes
from possession.features import build_features
from possession.model import PossessionModel
from possession.predict import predict_features
from possession.schema import FrameTable
from possession.storage import sha256_file


def _decoder():
    return {
        "version": 1, "reference_mode": "default", "reference": {},
        "blend_mode": "static", "neural_weight": .5,
        "visible_neural_weight": .5, "missing_neural_weight": .5,
        "visible_live_weight": .5, "missing_live_weight": .5,
        "visible_owner_weight": .5, "missing_owner_weight": .5,
        "transition_seconds": 2., "live_temperature": 1., "owner_temperature": 1.,
        "boundary_temperature": 1., "switch_cost_type": "typed",
        "switch_cost": {"owner_switch": .2, "to_dead": 2., "from_dead": 2.},
        "boundary_scale": .5,
    }


def _selection():
    return {"kind": "convex_pair", "blend_weight": .5, "decoder": _decoder(),
            "members": [{"id": member_id, "steps": 7200,
                         "spec": {"model_kwargs": {
                             "player_dim": 8, "global_dim": 9,
                             "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                             "width": 32, "pooling": pooling}}}
                        for member_id, pooling in (("mean_007", "mean"),
                                                   ("ball_attention_002", "ball_attention"))]}


def _bundle(directory, seed, omit_source=None):
    directory.mkdir(parents=True)
    selection = _selection()
    sources = _source_hashes()
    if omit_source is not None:
        sources.pop(omit_source)
    identity = {"seed": seed, "prepared_sha256": {"cache": "public-cache-hash"},
                "source_sha256": sources, "runtime": _execution_environment(),
                "selection": selection}
    recipe = {"schema": "campaign_ensemble_recipe_v1", "selection": selection,
              "source_sha256": identity["source_sha256"], "runtime": identity["runtime"]}
    campaign_protocol.publish(directory / "plan.json", identity)
    campaign_protocol.publish(directory / "recipe.json", recipe)
    members, files = {}, {name: sha256_file(directory / name)
                         for name in ("plan.json", "recipe.json")}
    for member in selection["members"]:
        member_id = member["id"]
        path = directory / "members" / member_id
        path.mkdir(parents=True)
        model_kwargs = {"player_dim": 8, "global_dim": 9,
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                        "width": 32, "pooling": member_id == "ball_attention_002" and
                        "ball_attention" or "mean"}
        config = {"train_matches": list(campaign_refit.FIT_MATCHES),
                  "selection_matches": list(campaign_refit.SELECTION_MATCHES),
                  "campaign_member_id": member_id, "campaign_spec": member["spec"],
                  "steps": 7200, "seed": seed,
                  "campaign_identity_sha256": campaign_refit._digest(identity),
                  "campaign_recipe_sha256": campaign_refit._digest(member["spec"]),
                  "selection_decoder": selection["decoder"],
                  "selection_blend_kind": selection["kind"],
                  "selection_blend_weight": selection["blend_weight"]}
        checkpoint = path / "model.pt"
        torch.save({"state_dict": {}, "model_kwargs": model_kwargs,
                    "prepared_sha256": identity["prepared_sha256"],
                    "training_config": config}, checkpoint)
        fit_path = path / "fit.json"
        fit = {"member_id": member_id, "checkpoint_sha256": sha256_file(checkpoint),
               "training_config_sha256": campaign_refit._digest(config),
               "identity_sha256": campaign_refit._digest(identity),
               "prepared_sha256": identity["prepared_sha256"]}
        campaign_protocol.publish(fit_path, fit)
        checkpoint_rel, fit_rel = (f"members/{member_id}/model.pt",
                                   f"members/{member_id}/fit.json")
        members[member_id] = {"checkpoint": checkpoint_rel, "sha256": sha256_file(checkpoint),
                              "fit_record": fit_rel, "fit_record_sha256": sha256_file(fit_path),
                              "model_kwargs": model_kwargs,
                              "training_config_sha256": campaign_refit._digest(config)}
        files[checkpoint_rel], files[fit_rel] = sha256_file(checkpoint), sha256_file(fit_path)
    manifest = {"schema": "campaign_ensemble_bundle_v1", "files": files,
                "members": members, "selection": selection,
                "seed": seed, "steps": 7200, "decoder": selection["decoder"],
                "blend_kind": selection["kind"], "blend_weight": selection["blend_weight"],
                "provenance_sha256": campaign_refit._digest(identity),
                "source_sha256": identity["source_sha256"], "runtime": identity["runtime"],
                "selection_matches": list(campaign_refit.SELECTION_MATCHES),
                "fit_matches": list(campaign_refit.FIT_MATCHES)}
    campaign_protocol.publish(directory / "manifest.json", manifest)
    return selection


def _study(path, *, eligible=True, outer_pass=True, candidate_mismatch=False,
           protocol_mismatch=False, source_mismatch=False):
    baseline = _selection()
    candidate = copy.deepcopy(baseline)
    candidate["blend_weight"] = .6
    if candidate_mismatch:
        candidate["members"][0]["spec"]["model_kwargs"]["pooling"] = "interaction"
    protocol = {"baseline_selection": baseline, "candidate_selection": candidate,
                "source_sha256": _source_hashes(), "runtime": _execution_environment(),
                "mask_batches": [{"short_seed": 41, "long_seed": 42,
                                  "scenarios": {"replay": {"seed": 5102}}},
                                 {"short_seed": 51, "long_seed": 52,
                                  "scenarios": {"replay": {"seed": 5102}}}]}
    if protocol_mismatch:
        protocol["candidate_selection"] = copy.deepcopy(candidate)
        protocol["candidate_selection"]["blend_weight"] = .7
    if source_mismatch:
        protocol["source_sha256"]["experiments/campaign_submission.py"] = "wrong-hash"
    campaign_protocol.publish(path / "protocol.json", protocol)
    checks = {"passed": True, "checks": {"objective": True, "matches": True},
              "objective_delta": .1}
    batches = []
    for short_seed, long_seed in ((41, 42), (51, 52)):
        batches.append({"mask_seeds": {"short": short_seed, "long": long_seed, "replay": 5102},
                        "baseline": [{"objective": .5}, {"objective": .5}],
                        "candidate": {"objective": .6}, "checks": checks,
                        "outer": {"baseline": [{"objective": .5}, {"objective": .5}],
                                  "candidate": {"objective": .6},
                                  "checks": {"passed": outer_pass,
                                             "checks": {"outer_gain": outer_pass}}}})
    result = {"eligible": eligible, "candidate_selection": candidate,
              "baseline_selection": baseline, "batches": batches,
              "protocol_sha256": campaign_refit._digest(protocol), "notes": []}
    campaign_protocol.publish(path / "result.json", result)


def test_four_way_blend_matches_explicit_two_level_campaign_blend():
    frames = FrameTable(
        frame_id=np.arange(24), players_xy=np.ones((24, 2, 3, 2), dtype=np.float32),
        player_valid=np.ones((24, 2, 3), dtype=bool),
        ball_xy=np.column_stack((np.arange(24, dtype=np.float32), np.ones(24, dtype=np.float32))),
        ball_valid=np.ones(24, dtype=bool), graded=np.ones(24, dtype=bool),
        match_id="synthetic", period=1)
    features = build_features(frames)
    selection = _selection()
    predictions, models = {seed: {} for seed in bagging.SEEDS}, {}
    for seed in bagging.SEEDS:
        models[seed] = {}
        for member, pooling in zip(selection["members"], ("mean", "ball_attention"), strict=True):
            torch.manual_seed(seed + (0 if pooling == "mean" else 1))
            model = PossessionModel(width=4, dilations=(1,), pooling=pooling)
            models[seed][member["id"]] = model
            predictions[seed][member["id"]] = predict_features(
                features, model, center_size=24, halo=model.radius)
    actual = bagging.combine_four_predictions(predictions, selection)
    seed_means = {
        member["id"]: campaign_submission.blend_neural(
            predictions[bagging.SEEDS[0]][member["id"]],
            predictions[bagging.SEEDS[1]][member["id"]], .5)
        for member in selection["members"]}
    expected = campaign_submission.blend_neural(
        seed_means[selection["members"][0]["id"]],
        seed_means[selection["members"][1]["id"]], selection["blend_weight"])
    for name in ("live_probability", "home_probability", "boundary"):
        np.testing.assert_array_equal(actual[name], expected[name])


def test_failed_gate_rejected_before_output_or_model_copy(tmp_path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    _study(study_dir, eligible=False)
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="eligibility"):
        bagging.export_bagged_bundle(study_dir, tmp_path / "missing-a",
                                     tmp_path / "missing-b", output)
    assert not output.exists()


def test_outer_gate_failure_rejected_before_output(tmp_path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    _study(study_dir, outer_pass=False)
    with pytest.raises(ValueError, match="outer checks"):
        bagging.export_bagged_bundle(study_dir, tmp_path / "missing-a",
                                     tmp_path / "missing-b", tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_export_is_portable_and_copies_only_verified_files(tmp_path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    _study(study_dir)
    first, second, output = tmp_path / "first", tmp_path / "second", tmp_path / "portable"
    _bundle(first, bagging.SEEDS[0])
    _bundle(second, bagging.SEEDS[1])
    result = bagging.export_bagged_bundle(study_dir, first, second, output)
    verified = bagging.verify_bagged_bundle(output)
    assert result["manifest"]["candidate_architecture_blend_weight"] == .6
    assert set(verified["children"]) == set(bagging.SEEDS)
    assert "seed_20261116/members/mean_007/model.pt" in verified["manifest"]["files"]
    assert "bagging-result.json" in verified["manifest"]["files"]
    (output / "seed_20261117" / "members" / "mean_007" / "model.pt").write_bytes(b"bad")
    with pytest.raises(ValueError, match="file differs"):
        bagging.verify_bagged_bundle(output)


def test_additive_refit_source_entries_are_accepted(tmp_path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    _study(study_dir)
    first, second = tmp_path / "first", tmp_path / "second"
    _bundle(first, bagging.SEEDS[0], omit_source="experiments/ensemble_bagging_submission.py")
    _bundle(second, bagging.SEEDS[1])
    result = bagging.export_bagged_bundle(study_dir, first, second, tmp_path / "portable")
    assert "experiments/ensemble_bagging_submission.py" not in result["manifest"][
        "refit_source_sha256"][str(bagging.SEEDS[0])]
    assert "experiments/ensemble_bagging_submission.py" in result["manifest"][
        "refit_source_sha256"][str(bagging.SEEDS[1])]


@pytest.mark.parametrize("protocol_mismatch,source_mismatch", [(True, False), (False, True)])
def test_study_protocol_selection_or_source_mismatch_fails_before_copy(
        tmp_path, protocol_mismatch, source_mismatch):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    _study(study_dir, protocol_mismatch=protocol_mismatch, source_mismatch=source_mismatch)
    with pytest.raises(ValueError, match="protocol selections|source hashes"):
        bagging.export_bagged_bundle(study_dir, tmp_path / "missing-a",
                                     tmp_path / "missing-b", tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_wrong_seed_and_member_specs_are_rejected(tmp_path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    _study(study_dir)
    first, second = tmp_path / "first", tmp_path / "second"
    _bundle(first, bagging.SEEDS[0])
    _bundle(second, 20261118)
    with pytest.raises(ValueError, match="frozen refit seeds"):
        bagging.export_bagged_bundle(study_dir, first, second, tmp_path / "wrong-seed")


def test_candidate_member_spec_mismatch_is_rejected(tmp_path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    _study(study_dir, candidate_mismatch=True)
    first, second = tmp_path / "first", tmp_path / "second"
    _bundle(first, bagging.SEEDS[0])
    _bundle(second, bagging.SEEDS[1])
    with pytest.raises(ValueError, match="member specs"):
        bagging.export_bagged_bundle(study_dir, first, second, tmp_path / "mismatch")
    assert not (tmp_path / "mismatch").exists()


def test_inference_resets_each_half_and_marks_outside_frames_x(tmp_path, monkeypatch):
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "manifest.json").write_text("portable bundle")
    selection = _selection()
    children = {seed: {"member_paths": {member: f"{seed}-{member}.pt"
                                        for member in bagging.MEMBERS}}
                for seed in bagging.SEEDS}
    monkeypatch.setattr(bagging, "verify_bagged_bundle", lambda path: {
        "manifest": {"candidate_selection": selection}, "children": children})
    monkeypatch.setattr(bagging.campaign_submission, "_verified_inputs", lambda path: {
        "raw": {"path": "raw.xml", "sha256": "raw"},
        "metadata": {"path": "meta.json", "sha256": "meta"}})
    frames = FrameTable(np.arange(7), np.ones((7, 2, 2, 2), dtype=np.float32),
                        np.ones((7, 2, 2), dtype=bool), np.ones((7, 2), dtype=np.float32),
                        np.ones(7, dtype=bool), np.ones(7, dtype=bool), match_id="target")
    monkeypatch.setattr(bagging, "read_atd", lambda *args, **kwargs: frames)
    monkeypatch.setattr(bagging, "N_FRAMES", 7)
    monkeypatch.setattr(bagging, "TARGET_SPANS", ((1, 3), (4, 6)))
    monkeypatch.setattr(bagging, "load_model_checkpoint", lambda path, device: (object(), {}))
    visited = []

    def fake_predict(half, models, candidate):
        visited.append((half.period, half.frame_id.tolist()))
        return np.asarray([0, 1])

    monkeypatch.setattr(bagging, "_predict_half", fake_predict)
    written = {}

    def write_submission(path, codes, *, n_expected):
        written["codes"] = codes.copy()
        path.write_text("synthetic csv")

    monkeypatch.setattr(bagging.campaign_submission, "_validate_output", lambda path: written["codes"])
    monkeypatch.setattr("possession.cli.write_submission", write_submission)
    report = bagging.infer_bagged_submission(tmp_path / "bundle", tmp_path / "data",
                                             tmp_path / "submission.csv")
    assert visited == [(1, [1, 2]), (2, [4, 5])]
    np.testing.assert_array_equal(written["codes"], np.array([3, 0, 1, 3, 0, 1, 3], dtype=np.int8))
    assert report["ground_truth_used"] is False
