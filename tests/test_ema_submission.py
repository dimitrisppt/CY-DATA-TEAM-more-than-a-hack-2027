import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import campaign_protocol, campaign_refit, ema_submission
from experiments.robustness import _execution_environment, _source_hashes
from possession.model import PossessionModel
from possession.storage import sha256_file


def _score(offset=0.0):
    names = ("clean", "short", "long", "replay")
    by_match = {
        f"match_{index}": {name: {"spell_f1_mean": .4 + offset,
                                 "transition_f1_at_12": .3 + offset}
                           for name in names}
        for index in range(6)
    }
    return {"by_match": by_match,
            "means": {name: {"spell_f1_mean": .4 + offset,
                             "transition_f1_at_12": .3 + offset}
                      for name in names},
            "objective": .5 + offset}


def _selection():
    members = []
    for member_id, pooling in (("mean_007", "mean"),
                               ("ball_attention_002", "ball_attention")):
        spec = {"model_kwargs": {"width": 4, "pooling": pooling},
                "training_args": {}, "outage_config": {},
                "augmentation_config": {},
                "schedule": {"total_steps": 7200}}
        members.append({"id": member_id, "spec": spec, "steps": 7200})
    return {"kind": "convex_pair", "members": members, "blend_weight": .5,
            "decoder": {"version": 1, "reference_mode": "default", "reference": {},
                        "blend_mode": "static", "neural_weight": .5,
                        "visible_neural_weight": .5, "missing_neural_weight": .5,
                        "visible_live_weight": .5, "missing_live_weight": .5,
                        "visible_owner_weight": .5, "missing_owner_weight": .5,
                        "transition_seconds": 2., "live_temperature": 1.,
                        "owner_temperature": 1., "boundary_temperature": 1.,
                        "switch_cost_type": "typed", "switch_cost": {
                            "owner_switch": .01, "to_dead": 1., "from_dead": 1.},
                        "boundary_scale": 1.}}


def _write_study(root: Path, baseline: dict, candidate: dict | None = None, *, passed=True):
    candidate = baseline if candidate is None else candidate
    protocol = {"schema": "ema_stability_v1", "tail": {"start_steps": 7000,
                                                            "end_steps": 7200,
                                                            "updates": 200,
                                                            "ema_decay": .99},
                "baseline_selection": baseline, "candidate_selection": candidate,
                "runtime": _execution_environment(), "source_sha256": _source_hashes()}
    protocol_sha = campaign_refit._digest(protocol)
    root.mkdir(parents=True, exist_ok=True)
    campaign_protocol.publish(root / "protocol.json", protocol)
    batches = []
    for index in (1, 2):
        raw, ema = _score(), _score(.01 if passed else 0.)
        inner = ema_submission.ema_stability._gate(raw, ema)
        outer = ema_submission.ema_stability._outer_gate(raw, ema)
        if not passed:
            inner["passed"] = False
        row = {"schema": "ema_stability_v1", "protocol_sha256": protocol_sha,
               "batch": f"batch_{index}", "raw_v4_inner": raw, "ema_inner": ema,
               "inner_checks": inner, "raw_v4_outer": raw, "ema_outer": ema,
               "outer_checks": outer, "artifact_sha256": {},
               "target_ground_truth_used": False}
        path = root / "batches" / row["batch"] / "result.json"
        campaign_protocol.publish(path, row)
        batches.append(row)
    result = {"schema": "ema_stability_v1", "protocol_sha256": protocol_sha,
              "baseline_selection": baseline, "candidate_selection": candidate,
              "batches": batches, "eligible": passed}
    campaign_protocol.publish(root / "result.json", result)
    return protocol, result


def _write_source_bundle(root: Path, selection: dict, seed: int,
                         *, wrong_member=None, wrong_split=False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    fit_matches = list(campaign_refit.FIT_MATCHES)
    plan = {"seed": seed, "fit_matches": fit_matches,
            "selection_matches": list(campaign_refit.SELECTION_MATCHES),
            "selection": selection, "prepared_sha256": {}}
    campaign_protocol.publish(root / "plan.json", plan)
    campaign_protocol.publish(root / "recipe.json", {"selection": selection})
    members = {}
    for member in selection["members"]:
        member_id, spec = member["id"], member["spec"]
        kwargs = {"player_dim": 8, "global_dim": 9,
                  "dilations": [1, 2, 4, 8, 16, 32, 64, 128], **spec["model_kwargs"]}
        model = PossessionModel(**kwargs)
        state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
        training_config = {"seed": seed, "steps": 7200, "campaign_spec": spec,
                           "train_matches": fit_matches,
                           "validation_matches": ["bad"] if wrong_split and member_id == wrong_member else [],
                           "campaign_identity_sha256": campaign_refit._digest(plan),
                           "campaign_recipe_sha256": campaign_refit._digest(spec),
                           "campaign_member_id": member_id}
        checkpoint = root / "members" / member_id / "model.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state, "model_kwargs": kwargs,
                    "training_config": training_config}, checkpoint)
        checkpoint_hash = sha256_file(checkpoint)
        fit_record = {"member_id": member_id, "identity_sha256": campaign_refit._digest(plan),
                      "recipe_sha256": campaign_refit._digest(spec),
            "seed": seed, "steps": 7200, "selection_matches": plan["selection_matches"],
            "fit_matches": fit_matches, "checkpoint_sha256": checkpoint_hash}
        fit_record["training_config_sha256"] = campaign_refit._digest(training_config)
        fit_path = checkpoint.parent / "fit.json"
        campaign_protocol.publish(fit_path, fit_record)
        members[member_id] = {"checkpoint": f"members/{member_id}/model.pt",
                              "sha256": checkpoint_hash,
                              "fit_record": f"members/{member_id}/fit.json",
                              "fit_record_sha256": sha256_file(fit_path),
                              "model_kwargs": kwargs, "training_config_sha256":
                              campaign_refit._digest(training_config)}
    manifest = {"schema": "campaign_ensemble_bundle_v1", "seed": seed,
                "steps": 7200, "fit_matches": fit_matches,
                "selection_matches": plan["selection_matches"], "selection": selection,
                "provenance_sha256": campaign_refit._digest(plan), "members": members}
    campaign_protocol.publish(root / "manifest.json", manifest)
    return manifest


def _write_portable_bundle(root: Path, *, bad_seed=None, bad_split=False,
                           bad_spec=False, bad_parity=False) -> dict:
    selection = _selection()
    candidate = copy.deepcopy(selection)
    candidate["blend_weight"] = .619862
    study = root / "proof" / "study"
    _write_study(study, selection, candidate)
    files = {}
    source_manifest_sha = {}
    for seed in ema_submission.SEEDS:
        source_root = root / "proof" / f"source_seed_{seed}"
        source_selection = copy.deepcopy(selection)
        if bad_spec and seed == ema_submission.SEEDS[0]:
            source_selection["members"][0]["spec"]["model_kwargs"]["width"] = 8
        source_manifest = _write_source_bundle(
            source_root, source_selection, seed,
            wrong_member="mean_007" if bad_split and seed == ema_submission.SEEDS[0] else None,
            wrong_split=bad_split)
        source_manifest_sha[str(seed)] = sha256_file(source_root / "manifest.json")
        for path in source_root.rglob("*"):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = sha256_file(path)
    for path in study.rglob("*"):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    for index in (1, 2):
        source = study / "batches" / f"batch_{index}" / "result.json"
        target = study / f"batch_{index}.json"
        target.write_bytes(source.read_bytes())
        files[target.relative_to(root).as_posix()] = sha256_file(target)

    members = {}
    parity = {"raw_weights_match_endpoint_state": True,
              "raw_weights_match_baseline_checkpoint": True,
              "raw_training_progress_matches_endpoint": True,
              "compared_optimizer_rng_transcript_and_counters": True,
              "start_steps": 7000, "end_steps": 7200}
    if bad_parity:
        parity["raw_training_progress_matches_endpoint"] = False
    for seed in ema_submission.SEEDS:
        members[str(seed)] = {}
        source_root = root / "proof" / f"source_seed_{seed}"
        plan = campaign_protocol.read_record(source_root / "plan.json")
        source_manifest = campaign_protocol.read_record(source_root / "manifest.json")
        for member in selection["members"]:
            member_id, spec = member["id"], member["spec"]
            source_row = source_manifest["members"][member_id]
            source_fit_path = source_root / source_row["fit_record"]
            source_ckpt = source_root / source_row["checkpoint"]
            source_fit = campaign_protocol.read_record(source_fit_path)
            raw = torch.load(source_ckpt, map_location="cpu", weights_only=True)
            average = {name: tensor.clone() for name, tensor in raw["state_dict"].items()}
            if bad_seed == seed:
                training_seed = seed + 99
            else:
                training_seed = seed
            meta = {"kind": "tail_ema", "decay": .99, "start_steps": 7000,
                    "end_steps": 7200, "updates": 200,
                    "initialization": "raw model state at step 7000",
                    "interpretation": "200-update tail EMA, not full-run EMA",
                    "raw_parity": parity}
            kwargs = raw["model_kwargs"]
            checkpoint = root / "members" / str(seed) / member_id / "model.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            config = dict(raw["training_config"])
            config["seed"] = training_seed
            payload = {**raw, "state_dict": average, "training_config": config,
                       "ema_metadata": meta}
            torch.save(payload, checkpoint)
            checkpoint_sha = sha256_file(checkpoint)
            fit = {"schema": "final_tail_ema_fit_v1", "member_id": member_id,
                   "seed": training_seed, "identity_sha256": campaign_refit._digest(
                       plan), "steps": 7200,
                   "train_matches": list(campaign_refit.FIT_MATCHES), "validation_matches": [],
                   "raw_parity": parity, "tail_ema": meta,
                   "ema_sha256": ema_submission._tensor_digest(average),
                   "training_config_sha256": campaign_refit._digest(config),
                   "checkpoint_sha256": checkpoint_sha,
                   "inputs": {"source_bundle_manifest_sha256": source_manifest_sha[str(seed)],
                              "source_fit_sha256": sha256_file(source_fit_path),
                              "baseline_checkpoint_sha256": source_fit["checkpoint_sha256"]}}
            fit_path = checkpoint.parent / "fit.json"
            campaign_protocol.publish(fit_path, fit)
            rel_ckpt, rel_fit = (checkpoint.relative_to(root).as_posix(),
                                 fit_path.relative_to(root).as_posix())
            files[rel_ckpt], files[rel_fit] = checkpoint_sha, sha256_file(fit_path)
            members[str(seed)][member_id] = {"checkpoint": rel_ckpt,
                                             "sha256": checkpoint_sha,
                                             "fit_record": rel_fit,
                                             "fit_record_sha256": sha256_file(fit_path),
                                             "model_kwargs": kwargs, "spec": spec,
                                             "seed": training_seed, "steps": 7200,
                                             "ema_sha256": fit["ema_sha256"]}
    protocol_path = study / "protocol.json"
    protocol = campaign_protocol.read_record(protocol_path)
    manifest = {"schema": ema_submission._SCHEMA, "portable": True,
                "development_only": True, "seeds": list(ema_submission.SEEDS),
                "member_ids": list(ema_submission.MEMBERS),
                "seed_weights": {str(seed): .5 for seed in ema_submission.SEEDS},
                "members": members, "baseline_selection": selection,
                "candidate_selection": candidate,
                "ema_protocol_sha256": campaign_refit._digest(protocol),
                "ema_protocol_file_sha256": sha256_file(study / "protocol.json"),
                "ema_result_file_sha256": sha256_file(study / "result.json"),
                "study_source_sha256": protocol["source_sha256"],
                "study_runtime": protocol["runtime"],
                "source_bundle_manifest_sha256": source_manifest_sha,
                "files": files, "steps": 7200,
                "ema": {"decay": .99, "start_steps": 7000,
                        "end_steps": 7200, "updates": 200}}
    campaign_protocol.publish(root / "manifest.json", manifest)
    return manifest


def test_portable_bundle_verifies_all_embedded_proofs(tmp_path):
    _write_portable_bundle(tmp_path)
    verified = ema_submission.verify_ema_bundle(tmp_path)
    assert verified["manifest"]["schema"] == "ema_ensemble_submission_v1"
    assert verified["result"]["eligible"] is True


@pytest.mark.parametrize("options", [
    {"bad_seed": 20261116},
    {"bad_split": True},
    {"bad_spec": True},
    {"bad_parity": True},
])
def test_portable_bundle_rejects_seed_split_spec_and_parity_drift(tmp_path, options):
    _write_portable_bundle(tmp_path, **options)
    with pytest.raises(ValueError):
        ema_submission.verify_ema_bundle(tmp_path)


def test_portable_bundle_rejects_checkpoint_byte_tampering(tmp_path):
    _write_portable_bundle(tmp_path)
    checkpoint = tmp_path / "members" / "20261116" / "mean_007" / "model.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
    with pytest.raises(ValueError, match="file hash differs"):
        ema_submission.verify_ema_bundle(tmp_path)


def test_study_must_have_both_completed_ema_batches(tmp_path):
    selection = _selection()
    study = tmp_path / "study"
    _write_study(study, selection, passed=False)
    with pytest.raises(ValueError, match="eligible|gate|differs"):
        ema_submission._verify_study(study, selection)


def test_ema_study_allows_refined_blend_but_keeps_original_as_baseline(tmp_path):
    baseline = _selection()
    candidate = copy.deepcopy(baseline)
    candidate["blend_weight"] = .619862
    study = tmp_path / "study"
    _write_study(study, baseline, candidate)
    protocol, result, _ = ema_submission._verify_study(study, baseline)
    assert protocol["baseline_selection"] == baseline
    assert result["candidate_selection"] == candidate
    with pytest.raises(ValueError, match="baseline|differs"):
        ema_submission._verify_study(study, candidate)


def test_final_identity_requires_seed_spec_and_full_seven_match_split():
    selection = _selection()
    member = selection["members"][0]
    plan = {"seed": 20261116, "fit_matches": list(campaign_refit.FIT_MATCHES),
            "selection_matches": list(campaign_refit.SELECTION_MATCHES),
            "selection": selection}
    assert ema_submission._final_identity(plan, member["id"], 20261116,
                                          member["spec"])["seed"] == 20261116
    for changed in ({**plan, "seed": 1},
                    {**plan, "fit_matches": list(campaign_refit.SELECTION_MATCHES)}):
        with pytest.raises(ValueError):
            ema_submission._final_identity(changed, member["id"], 20261116,
                                           member["spec"])
    with pytest.raises(ValueError, match="spec"):
        ema_submission._final_identity(plan, member["id"], 20261116,
                                       {**member["spec"], "model_kwargs": {"width": 9}})


def test_raw_tail_parity_checks_weights_and_complete_training_state():
    state = {"steps": 7200, "optimizer": {"step": 3},
             "rng": {"bytes": b"fixed"}, "model": {"weight": torch.tensor([1., 2.])}}
    parity = ema_submission._assert_raw_tail_parity(state, copy.deepcopy(state),
                                                     state["model"])
    assert parity["raw_training_progress_matches_endpoint"] is True
    changed_progress = copy.deepcopy(state)
    changed_progress["optimizer"]["step"] = 4
    with pytest.raises(ValueError, match="parity"):
        ema_submission._assert_raw_tail_parity(state, changed_progress, state["model"])
    changed_weights = {"weight": torch.tensor([9., 9.])}
    with pytest.raises(ValueError, match="parity"):
        ema_submission._assert_raw_tail_parity(state, state, changed_weights)


def test_output_directory_cannot_overlap_sources(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    output = source / "out"
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("verification should not run")

    monkeypatch.setattr(ema_submission.campaign_refit, "verify_bundle", forbidden)
    with pytest.raises(ValueError, match="separate"):
        ema_submission.export_ema_bundle(source, source, source, source, output)
    assert called is False


def test_inference_preserves_half_boundaries_and_emits_valid_csv(tmp_path, monkeypatch):
    bundle_dir, target_dir = tmp_path / "bundle", tmp_path / "target"
    bundle_dir.mkdir()
    target_dir.mkdir()
    _write_portable_bundle(bundle_dir)
    output = tmp_path / "submission.csv"

    class Frames:
        def __init__(self, n):
            self.frame_id = np.arange(n)
            self.ball_valid = np.zeros(n, dtype=bool)
            self.fps = 25.

        def subset(self, start, stop):
            return Frames(stop - start)

    monkeypatch.setattr(ema_submission, "N_FRAMES", 10)
    monkeypatch.setattr(ema_submission, "TARGET_SPANS", ((1, 4), (6, 9)))
    monkeypatch.setattr(ema_submission, "read_atd", lambda *args, **kwargs: Frames(10))
    monkeypatch.setattr(ema_submission.campaign_submission, "N_FRAMES", 10)
    monkeypatch.setattr(ema_submission.campaign_submission, "TARGET_SPANS", ((1, 4), (6, 9)))
    monkeypatch.setattr(ema_submission.campaign_submission, "_verified_inputs",
                        lambda path: {"raw": {"path": "raw.atd"},
                                      "metadata": {"path": "meta.json"}})
    monkeypatch.setattr(ema_submission, "_predict_half",
                        lambda frames, models, selection:
                        np.zeros(len(frames.frame_id), dtype=np.int8))
    result = ema_submission.infer_ema_submission(bundle_dir, target_dir, output)
    rows = output.read_text().splitlines()
    assert len(rows) == 11
    labels = [row.split(",")[1] for row in rows[1:]]
    assert labels == ["X", "H", "H", "H", "X", "X", "H", "H", "H", "X"]
    assert result["format_validated"] is True

