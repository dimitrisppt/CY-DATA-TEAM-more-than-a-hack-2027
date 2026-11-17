"""Release arithmetic, halftime isolation and model-integrity regression checks."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from test_campaign_submission import _frames

from experiments.campaign_decode import decode_arrays, reference_basis
from experiments.campaign_evaluation import _gap_seconds
from experiments.campaign_protocol import publish, read_record
from experiments.coarse_context import predict_coarse_native
from experiments.gpu_iou_evaluation import _blend
from experiments.gpu_tuning_evaluation import coarse_blend
from experiments.tune import _seed_for_fold
from possession.features import build_features
from possession.model import PossessionModel
from possession.predict import predict_features
from possession.storage import sha256_file
from scripts import v7_bundle


def recipe():
    return read_record(Path(__file__).resolve().parents[1] / "configs/v7-recipe.json")


def test_release_blend_has_selected_logit_weights_and_visibility_gate():
    r = recipe()
    rng = np.random.default_rng(17)
    predictions = {(name, seed): {"neural_live": rng.uniform(.1, .9, 5),
                                  "neural_home": rng.uniform(.1, .9, 5),
                                  "neural_boundary": rng.normal(size=(5, 3, 3))}
                   for name, seeds in [(m["id"], r["incumbent_seeds"])
                                       for m in r["selection"]["members"]]
                   + [("coarse", r["incumbent_seeds"]), ("candidate", r["candidate_seeds"])]
                   for seed in seeds}
    base = {"ball_valid": np.array([True, False, False, True, False]),
            "nearest_ball_seconds": np.array([0., 1., 20., 0., 3.])}
    actual = v7_bundle.combine_predictions(base, predictions, r)
    alpha = r["pipeline"]["coarse_weight"] * np.minimum(
        base["nearest_ball_seconds"] / r["pipeline"]["coarse_seconds"], 1)
    w = r["pipeline"]["candidate_weight"]
    b = r["selection"]["blend_weight"]
    def logit(p):
        return np.log(p) - np.log1p(-p)
    for key in ("neural_live", "neural_home"):
        by_role = {name: np.mean([logit(predictions[name, seed][key]) for seed in seeds], axis=0)
                   for name, seeds in [(m["id"], r["incumbent_seeds"])
                                       for m in r["selection"]["members"]]
                   + [("coarse", r["incumbent_seeds"]), ("candidate", r["candidate_seeds"])]}
        fine = (1-w) * (b*by_role["mean_007"] + (1-b)*by_role["ball_attention_002"])
        fine += w*by_role["candidate"]
        expected = 1 / (1 + np.exp(-((1-alpha)*fine + alpha*by_role["coarse"])))
        np.testing.assert_allclose(actual[key], expected, atol=1e-14, rtol=0)


def test_network_inference_matches_experiment_pipeline():
    torch.set_num_threads(1)
    torch.manual_seed(27)
    frames, r = _frames(83), recipe()
    models = {(name, seed): PossessionModel(width=4, dilations=[1, 2], pooling=pooling).eval()
              for name, seeds, pooling in [
                  ("mean_007", r["incumbent_seeds"], "mean"),
                  ("ball_attention_002", r["incumbent_seeds"], "ball_attention"),
                  ("coarse", r["incumbent_seeds"], "mean"),
                  ("candidate", r["candidate_seeds"], "ball_attention")]
              for seed in seeds}
    features = build_features(frames)
    base = {"ball_valid": frames.ball_valid,
            "nearest_ball_seconds": _gap_seconds(frames.ball_valid, frames.fps),
            **reference_basis(frames, features)}
    predictions = {}
    for (name, seed), model in models.items():
        if name == "coarse":
            p = predict_coarse_native(frames, model)
            p["boundary"] = np.zeros((83, 3, 3))
        else:
            p = predict_features(features, model)
        predictions[name, seed] = v7_bundle.neural_arrays(p)
    fine = [_blend(*(predictions[m["id"], s] for s in r["incumbent_seeds"]), .5)
            for m in r["selection"]["members"]]
    incumbent = _blend(*fine, r["selection"]["blend_weight"])
    candidate = _blend(*(predictions["candidate", s] for s in r["candidate_seeds"]), .5)
    coarse = _blend(*(predictions["coarse", s] for s in r["incumbent_seeds"]), .5)
    arrays = {**base, **_blend(candidate, incumbent, r["pipeline"]["candidate_weight"]),
              "coarse_live": coarse["neural_live"], "coarse_home": coarse["neural_home"]}
    expected = decode_arrays(coarse_blend(arrays, r["pipeline"]["coarse_weight"],
                                          r["pipeline"]["coarse_seconds"]),
                             r["pipeline"]["decoder"])
    np.testing.assert_array_equal(v7_bundle.predict_half(frames, models, r, "cpu"), expected)


def test_half_context_is_reset_and_excluded_frames_remain_x(monkeypatch):
    calls = []
    def predict(frames, *args):
        calls.append((frames.period, frames.frame_id.tolist()))
        return np.full(len(frames.frame_id), frames.period-1, dtype=np.int8)
    monkeypatch.setattr(v7_bundle, "predict_half", predict)
    codes, reports = v7_bundle.infer_spans(_frames(40), {}, {}, [(3, 15), (24, 36)])
    assert calls == [(1, list(range(3, 15))), (2, list(range(24, 36)))]
    np.testing.assert_array_equal(codes, [3]*3 + [0]*12 + [3]*9 + [1]*12 + [3]*4)
    assert len(reports) == 2


def make_bundle(root):
    r = copy.deepcopy(recipe())
    specs = {m["id"]: m["spec"] for m in r["selection"]["members"]}
    specs.update(coarse=r["coarse_spec"], candidate=r["candidate_recipe"]["spec"])
    for spec in specs.values():
        spec["model_kwargs"]["width"] = 4
    publish(root / "recipe.json", r)
    rows = []
    for name, spec in specs.items():
        for seed in r["candidate_seeds"] if name == "candidate" else r["incumbent_seeds"]:
            steps = 9600 if name == "candidate" else 7200 if name == "coarse" else 8400
            config = {"train_matches": r["matches"], "validation_matches": [], "steps": steps,
                      "seed": seed if name == "candidate" else _seed_for_fold(seed, "DFL-MAT-J03WMX", 0),
                      "campaign_spec": spec}
            if name == "candidate":
                config.update(tuning_export="raw", refinement={
                    k: r["candidate_recipe"][k] for k in ("spell_weight", "short_sampling_probability")})
            elif name != "coarse":
                config.update(refinement_export="tail_ema", tail_ema_updates=200,
                              refinement=r["settings"])
            model = PossessionModel(**spec["model_kwargs"])
            path = f"{name}-{seed}.pt"
            torch.save({"state_dict": model.state_dict(), "model_kwargs": spec["model_kwargs"],
                        "training_config": config}, root / path)
            halves = [f"{m}_half{p}.npz" for m in r["matches"] for p in (1, 2)]
            transcript = f"{name}-{seed}.txt"
            (root / transcript).write_text("".join(
                f"{halves[i % len(halves)]}:0:512\n" for i in range(steps)))
            rows.append({"name": name, "base_seed": seed, "steps": steps, "path": path,
                         "transcript_path": transcript,
                         "training_window_sha256": sha256_file(root / transcript)})
    manifest = {"schema": "gpu_tuned_submission_v1", "models": rows, "pipeline": r["pipeline"],
                "source_sha256": {}, "files": {p.name: sha256_file(p) for p in root.iterdir()}}
    publish(root / "manifest.json", manifest)
    return manifest


def test_portable_bundle_and_tampered_checkpoint(tmp_path):
    manifest = make_bundle(tmp_path)
    assert len(v7_bundle.verify_bundle(tmp_path)[0]["models"]) == 8
    (tmp_path / manifest["models"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        v7_bundle.verify_bundle(tmp_path)


def test_missing_model_and_changed_training_split_are_rejected(tmp_path):
    manifest = make_bundle(tmp_path)
    row = manifest["models"][0]
    path = tmp_path / row["path"]
    payload = torch.load(path, weights_only=True)
    payload["training_config"]["train_matches"] = ["challenge-match"]
    torch.save(payload, path)
    manifest["files"][row["path"]] = sha256_file(path)
    (tmp_path / "manifest.json").unlink()
    publish(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="training provenance"):
        v7_bundle.verify_bundle(tmp_path)
    manifest["models"].pop()
    (tmp_path / "manifest.json").unlink()
    publish(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="grid"):
        v7_bundle.verify_bundle(tmp_path)


def test_bundle_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        v7_bundle.contained(tmp_path, "../outside.pt")


def test_training_transcript_rejects_nonpublic_match_even_if_file_hash_updated(tmp_path):
    manifest = make_bundle(tmp_path)
    row = manifest["models"][0]
    path = tmp_path / row["transcript_path"]
    lines = path.read_text().splitlines(True)
    lines[0] = "challenge_half1.npz:0:512\n"
    path.write_text("".join(lines))
    manifest["files"][row["transcript_path"]] = sha256_file(path)
    row["training_window_sha256"] = sha256_file(path)
    (tmp_path / "manifest.json").unlink()
    publish(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="full public split"):
        v7_bundle.verify_bundle(tmp_path)
