"""Portable, label-free inference for the frozen eight-network GPU winner."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from experiments.campaign_decode import decode_arrays, reference_basis
from experiments.campaign_evaluation import _gap_seconds
from experiments.campaign_protocol import read_record
from experiments.coarse_context import predict_coarse_native
from experiments.gpu_iou_evaluation import _blend
from experiments.gpu_tuning_evaluation import coarse_blend
from experiments.tune import _seed_for_fold
from possession.features import build_features
from possession.predict import load_model_checkpoint, predict_features
from possession.storage import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def contained(root, name):
    path = Path(root).resolve() / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"bundle path escapes root: {name}")
    return path


def verify_bundle(directory, *, verify_source=True):
    directory = Path(directory).resolve()
    manifest = read_record(directory / "manifest.json")
    if not manifest or manifest.get("schema") != "gpu_tuned_submission_v1":
        raise ValueError("unsupported or missing v7 bundle manifest")
    for name, digest in manifest["files"].items():
        path = contained(directory, name)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"bundle file missing or changed: {name}")
    if verify_source:
        for name, digest in manifest["source_sha256"].items():
            path = contained(ROOT, name)
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"v7 source missing or changed: {name}")
    recipe = read_record(directory / "recipe.json")
    if recipe["version"] != "v7" or recipe["pipeline"] != manifest["pipeline"]:
        raise ValueError("v7 recipe and pipeline disagree")
    expected = {(member["id"], seed) for member in recipe["selection"]["members"]
                for seed in recipe["incumbent_seeds"]}
    expected |= {("coarse", seed) for seed in recipe["incumbent_seeds"]}
    expected |= {("candidate", seed) for seed in recipe["candidate_seeds"]}
    rows = manifest["models"]
    if len(rows) != 8 or {(r["name"], r["base_seed"]) for r in rows} != expected:
        raise ValueError("v7 model grid is incomplete or duplicated")
    for row in rows:
        if any(row[key] not in manifest["files"] for key in ("path", "transcript_path")):
            raise ValueError("model or transcript is not authenticated by bundle file map")
        transcript = contained(directory, row["transcript_path"])
        if sha256_file(transcript) != row["training_window_sha256"]:
            raise ValueError("sampled training windows differ from the fit receipt")
        lines = transcript.read_text().splitlines()
        halves = {line.split(":", 1)[0] for line in lines}
        expected_halves = {f"{match}_half{period}.npz"
                           for match in recipe["matches"] for period in (1, 2)}
        if len(lines) != row["steps"] or halves != expected_halves:
            raise ValueError("actual sampled windows do not cover the full public split")
        payload = torch.load(contained(directory, row["path"]),
                             map_location="cpu", weights_only=True)
        config = payload["training_config"]
        if row["name"] == "candidate":
            spec = recipe["candidate_recipe"]["spec"]
            seed, steps = row["base_seed"], spec["schedule"]["total_steps"]
            if (config.get("tuning_export") != recipe["candidate_recipe"]["export"]
                    or config["refinement"]["spell_weight"] != recipe["candidate_recipe"]["spell_weight"]
                    or config["refinement"]["short_sampling_probability"] !=
                    recipe["candidate_recipe"]["short_sampling_probability"]):
                raise ValueError("candidate refinement or export differs from selected recipe")
        else:
            seed = _seed_for_fold(row["base_seed"], "DFL-MAT-J03WMX", 0)
            if row["name"] == "coarse":
                spec, steps = recipe["coarse_spec"], 7200
            else:
                spec = next(m["spec"] for m in recipe["selection"]["members"]
                            if m["id"] == row["name"])
                steps = 8400
                if (config.get("refinement_export") != "tail_ema"
                        or config.get("tail_ema_updates") != 200):
                    raise ValueError("fine refinement requires the confirmed tail EMA export")
                for key in ("spell_weight", "short_sampling_probability", "lr_scale",
                            "start_steps", "total_extra_steps"):
                    if config["refinement"][key] != recipe["settings"][key]:
                        raise ValueError("fine refinement recipe differs")
        if (sorted(config["train_matches"]) != recipe["matches"]
                or config["validation_matches"]
                or config["steps"] != row["steps"] or row["steps"] != steps
                or config["seed"] != seed or config["campaign_spec"] != spec
                or payload["model_kwargs"] != spec["model_kwargs"]):
            raise ValueError("model full-data training provenance differs")
        if any(not torch.isfinite(v).all() for v in payload["state_dict"].values()):
            raise ValueError("nonfinite v7 weights")
    return manifest, recipe


def load_models(directory, manifest, device):
    return {(r["name"], r["base_seed"]): load_model_checkpoint(
        contained(directory, r["path"]), device=device)[0] for r in manifest["models"]}


def neural_arrays(prediction):
    return {"neural_live": prediction["live_probability"],
            "neural_home": prediction["home_probability"],
            "neural_boundary": prediction["boundary"]}


def combine_predictions(base_arrays, predictions, recipe):
    """Preserve the exact nesting, logit mixing and gate used in confirmation."""
    fine = [_blend(*(predictions[member["id"], seed]
                     for seed in recipe["incumbent_seeds"]), .5)
            for member in recipe["selection"]["members"]]
    incumbent = _blend(*fine, recipe["selection"]["blend_weight"])
    candidate = _blend(*(predictions["candidate", seed]
                         for seed in recipe["candidate_seeds"]), .5)
    coarse = _blend(*(predictions["coarse", seed]
                      for seed in recipe["incumbent_seeds"]), .5)
    pipeline = recipe["pipeline"]
    mixed = _blend(candidate, incumbent, pipeline["candidate_weight"])
    arrays = {**base_arrays, **mixed, "coarse_live": coarse["neural_live"],
              "coarse_home": coarse["neural_home"]}
    return coarse_blend(arrays, pipeline["coarse_weight"], pipeline["coarse_seconds"])


def predict_half(frames, models, recipe, device="cuda:0"):
    features = build_features(frames)
    predictions = {}
    for key, model in models.items():
        if key[0] == "coarse":
            result = predict_coarse_native(frames, model)
            result["boundary"] = np.zeros((len(frames.frame_id), 3, 3))
        else:
            result = predict_features(features, model, device=device, center_size=2048)
        predictions[key] = neural_arrays(result)
    basis = {"ball_valid": frames.ball_valid,
             "nearest_ball_seconds": _gap_seconds(frames.ball_valid, frames.fps),
             **reference_basis(frames, features)}
    arrays = combine_predictions(basis, predictions, recipe)
    return decode_arrays(arrays, recipe["pipeline"]["decoder"])


def infer_spans(frames, models, recipe, spans, device="cuda:0"):
    n = len(frames.frame_id)
    previous = 0
    for start, stop in spans:
        if not previous <= start < stop <= n:
            raise ValueError("half spans must be ordered, disjoint and inside the timeline")
        previous = stop
    if not spans:
        raise ValueError("at least one half span is required")
    codes = np.full(n, 3, dtype=np.int8)
    reports = []
    for period, (start, stop) in enumerate(spans, 1):
        half = frames.subset(start, stop)
        half.period = period
        path = predict_half(half, models, recipe, device)
        if path.shape != (stop - start,) or not np.isin(path, [0, 1, 2]).all():
            raise ValueError("invalid half predictions")
        codes[start:stop] = path
        reports.append({"period": period, "span": [start, stop],
                        "switches": int(np.count_nonzero(np.diff(path))),
                        "ball_observed_fraction": float(half.ball_valid.mean())})
    return codes, reports
