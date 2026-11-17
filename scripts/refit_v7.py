"""Refit the frozen GPU winner on all seven public matches, without target inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from experiments.campaign_protocol import publish, read_record
from experiments.coarse_context import prepare_coarse_data
from experiments.gpu_iou import train_parent, train_refinement
from experiments.gpu_runtime import accelerated_preprocessing, configure_device
from experiments.gpu_tuning_training import fit as fit_candidate
from experiments.robustness import _execution_environment, _source_hashes
from possession.prepared import verify_prepared
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache

ROOT = Path(__file__).resolve().parents[1]


def audit_fit(checkpoint, state_path, expected_matches, steps):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = payload["training_config"]
    if sorted(config["train_matches"]) != sorted(expected_matches) or config["steps"] != steps:
        raise ValueError("final model training split or update count differs")
    if any(not torch.isfinite(v).all() for v in payload["state_dict"].values()):
        raise ValueError("nonfinite final weights")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state = state.get("campaign", state)["state"]
    transcript = state["training_transcript"]
    lines = transcript.decode().splitlines()
    actual = sorted({line.split(":", 1)[0].rsplit("_half", 1)[0] for line in lines})
    if len(lines) != steps or actual != sorted(expected_matches):
        raise ValueError("actual final training windows differ from the full public split")
    return {"actual_train_matches": actual, "steps": steps,
            "training_window_sha256": hashlib.sha256(transcript).hexdigest(),
            "model_sha256": sha256_file(checkpoint), "state_sha256": sha256_file(state_path),
            "parameters": sum(v.numel() for v in payload["state_dict"].values())}


def train_job(plan, output, job):
    output = Path(output)
    configure_device("cuda:0")
    recipe = plan["recipe"]
    name, seed = job
    job_output = output / "jobs" / f"{name}-{seed}"
    job_output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data = Path(plan["coarse_data_dir"] if name == "coarse" else plan["data_dir"])
    cache = HalfCache(data, 200_000_000 if name == "coarse" else 2_000_000_000)
    with accelerated_preprocessing():
        if name == "candidate":
            def tick(**detail):
                write_json(job_output / "status.json", {
                    "job": name, "seed": seed, "updated_unix": time.time(), **detail})
            fitted = fit_candidate(
                data, recipe["candidate_recipe"], recipe["matches"], [], seed,
                recipe["candidate_recipe"]["spec"]["schedule"]["total_steps"],
                job_output / "fit", {"release": "v7", "protocol": plan["recipe_sha256"]},
                cache, tick)
            checkpoint, state = Path(fitted["model_path"]), Path(fitted["state_path"])
            steps = fitted["steps"]
        else:
            spec = (recipe["coarse_spec"] if name == "coarse" else
                    next(m["spec"] for m in recipe["selection"]["members"] if m["id"] == name))
            directory, fitted = train_parent(
                plan, job_output, 0, seed, name, spec, cache, coarse=name == "coarse")
            steps = 7200
            if name != "coarse":
                directory, fitted = train_refinement(
                    plan, job_output, directory, fitted, "spell_balanced", spec, cache)
                steps = 8400
            checkpoint = directory / "model.pt"
            state = directory / f"state_{steps:06d}.pt"
    audit = audit_fit(checkpoint, state, recipe["matches"], steps)
    result = {"name": name, "base_seed": seed, "checkpoint": str(checkpoint),
              "state": str(state), "fit": fitted, "audit": audit,
              "elapsed_seconds": time.time() - started}
    return publish(job_output / "complete.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=ROOT / "configs/v7-recipe.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/dfl")
    parser.add_argument("--coarse-data-dir", type=Path,
                        default=ROOT / "artifacts/gpu-iou-v1/coarse-data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/v7-refit")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error("workers must be one or two to keep the full public cache in RAM")
    recipe = read_record(args.recipe)
    data, coarse, output = (p.resolve() for p in
                            (args.data_dir, args.coarse_data_dir, args.output_dir))
    paths = sorted(data.glob("DFL-MAT-*_half*.npz"))
    matches = sorted({p.name.rsplit("_half", 1)[0] for p in paths})
    prepared = verify_prepared(data, paths)
    if (matches != recipe["matches"] or len(paths) != 14
            or prepared["status"] != "verified_manifest"):
        raise ValueError("final refit requires the seven manifested public matches")
    prepare_coarse_data(data, coarse)
    sources = _source_hashes()
    sources["scripts/refit_v7.py"] = sha256_file(Path(__file__))
    plan = {"schema": "v7_full_public_refit_v1", "recipe": recipe,
            "recipe_sha256": sha256_file(args.recipe), "data_dir": str(data),
            "coarse_data_dir": str(coarse), "prepared": prepared,
            "folds": [{"train": matches, "validation": []}],
            "settings": recipe["settings"], "parent_ema": recipe["parent_ema"],
            "resources": {"device": "cuda:0"}, "source_sha256": sources,
            "environment": _execution_environment(), "target_inputs_used": False,
            "target_ground_truth_used": False}
    publish(output / "protocol.json", plan)
    jobs = [(name, seed) for seed in recipe["incumbent_seeds"]
            for name in [m["id"] for m in recipe["selection"]["members"]] + ["coarse"]]
    jobs += [("candidate", seed) for seed in recipe["candidate_seeds"]]
    results = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(train_job, plan, output, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            progress = {"completed": len(results), "total": len(jobs),
                        "last": [result["name"], result["base_seed"]],
                        "elapsed_seconds": time.time() - started}
            write_json(output / "status.json", progress)
            print(json.dumps(progress), flush=True)
    results.sort(key=lambda r: (r["name"], r["base_seed"]))
    publish(output / "training.json", {"recipe_sha256": sha256_file(args.recipe),
                                       "fits": results, "completed_unix": time.time()})


if __name__ == "__main__":
    main()
