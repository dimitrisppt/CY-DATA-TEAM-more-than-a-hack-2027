"""Precompute historical short/long-mask CUDA predictions with the frozen evaluator.

This performs no scoring. It reuses the evaluator's inference and authenticated
cache format; clean/replay inputs are handled by reuse_iou_predictions instead.
"""

import argparse
import fcntl
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments import campaign_evaluation as inputs
from experiments.campaign_protocol import read_record
from experiments.gpu_iou import ROOT, verify_hash
from experiments.gpu_iou_evaluation import VARIANTS, _predict
from experiments.gpu_runtime import configure_device
from experiments.robustness import _digest
from possession.predict import load_model_checkpoint
from possession.schema import FeatureBatch
from possession.storage import sha256_file, write_json


def prefetch_fold(study, fold):
    study = Path(study)
    plan = read_record(study / "protocol.json")
    evaluation = read_record(study / "evaluation-protocol.json")
    for name, digest in evaluation["source_sha256"].items():
        verify_hash(ROOT / name, digest)
    audit = read_record(study / "training-audit.json")
    training = read_record(study / "training.json")
    if (not audit["all_54_fits_verified"] or audit["protocol_sha256"] != _digest(plan)
            or training["protocol_sha256"] != _digest(plan) or len(training["fits"]) != 54):
        raise ValueError("complete verified training is required before prefetch")
    device = plan["resources"]["device"]
    configure_device(device)
    output = study / "historical-masks"
    rows = []
    for path in sorted((output / "assessment").glob("*/input-manifest.json")):
        expected = json.loads(path.with_suffix(".sha256.json").read_text())["sha256"]
        verify_hash(path, expected)
        manifest = json.loads(path.read_text())
        if manifest["status"] != "complete":
            raise ValueError("finish historical input preparation before prefetch")
        rows.extend({"batch": path.parent.name, **row,
                     "input_path": str(inputs._input_path(path.parent, row["match"],
                                                          row["period"], row["scenario"]))}
                    for row in manifest["records"]
                    if row["match"] in plan["folds"][fold]["validation"]
                    and row["scenario"] in ("short", "long"))
    if len(rows) != 16:
        raise ValueError("expected both outage scenarios, batches, matches and halves")
    models = {}
    for fit in training["fits"]:
        identity = fit["identity"]
        if identity["fold"] != fold or identity["role"] == "coarse":
            continue
        filename = "ema.pt" if identity["role"] == "parent" else "model.pt"
        path = Path(fit["directory"]) / filename
        verify_hash(path, fit["files"][filename])
        model, payload = load_model_checkpoint(path, device=device)
        if set(payload["training_config"]["train_matches"]) & set(plan["folds"][fold]["validation"]):
            raise ValueError("assessment match in model training provenance")
        models[identity["base_seed"], identity["member"], identity["role"]] = (
            model, {**fit, "checkpoint_sha256": fit["files"][filename]})
    if len(models) != 16:
        raise ValueError("expected 16 fine checkpoints per fold")
    for i, source in enumerate(rows):
        arrays = inputs.load_arrays(source)
        features = FeatureBatch(*(arrays[k] for k in
                                  ("player_tokens", "player_mask", "global_tokens", "time_valid")))
        for variant in VARIANTS:
            role = "parent" if variant == "gpu_baseline" else variant
            for seed in plan["seeds"]:
                for member in plan["selection"]["members"]:
                    model, fit = models[seed, member["id"], role]
                    path = (output / "inference" / source["batch"] / variant / f"seed_{seed}"
                            / member["id"] / source["match"] / f"half{source['period']}"
                            / f"{source['scenario']}.npz")
                    _predict(arrays, features, model, source, fit, path, device)
        progress = {"fold": fold, "completed_rows": i + 1, "total_rows": len(rows),
                    "device": device, "code_sha256": sha256_file(__file__),
                    "evaluator_sha256": evaluation["source_sha256"]["experiments/gpu_iou_evaluation.py"],
                    "status": "complete" if i + 1 == len(rows) else "running"}
        write_json(output / f"prefetch-fold{fold}.json", progress)
        print(json.dumps(progress), flush=True)
    return {"fold": fold, "completed_rows": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("artifacts/gpu-iou-v1"))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    study = args.study_dir.resolve()
    with (study / "historical-masks/.prefetch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(prefetch_fold, study, fold) for fold in range(3)]
            print(json.dumps({"complete": [future.result() for future in futures]}), flush=True)


if __name__ == "__main__":
    main()
