"""Package completed full-data fits and compact public validation evidence for v7."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from experiments.campaign_protocol import publish, read_record
from experiments.robustness import _source_hashes
from possession.storage import sha256_file
from scripts.refit_v7 import audit_fit
from scripts.v7_bundle import ROOT, verify_bundle


def copy_checked(source, destination, digest=None):
    source, destination = Path(source), Path(destination)
    digest = digest or sha256_file(source)
    if sha256_file(source) != digest:
        raise ValueError(f"source artifact changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != digest:
            raise ValueError(f"refusing to replace differing artifact: {destination}")
    else:
        shutil.copyfile(source, destination)
    if sha256_file(destination) != digest:
        raise ValueError(f"copy differs: {destination}")


def export(refit, output, benchmark):
    refit, output, benchmark = (Path(p).resolve() for p in (refit, output, benchmark))
    plan = read_record(refit / "protocol.json")
    training = read_record(refit / "training.json")
    recipe = plan["recipe"]
    if training["recipe_sha256"] != plan["recipe_sha256"] or len(training["fits"]) != 8:
        raise ValueError("incomplete final training or recipe mismatch")
    for name, digest in plan["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"training source changed: {name}")
    copy_checked(ROOT / "configs/v7-recipe.json", output / "recipe.json", plan["recipe_sha256"])
    study = ROOT / "artifacts/gpu-tuning-8h-v1"
    followup = ROOT / "artifacts/gpu-iou-followup-v1/protocol.json"
    copy_checked(followup, output / "evidence/followup-protocol.json",
                 recipe["followup_protocol_sha256"])
    original = read_record(followup)["arms"]["confirmation"]
    for key in ("selection", "coarse_spec", "settings", "parent_ema"):
        if recipe[key] != original[key]:
            raise ValueError(f"release incumbent differs from the confirmed recipe: {key}")
    for name, digest in recipe["evidence"].items():
        copy_checked(study / name, output / "evidence" / name, digest)
    selection = read_record(output / "evidence/selection.json")
    if (selection["pipeline"] != recipe["pipeline"]
            or selection["proposal"]["recipe"] != recipe["candidate_recipe"]):
        raise ValueError("release recipe differs from the selected tuning winner")
    for scope in ("confirmation", "guard"):
        if not read_record(output / "evidence" / f"{scope}.json")["comparison"]["passed"]:
            raise ValueError("selected candidate failed its confirmation gate")
    copy_checked(refit / "protocol.json", output / "training/protocol.json")
    copy_checked(refit / "training.json", output / "training/results.json")
    models = []
    for fitted in training["fits"]:
        audit = audit_fit(fitted["checkpoint"], fitted["state"], recipe["matches"],
                          fitted["audit"]["steps"])
        if audit != fitted["audit"]:
            raise ValueError("training receipt audit differs")
        name = f"weights/{fitted['name']}-{fitted['base_seed']}.pt"
        copy_checked(fitted["checkpoint"], output / name, audit["model_sha256"])
        state = torch.load(fitted["state"], map_location="cpu", weights_only=True)
        transcript = state.get("campaign", state)["state"]["training_transcript"]
        transcript_name = f"training/windows/{fitted['name']}-{fitted['base_seed']}.txt"
        transcript_path = output / transcript_name
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        if transcript_path.exists() and transcript_path.read_bytes() != transcript:
            raise ValueError("existing training transcript differs")
        transcript_path.write_bytes(transcript)
        models.append({"name": fitted["name"], "base_seed": fitted["base_seed"],
                       "path": name, "steps": audit["steps"],
                       "transcript_path": transcript_name,
                       "training_window_sha256": audit["training_window_sha256"]})
    source = _source_hashes()
    for name in ("scripts/refit_v7.py", "scripts/v7_bundle.py", "scripts/reproduce_v7.py",
                 "scripts/reproduce.py", "scripts/export_v7.py", "scripts/evaluate_v7.py",
                 "scripts/package_v7.py"):
        source[name] = sha256_file(ROOT / name)
    files = {p.relative_to(output).as_posix(): sha256_file(p)
             for p in sorted(output.rglob("*")) if p.is_file() and p != output / "manifest.json"}
    manifest = {"schema": "gpu_tuned_submission_v1", "version": "v7", "models": models,
                "pipeline": recipe["pipeline"], "files": files, "source_sha256": source,
                "coordinator_environment": plan["environment"],
                "training_workers": {"torch_num_threads": 1, "device": "cuda:0",
                                     "deterministic": True, "precision": "float32",
                                     "tf32": False},
                "target_ground_truth_used": False, "all_public_matches_refit": True}
    publish(output / "manifest.json", manifest)
    verify_bundle(output)
    rows = []
    for scope in ("confirmation", "guard"):
        for batch in ("batch_1", "batch_2"):
            for variant in ("incumbent", "ensemble"):
                relative = Path(scope) / batch / variant
                result = read_record(study / relative / "result.json")
                copy_checked(study / relative / "labels.npz", benchmark / relative / "labels.npz",
                             result["labels_sha256"])
                copy_checked(study / relative / "result.json", benchmark / relative / "result.json")
                rows.append({"scope": scope, "batch": batch, "variant": variant,
                             "directory": relative.as_posix()})
    files = {p.relative_to(benchmark).as_posix(): sha256_file(p)
             for p in sorted(benchmark.rglob("*"))
             if p.is_file() and p != benchmark / "manifest.json"}
    expected = {scope: {name: read_record(study / f"{scope}.json")[name]
                       for name in ("candidate", "incumbent")}
                for scope in ("confirmation", "guard")}
    publish(benchmark / "manifest.json", {
        "schema": "v7_public_validation_reference_v1", "records": rows, "files": files,
        "expected": expected, "independent_test": False,
        "note": "OOF development recipe evidence; final full-data weights have no target score."})
    return {"model_count": len(models), "benchmark_records": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit", type=Path, default=ROOT / "artifacts/v7-refit")
    parser.add_argument("--output", type=Path, default=ROOT / "models/v7")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "benchmarks/v7")
    args = parser.parse_args()
    print(export(args.refit, args.output, args.benchmark))


if __name__ == "__main__":
    main()
