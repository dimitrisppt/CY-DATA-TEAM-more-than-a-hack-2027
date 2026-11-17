"""Fresh, resumable CUDA reconstruction of the full possession refinement study.

Run with ``python -m experiments.gpu_iou``. The historical CPU run is not resumed:
all 18 fold-specific parents and 36 continuations are rebuilt from public data.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path

import torch

from experiments import campaign_training as campaign
from experiments.campaign_protocol import WEIGHTS, publish, read_record, scenarios
from experiments.coarse_context import prepare_coarse_data
from experiments.cross_validation import nested_splits
from experiments.gpu_runtime import accelerated_preprocessing, build_session, configure_device
from experiments.possession_refinement import CAVEAT, SETTINGS, VARIANTS, validate_folds
from experiments.refinement_training import (
    _atomic_torch_save,
    configure_refinement,
    export_refinement_checkpoint,
    load_refinement_state,
    save_refinement_state,
)
from experiments.robustness import _digest, _execution_environment, _source_hashes
from experiments.tune import _seed_for_fold
from possession.predict import load_model_checkpoint
from possession.prepared import verify_prepared
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (8101, 8102)


def status(output, stage, **detail):
    value = {"status": "running", "stage": stage, "pid": os.getpid(),
             "updated_unix": time.time(), **detail}
    filename = f"status-fold{detail['fold']}.json" if "fold" in detail else "status.json"
    write_json(Path(output) / filename, value)
    print(json.dumps(value), flush=True)


def verify_hash(path, digest):
    if not Path(path).is_file() or sha256_file(path) != digest:
        raise ValueError(f"artifact missing or hash differs: {path}")


def protocol(data, target, output, device, cache_bytes, workers=3):
    release_path = ROOT / "models/v6/manifest.json"
    release = read_record(release_path)
    paths = sorted(data.glob("DFL-MAT-*_half*.npz"))
    prepared = verify_prepared(data, paths)
    matches = sorted({p.name.rsplit("_half", 1)[0] for p in paths})
    if len(matches) != 7 or len(paths) != 14 or prepared["status"] != "verified_manifest":
        raise ValueError("all seven manifested public DFL matches are required")
    fold = next(f for f in nested_splits(matches, 3, 0)
                if f["outer_test"] == ["DFL-MAT-J03WMX"])
    validate_folds(fold["inner"])
    current_source = _source_hashes()
    # The released recipe is read as configuration; its full-data weights are
    # never used for development assessment.
    for name, digest in release["source_sha256"].items():
        if current_source.get(name) != digest:
            raise ValueError(f"source-pinned release dependency differs: {name}")
    existing = read_record(output / "protocol.json")
    source = existing["source_sha256"] if existing else current_source
    if any(current_source.get(name) != digest for name, digest in source.items()):
        raise ValueError("training source changed since the GPU protocol was frozen")
    plan = {
        "schema": "fresh_gpu_possession_iou_v1", "development_only": True,
        "historical_reuse_caveat": CAVEAT,
        "historical_checkpoint_resume": False, "target_ground_truth_used": False,
        "release_manifest_sha256": sha256_file(release_path),
        "data_dir": str(data), "target_dir": str(target),
        "target_manifest_sha256": sha256_file(target / "input-manifest.json"),
        "prepared": prepared, "folds": fold["inner"], "guard": fold["outer_test"],
        "selection": {key: release["candidate_selection"][key]
                      for key in ("kind", "members", "blend_weight", "decoder")},
        "coarse_spec": release["coarse_spec"], "seeds": list(SEEDS),
        "variants": list(VARIANTS), "settings": SETTINGS,
        "parent_steps": 7200, "parent_ema": {"start": 7000, "decay": .99, "updates": 200},
        "coarse_steps": 7200, "continuation_steps": 1200,
        "batches": {"batch_1": scenarios(False), "batch_2": scenarios(True)},
        "scenario_weights": WEIGHTS, "iou_thresholds": [i / 10 for i in range(1, 10)],
        "official_iou_thresholds": [.3, .4, .5, .6, .7],
        "automatic_promotion": False, "source_sha256": source,
        "environment": _execution_environment(),
        "resources": {"device": device, "cache_bytes": cache_bytes, "threads": 1,
                      "workers": workers,
                      "deterministic": True, "precision": "float32", "tf32": False,
                      "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
                      "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
    }
    return publish(output / "protocol.json", plan)


def _identity(plan, fold_index, seed, member, role):
    fold = plan["folds"][fold_index]
    return {"protocol_sha256": _digest(plan), "fold": fold_index,
            "base_seed": seed, "seed": _seed_for_fold(seed, "DFL-MAT-J03WMX", fold_index),
            "member": member, "role": role, "train_matches": fold["train"],
            "validation_matches": fold["validation"]}


def _completed(directory, identity):
    fit = read_record(directory / "fit.json")
    if fit:
        if fit["identity"] != identity:
            raise ValueError("completed fit identity differs")
        for name, digest in fit["files"].items():
            verify_hash(directory / name, digest)
    return fit


def _states(directory, limit):
    return sorted(p for p in directory.glob("state_*.pt")
                  if int(p.stem.split("_")[-1]) <= limit
                  and p.with_suffix(".pt.sha256.json").is_file())


def train_parent(plan, output, fold_index, seed, member, spec, cache, *, coarse=False):
    role = "coarse" if coarse else "parent"
    identity = _identity(plan, fold_index, seed, member, role)
    directory = output / "fits" / role / f"fold_{fold_index}" / f"seed_{seed}" / member
    fit = _completed(directory, identity)
    if fit:
        return directory, fit
    started = time.monotonic()
    session = build_session(cache.data_dir, spec, identity["train_matches"], identity["seed"],
                            cache, device=plan["resources"]["device"])
    # A fine parent's incomplete last transaction is replayed from 7000 so its
    # online EMA can never become detached from the authenticated raw trajectory.
    states = _states(directory, 7200 if coarse else 7000)
    if states:
        campaign.load_training_state(states[-1], session, identity)
    ema = {}

    def update_ema(optimizer, args, kwargs, *, session=session, ema=ema):
        step = session.steps + 1
        if step == 7000:
            ema.update({k: v.detach().clone() for k, v in session.model.state_dict().items()})
        elif step > 7000:
            with torch.no_grad():
                for name, value in session.model.state_dict().items():
                    if value.is_floating_point():
                        ema[name].mul_(.99).add_(value.detach(), alpha=.01)
                    else:
                        ema[name].copy_(value)

    if not coarse:
        if session.steps == 7000:
            ema.update({k: v.detach().clone() for k, v in session.model.state_dict().items()})
        ema_hook = session.optimizer.register_step_post_hook(update_ema)
    while session.steps < 7200:
        status(output, "training", role=role, fold=fold_index, seed=seed,
               member=member, step=session.steps, end_step=7200)
        session.advance_to(min(7200, session.steps + 200))
        campaign.save_training_state(directory / f"state_{session.steps:06d}.pt", session, identity)
    raw = directory / "model.pt"
    session.save_checkpoint(raw, validation_matches=identity["validation_matches"],
                            extra_training_config={"gpu_study_identity": identity})
    names = ["model.pt", "state_007200.pt"]
    if not coarse:
        payload = torch.load(raw, map_location="cpu", weights_only=True)
        payload["state_dict"] = {k: v.detach().cpu().clone() for k, v in ema.items()}
        payload["ema_metadata"] = plan["parent_ema"]
        _atomic_torch_save(directory / "ema.pt", payload)
        names.append("ema.pt")
        ema_hook.remove()
    fit = publish(directory / "fit.json", {
        "identity": identity, "steps": 7200,
        "files": {name: sha256_file(directory / name) for name in names},
        "progress": session._progress(), "seconds_this_invocation": time.monotonic() - started})
    del session, ema
    gc.collect()
    return directory, fit


def train_refinement(plan, output, parent_dir, parent_fit, variant, spec, cache):
    parent = parent_fit["identity"]
    identity = {**parent, "role": variant,
                "parent_state_sha256": parent_fit["files"]["state_007200.pt"],
                "teacher_sha256": parent_fit["files"]["ema.pt"] if variant == "masked_teacher" else None}
    directory = (output / "fits" / variant / f"fold_{parent['fold']}"
                 / f"seed_{parent['base_seed']}" / parent["member"])
    fit = _completed(directory, identity)
    if fit:
        return directory, fit
    started = time.monotonic()
    device = plan["resources"]["device"]
    session = build_session(cache.data_dir, spec, parent["train_matches"], parent["seed"],
                            cache, device=device)
    campaign.load_training_state(parent_dir / "state_007200.pt", session, parent)
    teacher = None
    if variant == "masked_teacher":
        verify_hash(parent_dir / "ema.pt", identity["teacher_sha256"])
        teacher, _ = load_model_checkpoint(parent_dir / "ema.pt", device=device)
    configure_refinement(session, variant=variant, teacher=teacher, **plan["settings"])
    states = _states(directory, 8400)
    if states:
        load_refinement_state(states[-1], session, identity)
    while session.steps < 8400:
        status(output, "training", role=variant, fold=parent["fold"],
               seed=parent["base_seed"], member=parent["member"],
               step=session.steps, end_step=8400)
        session.advance_to(min(8400, session.steps + 200))
        save_refinement_state(directory / f"state_{session.steps:06d}.pt", session, identity)
    export_refinement_checkpoint(directory / "model.pt", session, identity)
    fit = publish(directory / "fit.json", {
        "identity": identity, "steps": 8400,
        "files": {name: sha256_file(directory / name)
                  for name in ("model.pt", "state_008400.pt")},
        "progress": session._progress(), "seconds_this_invocation": time.monotonic() - started})
    del session, teacher
    gc.collect()
    return directory, fit


def train_fold(plan, output, index):
    configure_device(plan["resources"]["device"])
    with accelerated_preprocessing():
        return _train_fold(plan, Path(output), index)


def _train_fold(plan, output, index):
    data = Path(plan["data_dir"])
    coarse_data = output / "coarse-data"
    fine_cache = HalfCache(data, plan["resources"]["cache_bytes"])
    coarse_cache = HalfCache(coarse_data, 200_000_000)
    fits = []
    for seed in plan["seeds"]:
        directory, fit = train_parent(plan, output, index, seed, "coarse_player_only",
                                       plan["coarse_spec"], coarse_cache, coarse=True)
        fits.append({"directory": str(directory), **fit})
        for member in plan["selection"]["members"]:
            directory, fit = train_parent(plan, output, index, seed, member["id"],
                                           member["spec"], fine_cache)
            fits.append({"directory": str(directory), **fit})
            for variant in plan["variants"]:
                child_dir, child = train_refinement(
                    plan, output, directory, fit, variant, member["spec"], fine_cache)
                fits.append({"directory": str(child_dir), **child})
    publish(output / f"training-fold{index}.json", {"protocol_sha256": _digest(plan), "fits": fits})
    status(output, "training_complete", fold=index, completed_fits=len(fits))
    return fits


def train_all(plan, output):
    status(output, "preparing_coarse")
    prepare_coarse_data(Path(plan["data_dir"]), output / "coarse-data")
    fits = []
    with ProcessPoolExecutor(max_workers=plan["resources"]["workers"],
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(train_fold, plan, str(output), i) for i in range(3)}
        while pending:
            done, pending = wait(pending, timeout=30)
            for future in done:
                fits.extend(future.result())
            status(output, "training", completed_fits=len(list((output / "fits").rglob("fit.json"))),
                   total_fits=54, active_folds=len(pending))
    fits.sort(key=lambda row: (row["identity"]["fold"], row["identity"]["base_seed"],
                              row["identity"]["member"], row["identity"]["role"]))
    if len(fits) != 54:
        raise AssertionError("complete training requires 18 parents and 36 continuations")
    publish(output / "training.json", {"protocol_sha256": _digest(plan), "fits": fits})
    status(output, "training_complete", completed_fits=len(fits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/dfl")
    parser.add_argument("--target-dir", type=Path, default=ROOT / "data/target/2027")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/gpu-iou-v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-bytes", type=int, default=3_000_000_000)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--stage", choices=("all", "train", "evaluate", "plan"), default="all")
    args = parser.parse_args()
    if args.cache_bytes < 1:
        parser.error("cache-bytes must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            device = configure_device(args.device)
            plan = protocol(args.data_dir.resolve(), args.target_dir.resolve(),
                            output, device, args.cache_bytes, args.workers)
            if args.stage == "plan":
                status(output, "planned", total_fits=54)
                return
            if args.stage in ("train", "all"):
                with accelerated_preprocessing():
                    train_all(plan, output)
            if args.stage in ("evaluate", "all"):
                from experiments.gpu_iou_evaluation import evaluate
                evaluate(plan, output)
        except BaseException as exc:
            status(output, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                   error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
