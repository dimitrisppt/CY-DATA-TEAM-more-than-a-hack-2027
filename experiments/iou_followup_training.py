"""Frozen fresh-seed confirmation, factorial continuations, and WMX guard.

Adds adapters without changing any source pinned by the completed GPU study.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from experiments import campaign_training as campaign
from experiments.campaign_protocol import publish, read_record
from experiments.gpu_iou import (
    ROOT,
    _completed,
    _states,
    status,
    train_parent,
    train_refinement,
    verify_hash,
)
from experiments.gpu_runtime import accelerated_preprocessing, build_session, configure_device
from experiments.refinement_training import (
    RefinedTrainingSession,
    configure_refinement,
    export_refinement_checkpoint,
    load_refinement_state,
    save_refinement_state,
)
from experiments.robustness import _digest
from possession.storage import sha256_file
from possession.training_session import HalfCache

SOURCE = ROOT / "artifacts/gpu-iou-v1"
OUTPUT = ROOT / "artifacts/gpu-iou-followup-v1"
MODES = ("sampling_only", "loss_only")


class FactorialSession(RefinedTrainingSession):
    def _sample_start(self, frames, targets, features, transitions, size):
        if self._refinement_config["factorial_mode"] != "loss_only":
            return super()._sample_start(frames, targets, features, transitions, size)
        # Invoke the exact original sampler without the spell branch's extra
        # probability draw. This preserves the control's windows and RNG stream.
        self._refinement_counters["original_samples"] += 1
        self._refinement_config["variant"] = "control"
        try:
            return super()._sample_start(frames, targets, features, transitions, size)
        finally:
            self._refinement_config["variant"] = "spell_balanced"


def configure_factorial(session, mode, **settings):
    if mode not in MODES:
        raise ValueError("unknown factorial mode")
    settings = dict(settings)
    if mode == "sampling_only":
        settings["spell_weight"] = 0.
    else:
        settings["short_sampling_probability"] = 0.
    configure_refinement(session, variant="spell_balanced", **settings)
    session.__class__ = FactorialSession
    session._refinement_config["factorial_mode"] = mode
    session.config["refinement"] = copy.deepcopy(session._refinement_config)
    return session


def freeze(output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    previous = read_record(output / "protocol.json")
    if previous:
        for name, digest in previous["source_sha256"].items():
            verify_hash(ROOT / name, digest)
        for name, digest in previous["upstream_sha256"].items():
            verify_hash(SOURCE / name, digest)
        return previous
    original = read_record(SOURCE / "protocol.json")
    masks = read_record(SOURCE / "historical-masks/protocol.json")
    for name, digest in original["source_sha256"].items():
        verify_hash(ROOT / name, digest)
    common = {key: copy.deepcopy(original[key]) for key in (
        "data_dir", "target_dir", "selection", "coarse_spec", "settings", "parent_ema",
        "scenario_weights", "iou_thresholds", "official_iou_thresholds", "resources")}
    common["resources"].update(cache_bytes=1_300_000_000, workers=3)
    common["batches"] = masks["batches"]
    common["coarse_data_dir"] = str(SOURCE / "coarse-data")
    common["training_source_sha256"] = {
        **original["source_sha256"],
        "experiments/iou_followup_training.py": sha256_file(Path(__file__)),
    }
    inner = sorted({m for f in original["folds"] for m in f["validation"]})
    arms = {}
    for arm, seeds, folds, variants in (
        ("confirmation", [8201, 8202], original["folds"], ["control", "spell_balanced"]),
        ("ablation", original["seeds"], original["folds"],
         ["control", "spell_balanced", *MODES]),
        ("guard", [9201, 9202], [{"train": inner, "validation": ["DFL-MAT-J03WMX"]}],
         ["control", "spell_balanced"]),
    ):
        arms[arm] = {**copy.deepcopy(common), "arm": arm, "seeds": seeds, "folds": folds,
                     "variants": variants,
                     "input_root": str(SOURCE / "historical-masks/assessment")
                     if arm != "guard" else str(output / "guard/assessment")}
    plan = {
        "schema": "gpu_iou_followup_v1", "frozen_unix": time.time(), "arms": arms,
        "source_sha256": common["training_source_sha256"],
        "upstream_sha256": {name: sha256_file(SOURCE / name) for name in (
            "protocol.json", "training.json", "historical-masks/protocol.json",
            "historical-masks/reference/manifest.json", "historical-masks/reference/labels.npz")},
        "expected_new_fits": {"confirmation": 42, "ablation": 24, "guard": 14},
        "expected_new_updates": 240000,
        "prespecified_analysis": {
            "candidate": "spell_balanced; fixed before any new results; never replace with ablation winner",
            "confirmation": "same comparison_gate against raw-parent EMA baseline and control: gain > .002, "
                            "missing gain > 0, wins >= 4/6, both batches positive, each scenario >= -.01",
            "guard": "WMX candidate versus paired ensemble control and parent baseline: "
                     "objective >= reference-.01 and every scenario >= reference-.01; "
                     "also report both batches and individual fine seeds; no single-match CI",
            "factorial": "sampling_only-control, loss_only-control, combined-control, "
                         "interaction=combined-sampling_only-loss_only+control; exploratory; no retuning",
            "sampling_pairing": "loss_only matches control windows; sampling_only matches combined windows",
            "bootstrap": {"unit": "whole match", "draws": 5000, "seed": 20261116,
                          "percentiles": [.025, .975], "interpretation": "descriptive; six reused matches"},
            "audit": "original ensemble, clean batch_1, official IoU .5; all six matches; "
                     "same-team overlap degree >1 flags split/merge; dominant opponent overlap flags "
                     "wrong-team; unmatched same-team overlap flags timing; otherwise phantom; "
                     "exclusive precedence wrong-team, split, merge, timing, phantom; flags retained; "
                     "WOH timeline uses largest increase in spurious events in fixed 60-second bins",
            "independent_test": False, "automatic_promotion": False,
            "full_recipe_seed_confirmation": "fresh fine and coarse models, same masks and decoder",
            "seed_mapping": "existing _seed_for_fold(base_seed, WMX, fold_index), including guard fold 0",
        },
    }
    return publish(output / "protocol.json", plan)


def train_factorial(plan, output, parent_dir, parent_fit, mode, spec, cache):
    parent = parent_fit["identity"]
    identity = {**parent, "role": mode, "followup_protocol_sha256": _digest(plan),
                "parent_state_sha256": parent_fit["files"]["state_007200.pt"],
                "teacher_sha256": None}
    directory = output / "fits" / mode / f"fold_{parent['fold']}" / f"seed_{parent['base_seed']}" / parent["member"]
    fit = _completed(directory, identity)
    if fit:
        return directory, fit
    started = time.monotonic()
    session = build_session(cache.data_dir, spec, parent["train_matches"], parent["seed"],
                            cache, device=plan["resources"]["device"])
    campaign.load_training_state(parent_dir / "state_007200.pt", session, parent)
    configure_factorial(session, mode, **plan["settings"])
    states = _states(directory, 8400)
    if states:
        load_refinement_state(states[-1], session, identity)
    while session.steps < 8400:
        status(output, "training", fold=parent["fold"], seed=parent["base_seed"],
               role=mode, member=parent["member"], step=session.steps)
        session.advance_to(min(8400, session.steps + 200))
        save_refinement_state(directory / f"state_{session.steps:06d}.pt", session, identity)
    export_refinement_checkpoint(directory / "model.pt", session, identity)
    fit = publish(directory / "fit.json", {
        "identity": identity, "steps": 8400,
        "files": {name: sha256_file(directory / name) for name in ("model.pt", "state_008400.pt")},
        "progress": session._progress(), "seconds_this_invocation": time.monotonic() - started})
    del session
    gc.collect()
    return directory, fit


def train_fold(plan, output, fold):
    output = Path(output)
    configure_device(plan["resources"]["device"])
    fine = HalfCache(Path(plan["data_dir"]), plan["resources"]["cache_bytes"])
    coarse = HalfCache(Path(plan["coarse_data_dir"]), 200_000_000)
    fits = []
    old = read_record(SOURCE / "training.json")["fits"] if plan["arm"] == "ablation" else []
    with accelerated_preprocessing():
        for seed in plan["seeds"]:
            if not old:
                directory, fit = train_parent(plan, output, fold, seed, "coarse_player_only",
                                               plan["coarse_spec"], coarse, coarse=True)
                fits.append({"directory": str(directory), **fit})
            for member in plan["selection"]["members"]:
                if old:
                    row = next(r for r in old if r["identity"]["fold"] == fold
                               and r["identity"]["base_seed"] == seed
                               and r["identity"]["member"] == member["id"]
                               and r["identity"]["role"] == "parent")
                    directory = Path(row["directory"])
                    fit = _completed(directory, row["identity"])
                    variants = MODES
                else:
                    directory, fit = train_parent(plan, output, fold, seed, member["id"], member["spec"], fine)
                    fits.append({"directory": str(directory), **fit})
                    variants = plan["variants"]
                for variant in variants:
                    train = train_factorial if old else train_refinement
                    child_dir, child = train(plan, output, directory, fit, variant, member["spec"], fine)
                    fits.append({"directory": str(child_dir), **child})
    publish(output / f"training-fold{fold}.json", {"protocol_sha256": _digest(plan), "fits": fits})
    status(output, "training_complete", fold=fold, completed_fits=len(fits))
    return plan["arm"], fits


def train_all(plan, output):
    collected = {arm: [] for arm in plan["arms"]}
    # Fix confirmation first. Guard training uses no outcomes; scoring is last.
    jobs = [(arm, index) for arm in ("confirmation", "ablation", "guard")
            for index in range(len(plan["arms"][arm]["folds"]))]
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(train_fold, plan["arms"][arm], str(output / arm), index)
                   for arm, index in jobs]
        for future in as_completed(futures):
            arm, fits = future.result()
            collected[arm].extend(fits)
            status(output, "training", completed_fits=sum(map(len, collected.values())), total_fits=80)
    for arm, fits in collected.items():
        if len(fits) != plan["expected_new_fits"][arm]:
            raise ValueError("incomplete new training grid")
        fits.sort(key=lambda r: (r["identity"]["fold"], r["identity"]["base_seed"],
                                 r["identity"]["member"], r["identity"]["role"]))
        publish(output / arm / "training.json", {"protocol_sha256": _digest(plan["arms"][arm]), "fits": fits})
    status(output, "training_complete", completed_fits=80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=("plan", "train"), default="train")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configure_device("cuda")
        plan = freeze(output)
        if args.stage == "train":
            train_all(plan, output)


if __name__ == "__main__":
    main()
