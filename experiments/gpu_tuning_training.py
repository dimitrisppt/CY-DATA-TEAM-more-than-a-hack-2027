"""Resumable GPU training; existing source-pinned experiments remain untouched."""
from __future__ import annotations

import gc
from pathlib import Path

import torch

from experiments import campaign_training as campaign
from experiments.campaign_protocol import publish, read_record
from experiments.gpu_runtime import build_session
from experiments.refinement_training import (
    configure_refinement,
    load_refinement_state,
    save_refinement_state,
)
from possession.storage import sha256_file


def make_session(data, recipe, matches, seed, cache=None, device="cuda:0"):
    spec = recipe["spec"]
    session = build_session(data, spec, matches, seed, cache, device=device)
    variant = ("control" if recipe["spell_weight"] == 0
               and recipe["short_sampling_probability"] == 0 else "spell_balanced")
    configure_refinement(session, variant=variant, start_steps=0,
                         total_extra_steps=spec["schedule"]["total_steps"], lr_scale=1.,
                         spell_weight=recipe["spell_weight"],
                         short_sampling_probability=recipe["short_sampling_probability"])
    # The original continuation adapter deliberately uses constant LR. This new
    # from-scratch study restores the sampled campaign schedule explicitly.
    session._refinement_lr_hook.remove()
    session._campaign_schedule_hook = campaign._schedule_hook(session, spec)
    return session


def check_file(path, digest):
    if not Path(path).is_file() or sha256_file(path) != digest:
        raise ValueError(f"missing or changed artifact: {path}")


def fit(data, recipe, matches, validation, seed, steps, directory, identity, cache, tick):
    directory = Path(directory)
    if set(matches) & set(validation):
        raise ValueError("training and assessment matches overlap")
    identity = {**identity, "recipe": recipe, "train_matches": matches,
                "validation_matches": validation, "seed": seed}
    destination = directory / f"rung_{steps}"
    saved = read_record(destination / "fit.json")
    if saved:
        if saved["identity"] != identity or saved["steps"] != steps:
            raise ValueError("completed fit identity differs")
        check_file(destination / "model.pt", saved["model_sha256"])
        return saved
    tick()
    session = make_session(data, recipe, matches, seed, cache)
    states = sorted(p for p in directory.glob("state_*.pt")
                    if p.with_suffix(".pt.sha256.json").exists())
    state = states[-1] if states else None
    if states:
        load_refinement_state(states[-1], session, identity)
    if session.steps > steps:
        raise ValueError("earlier rung checkpoint is missing")
    while session.steps < steps:
        tick(steps=session.steps, target_steps=steps)
        session.advance_to(min(steps, session.steps + 200))
        state = directory / f"state_{session.steps:06d}.pt"
        save_refinement_state(state, session, identity)
        # Only files owned by this new fit are removed; two transactional
        # generations survive while training, one after a completed rung.
        for obsolete in sorted(directory.glob("state_*.pt"))[:-2]:
            obsolete.unlink()
            obsolete.with_suffix(".pt.sha256.json").unlink(missing_ok=True)
    lines = session._training_window_hash.transcript.decode().splitlines()
    used = {line.split(":", 1)[0].rsplit("_half", 1)[0] for line in lines}
    if len(lines) != steps or not used <= set(matches):
        raise ValueError("actual training transcript violates split or update budget")
    use_ema = steps == recipe["spec"]["schedule"]["total_steps"] and recipe["export"] == "ema"
    if use_ema:
        if session._tail_ema_updates != min(200, steps):
            raise ValueError("tail EMA update budget differs")
        session.model.load_state_dict(session._tail_ema, strict=True)
    if any(not torch.isfinite(value).all() for value in session.model.state_dict().values()):
        raise ValueError("nonfinite model weights")
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.pt"
    session.save_checkpoint(model_path, validation_matches=validation,
                            extra_training_config={"tuning_identity": identity,
                                                   "tuning_export": "ema" if use_ema else "raw"})
    result = publish(destination / "fit.json", {
        "identity": identity, "steps": steps, "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "state_path": str(state), "state_sha256": sha256_file(state),
        "parameters": sum(p.numel() for p in session.model.parameters()),
        "actual_train_matches": sorted(used),
        "training_window_sha256": session._training_window_hash.hexdigest(),
        "export": "ema" if use_ema else "raw", "progress": session._progress()})
    for obsolete in sorted(directory.glob("state_*.pt"))[:-1]:
        obsolete.unlink()
        obsolete.with_suffix(".pt.sha256.json").unlink(missing_ok=True)
    del session
    gc.collect()
    return result
