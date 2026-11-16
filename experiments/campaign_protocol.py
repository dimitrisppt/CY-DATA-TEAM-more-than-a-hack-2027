"""Frozen development protocol, wall-clock budget, and paired promotion gates."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from experiments.cross_validation import nested_splits
from experiments.robustness import _digest, _execution_environment, _source_hashes
from possession.prepared import verify_prepared
from possession.storage import sha256_file, write_json

POOLINGS = ("mean", "ball_attention", "interaction")
WEIGHTS = {"clean": .5, "short": .2, "long": .15, "replay": .15}
SEARCH_SEED = 7100
CONFIRMATION_SEEDS = (8101, 8102)
OUTER_SEEDS = (9101, 9102)


def scenarios(confirmation=False):
    return {
        "clean": {"kind": "clean"},
        "short": {"kind": "synthetic", "seed": 73102 if confirmation else 73101,
                  "missing_fraction": .41, "min_seconds": 1., "max_seconds": 10.},
        "long": {"kind": "synthetic", "seed": 74102 if confirmation else 74101,
                 "missing_fraction": .41, "min_seconds": 10., "max_seconds": 30.},
        "replay": {"kind": "target_replay", "seed": 5102 if confirmation else 5101},
    }


def read_record(path):
    path = Path(path)
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if set(value) != {"sha256", "payload"} or value["sha256"] != _digest(value["payload"]):
        raise ValueError(f"campaign record hash differs: {path}")
    return value["payload"]


def publish(path, payload):
    payload = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    previous = read_record(path)
    if previous is not None:
        if previous != payload:
            raise ValueError(f"completed campaign record differs: {path}")
        return previous
    write_json(path, {"sha256": _digest(payload), "payload": payload})
    return payload


def make_protocol(data_dir, target_dir, baseline_path, *, hours=12., per_family=12,
                  decoder_trials=(12, 32, 64), rungs=(1200, 3600, 7200), cache_bytes=1_000_000_000):
    if not np.isfinite(hours) or not 0 < hours <= 12:
        raise ValueError("local campaign must have a positive budget of at most 12 hours")
    if not isinstance(per_family, int) or not 6 <= per_family <= 24:
        raise ValueError("per_family must be between 6 and 24")
    if len(rungs) != 3 or tuple(sorted(set(rungs))) != tuple(rungs) or rungs[0] < 1:
        raise ValueError("three strictly increasing positive rungs are required")
    if len(decoder_trials) != 3 or any(x < 3 for x in decoder_trials):
        raise ValueError("each rung needs at least three decoder anchors")
    data_dir, target_dir = Path(data_dir).resolve(), Path(target_dir).resolve()
    paths = sorted(data_dir.glob("DFL-MAT-*_half*.npz"))
    provenance = verify_prepared(data_dir, paths)
    if provenance["status"] != "verified_manifest":
        raise ValueError("the overnight campaign requires manifested public data")
    matches = sorted({path.name.rsplit("_half", 1)[0] for path in paths})
    if len(matches) != 7 or len(paths) != 14:
        raise ValueError("this protocol is defined for the seven public matches")
    fold = next(fold for fold in nested_splits(matches, inner_folds=3, seed=0)
                if fold["outer_test"] == ["DFL-MAT-J03WMX"])
    target_manifest = target_dir / "input-manifest.json"
    return {
        "version": "overnight_joint_v1_development", "development_only": True,
        "interpretation": "All seven matches were used historically; no independent test claim.",
        "fold": fold, "prepared": provenance,
        "baseline_sha256": sha256_file(baseline_path),
        "target_input_manifest_sha256": sha256_file(target_manifest),
        "source_sha256": _source_hashes(), "environment": _execution_environment(),
        "resources": {"hours": float(hours), "per_family": per_family,
                      "rungs": list(rungs), "decoder_trials": list(decoder_trials),
                      "cache_bytes": cache_bytes, "workers": 1, "cpu_threads": 4,
                      "screen_cutoff_fraction": .45, "promote_cutoff_fraction": .65,
                      "refine_cutoff_fraction": .75, "confirmation_reserve_fraction": .25},
        "poolings": list(POOLINGS), "search_seed": SEARCH_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS), "outer_seeds": list(OUTER_SEEDS),
        "search_scenarios": scenarios(), "confirmation_scenarios": scenarios(True),
        "scenario_weights": WEIGHTS,
        "selection": {"objective": "mean_match(.5 clean + .2 short + .15 long + .15 replay) spell F1",
                      "promotion": "middle: protect unaugmented anchor plus best two others per architecture; final: best one each",
                      "rung_selection": "best complete middle or final configuration; baseline always eligible",
                      "paired_training_seeds": True, "complete_rounds_only_for_screening": True,
                      "complete_rounds_only_for_promotion": True,
                      "phase_cutoff": "close stage permanently at its deadline; exclude partial architecture rounds",
                      "sampler": "seeded grouped multivariate TPE, separate equal-budget architecture studies",
                      "pruning": "synchronous successive halving after all three inner folds",
                      "min_screen_rounds": 6, "reference_only_trials": 64,
                      "pair_blend_min_gain": .005, "pair_blend_weights": [.25, .5, .75]},
        "fixed": {"optimizer": "AdamW", "loss": "masked BCE", "gradient_clip": 1.,
                  "training_center_frames": 512, "inference_center_frames": 2048,
                  "ball_age_feature_cap_seconds": 10., "team_swap": True,
                  "rate": "native 25 Hz; full halves for inference and official scoring",
                  "target_possession_labels": "never opened or used"},
        "gate": {"objective_gain_gt": 0., "missing_gain_gt": 0.,
                 "positive_missing_matches_at_least": 4, "clean_loss_at_most": .01,
                 "long_loss_at_most": .01, "replay_loss_at_most": .01,
                 "missing_transition_loss_at_most": .01,
                 "each_confirmation_seed_objective_gain_gt": 0.,
                 "outer_objective_loss_at_most": .01, "automatic_deployment": False},
    }


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """One persistent wall-clock deadline, including downtime across resume."""

    def __init__(self, output, hours, *, now=None):
        if not np.isfinite(hours) or not 0 < hours <= 12:
            raise ValueError("budget must be positive and at most 12 hours")
        self.output = Path(output)
        start = time.time() if now is None else float(now)
        path = self.output / "clock.json"
        existing = read_record(path)
        self.clock = existing or publish(path, {
            "started_unix": start, "hours": hours, "deadline_unix": start + 3600 * hours})
        if self.clock["hours"] != hours:
            raise ValueError("resume cannot reset or extend the compute budget")
        self.stage = "starting"

    def seconds_left(self, fraction=1.):
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("budget fraction must lie in (0, 1]")
        cutoff = self.clock["started_unix"] + 3600 * self.clock["hours"] * fraction
        return cutoff - time.time()

    def check(self, fraction=1.):
        if self.seconds_left(fraction) <= 0:
            raise BudgetExceeded(f"{self.stage}: budget boundary {fraction:.2f} reached")

    def status(self, stage, **detail):
        self.stage = stage
        write_json(self.output / "status.json", {
            "status": "running", "stage": stage, "pid": os.getpid(),
            **self.clock, "updated_unix": time.time(),
            "remaining_seconds": max(0., self.seconds_left()), **detail})


def paired_gate(baselines, candidates):
    """Evaluate a frozen candidate on paired confirmation seeds and matches."""
    if len(baselines) != 2 or len(candidates) != 2:
        raise ValueError("two paired confirmation seeds are required")
    matches = sorted(baselines[0]["by_match"])
    if len(matches) != 6 or any(sorted(row["by_match"]) != matches
                                for row in baselines + candidates):
        raise ValueError("confirmation must contain the same six development matches")
    deltas = {name: float(np.mean([
        candidate["means"][name]["spell_f1_mean"] - baseline["means"][name]["spell_f1_mean"]
        for baseline, candidate in zip(baselines, candidates, strict=True)])) for name in WEIGHTS}
    objective = [candidate["objective"] - baseline["objective"]
                 for baseline, candidate in zip(baselines, candidates, strict=True)]
    match_deltas = {}
    transition_deltas = []
    for match in matches:
        match_deltas[match] = float(np.mean([
            sum(WEIGHTS[name] * (candidate["by_match"][match][name]["spell_f1_mean"]
                                - baseline["by_match"][match][name]["spell_f1_mean"])
                for name in WEIGHTS if name != "clean") / .5
            for baseline, candidate in zip(baselines, candidates, strict=True)]))
    for baseline, candidate in zip(baselines, candidates, strict=True):
        transition_deltas.append(sum(WEIGHTS[name] * (
            candidate["means"][name]["transition_f1_at_12"]
            - baseline["means"][name]["transition_f1_at_12"])
            for name in WEIGHTS if name != "clean") / .5)
    missing = sum(WEIGHTS[name] * deltas[name] for name in WEIGHTS if name != "clean") / .5
    checks = {"objective_improved": np.mean(objective) > 0,
              "both_seeds_improved": all(value > 0 for value in objective),
              "missing_improved": missing > 0,
              "four_matches_improved": sum(value > 0 for value in match_deltas.values()) >= 4,
              "clean_preserved": deltas["clean"] >= -.01,
              "long_preserved": deltas["long"] >= -.01,
              "replay_preserved": deltas["replay"] >= -.01,
              "transitions_preserved": np.mean(transition_deltas) >= -.01}
    rng = np.random.default_rng(20271027)
    boot = rng.choice(list(match_deltas.values()), size=(5000, 6), replace=True).mean(axis=1)
    return {"passed": bool(all(checks.values())),
            "checks": {name: bool(value) for name, value in checks.items()},
            "objective_delta": float(np.mean(objective)), "seed_objective_deltas": objective,
            "scenario_deltas": deltas, "missing_delta": missing,
            "missing_by_match_deltas": match_deltas,
            "missing_match_bootstrap_95_interval": np.quantile(boot, [.025, .975]).tolist(),
            "interval_note": "Descriptive paired match bootstrap; six reused matches, not a test-set claim."}
