"""Bounded joint model, loss, augmentation and decoder search spaces."""
from __future__ import annotations

import copy

from possession.augmentation import AugmentationConfig
from possession.outages import OutageConfig


def sample_recipe(trial):
    depth = trial.suggest_categorical("depth", [7, 8, 9])
    steps = trial.suggest_categorical("total_steps", [7200, 9600])
    schedule = trial.suggest_categorical("schedule", ["constant", "cosine"])
    policy = trial.suggest_categorical("augmentation", [
        "none", "coordinate_noise", "player_dropout", "geometry", "sensor", "combined"])
    operations = {"none": [], "coordinate_noise": ["coordinate_noise"],
                  "player_dropout": ["player_dropout"],
                  "geometry": ["translation", "rotation", "reflection"],
                  "sensor": ["frame_dropout", "frame_hold", "temporal_blur"],
                  "combined": ["coordinate_noise", "player_dropout", "frame_hold"]}[policy]
    mode = ("none" if not operations else "temporal" if policy == "sensor" else
            "combined" if policy == "combined" else "spatial")
    augmentation = AugmentationConfig(
        mode=mode, operations=tuple(operations),
        clean_probability=trial.suggest_float("augmentation_clean", .2, .8),
        strength=trial.suggest_float("augmentation_strength", .2, 1.),
        max_transforms=1, noise_player_std=.08, noise_ball_std=.15,
        player_dropout_fraction=.05, dropout_max_seconds=.5, warmup_steps=400).to_dict()
    outage = OutageConfig(
        mode=trial.suggest_categorical("outage", ["legacy", "correlated"]),
        clean_probability=trial.suggest_float("outage_clean", .15, .4),
        long_gap_probability=trial.suggest_float("outage_long", 0., .3)).to_dict()
    spec = {
        "model_kwargs": {"player_dim": 8, "global_dim": 9,
                         "width": trial.suggest_categorical("width", [32, 48, 64]),
                         "pooling": trial.suggest_categorical(
                             "pooling", ["mean", "ball_attention", "interaction"]),
                         "dilations": [2 ** i for i in range(depth)]},
        "training_args": {
            "learning_rate": trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True),
            "boundary_weight": trial.suggest_categorical("boundary_weight", [0., .1, .25, .5]),
            "boundary_pos_weight": trial.suggest_float("boundary_pos_weight", 1., 20., log=True),
            "center_size": trial.suggest_categorical("center_size", [256, 512, 768]),
            "augment": True},
        "augmentation_config": augmentation, "outage_config": outage,
        "schedule": {"kind": schedule, "total_steps": steps,
                     "warmup_steps": 400 if schedule == "cosine" else 0,
                     "min_lr_ratio": trial.suggest_float("lr_floor", .05, .35)
                     if schedule == "cosine" else 1.},
    }
    return {"spec": spec,
            "spell_weight": trial.suggest_categorical("spell_weight", [0., .1, .25, .5]),
            "short_sampling_probability": trial.suggest_categorical(
                "short_sampling_probability", [0., .15, .3, .5]),
            "export": trial.suggest_categorical("export", ["ema", "raw"])}


def anchor_recipe(member, balanced=True):
    return {"spec": copy.deepcopy(member["spec"]),
            "spell_weight": .25 if balanced else 0.,
            "short_sampling_probability": .25 if balanced else 0., "export": "ema"}


def pipeline_anchor(decoder, weight=.5):
    return {"decoder": copy.deepcopy(decoder), "candidate_weight": weight,
            "coarse_weight": .25, "coarse_seconds": 10.}


def sample_pipeline(trial, baseline):
    """Search around the incumbent while allowing calibrated asymmetric switches."""
    decoder = copy.deepcopy(baseline)
    mode = trial.suggest_categorical("blend_mode", ["static", "quality_separate"])
    decoder["blend_mode"] = mode
    decoder["neural_weight"] = trial.suggest_float("neural_weight", .3, .9)
    if mode == "quality_separate":
        for key in ("visible_live_weight", "missing_live_weight",
                    "visible_owner_weight", "missing_owner_weight"):
            decoder[key] = trial.suggest_float(key, .2, .95)
        decoder["transition_seconds"] = trial.suggest_float("transition_seconds", .5, 15., log=True)
    decoder["live_temperature"] = trial.suggest_float("live_temperature", 1., 6., log=True)
    decoder["owner_temperature"] = trial.suggest_float("owner_temperature", .5, 2., log=True)
    decoder["boundary_scale"] = trial.suggest_float("boundary_scale", 0., 2.)
    decoder["switch_cost_type"] = "typed"
    decoder["switch_cost"] = {
        "owner_switch": trial.suggest_float("owner_switch", 0., 2.),
        "to_dead": trial.suggest_float("to_dead", 1., 7.),
        "from_dead": trial.suggest_float("from_dead", .5, 6.)}
    reference_mode = trial.suggest_categorical("reference_mode", ["default", "tuned"])
    decoder["reference_mode"] = reference_mode
    decoder["reference"] = {} if reference_mode == "default" else {
        "owner_smoothing_seconds": trial.suggest_float("owner_smoothing_seconds", .2, 1.5, log=True),
        "activity_smoothing_seconds": trial.suggest_float("activity_smoothing_seconds", .2, 2., log=True),
        "owner_strength": trial.suggest_float("owner_strength", 1., 6., log=True)}
    return {"decoder": decoder,
            "candidate_weight": trial.suggest_float("candidate_weight", .1, 1.),
            "coarse_weight": trial.suggest_float("coarse_weight", 0., .5),
            "coarse_seconds": trial.suggest_float("coarse_seconds", 3., 20., log=True)}
