"""Canonical joint search space for the isolated training campaign."""

from __future__ import annotations

import json

from possession.augmentation import AugmentationConfig, normalize_augmentation
from possession.outages import OutageConfig

TOTAL_STEPS = 7200
DILATIONS = {
    7: [1, 2, 4, 8, 16, 32, 64],
    8: [1, 2, 4, 8, 16, 32, 64, 128],
    9: [1, 2, 4, 8, 16, 32, 64, 128, 256],
}
OPS = (
    "translation", "rotation", "reflection", "coordinate_noise", "player_dropout",
    "spatial_occlusion", "time_warp", "time_shift", "frame_dropout", "frame_hold",
    "temporal_blur",
)
FAMILIES = {
    "geometry": ("translation", "rotation", "reflection"),
    "noise": ("coordinate_noise",),
    "occlusion": ("player_dropout", "spatial_occlusion"),
    "time": ("time_warp", "time_shift"),
    "sensor": ("frame_dropout", "frame_hold", "temporal_blur"),
    "combined": OPS,
}
POLICIES = ("none", *OPS, *FAMILIES)


def _categorical(trial, name, values):
    return trial.suggest_categorical(name, list(values))


def _augmentation(trial):
    policy = _categorical(trial, "augmentation_policy", POLICIES)
    if policy == "none":
        return AugmentationConfig().to_dict()
    operations = FAMILIES.get(policy, (policy,))
    temporal = {"time_warp", "time_shift", "frame_dropout", "frame_hold", "temporal_blur"}
    spatial = {"translation", "rotation", "reflection", "coordinate_noise",
               "player_dropout", "spatial_occlusion"}
    mode = "combined" if set(operations) & temporal and set(operations) & spatial else (
        "temporal" if set(operations) <= temporal else "spatial")
    values = {
        "mode": mode, "operations": list(operations),
        "clean_probability": _categorical(trial, "augmentation_clean_probability", (.2, .5, .75)),
        "max_transforms": (1 if len(operations) == 1 else
                           _categorical(trial, "augmentation_max_transforms", (1, 2))),
        "strength": _categorical(trial, "augmentation_strength", (.25, .5, 1.0)),
        "warmup_steps": 400,
    }
    # Magnitudes are suggested only when their operation is active.
    if "coordinate_noise" in operations:
        values.update(noise_player_std=_categorical(trial, "noise_player_std", (.03, .06, .08)),
                      noise_ball_std=_categorical(trial, "noise_ball_std", (.05, .1, .15)),
                      noise_correlation=_categorical(trial, "noise_correlation", (.5, .8, .95)))
    if "player_dropout" in operations:
        values["player_dropout_fraction"] = _categorical(
            trial, "player_dropout_fraction", (.05, .1, .15))
    if set(operations) & {"player_dropout", "spatial_occlusion"}:
        values["dropout_max_seconds"] = _categorical(
            trial, "dropout_max_seconds", (.5, 1., 2.))
    if "spatial_occlusion" in operations:
        values["patch_fraction"] = _categorical(trial, "patch_fraction", (.05, .1, .15))
    if "translation" in operations:
        values["translation_max_meters"] = _categorical(trial, "translation_max_meters", (.25, .5, 1.))
    if "rotation" in operations:
        values["rotation_max_degrees"] = _categorical(trial, "rotation_max_degrees", (.5, 1., 2.))
    if "time_warp" in operations:
        values.update(time_warp_max_rate=_categorical(trial, "time_warp_max_rate", (.025, .05, .1)),
                      time_warp_knots=_categorical(trial, "time_warp_knots", (3, 4, 6)))
    if "time_shift" in operations:
        values["temporal_shift_max_seconds"] = _categorical(
            trial, "temporal_shift_max_seconds", (.04, .08, .12))
    if "frame_dropout" in operations:
        values["frame_dropout_fraction"] = _categorical(
            trial, "frame_dropout_fraction", (.01, .025, .05))
    if "frame_hold" in operations:
        values["hold_max_frames"] = _categorical(trial, "hold_max_frames", (2, 3, 5))
    if "temporal_blur" in operations:
        values["blur_radius"] = _categorical(trial, "blur_radius", (1, 2, 3))
    return AugmentationConfig(**values).to_dict()


def _outage(trial):
    mode = _categorical(trial, "outage_mode", ("legacy", "correlated"))
    if mode == "legacy":
        return OutageConfig().to_dict()
    clean = _categorical(trial, "outage_clean_probability", (.1, .2, .35))
    full = _categorical(trial, "outage_full_probability", (0., .05, .1))
    if clean + full > 1:
        raise ValueError("invalid outage probability combination")
    fraction_min = _categorical(trial, "outage_fraction_min", (.2, .3, .4))
    fraction_max = _categorical(trial, "outage_fraction_max", (.5, .66, .8))
    if fraction_min > fraction_max:
        fraction_min, fraction_max = fraction_max, fraction_min
    short_min = _categorical(trial, "outage_min_gap_seconds", (.5, 1., 2.))
    short_max = _categorical(trial, "outage_max_gap_seconds", (4., 8., 10.))
    if short_min > short_max:
        short_min, short_max = short_max, short_min
    long_min = _categorical(trial, "outage_long_min_seconds", (8., 10., 15.))
    long_max = _categorical(trial, "outage_long_max_seconds", (20., 30.))
    if long_min > long_max:
        long_min, long_max = long_max, long_min
    return OutageConfig(
        mode="correlated", clean_probability=clean, full_probability=full,
        fraction_min=fraction_min, fraction_max=fraction_max,
        min_gap_seconds=short_min, max_gap_seconds=short_max,
        long_gap_probability=_categorical(trial, "outage_long_gap_probability", (0., .1, .25)),
        long_gap_min_seconds=long_min, long_gap_max_seconds=long_max,
    ).to_dict()


def sample_spec(trial, pooling: str) -> dict:
    """Sample one valid, JSON-canonical joint model/training recipe."""
    if pooling not in ("mean", "ball_attention", "interaction"):
        raise ValueError(f"unsupported pooling: {pooling}")
    depth = _categorical(trial, "depth", (7, 8, 9))
    lr = trial.suggest_float("learning_rate", 3e-4, 5e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    kind = _categorical(trial, "schedule_kind", ("constant", "cosine"))
    warmup = 0 if kind == "constant" else _categorical(trial, "warmup_steps", (0, 200, 600))
    ratio = 1.0 if kind == "constant" else _categorical(trial, "min_lr_ratio", (.05, .1, .25))
    boundary_weight = _categorical(trial, "boundary_weight", (0., .1, .25, .5))
    boundary_pos_weight = (trial.suggest_float("boundary_pos_weight", 1., 100., log=True)
                           if boundary_weight > 0 else 1.0)
    spec = {
        "model_kwargs": {"player_dim": 8, "global_dim": 9,
                         "width": _categorical(trial, "width", (32, 64, 96)),
                         "pooling": pooling, "dilations": DILATIONS[depth]},
        "training_args": {
            "learning_rate": lr, "weight_decay": wd,
            "boundary_weight": boundary_weight,
            "boundary_pos_weight": boundary_pos_weight,
            "center_size": 512, "augment": True,
        },
        "outage_config": _outage(trial),
        "augmentation_config": _augmentation(trial),
        "schedule": {"kind": kind, "warmup_steps": warmup,
                     "min_lr_ratio": ratio, "total_steps": TOTAL_STEPS},
    }
    return json.loads(json.dumps(spec, sort_keys=True, allow_nan=False))


def baseline_spec(focused_baseline: dict) -> dict:
    """Expand the existing frozen focused recipe with a constant-LR schedule."""
    source = focused_baseline.get("selected_model_recipe", focused_baseline)
    model = dict(source["model_kwargs"])
    args = dict(source["training_args"])
    steps = int(args.pop("steps", focused_baseline.get("steps", TOTAL_STEPS)))
    model.setdefault("player_dim", 8)
    model.setdefault("global_dim", 9)
    model.setdefault("pooling", "mean")
    model.setdefault("dilations", DILATIONS[8])
    for key in ("learning_rate", "weight_decay", "boundary_weight", "boundary_pos_weight",
                "center_size", "augment"):
        args.setdefault(key, {"learning_rate": 1e-3, "weight_decay": 1e-4,
                              "boundary_weight": .25, "boundary_pos_weight": 1.,
                              "center_size": 512, "augment": True}[key])
    args.pop("outage_config", None)
    args.pop("augmentation_config", None)
    return json.loads(json.dumps({
        "model_kwargs": model, "training_args": args,
        "outage_config": OutageConfig(**focused_baseline.get(
            "outage_config", {"mode": "legacy"})).to_dict(),
        "augmentation_config": normalize_augmentation(focused_baseline.get(
            "augmentation_config", {"mode": "none"})).to_dict(),
        "schedule": {"kind": "constant", "warmup_steps": 0,
                     "min_lr_ratio": 1.0, "total_steps": steps},
    }, sort_keys=True, allow_nan=False))
