"""Opt-in, label-preserving augmentation of raw football tracking windows."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .schema import FrameTable, TrainingTargets

_TEMPORAL = ("time_warp", "time_shift", "frame_dropout", "frame_hold", "temporal_blur")
_SPATIAL = ("translation", "rotation", "reflection", "coordinate_noise",
            "player_dropout", "spatial_occlusion")
_OPERATIONS = _TEMPORAL + _SPATIAL


def _number(name: str, value: Any, *, minimum: float = 0.0,
            maximum: float | None = None) -> float:
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")
    return number


def _integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


@dataclass(frozen=True)
class AugmentationConfig:
    mode: str = "none"
    operations: tuple[str, ...] = ()
    clean_probability: float = 0.2
    max_transforms: int = 2
    strength: float = 1.0
    warmup_steps: int = 400
    noise_player_std: float = 0.08
    noise_ball_std: float = 0.15
    noise_correlation: float = 0.95
    player_dropout_fraction: float = 0.15
    dropout_max_seconds: float = 2.0
    patch_fraction: float = 0.15
    translation_max_meters: float = 1.0
    rotation_max_degrees: float = 2.0
    time_warp_max_rate: float = 0.1
    time_warp_knots: int = 4
    temporal_shift_max_seconds: float = 0.12
    frame_dropout_fraction: float = 0.05
    hold_max_frames: int = 3
    blur_radius: int = 2

    def __post_init__(self) -> None:
        if self.mode not in ("none", "spatial", "temporal", "combined"):
            raise ValueError("mode must be none, spatial, temporal, or combined")
        if not isinstance(self.operations, (tuple, list)) or any(
                not isinstance(op, str) for op in self.operations):
            raise TypeError("operations must be a sequence of operation names")
        operations = tuple(self.operations)
        if len(operations) != len(set(operations)):
            raise ValueError("operations must not contain duplicates")
        if any(op not in _OPERATIONS for op in operations):
            raise ValueError("operations contain an unknown name")
        allowed = set(_TEMPORAL if self.mode == "temporal" else
                      _SPATIAL if self.mode == "spatial" else
                      _OPERATIONS if self.mode == "combined" else ())
        if any(op not in allowed for op in operations):
            raise ValueError("operations must belong to the selected mode")
        object.__setattr__(self, "operations", operations)
        for name in ("clean_probability", "strength", "player_dropout_fraction",
                     "patch_fraction", "frame_dropout_fraction"):
            _number(name, getattr(self, name), maximum=1.0)
        _number("noise_correlation", self.noise_correlation, maximum=1.0)
        for name in ("noise_player_std", "noise_ball_std", "dropout_max_seconds",
                     "translation_max_meters", "rotation_max_degrees",
                     "time_warp_max_rate", "temporal_shift_max_seconds"):
            _number(name, getattr(self, name))
        if self.time_warp_max_rate >= 1:
            raise ValueError("time_warp_max_rate must be below 1")
        _integer("max_transforms", self.max_transforms, minimum=1)
        _integer("warmup_steps", self.warmup_steps, minimum=0)
        _integer("time_warp_knots", self.time_warp_knots, minimum=2)
        _integer("hold_max_frames", self.hold_max_frames, minimum=2)
        _integer("blur_radius", self.blur_radius, minimum=0)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["operations"] = list(self.operations)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AugmentationConfig:
        if not isinstance(value, dict):
            raise TypeError("augmentation config must be a dictionary")
        return cls(**value)


def normalize_augmentation(
        config: AugmentationConfig | dict[str, Any] | None) -> AugmentationConfig:
    if config is None:
        return AugmentationConfig()
    if isinstance(config, AugmentationConfig):
        return config
    if isinstance(config, dict):
        return AugmentationConfig.from_dict(config)
    raise TypeError("augmentation config must be AugmentationConfig, dict, or None")


def _copy_frames(frames: FrameTable) -> FrameTable:
    return frames.subset(0, len(frames.frame_id))


def _observation_arrays(frames: FrameTable, targets: TrainingTargets) -> tuple[np.ndarray, ...]:
    return (frames.players_xy, frames.player_valid, frames.ball_xy, frames.ball_valid,
            frames.graded, targets.live, targets.live_valid, targets.owner,
            targets.owner_valid, targets.state, targets.state_valid)


def _coordinate_noise(frames: FrameTable, rng: np.random.Generator,
                      config: AugmentationConfig, strength: float) -> dict[str, Any]:
    rho = float(config.noise_correlation)
    touched = 0
    for xy, valid, std in ((frames.players_xy, frames.player_valid, config.noise_player_std),
                           (frames.ball_xy[:, None, :], frames.ball_valid[:, None],
                            config.noise_ball_std)):
        sigma = float(std) * strength
        if sigma <= 0:
            continue
        tracks = valid.reshape(len(valid), -1)
        points = xy.reshape(len(valid), -1, 2)
        for track in range(tracks.shape[1]):
            prev = np.zeros(2, dtype=np.float64)
            contiguous = False
            for t in range(len(valid)):
                if not frames.graded[t] or not tracks[t, track]:
                    contiguous = False
                    prev[:] = 0
                    continue
                innovation_std = sigma * math.sqrt(max(0.0, 1.0 - rho * rho)) if contiguous else sigma
                prev = rho * prev + np.clip(rng.normal(0, innovation_std, 2),
                                             -3 * sigma, 3 * sigma)
                points[t, track] += prev.astype(np.float32)
                contiguous = True
                touched += 1
    return {"observations": touched}


def _player_dropout(frames: FrameTable, rng: np.random.Generator,
                    config: AugmentationConfig, strength: float) -> dict[str, Any]:
    n = len(frames.frame_id)
    tracks = frames.player_valid.reshape(n, -1)
    eligible = np.flatnonzero(np.any(tracks & frames.graded[:, None], axis=0))
    fraction = config.player_dropout_fraction * strength
    if not len(eligible) or fraction <= 0 or n == 0 or config.dropout_max_seconds == 0:
        return {"observations_hidden": 0, "tracks_selected": 0}
    count = min(len(eligible), max(1, math.ceil(fraction * tracks.shape[1])))
    selected = rng.choice(eligible, size=count, replace=False)
    max_span = max(1, round(config.dropout_max_seconds * frames.fps * strength))
    hidden = 0
    for track in np.atleast_1d(selected):
        possible = np.flatnonzero(tracks[:, track] & frames.graded)
        if not len(possible):
            continue
        anchor = int(rng.choice(possible))
        span = int(rng.integers(1, min(n, max_span) + 1))
        start = max(0, min(anchor - int(rng.integers(span)), n - span))
        end = start + span
        region = tracks[start:end, track]
        hide = region & frames.graded[start:end]
        hidden += int(hide.sum())
        region[hide] = False
    frames.players_xy[~frames.player_valid] = 0
    return {"observations_hidden": hidden, "tracks_selected": int(count)}


def _spatial_occlusion(frames: FrameTable, rng: np.random.Generator,
                       config: AugmentationConfig, strength: float) -> dict[str, Any]:
    n = len(frames.frame_id)
    area = config.patch_fraction * strength
    if n == 0 or area <= 0 or config.dropout_max_seconds == 0:
        return {"observations_hidden": 0}
    player_sites = np.argwhere(frames.player_valid & frames.graded[:, None, None])
    ball_sites = np.flatnonzero(frames.ball_valid & frames.graded)
    total = len(player_sites) + len(ball_sites)
    if total == 0:
        return {"observations_hidden": 0}
    pick = int(rng.integers(total))
    if pick < len(player_sites):
        t, team, player = (int(v) for v in player_sites[pick])
        center = frames.players_xy[t, team, player].astype(np.float64)
    else:
        t = int(ball_sites[pick - len(player_sites)])
        center = frames.ball_xy[t].astype(np.float64)
    length, width = frames.pitch
    center = np.clip(center, [0.0, 0.0], [length, width])
    half_size = np.array([length, width]) * math.sqrt(area) / 2.0
    max_span = max(1, round(config.dropout_max_seconds * frames.fps * strength))
    span = int(rng.integers(1, min(n, max_span) + 1))
    start = max(0, min(t - int(rng.integers(span)), n - span))
    end = start + span
    players_in = np.all(np.abs(frames.players_xy[start:end] - center) <= half_size, axis=-1)
    balls_in = np.all(np.abs(frames.ball_xy[start:end] - center) <= half_size, axis=-1)
    players_hide = (frames.player_valid[start:end] & players_in &
                    frames.graded[start:end, None, None])
    balls_hide = frames.ball_valid[start:end] & balls_in & frames.graded[start:end]
    frames.player_valid[start:end] &= ~players_hide
    frames.ball_valid[start:end] &= ~balls_hide
    frames.players_xy[~frames.player_valid] = 0
    frames.ball_xy[~frames.ball_valid] = 0
    return {"observations_hidden": int(players_hide.sum() + balls_hide.sum()),
            "duration_frames": int(span)}


def _geometry(frames: FrameTable, rng: np.random.Generator, config: AugmentationConfig,
              strength: float, operation: str) -> dict[str, Any]:
    length, width = frames.pitch
    center = np.array([length / 2, width / 2], dtype=np.float64)
    if operation == "translation":
        delta = rng.uniform(-config.translation_max_meters,
                            config.translation_max_meters, 2) * strength
        for xy, valid in ((frames.players_xy, frames.player_valid),
                          (frames.ball_xy, frames.ball_valid)):
            xy[valid] += delta.astype(np.float32)
        return {"dx_m": float(delta[0]), "dy_m": float(delta[1])}
    if operation == "rotation":
        degrees = float(rng.uniform(-config.rotation_max_degrees,
                                     config.rotation_max_degrees) * strength)
        angle = math.radians(degrees)
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        for xy, valid in ((frames.players_xy, frames.player_valid),
                          (frames.ball_xy, frames.ball_valid)):
            xy[valid] = ((xy[valid] - center) @ rotation.T + center).astype(np.float32)
        return {"degrees": degrees}
    axis = ("x", "y", "both")[int(rng.integers(3))]
    for xy, valid in ((frames.players_xy, frames.player_valid),
                      (frames.ball_xy, frames.ball_valid)):
        if axis in ("x", "both"):
            xy[..., 0][valid] = length - xy[..., 0][valid]
        if axis in ("y", "both"):
            xy[..., 1][valid] = width - xy[..., 1][valid]
    return {"axis": axis}


def augment_tracking(
    frames: FrameTable,
    targets: TrainingTargets,
    rng: np.random.Generator,
    config: AugmentationConfig | dict[str, Any] | None,
    *,
    step: int = 0,
) -> tuple[FrameTable, TrainingTargets, dict[str, Any]]:
    """Apply at most ``max_transforms`` sampled operations to a tracking window.

    Temporal transforms run first so geometry and observation noise use the final
    time grid. The original frames and targets are never changed.
    """
    config = normalize_augmentation(config)
    if not isinstance(frames, FrameTable) or not isinstance(targets, TrainingTargets):
        raise TypeError("frames and targets must be FrameTable and TrainingTargets")
    if len(frames.frame_id) != len(targets.live):
        raise ValueError("frames and targets must have the same length")
    step = _integer("step", step, minimum=0)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator")
    pool = (config.operations or
            (_SPATIAL if config.mode == "spatial" else
             _TEMPORAL if config.mode == "temporal" else
             _OPERATIONS if config.mode == "combined" else ()))
    strength = float(config.strength) * (min(1.0, step / config.warmup_steps)
                                         if config.warmup_steps else 1.0)
    report: dict[str, Any] = {"selected": [], "applied": [], "strength": strength,
                              "details": {}}
    if not pool or not len(frames.frame_id) or strength <= 0:
        return frames, targets, report
    if rng.random() < config.clean_probability:
        return frames, targets, report
    count = int(rng.integers(1, min(config.max_transforms, len(pool)) + 1))
    selected = tuple(str(op) for op in rng.choice(pool, size=count, replace=False))
    report["selected"] = list(selected)
    current_frames = _copy_frames(frames)
    current_targets = targets.subset(0, len(targets.live))
    for operation in _OPERATIONS:
        if operation not in selected:
            continue
        before = tuple(value.copy() for value in _observation_arrays(current_frames, current_targets))
        if operation in _TEMPORAL:
            from .temporal_augmentation import apply_temporal

            current_frames, current_targets, detail = apply_temporal(
                current_frames, current_targets, rng, operation=operation, strength=strength,
                time_warp_max_rate=config.time_warp_max_rate,
                time_warp_knots=config.time_warp_knots,
                temporal_shift_max_seconds=config.temporal_shift_max_seconds,
                frame_dropout_fraction=config.frame_dropout_fraction,
                hold_max_frames=config.hold_max_frames,
                blur_radius=config.blur_radius)
        elif operation == "coordinate_noise":
            detail = _coordinate_noise(current_frames, rng, config, strength)
        elif operation == "player_dropout":
            detail = _player_dropout(current_frames, rng, config, strength)
        elif operation == "spatial_occlusion":
            detail = _spatial_occlusion(current_frames, rng, config, strength)
        else:
            detail = _geometry(current_frames, rng, config, strength, operation)
        changed = any(not np.array_equal(old, new) for old, new in
                      zip(before, _observation_arrays(current_frames, current_targets)))
        if changed:
            report["applied"].append(operation)
        detail["changed"] = changed
        report["details"][operation] = detail
    current_frames.validate()
    return current_frames, current_targets, report
