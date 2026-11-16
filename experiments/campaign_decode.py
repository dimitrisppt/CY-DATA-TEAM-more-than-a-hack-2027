"""Cached, label-free decoder search primitives for a new campaign.

The module deliberately does not own fitting, bundle storage, scoring, or study
management. A campaign can build one compact reference basis per half and reuse
it across many decoder trials.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from possession.calibrate import CalibrationConfig, apply_temperature_calibration
from possession.decode import decode_half, emissions_from_probabilities
from possession.features import build_features
from possession.reference import ReferenceConfig, _smooth_observed
from possession.schema import FeatureBatch, FrameTable

_REF_FIELDS = tuple(ReferenceConfig.__dataclass_fields__)
_REF_DEFAULTS = asdict(ReferenceConfig())

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only in minimal installs
    _smooth_compiled = None
else:
    @njit(cache=True, fastmath=False)
    def _smooth_numba(value, observed, fps, seconds):
        """Local exact recurrence; leaves possession.reference unchanged."""
        n = len(value)
        if n == 0:
            return value.copy(), np.zeros(0, dtype=np.float64)
        alpha = np.exp(-1.0 / max(seconds * fps, 1e-6))
        weighted = np.where(observed, value, 0.0)
        weight = observed.astype(np.float64)
        forward_sum = np.empty(n, dtype=np.float64)
        forward_count = np.empty(n, dtype=np.float64)
        backward_sum = np.empty(n, dtype=np.float64)
        backward_count = np.empty(n, dtype=np.float64)
        run_sum = 0.0
        run_count = 0.0
        for t in range(n):
            run_sum = weighted[t] + alpha * run_sum
            run_count = weight[t] + alpha * run_count
            forward_sum[t] = run_sum
            forward_count[t] = run_count
        run_sum = 0.0
        run_count = 0.0
        for t in range(n - 1, -1, -1):
            run_sum = weighted[t] + alpha * run_sum
            run_count = weight[t] + alpha * run_count
            backward_sum[t] = run_sum
            backward_count[t] = run_count
        result = np.zeros(n, dtype=np.float64)
        count = forward_count + backward_count - weight
        for t in range(n):
            total = forward_sum[t] + backward_sum[t] - weighted[t]
            if count[t] > 1e-12:
                result[t] = total / count[t]
        return result, count
    _smooth_compiled = _smooth_numba


def _smooth_basis(value: np.ndarray, observed: np.ndarray, fps: float,
                  seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Accelerated equivalent of reference._smooth_observed."""
    if _smooth_compiled is None:
        return _smooth_observed(value, observed, fps, seconds)
    return _smooth_compiled(np.asarray(value, dtype=np.float64),
                            np.asarray(observed, dtype=bool), fps, seconds)


def reference_basis(frames: FrameTable, features: FeatureBatch | None = None) -> dict[str, np.ndarray]:
    """Extract sufficient label-free arrays for any ``ReferenceConfig``.

    Stored per-frame values are distances and evidence components, not labels.
    Scalar ``ref_fps`` and ``ref_pitch`` are needed for smoothing and outside
    detection. Arrays are compact relative to raw player coordinates and can be
    persisted alongside the neural OOF probabilities.
    """
    if features is None:
        features = build_features(frames)
    valid = np.asarray(frames.player_valid, dtype=bool)
    ball_valid = np.asarray(frames.ball_valid, dtype=bool)
    ball = np.asarray(frames.ball_xy, dtype=np.float64)
    players = np.asarray(frames.players_xy, dtype=np.float64)
    n = len(ball_valid)
    distances = np.linalg.norm(players - ball[:, None, None, :], axis=-1)
    distances = np.where(valid & ball_valid[:, None, None], distances, np.inf)
    sorted_distance = np.sort(distances, axis=-1)
    nearest = sorted_distance[..., 0] if sorted_distance.shape[-1] else np.full((n, 2), np.inf)
    second = sorted_distance[..., 1] if sorted_distance.shape[-1] > 1 else np.full((n, 2), np.inf)

    player_motion = np.asarray(features.player_tokens[..., 4] > 0, dtype=bool)
    ball_motion = np.asarray(features.global_tokens[:, 5] > 0, dtype=bool)
    player_velocity = np.asarray(features.player_tokens[..., 2:4], dtype=np.float64) * 15.0
    ball_velocity = np.asarray(features.global_tokens[:, 3:5], dtype=np.float64) * 50.0
    relative_speed = np.linalg.norm(player_velocity - ball_velocity[:, None, None, :], axis=-1)
    nearest_slot = np.argmin(distances, axis=-1) if distances.shape[-1] else np.zeros((n, 2), dtype=int)
    if distances.shape[-1]:
        nearest_motion = np.take_along_axis(player_motion, nearest_slot[..., None], axis=-1)[..., 0]
        nearest_relative_speed = np.take_along_axis(relative_speed, nearest_slot[..., None], axis=-1)[..., 0]
    else:
        nearest_motion = np.zeros((n, 2), dtype=bool)
        nearest_relative_speed = np.zeros((n, 2), dtype=np.float64)
    motion_credible = ball_motion[:, None] & nearest_motion & np.isfinite(nearest)
    motion_factor = np.where(motion_credible,
                             0.85 + 0.15 * np.exp(-nearest_relative_speed / 8.0), 1.0)

    speed = np.linalg.norm(player_velocity, axis=-1)
    counts = player_motion.sum(axis=(1, 2))
    moving = ((speed > 0.5) & player_motion).sum(axis=(1, 2))
    activity = np.divide(moving, counts, out=np.zeros(n, dtype=np.float64), where=counts > 0)
    activity_observed = counts > 0
    length, width = frames.pitch
    outside = ball_valid & ((ball[:, 0] < 0) | (ball[:, 0] > length)
                            | (ball[:, 1] < 0) | (ball[:, 1] > width))
    return {
        "ref_nearest_m": nearest.astype(np.float64),
        "ref_second_m": second.astype(np.float64),
        "ref_motion_factor": motion_factor.astype(np.float64),
        "ref_owner_observed": (ball_valid & np.any(valid, axis=(1, 2))),
        "ref_activity": activity.astype(np.float64),
        "ref_activity_observed": activity_observed,
        "ref_ball_motion": ball_motion,
        "ref_ball_speed": np.linalg.norm(ball_velocity, axis=-1).astype(np.float64),
        "ref_outside": outside,
        "ref_fps": np.asarray(frames.fps, dtype=np.float64),
        "ref_pitch": np.asarray((length, width), dtype=np.float64),
    }


def _basis_arrays(basis: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int, float]:
    required = {"ref_nearest_m", "ref_second_m", "ref_motion_factor", "ref_owner_observed",
                "ref_activity", "ref_activity_observed", "ref_ball_motion", "ref_ball_speed",
                "ref_outside", "ref_fps", "ref_pitch"}
    missing = required - basis.keys()
    if missing:
        raise ValueError(f"reference basis lacks keys: {sorted(missing)}")
    arrays = {key: np.asarray(basis[key]) for key in required}
    nearest = arrays["ref_nearest_m"]
    if nearest.ndim != 2 or nearest.shape[1] != 2:
        raise ValueError("ref_nearest_m must have shape [T,2]")
    n = len(nearest)
    for key in ("ref_second_m", "ref_motion_factor"):
        if arrays[key].shape != (n, 2):
            raise ValueError(f"{key} must have shape [T,2]")
    for key in ("ref_owner_observed", "ref_activity", "ref_activity_observed",
                "ref_ball_motion", "ref_ball_speed", "ref_outside"):
        if arrays[key].shape != (n,):
            raise ValueError(f"{key} must have shape [T]")
    if arrays["ref_pitch"].shape != (2,) or arrays["ref_fps"].ndim != 0:
        raise ValueError("ref_pitch and ref_fps must have shapes [2] and scalar")
    if (not np.isfinite(arrays["ref_fps"]).all() or float(arrays["ref_fps"]) <= 0
            or not np.isfinite(arrays["ref_pitch"]).all()
            or np.any(arrays["ref_pitch"] <= 0)):
        raise ValueError("reference basis fps and pitch must be finite and positive")
    for key in ("ref_nearest_m", "ref_second_m", "ref_motion_factor", "ref_activity", "ref_ball_speed"):
        values = arrays[key]
        # Infinite distances represent an unavailable team; other fields must be finite.
        if key in ("ref_nearest_m", "ref_second_m"):
            if np.isnan(values).any() or np.isneginf(values).any():
                raise ValueError(f"{key} contains an invalid distance")
        elif not np.isfinite(values).all():
            raise ValueError(f"{key} must be finite")
    return arrays, n, float(arrays["ref_fps"])


def reference_from_basis(basis: dict[str, np.ndarray], config: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Reproduce ``reference_scores`` from a cached basis and optional config."""
    arrays, n, fps = _basis_arrays(basis)
    values = dict(_REF_DEFAULTS)
    if config is not None:
        unknown = set(config) - set(_REF_FIELDS)
        if unknown:
            raise ValueError(f"unknown reference config fields: {sorted(unknown)}")
        values.update(config)
    cfg = ReferenceConfig(**values)
    for name in _REF_FIELDS:
        value = getattr(cfg, name)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if min(cfg.owner_smoothing_seconds, cfg.activity_smoothing_seconds,
           cfg.nearest_scale_m, cfg.second_scale_m) <= 0:
        raise ValueError("smoothing times and distance scales must be positive")
    nearest = arrays["ref_nearest_m"].astype(np.float64)
    second = arrays["ref_second_m"].astype(np.float64)
    quality = (np.exp(-nearest / cfg.nearest_scale_m)
               + cfg.second_weight * np.exp(-second / cfg.second_scale_m))
    quality *= arrays["ref_motion_factor"]
    owner_evidence = cfg.owner_strength * (quality[:, 0] - quality[:, 1])
    owner_context, _ = _smooth_basis(owner_evidence, arrays["ref_owner_observed"], fps,
                                     cfg.owner_smoothing_seconds)
    home_probability = _sigmoid(owner_context)
    activity_context, activity_support = _smooth_basis(
        arrays["ref_activity"], arrays["ref_activity_observed"], fps,
        cfg.activity_smoothing_seconds)
    live_logit = np.full(n, cfg.live_bias, dtype=np.float64)
    live_logit += np.where(activity_support > 1e-12,
                           cfg.activity_strength * (activity_context - 0.25), 0.0)
    live_logit += cfg.ball_motion_bonus * (arrays["ref_ball_speed"] > 1.0) * arrays["ref_ball_motion"]
    live_logit -= cfg.outside_penalty * arrays["ref_outside"]
    return {"live_probability": _sigmoid(live_logit),
            "home_probability": home_probability,
            "boundary": np.zeros((n, 3, 3), dtype=np.float64)}


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _switch_matrix(config: dict[str, Any]) -> float | np.ndarray:
    kind = config["switch_cost_type"]
    cost = config["switch_cost"]
    if kind == "scalar":
        value = float(cost)
        if not np.isfinite(value) or not 0 <= value <= 8:
            raise ValueError("scalar switch_cost must lie in [0,8]")
        return value
    if kind != "typed" or not isinstance(cost, dict):
        raise ValueError("switch_cost_type must be scalar or typed")
    if set(cost) != {"owner_switch", "to_dead", "from_dead"}:
        raise ValueError("typed switch_cost requires owner_switch, to_dead, and from_dead")
    values = {key: float(value) for key, value in cost.items()}
    if any(not np.isfinite(value) or not 0 <= value <= 8 for value in values.values()):
        raise ValueError("typed switch costs must lie in [0,8]")
    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix[0, 1] = matrix[1, 0] = values["owner_switch"]
    matrix[0, 2] = matrix[1, 2] = values["to_dead"]
    matrix[2, 0] = matrix[2, 1] = values["from_dead"]
    return matrix


def baseline_decoder(focused_baseline: dict[str, Any]) -> dict[str, Any]:
    """Convert focused-study ``{ensemble, calibration, decoder}`` to canonical form."""
    if not isinstance(focused_baseline, dict):
        raise TypeError("focused_baseline must be a dictionary")
    for key in ("ensemble", "calibration", "decoder"):
        if not isinstance(focused_baseline.get(key), dict):
            raise TypeError(f"focused_baseline requires {key} mapping")
    ensemble, calibration, decoder = (focused_baseline[k] for k in ("ensemble", "calibration", "decoder"))
    weight = float(ensemble["neural_weight"])
    boundary_temperature = float(calibration.get("boundary_temperature", 1.0))
    if not np.isfinite(boundary_temperature) or boundary_temperature <= 0:
        raise ValueError("focused baseline boundary temperature must be positive and finite")
    switch = decoder["switch_cost"]
    kind = "scalar" if np.ndim(switch) == 0 else "typed"
    if kind == "typed":
        matrix = np.asarray(switch, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("focused baseline switch_cost must be scalar or [3,3]")
        switch = {"owner_switch": float(matrix[0, 1]), "to_dead": float(matrix[0, 2]),
                  "from_dead": float(matrix[2, 0])}
    config = {
        "version": 1, "reference_mode": "default", "reference": {},
        "blend_mode": "static", "neural_weight": weight,
        "visible_neural_weight": weight, "missing_neural_weight": weight,
        "visible_live_weight": weight, "missing_live_weight": weight,
        "visible_owner_weight": weight, "missing_owner_weight": weight,
        "transition_seconds": 2.0,
        "live_temperature": float(calibration.get("live_temperature", 1.0)),
        "owner_temperature": float(calibration.get("owner_temperature", 1.0)),
        # Decoder scale and calibration temperature are only identifiable as
        # this ratio; canonicalize to one effective boundary multiplier.
        "boundary_temperature": 1.0,
        "switch_cost_type": kind, "switch_cost": switch,
        "boundary_scale": float(decoder.get("boundary_scale", 0.0)) / boundary_temperature,
    }
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    required = {"version", "reference_mode", "reference", "blend_mode", "neural_weight",
                "visible_neural_weight", "missing_neural_weight", "visible_live_weight",
                "missing_live_weight", "visible_owner_weight", "missing_owner_weight",
                "transition_seconds", "live_temperature", "owner_temperature",
                "boundary_temperature", "switch_cost_type", "switch_cost", "boundary_scale"}
    if set(config) != required or config["version"] != 1:
        raise ValueError("decoder config has missing/unknown keys or unsupported version")
    if config["reference_mode"] not in ("default", "tuned") or not isinstance(config["reference"], dict):
        raise ValueError("reference_mode must be default or tuned with a reference mapping")
    if set(config["reference"]) - set(_REF_FIELDS):
        raise ValueError("reference config contains unknown fields")
    if config["blend_mode"] not in ("static", "quality_shared", "quality_separate"):
        raise ValueError("unknown blend_mode")
    for key in ("neural_weight", "visible_neural_weight", "missing_neural_weight",
                "visible_live_weight", "missing_live_weight", "visible_owner_weight",
                "missing_owner_weight"):
        value = float(config[key])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    for key in ("live_temperature", "owner_temperature", "boundary_temperature", "transition_seconds"):
        value = float(config[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    scale = float(config["boundary_scale"])
    if not np.isfinite(scale) or scale < 0 or scale > 2:
        raise ValueError("boundary_scale must lie in [0,2]")
    _switch_matrix(config)


def sample_decoder(trial: Any, *, reference_only: bool = False) -> dict[str, Any]:
    """Sample one conditional decoder recipe from a TPE-compatible trial.

    ``reference_only`` produces a geometric baseline while still allowing
    reference parameters, calibration temperatures, and decoder costs to vary.
    It omits blend/gate/boundary dimensions that cannot affect that baseline.
    """
    reference_mode = trial.suggest_categorical("reference_mode", ["default", "tuned"])
    reference = {}
    if reference_mode == "tuned":
        reference = {
            "owner_smoothing_seconds": trial.suggest_float("ref_owner_smoothing_seconds", .2, 1.5, log=True),
            "activity_smoothing_seconds": trial.suggest_float("ref_activity_smoothing_seconds", .2, 2.0, log=True),
            "nearest_scale_m": trial.suggest_float("ref_nearest_scale_m", 2.0, 8.0, log=True),
            "second_scale_m": trial.suggest_float("ref_second_scale_m", 4.0, 16.0, log=True),
            "second_weight": trial.suggest_float("ref_second_weight", 0.0, .8),
            "owner_strength": trial.suggest_float("ref_owner_strength", 1.0, 8.0, log=True),
            "live_bias": trial.suggest_float("ref_live_bias", -.5, 1.0),
            "activity_strength": trial.suggest_float("ref_activity_strength", .5, 4.0, log=True),
            "outside_penalty": trial.suggest_float("ref_outside_penalty", 0.0, 1.5),
            "ball_motion_bonus": trial.suggest_float("ref_ball_motion_bonus", 0.0, .8),
        }
    if reference_only:
        blend_mode = "static"
        neural_weight = shared_visible = shared_missing = 0.0
        visible_live = missing_live = visible_owner = missing_owner = 0.0
    else:
        blend_mode = trial.suggest_categorical(
            "blend_mode", ["static", "quality_shared", "quality_separate"])
        neural_weight = .5
        shared_visible = shared_missing = .5
        visible_live = missing_live = visible_owner = missing_owner = .5
        if blend_mode == "static":
            neural_weight = trial.suggest_float("neural_weight", 0.0, 1.0)
            shared_visible = shared_missing = neural_weight
            visible_live = missing_live = visible_owner = missing_owner = neural_weight
        elif blend_mode == "quality_shared":
            shared_visible = trial.suggest_float("visible_neural_weight", 0.0, 1.0)
            shared_missing = trial.suggest_float("missing_neural_weight", 0.0, 1.0)
            neural_weight = shared_visible
            visible_live = visible_owner = shared_visible
            missing_live = missing_owner = shared_missing
        else:
            visible_live = trial.suggest_float("visible_live_weight", 0.0, 1.0)
            missing_live = trial.suggest_float("missing_live_weight", 0.0, 1.0)
            visible_owner = trial.suggest_float("visible_owner_weight", 0.0, 1.0)
            missing_owner = trial.suggest_float("missing_owner_weight", 0.0, 1.0)
            shared_visible = (visible_live + visible_owner) / 2
            shared_missing = (missing_live + missing_owner) / 2
            neural_weight = shared_visible
    transition_seconds = (2.0 if blend_mode == "static" else
                          trial.suggest_float("transition_seconds", .25, 8.0, log=True))
    live_temperature = trial.suggest_float("live_temperature", .5, 2.5, log=True)
    owner_temperature = trial.suggest_float("owner_temperature", .5, 2.5, log=True)
    switch_cost_type = trial.suggest_categorical("switch_cost_type", ["scalar", "typed"])
    if switch_cost_type == "scalar":
        switch_cost: float | dict[str, float] = trial.suggest_float("switch_cost", 0.0, 8.0)
    else:
        switch_cost = {key: trial.suggest_float(f"switch_{key}", 0.0, 8.0)
                       for key in ("owner_switch", "to_dead", "from_dead")}
    # scale and temperature enter only through scale / temperature, so search
    # that effective multiplier once and keep boundary_temperature at one.
    boundary_scale = (0.0 if reference_only else
                      trial.suggest_float("boundary_scale", 0.0, 2.0))
    boundary_temperature = 1.0
    config = {"version": 1, "reference_mode": reference_mode, "reference": reference,
              "blend_mode": blend_mode, "neural_weight": neural_weight,
              "visible_neural_weight": shared_visible, "missing_neural_weight": shared_missing,
              "visible_live_weight": visible_live, "missing_live_weight": missing_live,
              "visible_owner_weight": visible_owner, "missing_owner_weight": missing_owner,
              "transition_seconds": transition_seconds, "live_temperature": live_temperature,
              "owner_temperature": owner_temperature, "boundary_temperature": boundary_temperature,
              "switch_cost_type": switch_cost_type, "switch_cost": switch_cost,
              "boundary_scale": boundary_scale}
    _validate_config(config)
    return config


def decode_arrays(arrays: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    """Blend cached neural/reference scores, calibrate, then decode one half."""
    _validate_config(config)
    required = {"neural_live", "neural_home", "neural_boundary", "ball_valid",
                "nearest_ball_seconds"}
    if required - arrays.keys():
        raise ValueError(f"decoder arrays lack keys: {sorted(required - arrays.keys())}")
    live = np.asarray(arrays["neural_live"], dtype=np.float64)
    home = np.asarray(arrays["neural_home"], dtype=np.float64)
    boundary = np.asarray(arrays["neural_boundary"], dtype=np.float64)
    ball_valid = np.asarray(arrays["ball_valid"])
    seconds = np.asarray(arrays["nearest_ball_seconds"], dtype=np.float64)
    if live.ndim != 1 or home.shape != live.shape or boundary.shape != (len(live), 3, 3):
        raise ValueError("neural arrays must have shapes [T], [T], and [T,3,3]")
    if ball_valid.shape != live.shape or ball_valid.dtype.kind != "b" or seconds.shape != live.shape:
        raise ValueError("ball_valid and nearest_ball_seconds must have shape [T]")
    if (not np.isfinite(live).all() or not np.isfinite(home).all()
            or not np.isfinite(boundary).all() or not np.isfinite(seconds).all()
            or np.any(seconds < 0) or np.any((live < 0) | (live > 1))
            or np.any((home < 0) | (home > 1))):
        raise ValueError("decoder arrays contain invalid values")
    basis = {key: value for key, value in arrays.items() if key.startswith("ref_")}
    reference = reference_from_basis(basis, config["reference"] if config["reference_mode"] == "tuned" else None)
    if len(reference["live_probability"]) != len(live):
        raise ValueError("reference basis timeline length differs from neural scores")
    mode = config["blend_mode"]
    gate = np.minimum(seconds, config["transition_seconds"]) / config["transition_seconds"]
    gate[ball_valid] = 0.0
    if mode == "static":
        weight_live = weight_owner = np.full(len(live), config["neural_weight"])
    elif mode == "quality_shared":
        w = config["visible_neural_weight"] + (config["missing_neural_weight"]
                                               - config["visible_neural_weight"]) * gate
        weight_live = weight_owner = w
    else:
        weight_live = config["visible_live_weight"] + (config["missing_live_weight"]
                                                        - config["visible_live_weight"]) * gate
        weight_owner = config["visible_owner_weight"] + (config["missing_owner_weight"]
                                                          - config["visible_owner_weight"]) * gate
    def blend(neural_p: np.ndarray, reference_p: np.ndarray, weights: np.ndarray) -> np.ndarray:
        eps = 1e-6
        p = np.clip(neural_p, eps, 1 - eps)
        r = np.clip(reference_p, eps, 1 - eps)
        logit = weights * (np.log(p) - np.log1p(-p)) + (1 - weights) * (
            np.log(r) - np.log1p(-r))
        output = _sigmoid(logit)
        output[weights == 0] = reference_p[weights == 0]
        output[weights == 1] = neural_p[weights == 1]
        return output
    blended_live = blend(live, reference["live_probability"], weight_live)
    blended_home = blend(home, reference["home_probability"], weight_owner)
    boundary_weight = (weight_live + weight_owner) / 2
    boundary_scores = boundary * boundary_weight[:, None, None] * config["boundary_scale"]
    calibration = CalibrationConfig(config["live_temperature"], config["owner_temperature"],
                                    config["boundary_temperature"])
    calibrated_live, calibrated_home, calibrated_boundary = apply_temperature_calibration(
        blended_live, blended_home, boundary_scores, calibration)
    emissions = emissions_from_probabilities(calibrated_live, calibrated_home)
    return decode_half(emissions, calibrated_boundary,
                       switch_cost=_switch_matrix(config))
