"""Label-free, framewise blend gated by distance from observed ball tracking."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

import numpy as np

from possession.ensemble import _checked, combine_scores


def _finite_number(name: str, value, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_, str)) or np.ndim(value) != 0:
        raise TypeError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return number


@dataclass(frozen=True)
class QualityBlendConfig:
    visible_neural_weight: float = .5
    missing_neural_weight: float = .5
    transition_seconds: float = 2.0

    def __post_init__(self):
        for name in ("visible_neural_weight", "missing_neural_weight"):
            value = _finite_number(name, getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0,1]")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "transition_seconds", _finite_number(
            "transition_seconds", self.transition_seconds, positive=True))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping) -> QualityBlendConfig:
        if not isinstance(value, Mapping):
            raise TypeError("quality blend config must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown quality blend config fields: {sorted(map(str, unknown))}")
        return cls(**value)


def combine_quality_scores(neural, reference, ball_visible, nearest_ball_seconds,
                           config: QualityBlendConfig) -> dict[str, np.ndarray]:
    """Blend probabilities with one team-symmetric weight per timeline frame.

    ``nearest_ball_seconds`` is an offline, bidirectional distance to observed
    ball tracking. It is a label-free feature and must be finite even when the
    entire supplied timeline has missing ball observations.
    """
    if isinstance(config, Mapping):
        config = QualityBlendConfig.from_dict(config)
    if not isinstance(config, QualityBlendConfig):
        raise TypeError("config must be a QualityBlendConfig or mapping")

    neural = _checked(neural, "neural")
    reference = _checked(reference, "reference")
    n = len(neural["live_probability"])
    if len(reference["live_probability"]) != n:
        raise ValueError("neural and reference score shapes differ")
    if ball_visible is None:
        raise ValueError("ball_visible is required")
    visible = np.asarray(ball_visible)
    if visible.shape != (n,) or visible.dtype.kind != "b":
        raise ValueError("ball_visible must be a boolean [T] array")
    if nearest_ball_seconds is None:
        raise ValueError("nearest_ball_seconds is required")
    try:
        raw_seconds = np.asarray(nearest_ball_seconds)
        if raw_seconds.dtype.kind not in "iuf":
            raise ValueError("nearest_ball_seconds must be numeric [T]")
        seconds = raw_seconds.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("nearest_ball_seconds must be numeric [T]") from exc
    if seconds.shape != (n,) or not np.isfinite(seconds).all() or np.any(seconds < 0):
        raise ValueError("nearest_ball_seconds must be nonnegative finite [T]")

    gate = np.minimum(seconds, config.transition_seconds) / config.transition_seconds
    gate[visible] = 0.0
    weight = (config.visible_neural_weight
              + (config.missing_neural_weight - config.visible_neural_weight) * gate)
    # Preserve the existing blend's exact endpoint and constant-weight behavior.
    if n == 0 or np.all(weight == weight[0]):
        return combine_scores(neural, reference,
                              config.visible_neural_weight if n == 0 else float(weight[0]))

    def blend_probability(neural_p, reference_p):
        eps = 1e-6
        neural_clipped = np.clip(neural_p, eps, 1 - eps)
        reference_clipped = np.clip(reference_p, eps, 1 - eps)
        neural_logit = np.log(neural_clipped) - np.log1p(-neural_clipped)
        reference_logit = np.log(reference_clipped) - np.log1p(-reference_clipped)
        logit = weight * neural_logit + (1 - weight) * reference_logit
        blended = 1.0 / (1.0 + np.exp(-logit))
        blended[weight == 0] = reference_p[weight == 0]
        blended[weight == 1] = neural_p[weight == 1]
        return blended

    boundary = neural["boundary"] * weight[:, None, None]
    boundary[weight == 0] = reference["boundary"][weight == 0]
    return {
        "live_probability": blend_probability(neural["live_probability"],
                                               reference["live_probability"]),
        "home_probability": blend_probability(neural["home_probability"],
                                               reference["home_probability"]),
        "boundary": boundary,
    }
