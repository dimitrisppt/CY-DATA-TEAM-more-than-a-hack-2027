"""Preselected, team-symmetric temperature transforms for inference scores.

Choose the temperatures on eligible inner validation data and persist them in
the run configuration. Applying this transform never accesses target labels.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationConfig:
    live_temperature: float = 1.0
    owner_temperature: float = 1.0
    boundary_temperature: float = 1.0
    eps: float = 1e-6

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def apply_temperature_calibration(
    live, home, boundary=None, config: CalibrationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Scale live/owner logits and all six boundary scores with shared scalars."""
    cfg = CalibrationConfig() if config is None else config
    for name in ("live_temperature", "owner_temperature", "boundary_temperature"):
        value = getattr(cfg, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not np.isfinite(cfg.eps) or not 0 < cfg.eps < 0.5:
        raise ValueError("eps must be finite and between zero and one half")
    p_live = np.asarray(live, dtype=np.float64)
    p_home = np.asarray(home, dtype=np.float64)
    if p_live.ndim != 1 or p_home.shape != p_live.shape:
        raise ValueError("live and home must have equal one-dimensional shapes")
    if not np.isfinite(p_live).all() or not np.isfinite(p_home).all():
        raise ValueError("probabilities must be finite")
    if np.any((p_live < 0) | (p_live > 1) | (p_home < 0) | (p_home > 1)):
        raise ValueError("probabilities must lie in [0,1]")

    def scale(p: np.ndarray, temperature: float) -> np.ndarray:
        clipped = np.clip(p, cfg.eps, 1 - cfg.eps)
        logit = np.log(clipped) - np.log1p(-clipped)
        return 1.0 / (1.0 + np.exp(-logit / temperature))

    scaled_boundary = None
    if boundary is not None:
        scores = np.asarray(boundary, dtype=np.float64)
        if scores.shape != (len(p_live), 3, 3) or not np.isfinite(scores).all():
            raise ValueError("boundary must be finite with shape [T,3,3]")
        scaled_boundary = scores / cfg.boundary_temperature
    return (scale(p_live, cfg.live_temperature),
            scale(p_home, cfg.owner_temperature), scaled_boundary)
