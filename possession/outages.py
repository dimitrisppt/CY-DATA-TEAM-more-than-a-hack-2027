"""Label-free, half-local ball outage sampling and diagnostics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


def _finite_number(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return number


@dataclass(frozen=True)
class OutageConfig:
    mode: str = "legacy"
    clean_probability: float = 0.2
    full_probability: float = 0.05
    fraction_min: float = 0.30
    fraction_max: float = 0.66
    min_gap_seconds: float = 1.0
    max_gap_seconds: float = 10.0
    long_gap_probability: float = 0.2
    long_gap_min_seconds: float = 10.0
    long_gap_max_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.mode not in ("legacy", "correlated"):
            raise ValueError("mode must be legacy or correlated")
        for name in ("clean_probability", "full_probability", "fraction_min",
                     "fraction_max", "long_gap_probability"):
            value = _finite_number(name, getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.clean_probability + self.full_probability > 1:
            raise ValueError("clean and full probabilities must sum to at most one")
        if self.fraction_min > self.fraction_max:
            raise ValueError("fraction_min must not exceed fraction_max")
        for low, high in (("min_gap_seconds", "max_gap_seconds"),
                          ("long_gap_min_seconds", "long_gap_max_seconds")):
            if _finite_number(low, getattr(self, low), positive=True) > _finite_number(
                    high, getattr(self, high), positive=True):
                raise ValueError(f"{low} must not exceed {high}")

    def to_dict(self) -> dict:
        return asdict(self)


def generate_outage_mask(
    ball_valid: np.ndarray,
    fps: float,
    rng: np.random.Generator,
    *,
    missing_fraction: float,
    min_seconds: float = 1,
    max_seconds: float = 10,
    long_probability: float = 0,
    long_min_seconds: float = 10,
    long_max_seconds: float = 30,
) -> np.ndarray:
    """Return additional missing frames, reaching floor(fraction * half length).

    Each proposal is a contiguous interval. Sampling an available anchor gives
    guaranteed progress even when proposed intervals overlap existing outages.
    """
    valid = np.asarray(ball_valid)
    if valid.ndim != 1 or valid.dtype != np.bool_:
        raise ValueError("ball_valid must be a one-dimensional bool array")
    fps = _finite_number("fps", fps, positive=True)
    fraction = _finite_number("missing_fraction", missing_fraction)
    if not 0 <= fraction <= 1:
        raise ValueError("missing_fraction must be between zero and one")
    probability = _finite_number("long_probability", long_probability)
    if not 0 <= probability <= 1:
        raise ValueError("long_probability must be between zero and one")
    for low, high in (("min_seconds", "max_seconds"),
                      ("long_min_seconds", "long_max_seconds")):
        if _finite_number(low, locals()[low], positive=True) > _finite_number(
                high, locals()[high], positive=True):
            raise ValueError(f"{low} must not exceed {high}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator")

    n = len(valid)
    additional = np.zeros(n, dtype=bool)
    needed = max(0, math.floor(fraction * n) - int((~valid).sum()))
    while needed:
        available = np.flatnonzero(valid & ~additional)
        anchor = int(rng.choice(available))
        if rng.random() < probability:
            low, high = long_min_seconds, long_max_seconds
        else:
            low, high = min_seconds, max_seconds
        duration = float(rng.uniform(low, high)) if low != high else float(low)
        length = min(n, max(1, round(duration * fps)))
        start = max(0, min(anchor - int(rng.integers(length)), n - length))
        candidates = np.flatnonzero(valid[start:start + length] & ~additional[start:start + length]) + start
        # A uniformly placed final interval avoids favoring either half edge.
        if len(candidates) > needed:
            offset = int(rng.integers(len(candidates) - needed + 1))
            candidates = candidates[offset:offset + needed]
        additional[candidates] = True
        needed -= len(candidates)
    return additional


def gap_statistics(missing: np.ndarray, fps: float) -> dict:
    """Summarize contiguous missing runs in one half, including edge runs."""
    mask = np.asarray(missing)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("missing must be a one-dimensional bool array")
    fps = _finite_number("fps", fps, positive=True)
    ends = np.diff(np.r_[False, mask, False].astype(np.int8))
    lengths = np.flatnonzero(ends == -1) - np.flatnonzero(ends == 1)
    frames = {
        "min_frames": int(lengths.min()) if len(lengths) else 0,
        "median_frames": float(np.median(lengths)) if len(lengths) else 0.0,
        "p95_frames": float(np.percentile(lengths, 95)) if len(lengths) else 0.0,
        "max_frames": int(lengths.max()) if len(lengths) else 0,
    }
    return {
        "count": len(lengths),
        "missing_frames": int(mask.sum()),
        "fraction": float(mask.mean()) if len(mask) else 0.0,
        **frames,
        **{key.replace("frames", "seconds"): value / fps for key, value in frames.items()},
    }
