"""Finite three-state half-local Viterbi decoding (H=0, A=1, D=2)."""

import logging
from dataclasses import dataclass

import numpy as np

from possession._viterbi import viterbi

_LOG = logging.getLogger(__name__)


def emissions_from_probabilities(live, home, eps: float = 1e-6) -> np.ndarray:
    """Factor P(H/A/D) into live and conditional home probabilities."""
    p_live = np.asarray(live, dtype=np.float64)
    p_home = np.asarray(home, dtype=np.float64)
    if p_live.ndim != 1 or p_home.shape != p_live.shape:
        raise ValueError("live and home must have equal one-dimensional shapes")
    if not 0 < eps < 0.5 or not np.isfinite(eps):
        raise ValueError("eps must be finite and between zero and one half")
    if not np.isfinite(p_live).all() or not np.isfinite(p_home).all():
        raise ValueError("probabilities must be finite")
    if np.any((p_live < 0) | (p_live > 1) | (p_home < 0) | (p_home > 1)):
        raise ValueError("probabilities must lie in [0,1]")
    p_live = np.clip(p_live, eps, 1 - eps)
    p_home = np.clip(p_home, eps, 1 - eps)
    return np.stack((np.log(p_live) + np.log(p_home),
                     np.log(p_live) + np.log1p(-p_home),
                     np.log1p(-p_live)), axis=1)


@dataclass(frozen=True)
class DecodeDetails:
    path: np.ndarray
    score: float
    switches: int
    tie_count: int


def decode_half_details(log_emit, boundary=None, switch_cost=2.0) -> DecodeDetails:
    """Decode one contiguous half, exposing the deterministic tie audit.

    At each exact score tie, the path with fewer switches wins. Remaining
    exact ties prefer lower-index predecessor states, then lower-index final
    states (H, A, D). This final rule cannot be H/A equivariant on ties.
    """
    emit = np.asarray(log_emit, dtype=np.float64)
    if emit.ndim != 2 or emit.shape[1] != 3 or not np.isfinite(emit).all():
        raise ValueError("log_emit must be finite with shape [T,3]")
    n = len(emit)
    if boundary is None:
        transition = np.zeros((n, 3, 3), dtype=np.float64)
    else:
        transition = np.asarray(boundary, dtype=np.float64)
        if transition.shape != (n, 3, 3) or not np.isfinite(transition).all():
            raise ValueError("boundary must be finite with shape [T,3,3]")
    cost = np.asarray(switch_cost, dtype=np.float64)
    if cost.ndim == 0:
        cost = np.full((3, 3), float(cost), dtype=np.float64)
    if cost.shape != (3, 3) or not np.isfinite(cost).all():
        raise ValueError("switch_cost must be a finite scalar or [3,3] matrix")
    if n == 0:
        return DecodeDetails(np.empty(0, dtype=np.int8), 0.0, 0, 0)
    path, score, switches, ties = viterbi(emit, transition, cost)
    if ties:
        _LOG.info("Viterbi encountered %d exact score ties", ties)
    return DecodeDetails(path, score, switches, ties)


def decode_half(log_emit, boundary=None, switch_cost=2.0) -> np.ndarray:
    """Return H/A/D states for one half; split halves before calling."""
    return decode_half_details(log_emit, boundary, switch_cost).path
