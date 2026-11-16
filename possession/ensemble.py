"""One-parameter, team-symmetric blend of neural and geometry-only scores."""

from collections.abc import Mapping

import numpy as np


def _checked(scores: Mapping, name: str) -> dict[str, np.ndarray]:
    result = {}
    for key in ("live_probability", "home_probability"):
        if key not in scores:
            raise ValueError(f"{name} lacks {key}")
        value = np.asarray(scores[key], dtype=np.float64)
        if value.ndim != 1:
            raise ValueError(f"{name} {key} shape must be [T]")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} {key} must be finite")
        if np.any((value < 0) | (value > 1)):
            raise ValueError(f"{name} {key} must lie in [0,1]")
        result[key] = value
    if result["home_probability"].shape != result["live_probability"].shape:
        raise ValueError(f"{name} probability shape mismatch")
    if "boundary" not in scores:
        raise ValueError(f"{name} lacks boundary")
    boundary = np.asarray(scores["boundary"], dtype=np.float64)
    if boundary.shape != (len(result["live_probability"]), 3, 3):
        raise ValueError(f"{name} boundary shape must be [T,3,3]")
    if not np.isfinite(boundary).all():
        raise ValueError(f"{name} boundary must be finite")
    result["boundary"] = boundary
    return result


def combine_scores(neural, reference, neural_weight: float = 1.0) -> dict[str, np.ndarray]:
    """Interpolate live/owner logits with one shared weight.

    Endpoints preserve input probabilities exactly. Between endpoints the
    neural boundary head is scaled by the same weight; the reference's neutral
    boundary field contributes no learned transition evidence.
    """
    if not np.isfinite(neural_weight) or not 0 <= neural_weight <= 1:
        raise ValueError("neural_weight must be finite and in [0,1]")
    neural = _checked(neural, "neural")
    reference = _checked(reference, "reference")
    if neural["live_probability"].shape != reference["live_probability"].shape:
        raise ValueError("neural and reference score shapes differ")
    if neural_weight == 0:
        return {key: value.copy() for key, value in reference.items()}
    if neural_weight == 1:
        return {key: value.copy() for key, value in neural.items()}

    def blend_probability(neural_p, reference_p):
        eps = 1e-6
        neural_p = np.clip(neural_p, eps, 1 - eps)
        reference_p = np.clip(reference_p, eps, 1 - eps)
        neural_logit = np.log(neural_p) - np.log1p(-neural_p)
        reference_logit = np.log(reference_p) - np.log1p(-reference_p)
        logit = neural_weight * neural_logit + (1-neural_weight) * reference_logit
        return 1.0 / (1.0 + np.exp(-logit))

    return {
        "live_probability": blend_probability(neural["live_probability"],
                                               reference["live_probability"]),
        "home_probability": blend_probability(neural["home_probability"],
                                               reference["home_probability"]),
        "boundary": neural["boundary"] * neural_weight,
    }
