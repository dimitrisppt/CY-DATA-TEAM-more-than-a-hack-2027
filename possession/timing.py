"""Fold-local temporal calibration over frozen H/A/D boundary sequences.

Prediction features never read truth. Callers must audit upstream training ancestry
before fitting, and pass the same match ID for correlated outage scenarios.
Positive offsets move a boundary later on the unchanged integer frame grid.
SciPy is imported only by fitting, inference and fixed shifts require NumPy only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from evaluation.possession_metrics import D, X, transitions, valid_runs

FAMILIES = ("restart", "stoppage", "team_switch")
FEATURE_NAMES = (
    "offset", "absolute_offset", "squared_offset", "typed_boundary_score",
    "before_left", "after_right", "before_right", "after_left",
    "before_left_mean", "after_right_mean", "relative_slope", "entropy",
    "ball_visible", "local_visible", "nearest_ball_seconds", "disagreement",
    "log_left_length", "log_right_length", "boundary_contrast",
)


@dataclass(frozen=True)
class TimingConfig:
    radius: int = 25
    l2: float = .05
    family_l2: float = .2
    movement_penalty: float = .05
    evidence_weight: float = 0.
    max_iter: int = 200
    prior_strength: float = .05

    def __post_init__(self):
        _radius(self.radius)
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        for name in ("l2", "family_l2", "movement_penalty", "evidence_weight", "prior_strength"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class BoundaryBatch:
    features: np.ndarray
    candidate_times: np.ndarray
    valid: np.ndarray
    boundaries: np.ndarray
    families: np.ndarray
    before: np.ndarray
    after: np.ndarray
    run_ids: np.ndarray
    runs: tuple[tuple[int, int], ...]
    original_labels: np.ndarray
    state_probability: np.ndarray
    evidence_available: np.ndarray
    radius: int


@dataclass(frozen=True)
class TimingTargets:
    offsets: np.ndarray
    matched: np.ndarray
    ambiguous: np.ndarray

    @property
    def unmatched(self):
        return ~self.matched & ~self.ambiguous


@dataclass(frozen=True)
class TimingPrediction:
    """Probabilities already include the reliability mixture with an atom at zero."""

    probabilities: np.ndarray
    reliability: np.ndarray


def _radius(value):
    if not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("radius must be a nonnegative integer")
    return int(value)


def _codes(labels):
    values = np.asarray(labels)
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any((values < 0) | (values > X)):
        raise ValueError("labels must be a one-dimensional integer H/A/D/X code array")
    return values.copy()


def _runs(labels, run_ids=None):
    if run_ids is None:
        return tuple(valid_runs(labels))
    ids = np.asarray(run_ids)
    if ids.shape != labels.shape or ids.dtype.kind not in "iu":
        raise ValueError("run_ids must be an integer vector on the label grid")
    # Every ID change is an immovable half or source-run edge, even without X.
    result = []
    edges = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(labels)]
    for left, right in pairwise(edges):
        result.extend((left + start, left + stop) for start, stop in valid_runs(labels[left:right]))
    return tuple((int(start), int(stop)) for start, stop in result)


def _family(before, after):
    return 0 if before == D else 1 if after == D else 2


def _array(arrays, name, shape, fill):
    if name not in arrays:
        raise ValueError(f"missing timing evidence: {name}")
    raw = np.asarray(arrays[name], dtype=np.float64)
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    finite = np.isfinite(raw)
    return np.where(finite, raw, fill), finite


def make_boundary_batch(labels, arrays: Mapping, radius=25, *, run_ids=None):
    """Extract relative team features inside explicit runs, with no truth access.

    ``run_ids`` or ``arrays['run_ids']`` splits adjacent halves even if no X
    separates them. X is always an immovable exclusion. Neural arrays are required,
    nonfinite evidence gets neutral fills and unavailable rows fall back to identity.
    Live/home inputs are probabilities. Typed boundary inputs are signed model
    scores retained in [-30, 30], then standardized using fit batches only.
    ``fine_coarse_disagreement`` is optional and must have shape [T].
    """
    radius = _radius(radius)
    labels = _codes(labels)
    n = len(labels)
    runs = _runs(labels, arrays.get("run_ids") if run_ids is None else run_ids)
    live, live_ok = _array(arrays, "neural_live", (n,), .5)
    home, home_ok = _array(arrays, "neural_home", (n,), .5)
    boundary, boundary_ok = _array(arrays, "neural_boundary", (n, 3, 3), 0.)
    ball, _ = _array(arrays, "ball_valid", (n,), 0.)
    gap, _ = _array(arrays, "nearest_ball_seconds", (n,), 10.)
    disagreement = np.zeros(n)
    if "fine_coarse_disagreement" in arrays:
        disagreement, _ = _array(arrays, "fine_coarse_disagreement", (n,), 0.)
    live, home = np.clip(live, 0, 1), np.clip(home, 0, 1)
    state = np.column_stack((live * home, live * (1 - home), 1 - live))
    entropy = -np.sum(state * np.log(np.maximum(state, 1e-12)), axis=1)
    ball, gap = np.clip(ball, 0, 1), np.clip(gap, 0, 10)
    # The frozen boundary head emits signed scores, not probabilities. Preserve
    # their ordering and negative peaks, with an explicit stability bound only.
    boundary = np.clip(boundary, -30, 30)
    prefix = np.vstack((np.zeros((1, 3)), np.cumsum(state, axis=0)))
    visible_prefix = np.r_[0., np.cumsum(ball)]
    contrasts = {(before, after): state[:, after] - state[:, before]
                 for before in range(3) for after in range(3) if before != after}
    rows, features, candidate_times, masks, available = [], [], [], [], []
    delta = np.arange(-radius, radius + 1)
    for run_id, (start, stop) in enumerate(runs):
        typed = transitions(labels, [(start, stop)])
        times = [start] + [item[0] for item in typed] + [stop]
        for position, (time, before, after) in enumerate(typed):
            candidates = time + delta
            valid = (candidates > start) & (candidates < stop)
            clipped = np.clip(candidates, start + 1, stop - 1)
            left, right = np.maximum(start, clipped - 3), np.minimum(stop, clipped + 3)
            before_mean = (prefix[clipped, before] - prefix[left, before]) / (clipped - left)
            after_mean = (prefix[right, after] - prefix[clipped, after]) / (right - clipped)
            contrast = contrasts[before, after]
            row = np.column_stack((
                delta / max(radius, 1), np.abs(delta) / max(radius, 1),
                (delta / max(radius, 1)) ** 2, boundary[clipped, before, after],
                state[clipped - 1, before], state[clipped, after],
                state[clipped, before], state[clipped - 1, after], before_mean, after_mean,
                contrast[right - 1] - contrast[left], entropy[clipped], ball[clipped],
                (visible_prefix[right] - visible_prefix[left]) / (right - left), gap[clipped],
                np.clip(disagreement[clipped], 0, 10),
                np.full(len(delta), np.log1p(time - times[position])),
                np.full(len(delta), np.log1p(times[position + 2] - time)),
                contrast[clipped] - contrast[clipped - 1],
            ))
            features.append(row)
            candidate_times.append(candidates)
            masks.append(valid)
            available.append(bool(np.any(valid & live_ok[clipped] & home_ok[clipped]
                                         & boundary_ok[clipped, before, after])))
            rows.append((time, _family(before, after), before, after, run_id))
    row_array = np.asarray(rows, dtype=np.int64).reshape(-1, 5)
    return BoundaryBatch(
        features=np.asarray(features, dtype=np.float64).reshape(-1, len(delta), len(FEATURE_NAMES)),
        candidate_times=np.asarray(candidate_times, dtype=np.int64).reshape(-1, len(delta)),
        valid=np.asarray(masks, dtype=bool).reshape(-1, len(delta)),
        boundaries=row_array[:, 0], families=row_array[:, 1], before=row_array[:, 2],
        after=row_array[:, 3], run_ids=row_array[:, 4], runs=runs, original_labels=labels,
        state_probability=state, evidence_available=np.asarray(available, bool), radius=radius,
    )


def associate_targets(batch: BoundaryBatch, gt):
    """Conservative ordered exact-pair matching, omitting all competing edges.

    Any multiple candidate, competing prediction or crossing edge is explicitly
    ambiguous, with no offset supervision. Missing or out-of-radius transitions
    remain unmatched. Truth X runs also bound association and are never bridged.
    """
    gt = _codes(gt)
    if gt.shape != batch.original_labels.shape:
        raise ValueError("truth and predictions must share the same grid")
    count = len(batch.boundaries)
    offsets = np.zeros(count, np.int64)
    matched = np.zeros(count, bool)
    ambiguous = np.zeros(count, bool)
    for run_id, (start, stop) in enumerate(batch.runs):
        pred_indices = np.flatnonzero(batch.run_ids == run_id)
        truth_runs = [(start + left, start + right) for left, right in valid_runs(gt[start:stop])]
        truth_events = transitions(gt, truth_runs)
        truth_ids = np.full(stop - start, -1, np.int64)
        for truth_id, (left, right) in enumerate(truth_runs):
            truth_ids[left - start:right - start] = truth_id
        event_ids = {event[0]: truth_ids[event[0] - start] for event in truth_events}
        by_type = {}
        for event_index, (time, before, after) in enumerate(truth_events):
            by_type.setdefault((before, after), []).append((time, event_index))
        edges = {}
        reverse = {}
        for index in pred_indices:
            time = batch.boundaries[index]
            truth_id = truth_ids[time - start]
            if truth_id < 0 or truth_ids[time - start - 1] != truth_id:
                continue
            typed = by_type.get((batch.before[index], batch.after[index]), [])
            if not typed:
                continue
            typed_times = np.array([item[0] for item in typed])
            left = np.searchsorted(typed_times, time - batch.radius)
            right = np.searchsorted(typed_times, time + batch.radius, side="right")
            possible = [event_index for target_time, event_index in typed[left:right]
                        if event_ids[target_time] == truth_id]
            edges[index] = possible
            for event_index in possible:
                reverse.setdefault(event_index, []).append(index)
            if len(possible) > 1:
                ambiguous[index] = True
        for indices in reverse.values():
            if len(indices) > 1:
                ambiguous[indices] = True
        unique = [(index, candidates[0]) for index, candidates in edges.items()
                  if len(candidates) == 1 and not ambiguous[index]]
        # Reject unique pair edges whose truth order conflicts with prediction order.
        if unique:
            truth_order = np.array([event for _, event in unique])
            previous = np.r_[-1, np.maximum.accumulate(truth_order)[:-1]]
            following = np.r_[np.minimum.accumulate(truth_order[::-1])[::-1][1:], len(truth_events)]
            for (index, event_index), lower, upper in zip(unique, previous, following):
                if event_index <= lower or event_index >= upper:
                    ambiguous[index] = True
                else:
                    matched[index] = True
                    offsets[index] = truth_events[event_index][0] - batch.boundaries[index]
    return TimingTargets(offsets, matched, ambiguous)


def _softmax(scores, valid):
    if not len(scores):
        return np.empty_like(scores)
    values = np.where(valid, scores, -np.inf)
    values -= np.max(values, axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def _context(batch):
    if not len(batch.boundaries):
        return np.empty((0, 2 * len(FEATURE_NAMES)))
    mean = np.sum(batch.features * batch.valid[:, :, None], axis=1) / batch.valid.sum(axis=1)[:, None]
    maximum = np.max(np.where(batch.valid[:, :, None], batch.features, -np.inf), axis=1)
    return np.column_stack((mean, maximum))


def _standardizer(values, weights):
    if not len(values) or weights.sum() == 0:
        return np.zeros(values.shape[-1]), np.ones(values.shape[-1])
    weights = weights / weights.sum()
    mean = np.sum(values * weights[:, None], axis=0)
    variance = np.sum((values - mean) ** 2 * weights[:, None], axis=0)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.
    return mean, scale


@dataclass
class TimingModel:
    config: TimingConfig
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    reliability_mean: np.ndarray
    reliability_scale: np.ndarray
    reliability_coefficients: np.ndarray
    fit_diagnostics: dict

    def predict(self, batch: BoundaryBatch):
        if batch.radius != self.config.radius:
            raise ValueError("candidate radius differs from fitted model")
        features = (batch.features - self.feature_mean) / self.feature_scale
        effective = self.coefficients[0] + self.coefficients[1:][batch.families]
        score = np.einsum("nkf,nf->nk", features, effective)
        score -= self.config.prior_strength * np.abs(batch.candidate_times - batch.boundaries[:, None])
        probability = _softmax(score, batch.valid)
        context = (_context(batch) - self.reliability_mean) / self.reliability_scale
        logits = context @ self.reliability_coefficients[1:] + self.reliability_coefficients[0]
        reliability = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        reliability *= batch.evidence_available
        probability *= reliability[:, None]
        probability[:, batch.radius] += 1 - reliability
        return TimingPrediction(probability, reliability)

    def save(self, path):
        """Save JSON or pickle-free NPZ with JSON schema and configuration metadata."""
        path = Path(path)
        arrays = {name: getattr(self, name) for name in (
            "feature_mean", "feature_scale", "coefficients", "reliability_mean",
            "reliability_scale", "reliability_coefficients")}
        metadata = {"version": 2, "config": asdict(self.config),
                    "feature_names": list(FEATURE_NAMES), "fit_diagnostics": self.fit_diagnostics}
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps({**metadata, "arrays": {name: value.tolist()
                            for name, value in arrays.items()}}, sort_keys=True) + "\n")
        else:
            with path.open("wb") as handle:
                np.savez_compressed(handle, metadata=np.array(json.dumps(metadata, sort_keys=True)), **arrays)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix.lower() == ".json":
            metadata = json.loads(path.read_text())
            arrays = {name: np.asarray(value, np.float64) for name, value in metadata.pop("arrays").items()}
        else:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"]))
                arrays = {name: data[name].copy() for name in data.files if name != "metadata"}
        if metadata["version"] != 2 or metadata["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("unsupported timing model schema")
        return cls(TimingConfig(**metadata["config"]), **arrays,
                   fit_diagnostics=metadata["fit_diagnostics"])


def _balanced_weights(base, groups, eligible):
    result = np.zeros(len(base))
    for group in dict.fromkeys(groups.tolist()):
        selected = (groups == group) & eligible
        mass = base[selected].sum()
        if mass > 0:
            result[selected] = base[selected] / mass
    total = result.sum()
    return result / total if total else result


def fit_timing_model(batches: Sequence[BoundaryBatch], targets: Sequence[TimingTargets], *,
                     match_ids=None, weights=None, config=None):
    """Fit shared linear candidate scores, regularized family effects and matchability.

    ``weights`` is one optional nonnegative boundary-weight array per batch.
    Each match contributes equal total loss to each head, regardless of frame
    count or repeated scenarios. Scaling uses training batches only. Match IDs
    do not enter features. Upstream ancestry validation belongs to the runner.
    """
    from scipy.optimize import minimize

    batches, targets = list(batches), list(targets)
    if not batches or len(batches) != len(targets):
        raise ValueError("provide equal nonempty lists of batches and targets")
    config = config or TimingConfig(radius=batches[0].radius)
    if any(batch.radius != config.radius for batch in batches):
        raise ValueError("all training batches must use the configured radius")
    if match_ids is None:
        match_ids = list(range(len(batches)))
    if len(match_ids) != len(batches):
        raise ValueError("one match ID is required for each batch")
    if weights is None:
        weights = [np.ones(len(batch.boundaries)) for batch in batches]
    if len(weights) != len(batches):
        raise ValueError("one weight vector is required per batch")
    for batch, target, weight in zip(batches, targets, weights):
        n = len(batch.boundaries)
        if any(np.shape(getattr(target, name)) != (n,) for name in ("offsets", "matched", "ambiguous")):
            raise ValueError("target vectors must align with boundaries")
        if np.shape(weight) != (n,) or not np.isfinite(weight).all() or np.any(np.asarray(weight) < 0):
            raise ValueError("weights must be finite nonnegative boundary vectors")
        selected = target.matched & ~target.ambiguous
        offset = np.asarray(target.offsets)
        if offset.dtype.kind not in "iu" or np.any(np.abs(offset[selected]) > config.radius):
            raise ValueError("matched offsets must be integer candidates inside the radius")
        if np.any(selected & ~batch.valid[np.arange(n), np.clip(offset + config.radius, 0, 2 * config.radius)]):
            raise ValueError("matched offset crosses a valid-run boundary")
    features = np.concatenate([batch.features for batch in batches])
    valid = np.concatenate([batch.valid for batch in batches])
    families = np.concatenate([batch.families for batch in batches])
    groups = np.concatenate([np.repeat(str(group), len(batch.boundaries))
                             for group, batch in zip(match_ids, batches)])
    base = np.concatenate([np.asarray(weight, np.float64) for weight in weights])
    matched = np.concatenate([target.matched & ~target.ambiguous for target in targets])
    offsets = np.concatenate([target.offsets for target in targets])
    all_weights = _balanced_weights(base, groups, np.ones(len(base), bool))
    candidate_weights = all_weights[:, None] * valid / valid.sum(axis=1, keepdims=True)
    feature_mean, feature_scale = _standardizer(features[valid], candidate_weights[valid])
    context = np.concatenate([_context(batch) for batch in batches])
    reliability_mean, reliability_scale = _standardizer(context, all_weights)
    context = (context - reliability_mean) / reliability_scale
    design = np.column_stack((np.ones(len(context)), context))
    f = len(FEATURE_NAMES)
    coefficients = np.zeros((4, f))
    offset_weights = _balanced_weights(base, groups, matched)
    prior = -config.prior_strength * np.abs(np.arange(-config.radius, config.radius + 1))
    chosen = np.flatnonzero(offset_weights > 0)
    z = (features[chosen] - feature_mean) / feature_scale
    chosen_families = families[chosen]
    chosen_valid = valid[chosen]
    del features
    regularization = np.full((4, f), config.family_l2)
    regularization[0] = config.l2

    def offset_objective(parameters):
        coefficients = parameters.reshape(4, f)
        effective = coefficients[0] + coefficients[1:][chosen_families]
        scores = np.einsum("nkf,nf->nk", z, effective) + prior
        probability = _softmax(scores, chosen_valid)
        rows = np.arange(len(chosen))
        columns = offsets[chosen] + config.radius
        log_loss = -np.sum(offset_weights[chosen] * np.log(np.maximum(probability[rows, columns], 1e-300)))
        residual = probability.copy()
        residual[rows, columns] -= 1
        residual *= offset_weights[chosen, None]
        gradient_by_row = np.einsum("nk,nkf->nf", residual, z)
        gradient = np.zeros_like(coefficients)
        gradient[0] = gradient_by_row.sum(axis=0)
        np.add.at(gradient[1:], chosen_families, gradient_by_row)
        gradient += regularization * coefficients
        penalty = .5 * np.sum(regularization * coefficients ** 2)
        return log_loss + penalty, gradient.ravel()

    offset_result = None
    if len(chosen):
        offset_result = minimize(offset_objective, coefficients.ravel(), method="L-BFGS-B", jac=True,
                                 options={"maxiter": config.max_iter, "ftol": 1e-10, "gtol": 1e-7})
        if not np.isfinite(offset_result.fun) or not np.isfinite(offset_result.x).all():
            raise ValueError("nonfinite timing fit")
        coefficients = offset_result.x.reshape(4, f)

    # Ambiguous and unmatched boundaries are negative matchability examples.
    reliability_coefficients = np.zeros(design.shape[1])
    reliability_result = None
    def reliability_objective(parameters):
        logits = design @ parameters
        probability = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        loss = np.sum(all_weights * (np.logaddexp(0, logits) - matched * logits))
        penalty = np.full(len(parameters), config.l2)
        penalty[0] = .001
        return (loss + .5 * np.sum(penalty * parameters ** 2),
                design.T @ (all_weights * (probability - matched)) + penalty * parameters)

    if len(base) and all_weights.sum():
        reliability_result = minimize(reliability_objective, reliability_coefficients,
                                      method="L-BFGS-B", jac=True,
                                      options={"maxiter": config.max_iter, "ftol": 1e-10, "gtol": 1e-7})
        if not np.isfinite(reliability_result.fun) or not np.isfinite(reliability_result.x).all():
            raise ValueError("nonfinite reliability fit")
        reliability_coefficients = reliability_result.x
    diagnostics = {
        "boundaries": len(base), "matched": int(matched.sum()),
        "ambiguous": int(sum(target.ambiguous.sum() for target in targets)),
        "offset_loss": float(offset_result.fun) if offset_result is not None else 0.,
        "offset_converged": bool(offset_result.success) if offset_result is not None else True,
        "offset_iterations": int(offset_result.nit) if offset_result is not None else 0,
        "offset_gradient_norm": float(np.linalg.norm(offset_result.jac, ord=np.inf)) if offset_result is not None else 0.,
        "offset_message": str(offset_result.message) if offset_result is not None else "no matched boundaries",
        "reliability_loss": float(reliability_result.fun) if reliability_result is not None else 0.,
        "reliability_iterations": int(reliability_result.nit) if reliability_result is not None else 0,
        "reliability_message": str(reliability_result.message) if reliability_result is not None else "no boundaries",
        "reliability_converged": bool(reliability_result.success) if reliability_result is not None else True,
        "matches": len(set(groups.tolist())),
    }
    return TimingModel(config, feature_mean, feature_scale, coefficients, reliability_mean,
                       reliability_scale, reliability_coefficients, diagnostics)


def _ordered_decode(batch, scores, movement):
    """Maximize additive boundary scores with strict ordering and stable ties."""
    selected = batch.boundaries.copy()
    for run_id in range(len(batch.runs)):
        indices = np.flatnonzero(batch.run_ids == run_id)
        if not len(indices):
            continue
        first = indices[0]
        dp = np.where(batch.valid[first], scores[first], -np.inf)
        travel = np.where(batch.valid[first], movement[first], np.inf)
        backpointers = []
        for previous, index in pairwise(indices):
            # Prefix winners reduce ordered transitions from K squared to K.
            # Stable ties prefer minimum cumulative movement, then first index.
            prefix_best = np.empty(len(dp), np.int64)
            best = -1
            for column in range(len(dp)):
                if np.isfinite(dp[column]) and (
                    best < 0 or dp[column] > dp[best]
                    or (dp[column] == dp[best] and travel[column] < travel[best])
                ):
                    best = column
                prefix_best[column] = best
            positions = np.searchsorted(batch.candidate_times[previous],
                                        batch.candidate_times[index], side="left") - 1
            back = prefix_best[np.maximum(positions, 0)]
            feasible = batch.valid[index] & (positions >= 0) & (back >= 0)
            dp = np.where(feasible, dp[back] + scores[index], -np.inf)
            travel = np.where(feasible, travel[back] + movement[index], np.inf)
            backpointers.append(back)
        choices = np.flatnonzero(np.isfinite(dp))
        if not len(choices):
            # Original boundaries are always feasible, including malformed external scores.
            continue
        choices = choices[dp[choices] == np.max(dp[choices])]
        column = choices[np.argmin(travel[choices])]
        for position in range(len(indices) - 1, -1, -1):
            index = indices[position]
            selected[index] = batch.candidate_times[index, column]
            if position:
                column = backpointers[position - 1][column]
    return selected


def refine_labels(labels, batch: BoundaryBatch, distributions, config=None):
    """Move existing boundaries jointly, preserving every run, state and segment.

    External probability matrices are accepted alongside TimingPrediction.
    Invalid rows fall back to a zero-offset atom. Point masses at zero reproduce
    the input exactly, even when frame evidence would prefer another sequence.
    """
    labels = _codes(labels)
    if not np.array_equal(labels, batch.original_labels):
        raise ValueError("batch does not describe these labels")
    config = config or TimingConfig(radius=batch.radius)
    if config.radius != batch.radius:
        raise ValueError("decoder radius differs from boundary batch")
    raw = distributions.probabilities if isinstance(distributions, TimingPrediction) else distributions
    probability = np.array(raw, dtype=np.float64, copy=True)
    if probability.shape != batch.valid.shape:
        raise ValueError("probabilities must align with candidates")
    probability = np.where(batch.valid & np.isfinite(probability) & (probability > 0), probability, 0.)
    mass = probability.sum(axis=1)
    bad = mass == 0
    probability[bad, batch.radius] = 1.
    mass[bad] = 1.
    probability /= mass[:, None]
    with np.errstate(divide="ignore"):
        score = np.log(probability)
    movement = np.abs(batch.candidate_times - batch.boundaries[:, None])
    score -= config.movement_penalty * movement / max(batch.radius, 1)
    if config.evidence_weight:
        log_state = np.log(np.maximum(batch.state_probability, 1e-12))
        prefix = np.vstack((np.zeros((1, 3)), np.cumsum(log_state, axis=0)))
        for index, (before, after) in enumerate(zip(batch.before, batch.after)):
            times = np.clip(batch.candidate_times[index], 0, len(labels))
            time = batch.boundaries[index]
            evidence = prefix[times, before] - prefix[times, after]
            evidence -= prefix[time, before] - prefix[time, after]
            score[index] += config.evidence_weight * evidence
    moved = _ordered_decode(batch, score, movement)
    return _render_labels(labels, batch, moved)


def _render_labels(labels, batch, moved):
    result = labels.copy()
    for run_id, (start, stop) in enumerate(batch.runs):
        indices = np.flatnonzero(batch.run_ids == run_id)
        if not len(indices):
            continue
        edges = np.r_[start, moved[indices], stop]
        states = np.r_[batch.before[indices[0]], batch.after[indices]]
        for left, right, state in zip(edges[:-1], edges[1:], states):
            result[left:right] = state
    return result


def shift_labels(labels, offsets_by_family, radius=25, *, run_ids=None):
    """Global or three-family fixed offset projected onto ordered run-local times.

    The projection minimizes total squared distance to requested boundary times.
    Equal optima prefer less total movement and deterministic integer ties.
    Family mappings use restart, stoppage and team_switch keys (missing means zero).
    """
    radius = _radius(radius)
    labels = _codes(labels)
    runs = _runs(labels, run_ids)
    rows = [(time, _family(before, after), before, after, run_id)
            for run_id, run in enumerate(runs)
            for time, before, after in transitions(labels, [run])]
    rows = np.asarray(rows, np.int64).reshape(-1, 5)
    candidates = rows[:, :1] + np.arange(-radius, radius + 1)
    valid = np.zeros(candidates.shape, bool)
    for run_id, (start, stop) in enumerate(runs):
        selected = rows[:, 4] == run_id
        valid[selected] = (candidates[selected] > start) & (candidates[selected] < stop)
    # Fixed baselines need only the integer grid, no frame evidence or features.
    batch = BoundaryBatch(
        features=np.empty((*candidates.shape, 0)), candidate_times=candidates, valid=valid,
        boundaries=rows[:, 0], families=rows[:, 1], before=rows[:, 2], after=rows[:, 3],
        run_ids=rows[:, 4], runs=runs, original_labels=labels,
        state_probability=np.empty((0, 3)), evidence_available=np.zeros(len(rows), bool),
        radius=radius,
    )
    if isinstance(offsets_by_family, Mapping):
        unknown = set(offsets_by_family) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown transition families: {sorted(unknown)}")
        offsets = np.array([offsets_by_family.get(name, 0) for name in FAMILIES], dtype=float)
    else:
        offsets = np.repeat(offsets_by_family, 3).astype(float)
    if offsets.shape != (3,) or not np.isfinite(offsets).all() or np.any(offsets != np.rint(offsets)):
        raise ValueError("fixed shifts must be finite integer frame offsets")
    if np.any(np.abs(offsets) > radius):
        raise ValueError("fixed shifts must be inside radius")
    desired = batch.boundaries + offsets[batch.families]
    score = -(batch.candidate_times - desired[:, None]) ** 2
    movement = np.abs(batch.candidate_times - batch.boundaries[:, None])
    selected = _ordered_decode(batch, score, movement)
    return _render_labels(labels, batch, selected)
