"""Deterministic CPU possession evidence, without learned or target labels.

This is an untrained reference. It uses ball-to-team proximity for owner
evidence and player motion for independent live/dead evidence. Missing ball
coordinates supply no owner or dead evidence. It cannot identify controlled
touches, unseen ball flight, or every restart convention from geometry alone.
"""

from dataclasses import asdict, dataclass

import numpy as np

from possession.features import build_features
from possession.schema import FrameTable


@dataclass(frozen=True)
class ReferenceConfig:
    owner_smoothing_seconds: float = 0.6
    activity_smoothing_seconds: float = 0.8
    nearest_scale_m: float = 4.0
    second_scale_m: float = 8.0
    second_weight: float = 0.4
    owner_strength: float = 5.0
    live_bias: float = 0.3
    activity_strength: float = 2.4
    outside_penalty: float = 0.7
    ball_motion_bonus: float = 0.3

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40., 40.)))


def _smooth_observed(value: np.ndarray, observed: np.ndarray, fps: float,
                     seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric exponential context using observed samples only, O(T)."""
    n = len(value)
    if n == 0:
        return value.copy(), np.zeros(0)
    alpha = np.exp(-1.0 / max(seconds * fps, 1e-6))
    weighted = np.where(observed, value, 0.0)
    weight = observed.astype(np.float64)
    forward_sum = np.empty(n)
    forward_count = np.empty(n)
    backward_sum = np.empty(n)
    backward_count = np.empty(n)
    run_sum = run_count = 0.0
    for t in range(n):
        run_sum = weighted[t] + alpha * run_sum
        run_count = weight[t] + alpha * run_count
        forward_sum[t], forward_count[t] = run_sum, run_count
    run_sum = run_count = 0.0
    for t in range(n - 1, -1, -1):
        run_sum = weighted[t] + alpha * run_sum
        run_count = weight[t] + alpha * run_count
        backward_sum[t], backward_count[t] = run_sum, run_count
    total = forward_sum + backward_sum - weighted
    count = forward_count + backward_count - weight
    result = np.divide(total, count, out=np.zeros(n), where=count > 1e-12)
    return result, count


def reference_scores(frames: FrameTable, config: ReferenceConfig | None = None) -> dict[str, np.ndarray]:
    """Return live, conditional-home, and neutral boundary scores for a half.

    The probabilities are heuristic evidence. Calibrate or evaluate them only
    on eligible external validation data; no performance level is implied.
    """
    cfg = ReferenceConfig() if config is None else config
    features = build_features(frames)
    valid = np.asarray(frames.player_valid, dtype=bool)
    ball_valid = np.asarray(frames.ball_valid, dtype=bool)
    n = len(ball_valid)

    # A team's nearest player and second player both inform proximity. Keeping
    # the second term separate avoids turning a single close duplicate into a
    # hard controller decision. Invalid slots contribute exactly zero.
    distances = np.linalg.norm(
        np.asarray(frames.players_xy, dtype=np.float64)
        - np.asarray(frames.ball_xy, dtype=np.float64)[:, None, None, :], axis=-1)
    distances = np.where(valid & ball_valid[:, None, None], distances, np.inf)
    sorted_distance = np.sort(distances, axis=-1)
    nearest = sorted_distance[..., 0] if sorted_distance.shape[-1] else np.full((n, 2), np.inf)
    second = (sorted_distance[..., 1] if sorted_distance.shape[-1] > 1
              else np.full((n, 2), np.inf))
    team_quality = (np.exp(-nearest / cfg.nearest_scale_m)
                    + cfg.second_weight * np.exp(-second / cfg.second_scale_m))

    # Relative motion is a small confidence adjustment, never a contact gate.
    ball_motion = features.global_tokens[:, 5] > 0
    player_motion = features.player_tokens[..., 4] > 0
    player_velocity = features.player_tokens[..., 2:4].astype(np.float64) * 15.0
    ball_velocity = features.global_tokens[:, 3:5].astype(np.float64) * 50.0
    relative_speed = np.linalg.norm(player_velocity - ball_velocity[:, None, None, :], axis=-1)
    nearest_slot = np.argmin(distances, axis=-1) if distances.shape[-1] else np.zeros((n, 2), dtype=int)
    nearest_has_motion = np.take_along_axis(player_motion, nearest_slot[..., None], axis=-1)[..., 0] if distances.shape[-1] else np.zeros((n, 2), dtype=bool)
    nearest_relative_speed = np.take_along_axis(relative_speed, nearest_slot[..., None], axis=-1)[..., 0] if distances.shape[-1] else np.zeros((n, 2))
    motion_credible = ball_motion[:, None] & nearest_has_motion & np.isfinite(nearest)
    motion_factor = np.where(motion_credible,
                             0.85 + 0.15 * np.exp(-nearest_relative_speed / 8.0), 1.0)
    quality = team_quality * motion_factor
    owner_evidence = cfg.owner_strength * (quality[:, 0] - quality[:, 1])
    observed_owner = ball_valid & np.any(valid, axis=(1, 2))
    owner_context, _ = _smooth_observed(owner_evidence, observed_owner, frames.fps,
                                        cfg.owner_smoothing_seconds)
    home_probability = _sigmoid(owner_context)

    # Activity uses all valid same-slot motions, symmetrically across teams.
    # An unavailable motion estimate is unknown, not stationary.
    speed = np.linalg.norm(player_velocity, axis=-1)
    counts = player_motion.sum(axis=(1, 2))
    moving = ((speed > 0.5) & player_motion).sum(axis=(1, 2))
    activity = np.divide(moving, counts, out=np.zeros(n), where=counts > 0)
    activity_context, activity_support = _smooth_observed(activity, counts > 0,
                                                           frames.fps, cfg.activity_smoothing_seconds)
    live_logit = np.full(n, cfg.live_bias, dtype=np.float64)
    live_logit += np.where(activity_support > 1e-12,
                           cfg.activity_strength * (activity_context - 0.25), 0.0)
    live_logit += cfg.ball_motion_bonus * (np.linalg.norm(ball_velocity, axis=-1) > 1.0) * ball_motion
    length, width = frames.pitch
    outside = ball_valid & (
        (frames.ball_xy[:, 0] < 0) | (frames.ball_xy[:, 0] > length)
        | (frames.ball_xy[:, 1] < 0) | (frames.ball_xy[:, 1] > width))
    live_logit -= cfg.outside_penalty * outside
    live_probability = _sigmoid(live_logit)

    return {
        "live_probability": live_probability,
        "home_probability": home_probability,
        "boundary": np.zeros((n, 3, 3), dtype=np.float64),
    }
