"""Synchronized time and tracking-sensor augmentations for a single half.

Coordinates and labels share the same source-time map. Unknown coordinates or
labels never become observations through interpolation.
"""

import numpy as np

from possession.schema import FrameTable, TrainingTargets


def _copy_frames(frames: FrameTable) -> FrameTable:
    return FrameTable(
        frames.frame_id.copy(), frames.players_xy.copy(), frames.player_valid.copy(),
        frames.ball_xy.copy(), frames.ball_valid.copy(), frames.graded.copy(),
        frames.fps, frames.pitch, frames.match_id, frames.period,
    )


def _copy_targets(targets: TrainingTargets) -> TrainingTargets:
    return TrainingTargets(*(getattr(targets, name).copy() for name in (
        "live", "live_valid", "owner", "owner_valid", "state", "state_valid",
    )))


def resample_tracking(
    frames: FrameTable, targets: TrainingTargets, source_positions: np.ndarray,
) -> tuple[FrameTable, TrainingTargets]:
    """Sample one half at fractional source-frame positions, without extrapolation.

    An exact source frame needs only its own mask. Fractional positions need
    both adjacent frames observed and graded. Labels use the nearest source
    frame, with both endpoint validity masks required at fractional positions.
    """
    n = len(frames.frame_id)
    if len(targets.live) != n:
        raise ValueError("frames and targets must have the same length")
    positions = np.asarray(source_positions, dtype=np.float64)
    if positions.shape != (n,) or not np.isfinite(positions).all():
        raise ValueError("source_positions must contain one finite position per frame")
    if np.any(np.diff(positions) < 0):
        raise ValueError("source_positions must be monotone")
    if n == 0:
        return _copy_frames(frames), _copy_targets(targets)

    supported = (positions >= 0.0) & (positions <= n - 1)
    safe = np.clip(positions, 0.0, n - 1)
    lower = np.floor(safe).astype(np.int64)
    fraction = safe - lower
    exact = fraction <= 1e-10
    upper = np.minimum(lower + (~exact), n - 1)
    weight = fraction.astype(np.float32)
    nearest = np.minimum(np.floor(safe + 0.5).astype(np.int64), n - 1)
    graded = supported & frames.graded[lower] & frames.graded[upper]

    player_valid = (graded[:, None, None] & frames.player_valid[lower]
                    & frames.player_valid[upper])
    ball_valid = graded & frames.ball_valid[lower] & frames.ball_valid[upper]
    players_xy = (frames.players_xy[lower] * (1.0 - weight[:, None, None, None])
                  + frames.players_xy[upper] * weight[:, None, None, None])
    ball_xy = (frames.ball_xy[lower] * (1.0 - weight[:, None])
               + frames.ball_xy[upper] * weight[:, None])
    players_xy[~player_valid] = 0.0
    ball_xy[~ball_valid] = 0.0
    frame_id = frames.frame_id[0] + np.arange(n, dtype=np.int64)
    out_frames = FrameTable(
        frame_id, players_xy, player_valid, ball_xy, ball_valid, graded,
        frames.fps, frames.pitch, frames.match_id, frames.period,
    )

    live_valid = graded & targets.live_valid[lower] & targets.live_valid[upper]
    # A known dead frame has no owner by definition. It still provides a known
    # endpoint when interpolating into an active frame with a known owner.
    owner_known = targets.owner_valid | (targets.live_valid & (targets.live == 0))
    owner_valid = (live_valid & targets.owner_valid[nearest]
                   & owner_known[lower] & owner_known[upper])
    state_valid = (live_valid & targets.state_valid[lower] & targets.state_valid[upper]
                   & ((targets.state[nearest] == 2) | owner_valid))
    live = targets.live[nearest].copy()
    owner = targets.owner[nearest].copy()
    state = targets.state[nearest].copy()
    live[~live_valid] = 0
    owner[~owner_valid] = 0
    state[~state_valid] = 0
    out_targets = TrainingTargets(live, live_valid, owner, owner_valid, state, state_valid)
    return out_frames, out_targets


def _warp_positions(n: int, rng: np.random.Generator, rate: float, knots: int) -> np.ndarray:
    positions = np.arange(n, dtype=np.float64)
    if n <= 2 or rate == 0.0:
        return positions
    times = np.linspace(0.0, n - 1.0, min(knots + 2, n))
    intervals = np.diff(times)
    deviations = rng.uniform(-1.0, 1.0, len(intervals))
    deviations -= np.average(deviations, weights=intervals)
    maximum = np.max(np.abs(deviations))
    if maximum > 0:
        deviations *= rate / maximum
    mapped_knots = np.r_[0.0, np.cumsum(intervals * (1.0 + deviations))]
    mapped_knots[-1] = n - 1.0
    return np.interp(positions, times, mapped_knots)


def _sensor_array(frames: FrameTable) -> tuple[np.ndarray, np.ndarray]:
    n = len(frames.frame_id)
    player_count = 2 * frames.players_xy.shape[2]
    coordinates = np.concatenate((frames.players_xy.reshape(n, player_count, 2),
                                  frames.ball_xy[:, None, :]), axis=1)
    valid = np.concatenate((frames.player_valid.reshape(n, player_count),
                            frames.ball_valid[:, None]), axis=1)
    return coordinates, valid


def _sensor_frames(frames: FrameTable, coordinates: np.ndarray, valid: np.ndarray) -> FrameTable:
    n = len(frames.frame_id)
    players = frames.players_xy.shape[2]
    player_count = 2 * players
    return FrameTable(
        frames.frame_id.copy(), coordinates[:, :player_count].reshape(n, 2, players, 2).copy(),
        valid[:, :player_count].reshape(n, 2, players).copy(),
        coordinates[:, -1].copy(), valid[:, -1].copy(), frames.graded.copy(),
        frames.fps, frames.pitch, frames.match_id, frames.period,
    )


def _hold(frames: FrameTable, rng: np.random.Generator, max_frames: int) -> FrameTable:
    coordinates, valid = _sensor_array(frames)
    result = coordinates.copy()
    eligible = valid & frames.graded[:, None]
    duration = int(rng.integers(2, max_frames + 1))
    phase = int(rng.integers(0, duration))
    # Each channel resets at a missing or ungraded frame; no stale value crosses it.
    last = np.full(valid.shape[1], -1, dtype=np.int64)
    for t in range(len(frames.frame_id)):
        last[~eligible[t]] = -1
        refresh = ((t - phase) % duration == 0) | (last < 0)
        last[eligible[t] & refresh] = t
        held = eligible[t] & ~refresh
        result[t, held] = coordinates[last[held], np.flatnonzero(held)]
    return _sensor_frames(frames, result, valid)


def _blur(frames: FrameTable, radius: int) -> FrameTable:
    coordinates, valid = _sensor_array(frames)
    result = coordinates.copy()
    eligible = valid & frames.graded[:, None]
    for channel in range(valid.shape[1]):
        boundary = np.flatnonzero(np.diff(np.r_[False, eligible[:, channel], False]))
        for start, stop in boundary.reshape(-1, 2):
            length = stop - start
            indices = np.arange(length)
            lo = np.maximum(indices - radius, 0)
            hi = np.minimum(indices + radius + 1, length)
            cumulative = np.vstack((np.zeros((1, 2), dtype=np.float64),
                                    np.cumsum(coordinates[start:stop, channel], axis=0,
                                              dtype=np.float64)))
            result[start:stop, channel] = (cumulative[hi] - cumulative[lo]) / (hi - lo)[:, None]
    return _sensor_frames(frames, result, valid)


def apply_temporal(
    frames: FrameTable,
    targets: TrainingTargets,
    rng: np.random.Generator,
    *,
    operation: str,
    strength: float = 1.0,
    time_warp_max_rate: float = 0.1,
    time_warp_knots: int = 4,
    temporal_shift_max_seconds: float = 0.12,
    frame_dropout_fraction: float = 0.05,
    hold_max_frames: int = 3,
    blur_radius: int = 2,
) -> tuple[FrameTable, TrainingTargets, dict]:
    """Apply one temporal augmentation with compact JSON-safe diagnostics."""
    operations = {"time_warp", "time_shift", "frame_dropout", "frame_hold", "temporal_blur"}
    if operation not in operations:
        raise ValueError(f"unknown temporal operation: {operation}")
    if len(targets.live) != len(frames.frame_id):
        raise ValueError("frames and targets must have the same length")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be nonnegative and finite")
    if not np.isfinite(time_warp_max_rate) or time_warp_max_rate < 0:
        raise ValueError("time_warp_max_rate must be nonnegative and finite")
    if time_warp_max_rate * strength >= 1:
        raise ValueError("effective time_warp_max_rate must be below 1")
    if isinstance(time_warp_knots, bool) or not isinstance(time_warp_knots, (int, np.integer)) or time_warp_knots < 1:
        raise ValueError("time_warp_knots must be a positive integer")
    if not np.isfinite(temporal_shift_max_seconds) or temporal_shift_max_seconds < 0:
        raise ValueError("temporal_shift_max_seconds must be nonnegative and finite")
    if not np.isfinite(frame_dropout_fraction) or not 0 <= frame_dropout_fraction <= 1:
        raise ValueError("frame_dropout_fraction must be between 0 and 1")
    if isinstance(hold_max_frames, bool) or not isinstance(hold_max_frames, (int, np.integer)) or hold_max_frames < 2:
        raise ValueError("hold_max_frames must be an integer of at least 2")
    if isinstance(blur_radius, bool) or not isinstance(blur_radius, (int, np.integer)) or blur_radius < 0:
        raise ValueError("blur_radius must be a nonnegative integer")

    n = len(frames.frame_id)
    details: dict[str, float | int] = {}
    if strength == 0 or n == 0:
        out_frames, out_targets = _copy_frames(frames), _copy_targets(targets)
    elif operation == "time_warp":
        positions = _warp_positions(n, rng, time_warp_max_rate * strength, time_warp_knots)
        out_frames, out_targets = resample_tracking(frames, targets, positions)
        if n > 1:
            details["max_source_rate_deviation"] = float(np.max(np.abs(np.diff(positions) - 1)))
    elif operation == "time_shift":
        bound = temporal_shift_max_seconds * frames.fps * strength
        shift = float(rng.uniform(-bound, bound)) if bound > 0 else 0.0
        positions = np.arange(n, dtype=np.float64) + shift
        out_frames, out_targets = resample_tracking(frames, targets, positions)
        details["shift_frames"] = shift
    elif operation == "frame_dropout":
        out_frames, out_targets = _copy_frames(frames), _copy_targets(targets)
        count = min(n, int(np.ceil(n * frame_dropout_fraction * strength)))
        if count:
            start = int(rng.integers(0, n - count + 1))
            stop = start + count
            out_frames.player_valid[start:stop] = False
            out_frames.ball_valid[start:stop] = False
            out_frames.players_xy[start:stop] = 0.0
            out_frames.ball_xy[start:stop] = 0.0
            details.update({"dropout_start": start, "dropout_frames": count})
    elif operation == "frame_hold":
        effective_max = max(2, min(hold_max_frames,
                                   1 + int(np.ceil((hold_max_frames - 1) * strength))))
        out_frames, out_targets = _hold(frames, rng, effective_max), _copy_targets(targets)
        details["hold_max_frames"] = effective_max
    else:
        effective_radius = min(blur_radius, int(np.ceil(blur_radius * strength)))
        out_frames, out_targets = _blur(frames, effective_radius), _copy_targets(targets)
        details["blur_radius"] = effective_radius

    before_coordinates, before_valid = _sensor_array(frames)
    after_coordinates, after_valid = _sensor_array(out_frames)
    changed = np.any(before_coordinates != after_coordinates, axis=-1) & after_valid
    metadata = {
        "operation": operation,
        "strength": float(strength),
        "coordinate_samples_changed": int(np.count_nonzero(changed)),
        "coordinate_samples_masked": int(np.count_nonzero(before_valid & ~after_valid)),
        "graded_frames": int(np.count_nonzero(out_frames.graded)),
        **details,
    }
    return out_frames, out_targets, metadata
