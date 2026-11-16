"""Label-free, mask-aware tracking features in metric pitch coordinates.

The caller supplies one half at a time. Slot motion is valid only across adjacent
source frames, and the token has no slot index or player identity embedding.
"""

import numpy as np

from possession.schema import FeatureBatch, FrameTable


def _ball_gap_ages(valid: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Seconds since the last and until the next ball observation (capped at 10)."""
    n = len(valid)
    before = np.full(n, 10.0, dtype=np.float32)
    after = np.full(n, 10.0, dtype=np.float32)
    last = -1
    for t in range(n):
        if valid[t]:
            last = t
            before[t] = 0
        elif last >= 0:
            before[t] = min((t - last) / fps, 10.0)
    next_seen = n
    for t in range(n - 1, -1, -1):
        if valid[t]:
            next_seen = t
            after[t] = 0
        elif next_seen < n:
            after[t] = min((next_seen - t) / fps, 10.0)
    return before / 10, after / 10


def build_features(frames: FrameTable) -> FeatureBatch:
    """Create eight player and nine global channels without reading targets.

    Player channels: normalized position, clipped-valid backward velocity,
    motion flag, ball-relative position, and ball visibility. Global channels:
    normalized ball position/visibility, valid backward ball velocity/flag,
    nearest observed-ball gap ages, and median valid player speed.
    """
    xy = np.asarray(frames.players_xy, dtype=np.float32)
    player_valid = np.asarray(frames.player_valid, dtype=bool)
    ball = np.asarray(frames.ball_xy, dtype=np.float32)
    ball_valid = np.asarray(frames.ball_valid, dtype=bool)
    frame_id = np.asarray(frames.frame_id)
    n, _, p, _ = xy.shape
    length, width = frames.pitch
    fps = float(frames.fps)

    player = np.zeros((n, 2, p, 8), dtype=np.float32)
    global_tokens = np.zeros((n, 9), dtype=np.float32)
    xy_clean = np.where(player_valid[..., None], xy, 0.0)
    ball_clean = np.where(ball_valid[:, None], ball, 0.0)
    scale = np.array([length, width], dtype=np.float32)
    player[..., :2] = xy_clean / scale
    player[..., 7] = player_valid & ball_valid[:, None, None]
    player[..., 5:7] = np.where(
        (player_valid & ball_valid[:, None, None])[..., None],
        (xy_clean - ball_clean[:, None, None, :]) / scale,
        0.0,
    )

    global_tokens[:, :2] = ball_clean / scale
    global_tokens[:, 2] = ball_valid
    gap_before, gap_after = _ball_gap_ages(ball_valid, fps)
    global_tokens[:, 6] = gap_before
    global_tokens[:, 7] = gap_after

    if n > 1:
        adjacent = np.diff(frame_id) == 1
        delta = (xy_clean[1:] - xy_clean[:-1]) * fps
        speeds = np.linalg.norm(delta, axis=-1)
        moving_valid = (player_valid[1:] & player_valid[:-1]
                        & adjacent[:, None, None] & np.isfinite(speeds) & (speeds <= 15.0))
        player[1:, ..., 2:4] = np.where(moving_valid[..., None], delta / 15.0, 0.0)
        player[1:, ..., 4] = moving_valid
        for t in range(1, n):
            valid_speeds = speeds[t - 1][moving_valid[t - 1]]
            if valid_speeds.size:
                global_tokens[t, 8] = np.median(valid_speeds) / 15.0

        ball_delta = (ball_clean[1:] - ball_clean[:-1]) * fps
        ball_speed = np.linalg.norm(ball_delta, axis=-1)
        ball_motion_valid = (ball_valid[1:] & ball_valid[:-1] & adjacent
                             & np.isfinite(ball_speed) & (ball_speed <= 50.0))
        global_tokens[1:, 3:5] = np.where(ball_motion_valid[:, None], ball_delta / 50.0, 0.0)
        global_tokens[1:, 5] = ball_motion_valid

    # Invalid player tokens remain exactly zero, even when unobserved inputs
    # contain NaN placeholders or a visible ball is present.
    player[~player_valid] = 0.0
    return FeatureBatch(player, player_valid.copy(), global_tokens,
                        np.asarray(frames.graded, dtype=bool).copy())
