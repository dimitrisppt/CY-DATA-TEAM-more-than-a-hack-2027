"""Tracking inputs and supervised targets with separate validity masks."""

from dataclasses import dataclass

import numpy as np


@dataclass
class FrameTable:
    frame_id: np.ndarray
    players_xy: np.ndarray
    player_valid: np.ndarray
    ball_xy: np.ndarray
    ball_valid: np.ndarray
    graded: np.ndarray
    fps: float = 25.0
    pitch: tuple[float, float] = (105.3, 68.0)
    match_id: str = ""
    period: int = 1

    def __post_init__(self):
        raw_frame_id = np.asarray(self.frame_id)
        if not np.issubdtype(raw_frame_id.dtype, np.integer):
            raise ValueError("frame_id must contain integers")
        self.frame_id = raw_frame_id.astype(np.int64, copy=False)
        self.players_xy = np.asarray(self.players_xy, dtype=np.float32)
        self.player_valid = np.asarray(self.player_valid, dtype=bool)
        self.ball_xy = np.asarray(self.ball_xy, dtype=np.float32)
        self.ball_valid = np.asarray(self.ball_valid, dtype=bool)
        self.graded = np.asarray(self.graded, dtype=bool)
        self.validate()

    def validate(self):
        n = len(self.frame_id)
        if self.frame_id.ndim != 1 or (n > 1 and not np.all(np.diff(self.frame_id) == 1)):
            raise ValueError("frame_id must be a contiguous ordered one-dimensional array")
        if self.players_xy.ndim != 4 or self.players_xy.shape[:2] != (n, 2) or self.players_xy.shape[-1] != 2:
            raise ValueError("players_xy must have shape [T,2,P,2]")
        if self.player_valid.shape != self.players_xy.shape[:-1]:
            raise ValueError("player_valid shape does not match players_xy")
        if self.ball_xy.shape != (n, 2) or self.ball_valid.shape != (n,):
            raise ValueError("ball_xy and ball_valid must have shapes [T,2] and [T]")
        if self.graded.shape != (n,):
            raise ValueError("graded must have shape [T]")
        if not np.isfinite(self.players_xy[self.player_valid]).all():
            raise ValueError("players_xy has nonfinite observed coordinates")
        if not np.isfinite(self.ball_xy[self.ball_valid]).all():
            raise ValueError("ball_xy has nonfinite observed coordinates")
        self.players_xy[~self.player_valid] = 0
        self.ball_xy[~self.ball_valid] = 0
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")
        if len(self.pitch) != 2 or any(not np.isfinite(v) or v <= 0 for v in self.pitch):
            raise ValueError("pitch must contain positive finite dimensions")
        return self

    def subset(self, start: int, stop: int):
        s = slice(start, stop)
        return FrameTable(self.frame_id[s], self.players_xy[s].copy(), self.player_valid[s].copy(),
                          self.ball_xy[s].copy(), self.ball_valid[s].copy(), self.graded[s].copy(),
                          self.fps, self.pitch, self.match_id, self.period)


@dataclass
class FeatureBatch:
    player_tokens: np.ndarray
    player_mask: np.ndarray
    global_tokens: np.ndarray
    time_valid: np.ndarray

    def __post_init__(self):
        self.player_tokens = np.asarray(self.player_tokens, dtype=np.float32)
        self.player_mask = np.asarray(self.player_mask, dtype=bool)
        self.global_tokens = np.asarray(self.global_tokens, dtype=np.float32)
        self.time_valid = np.asarray(self.time_valid, dtype=bool)
        n = len(self.time_valid)
        if (self.player_tokens.ndim != 4 or self.player_tokens.shape[:2] != (n, 2)
                or self.player_tokens.shape[-1] != 8):
            raise ValueError("player_tokens must have shape [T,2,P,8]")
        if self.player_mask.shape != self.player_tokens.shape[:-1]:
            raise ValueError("player_mask shape does not match player_tokens")
        if self.global_tokens.shape != (n, 9):
            raise ValueError("global_tokens must have shape [T,9]")


@dataclass
class TrainingTargets:
    live: np.ndarray
    live_valid: np.ndarray
    owner: np.ndarray
    owner_valid: np.ndarray
    state: np.ndarray
    state_valid: np.ndarray

    def __post_init__(self):
        self.live = np.asarray(self.live, dtype=np.int8)
        self.live_valid = np.asarray(self.live_valid, dtype=bool)
        self.owner = np.asarray(self.owner, dtype=np.int8)
        self.owner_valid = np.asarray(self.owner_valid, dtype=bool)
        self.state = np.asarray(self.state, dtype=np.int8)
        self.state_valid = np.asarray(self.state_valid, dtype=bool)
        n = len(self.live)
        if any(a.shape != (n,) for a in (self.live, self.live_valid, self.owner,
                                         self.owner_valid, self.state, self.state_valid)):
            raise ValueError("TrainingTargets fields must all have shape [T]")
        if not np.isin(self.live[self.live_valid], [0, 1]).all():
            raise ValueError("live valid entries must be 0 or 1")
        if not np.isin(self.owner[self.owner_valid], [0, 1]).all():
            raise ValueError("owner valid entries must be A=0 or H=1")
        if not np.isin(self.state[self.state_valid], [0, 1, 2]).all():
            raise ValueError("state valid entries must be H=0, A=1, or D=2")
        if np.any(self.owner_valid & (~self.live_valid | (self.live != 1))):
            raise ValueError("owner_valid requires a valid live frame")
        if np.any(self.state_valid & ~self.live_valid):
            raise ValueError("state_valid requires live_valid")
        dead = self.state_valid & (self.state == 2)
        active = self.state_valid & (self.state != 2)
        if np.any(dead & ((self.live != 0) | self.owner_valid)):
            raise ValueError("dead state requires live=0 and no valid owner")
        if np.any(active & ((self.live != 1) | ~self.owner_valid |
                            (self.owner != 1 - self.state))):
            raise ValueError("active state must agree with live and owner")

    def subset(self, start: int, stop: int):
        s = slice(start, stop)
        return TrainingTargets(*(getattr(self, name)[s].copy() for name in
                                 ("live", "live_valid", "owner", "owner_valid", "state", "state_valid")))
