"""CUDA session adapter and equivalent CPU preprocessing for fresh GPU studies.

The historical, source-pinned trainer stays intact. Accelerators are scoped to
this runner and tested against the original features, augmentation and RNG.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager

import numpy as np
import torch
from numba import njit

from experiments import campaign_training as campaign
from possession import augmentation, train
from possession.schema import FeatureBatch
from possession.training_session import TrainingSession


def configure_device(device="cuda", threads=1):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("the study supports CPU or CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing a silent CPU fallback")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        device = torch.device("cuda", device.index or 0)
        torch.cuda.set_device(device)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return str(device)


def build_session(data_dir, spec, train_matches, seed, cache=None, *, device="cuda"):
    if set(spec) != {"model_kwargs", "training_args", "outage_config",
                     "augmentation_config", "schedule"}:
        raise ValueError("campaign spec fields differ")
    session = TrainingSession(
        data_dir, train_matches=train_matches, seed=seed, device=device, cache=cache,
        model_kwargs=spec["model_kwargs"], outage_config=spec["outage_config"],
        augmentation_config=spec["augmentation_config"], **spec["training_args"])
    session._training_window_hash = campaign._TranscriptHash()
    session._campaign_spec = json.loads(json.dumps(spec, sort_keys=True, allow_nan=False))
    session.config["campaign_spec"] = session._campaign_spec
    session._campaign_schedule_hook = campaign._schedule_hook(session, spec)
    return session


@njit(cache=True, fastmath=False)
def _noise_kernel(points, tracks, graded, rng, rho, sigma):
    touched = 0
    for track in range(tracks.shape[1]):
        previous = np.zeros(2, dtype=np.float64)
        contiguous = False
        for t in range(len(graded)):
            if not graded[t] or not tracks[t, track]:
                contiguous = False
                previous[:] = 0
                continue
            scale = sigma * math.sqrt(max(0., 1. - rho * rho)) if contiguous else sigma
            innovation = rng.normal(0., scale, 2)
            for axis in range(2):
                previous[axis] = (rho * previous[axis]
                                  + min(max(innovation[axis], -3 * sigma), 3 * sigma))
                points[t, track, axis] += np.float32(previous[axis])
            contiguous = True
            touched += 1
    return touched


def coordinate_noise(frames, rng, config, strength):
    touched = 0
    for xy, valid, std in ((frames.players_xy, frames.player_valid, config.noise_player_std),
                           (frames.ball_xy[:, None, :], frames.ball_valid[:, None],
                            config.noise_ball_std)):
        sigma = float(std) * strength
        if sigma > 0:
            touched += _noise_kernel(xy.reshape(len(valid), -1, 2),
                                     valid.reshape(len(valid), -1), frames.graded,
                                     rng, float(config.noise_correlation), sigma)
    return {"observations": touched}


def build_features(frames):
    """Original feature arithmetic with vectorized gap ages and speed medians."""
    xy = np.asarray(frames.players_xy, dtype=np.float32)
    valid = np.asarray(frames.player_valid, dtype=bool)
    ball = np.asarray(frames.ball_xy, dtype=np.float32)
    seen = np.asarray(frames.ball_valid, dtype=bool)
    n, _, players, _ = xy.shape
    fps = float(frames.fps)
    player = np.zeros((n, 2, players, 8), dtype=np.float32)
    context = np.zeros((n, 9), dtype=np.float32)
    clean_xy = np.where(valid[..., None], xy, 0.)
    clean_ball = np.where(seen[:, None], ball, 0.)
    scale = np.array(frames.pitch, dtype=np.float32)
    player[..., :2] = clean_xy / scale
    player[..., 7] = valid & seen[:, None, None]
    player[..., 5:7] = np.where(
        (valid & seen[:, None, None])[..., None],
        (clean_xy - clean_ball[:, None, None, :]) / scale, 0.)
    context[:, :2] = clean_ball / scale
    context[:, 2] = seen
    index = np.arange(n)
    previous = np.maximum.accumulate(np.where(seen, index, -1))
    following = np.minimum.accumulate(np.where(seen, index, n)[::-1])[::-1]
    # Cast before dividing by 10, matching the original float32 buffer rounding.
    context[:, 6] = np.where(previous >= 0, np.minimum((index - previous) / fps, 10.),
                             10.).astype(np.float32) / 10
    context[:, 7] = np.where(following < n, np.minimum((following - index) / fps, 10.),
                             10.).astype(np.float32) / 10
    if n > 1:
        adjacent = np.diff(frames.frame_id) == 1
        delta = (clean_xy[1:] - clean_xy[:-1]) * fps
        speeds = np.linalg.norm(delta, axis=-1)
        moving = (valid[1:] & valid[:-1] & adjacent[:, None, None]
                  & np.isfinite(speeds) & (speeds <= 15.))
        player[1:, ..., 2:4] = np.where(moving[..., None], delta / 15., 0.)
        player[1:, ..., 4] = moving
        if players:
            ordered = np.sort(np.where(moving, speeds, np.inf).reshape(n - 1, -1), axis=1)
            counts = moving.reshape(n - 1, -1).sum(axis=1)
            rows = np.flatnonzero(counts)
            lo, hi = (counts[rows] - 1) // 2, counts[rows] // 2
            context[rows + 1, 8] = ((ordered[rows, lo] + ordered[rows, hi]) / 2) / 15.
        delta = (clean_ball[1:] - clean_ball[:-1]) * fps
        speeds = np.linalg.norm(delta, axis=-1)
        moving = seen[1:] & seen[:-1] & adjacent & np.isfinite(speeds) & (speeds <= 50.)
        context[1:, 3:5] = np.where(moving[:, None], delta / 50., 0.)
        context[1:, 5] = moving
    player[~valid] = 0.
    return FeatureBatch(player, valid.copy(), context, frames.graded.copy())


@contextmanager
def accelerated_preprocessing():
    """Use verified equivalents only within the explicit fresh-study scope."""
    old_noise, old_features = augmentation._coordinate_noise, train.build_features
    augmentation._coordinate_noise = coordinate_noise
    train.build_features = build_features
    try:
        yield
    finally:
        augmentation._coordinate_noise = old_noise
        train.build_features = old_features
