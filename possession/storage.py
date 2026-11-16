"""Versioned local caches. Features and targets remain separate objects."""

import hashlib
import json
import os
import tempfile
from dataclasses import fields
from pathlib import Path

import numpy as np

from .schema import FrameTable, TrainingTargets

FRAME_ARRAYS = ("frame_id", "players_xy", "player_valid", "ball_xy", "ball_valid", "graded")
TARGET_ARRAYS = tuple(field.name for field in fields(TrainingTargets))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Publish complete JSON only, with stable key order and no NaNs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_half(path, frames, targets):
    if len(frames.frame_id) != len(targets.state):
        raise ValueError("input and target length mismatch")
    metadata = {
        "schema_version": 1, "fps": frames.fps, "pitch": list(frames.pitch),
        "match_id": frames.match_id, "period": frames.period,
    }
    arrays = {name: getattr(frames, name) for name in FRAME_ARRAYS}
    arrays.update({"target_" + name: getattr(targets, name) for name in TARGET_ARRAYS})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, metadata=json.dumps(metadata, sort_keys=True), **arrays)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_half(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.pop("schema_version") != 1:
            raise ValueError("unsupported cache schema version")
        metadata["pitch"] = tuple(metadata["pitch"])
        frames = FrameTable(**{name: data[name] for name in FRAME_ARRAYS}, **metadata)
        targets = TrainingTargets(**{name: data["target_" + name] for name in TARGET_ARRAYS})
    if len(frames.frame_id) != len(targets.state):
        raise ValueError("input and target length mismatch")
    return frames, targets
