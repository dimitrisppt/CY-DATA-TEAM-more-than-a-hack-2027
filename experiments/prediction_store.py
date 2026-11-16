"""Lossless, half-local OOF arrays for development experiments.

The stored predictor intentionally supports ``boundary_scale == 0`` only: it
does not retain neural boundary scores. A bundle is meaningful only alongside
its parent fit result, which binds the checkpoint and protocol hashes. The
record here authenticates arrays and half identity, not model ancestry.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from possession.ensemble import _checked
from possession.storage import sha256_file, write_json

SCHEMA_VERSION = 1
_ARRAY_NAMES = ("neural_live", "neural_home", "reference_live", "reference_home",
                "gt", "ball_valid", "nearest_ball_seconds", "frame_id")
_RECORD_KEYS = {"path", "sha256", "schema_version", "match", "period", "scenario"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _identity(match, period, scenario):
    if not isinstance(match, str) or not match:
        raise ValueError("match must be a nonempty string")
    if isinstance(period, bool) or not isinstance(period, int) or period not in (1, 2):
        raise ValueError("period must be 1 or 2")
    if not isinstance(scenario, str) or not scenario:
        raise ValueError("scenario must be a nonempty string")
    return {"schema_version": SCHEMA_VERSION, "match": match, "period": period,
            "scenario": scenario, "supported_boundary_scale": 0}


def _vector(name, values, n, kind):
    if values is None:
        raise ValueError(f"{name} is required")
    raw = np.asarray(values)
    if raw.shape != (n,):
        raise ValueError(f"{name} must have shape [T]")
    if kind == "bool":
        if raw.dtype.kind != "b":
            raise ValueError(f"{name} must be boolean")
        return raw.astype(np.bool_, copy=True)
    if kind == "int":
        if raw.dtype.kind not in "iu":
            raise ValueError(f"{name} must be integer")
        return raw
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be numeric")
    return raw.astype(np.float64)


def _arrays(neural, reference, gt, ball_valid, nearest_ball_seconds, frame_id):
    neural = _checked(neural, "neural")
    reference = _checked(reference, "reference")
    n = len(neural["live_probability"])
    if len(reference["live_probability"]) != n:
        raise ValueError("neural and reference score shapes differ")
    codes = _vector("gt", gt, n, "int")
    if not np.isin(codes, np.arange(4)).all():
        raise ValueError("gt must contain official codes 0..3")
    visible = _vector("ball_valid", ball_valid, n, "bool")
    seconds = _vector("nearest_ball_seconds", nearest_ball_seconds, n, "float")
    if not np.isfinite(seconds).all() or np.any(seconds < 0):
        raise ValueError("nearest_ball_seconds must be nonnegative and finite")
    ids = _vector("frame_id", frame_id, n, "int")
    if n and (int(ids.min()) < np.iinfo(np.int64).min
              or int(ids.max()) > np.iinfo(np.int64).max):
        raise ValueError("frame_id must fit int64 without loss")
    ids = ids.astype(np.int64)
    if n > 1 and not (np.all(ids[1:] > ids[:-1]) and np.all(np.diff(ids) == 1)):
        raise ValueError("frame_id must be consecutive")
    return {
        "neural_live": neural["live_probability"].copy(),
        "neural_home": neural["home_probability"].copy(),
        "reference_live": reference["live_probability"].copy(),
        "reference_home": reference["home_probability"].copy(),
        "gt": codes.astype(np.int8), "ball_valid": visible,
        "nearest_ball_seconds": seconds, "frame_id": ids,
    }


def _record(path, digest, identity):
    return {"path": str(path), "sha256": digest,
            **{key: identity[key] for key in ("schema_version", "match", "period", "scenario")}}


def load_prediction_bundle(record) -> dict[str, np.ndarray]:
    """Verify sidecar, bytes, identity, and array schema before returning copies."""
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        raise ValueError("prediction record fields differ")
    path = Path(record["path"])
    if not path.is_absolute() or path.suffix != ".npz":
        raise ValueError("prediction path must be an absolute .npz path")
    identity = _identity(record["match"], record["period"], record["scenario"])
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("prediction schema version differs")
    digest = record["sha256"]
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("prediction digest is invalid")
    sidecar = path.with_suffix(".sha256.json")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("prediction bundle or hash sidecar is missing")
    if json.loads(sidecar.read_text()) != {"sha256": digest} or sha256_file(path) != digest:
        raise ValueError("prediction bundle hash differs")
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != {*_ARRAY_NAMES, "metadata"}:
            raise ValueError("prediction array schema differs")
        metadata = stored["metadata"]
        if metadata.ndim != 0 or json.loads(str(metadata)) != identity:
            raise ValueError("prediction bundle identity differs")
        arrays = {name: stored[name].copy() for name in _ARRAY_NAMES}
    n = len(arrays["gt"])
    types = {"neural_live": np.float64, "neural_home": np.float64,
             "reference_live": np.float64, "reference_home": np.float64,
             "gt": np.int8, "ball_valid": np.bool_,
             "nearest_ball_seconds": np.float64, "frame_id": np.int64}
    if any(arrays[name].shape != (n,) or arrays[name].dtype != dtype
           for name, dtype in types.items()):
        raise ValueError("prediction array shapes or dtypes differ")
    # Reuse the same domain checks as a new bundle. Boundary placeholders are
    # only validation inputs; no boundary arrays are retained in the store.
    zeros = np.zeros((n, 3, 3), dtype=np.float64)
    checked = _arrays(
        {"live_probability": arrays["neural_live"],
         "home_probability": arrays["neural_home"], "boundary": zeros},
        {"live_probability": arrays["reference_live"],
         "home_probability": arrays["reference_home"], "boundary": zeros},
        arrays["gt"], arrays["ball_valid"], arrays["nearest_ball_seconds"],
        arrays["frame_id"])
    if any(not np.array_equal(arrays[name], checked[name]) for name in _ARRAY_NAMES):
        raise ValueError("prediction values differ from validated schema")
    return arrays


def save_prediction_bundle(path, *, neural, reference, gt, ball_valid,
                           nearest_ball_seconds, frame_id, match, period, scenario) -> dict:
    """Write once atomically, or verify exact existing arrays and half identity."""
    path = Path(path).resolve()
    if path.suffix != ".npz":
        raise ValueError("prediction path must end in .npz")
    identity = _identity(match, period, scenario)
    arrays = _arrays(neural, reference, gt, ball_valid, nearest_ball_seconds, frame_id)
    sidecar = path.with_suffix(".sha256.json")
    if path.exists():
        if not sidecar.is_file():
            raise ValueError("existing prediction bundle lacks hash sidecar")
        digest = json.loads(sidecar.read_text()).get("sha256")
        existing = _record(path, digest, identity)
        loaded = load_prediction_bundle(existing)
        if any(not np.array_equal(loaded[name], arrays[name]) for name in _ARRAY_NAMES):
            raise ValueError("existing prediction bundle arrays differ")
        return existing
    if sidecar.exists():
        raise ValueError("orphan prediction hash sidecar exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz.tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, metadata=json.dumps(identity, sort_keys=True), **arrays)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    digest = sha256_file(path)
    write_json(sidecar, {"sha256": digest})
    return _record(path, digest, identity)
