"""Ball-free, one-hertz context inputs and native-rate prediction adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from possession.features import build_features
from possession.model import PossessionModel
from possession.predict import _slice_tensor
from possession.prepared import verify_prepared
from possession.schema import FrameTable, TrainingTargets
from possession.storage import load_half, save_half, sha256_file, write_json

COARSE_RATE_HZ = 1
NATIVE_RATE_HZ = 25
STRIDE = NATIVE_RATE_HZ // COARSE_RATE_HZ
BINNING_RULE = "half_local_exact_sample_at_offsets_0_25_50_..."
PROCESSOR = {
    "schema": "coarse_player_only_v1",
    "fps": COARSE_RATE_HZ,
    "native_fps": NATIVE_RATE_HZ,
    "stride": STRIDE,
    "binning_rule": BINNING_RULE,
    "ball_inputs": "zero_coordinates_and_false_visibility",
    "target_rule": "sample_exact_native_anchor_without_interpolation",
}
COARSE_MODEL_KWARGS = {
    "player_dim": 8,
    "global_dim": 9,
    "width": 32,
    "dilations": [1, 2, 4, 8, 16, 32, 64],
    "pooling": "mean",
}
_DERIVATION_FILENAME = "coarse-derivation.json"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read_derivation(path):
    path = Path(path)
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or set(value) != {"sha256", "payload"}
            or value["sha256"] != _digest(value["payload"])):
        raise ValueError(f"coarse derivation manifest hash differs: {path}")
    return value["payload"]


def _publish_derivation(path, payload):
    previous = _read_derivation(path)
    if previous is not None:
        if previous != payload:
            raise ValueError(f"coarse derivation manifest differs: {path}")
        return previous
    write_json(path, {"sha256": _digest(payload), "payload": payload})
    return payload


def coarsen_half(frames: FrameTable, targets: TrainingTargets | None = None):
    """Sample one exact native frame per second, resetting ball input channels.

    Coarse frame IDs start at zero and are contiguous. Sampling is anchored to
    each half's local first frame, never to a cross-half or match-global clock.
    Targets, when supplied for training data, are sampled at the same anchors.
    """
    if not isinstance(frames, FrameTable):
        raise TypeError("frames must be a FrameTable")
    if not np.isclose(float(frames.fps), NATIVE_RATE_HZ, rtol=0, atol=1e-6):
        raise ValueError("coarse context requires native 25 Hz frames")
    frame_id = np.asarray(frames.frame_id)
    if (frame_id.ndim != 1 or frame_id.dtype.kind not in "iu"
            or (len(frame_id) > 1 and not np.all(np.diff(frame_id) == 1))):
        raise ValueError("native frame IDs must be contiguous integers")
    if targets is not None and not isinstance(targets, TrainingTargets):
        raise TypeError("targets must be TrainingTargets or None")
    if targets is not None and len(targets.state) != len(frames.frame_id):
        raise ValueError("frame and target timeline lengths differ")
    anchors = np.arange(0, len(frames.frame_id), STRIDE, dtype=np.int64)
    if not len(anchors):
        raise ValueError("a half must contain at least one native frame")
    coarse = FrameTable(
        frame_id=np.arange(len(anchors), dtype=np.int64),
        players_xy=frames.players_xy[anchors].copy(),
        player_valid=frames.player_valid[anchors].copy(),
        ball_xy=np.zeros((len(anchors), 2), dtype=np.float32),
        ball_valid=np.zeros(len(anchors), dtype=bool),
        graded=frames.graded[anchors].copy(),
        fps=float(COARSE_RATE_HZ), pitch=frames.pitch,
        match_id=frames.match_id, period=frames.period,
    )
    coarse_targets = None
    if targets is not None:
        coarse_targets = TrainingTargets(
            live=targets.live[anchors].copy(),
            live_valid=targets.live_valid[anchors].copy(),
            owner=targets.owner[anchors].copy(),
            owner_valid=targets.owner_valid[anchors].copy(),
            state=targets.state[anchors].copy(),
            state_valid=targets.state_valid[anchors].copy(),
        )
    return coarse, coarse_targets, anchors


def _source_grid(data_dir):
    data_dir = Path(data_dir).resolve()
    paths = sorted(data_dir.glob("DFL-MAT-*_half*.npz"))
    if not paths:
        raise ValueError("prepared source directory contains no DFL halves")
    provenance = verify_prepared(data_dir, paths)
    if provenance.get("status") != "verified_manifest":
        raise ValueError("coarse preparation requires manifested prepared source data")
    grouped = {}
    for path in paths:
        stem, separator, period = path.stem.rpartition("_half")
        if not separator or period not in ("1", "2"):
            raise ValueError(f"unexpected prepared source name: {path.name}")
        grouped.setdefault(stem, {})[int(period)] = path
    if any(set(periods) != {1, 2} for periods in grouped.values()):
        raise ValueError("each source match must contain both halves")
    return data_dir, provenance, grouped


def _expected_derived_names(grouped):
    return {f"{match}_half{period}.npz" for match, periods in grouped.items()
            for period in periods}


def _validate_existing_derived(output_dir, provenance, source_hashes, grouped):
    output_dir = Path(output_dir)
    derivation = _read_derivation(output_dir / _DERIVATION_FILENAME)
    if derivation is None:
        stray = list(output_dir.glob("DFL-MAT-*_half*.npz"))
        if stray or (output_dir / "manifest.json").exists():
            raise ValueError("derived cache exists without a complete derivation record")
        return None
    if (derivation.get("processor") != PROCESSOR
            or derivation.get("source_prepared") != provenance
            or derivation.get("source_cache_sha256") != source_hashes):
        raise ValueError("coarse derivation source or processor identity differs")
    expected_names = _expected_derived_names(grouped)
    output_paths = [output_dir / name for name in sorted(expected_names)]
    verified = verify_prepared(output_dir, output_paths)
    if verified.get("status") != "verified_manifest":
        raise ValueError("coarse cache lacks a verified prepared-data manifest")
    if verified.get("sources") != provenance.get("sources"):
        raise ValueError("coarse prepared manifest source fingerprints differ")
    manifest_sha256 = sha256_file(output_dir / "manifest.json")
    if derivation.get("derived_manifest_sha256") != manifest_sha256:
        raise ValueError("coarse prepared manifest checksum differs")
    expected_derived = derivation.get("derived_cache_sha256")
    actual_derived = {path.name: sha256_file(path) for path in output_paths}
    if actual_derived != expected_derived:
        raise ValueError("coarse derived cache hashes differ")
    expected_from_provenance = {item["cache"] for item in verified["caches"]}
    if expected_from_provenance != expected_names:
        raise ValueError("coarse prepared manifest grid differs")
    return derivation


def prepare_coarse_data(data_dir, output_dir):
    """Create a separate verified prepared cache with ball-free 1 Hz halves.

    The v1 prepared manifest preserves source-position fingerprints while
    replacing cache hashes with hashes of the derived halves. A second,
    hash-wrapped derivation record links each new cache to its source cache and
    pins the exact processor rules. Source directories are never modified.
    """
    data_dir, provenance, grouped = _source_grid(data_dir)
    output_dir = Path(output_dir).resolve()
    if output_dir == data_dir or output_dir.is_relative_to(data_dir) or data_dir.is_relative_to(output_dir):
        raise ValueError("coarse output must be separate from prepared source data")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {path.name: sha256_file(path)
                     for periods in grouped.values() for path in periods.values()}
    existing = _validate_existing_derived(output_dir, provenance, source_hashes, grouped)
    if existing is not None:
        return existing
    if any(output_dir.iterdir()):
        raise ValueError("coarse output directory is not empty and has no complete derivation")

    output_matches = {}
    mapping = []
    for match in sorted(grouped):
        output_halves = []
        for period in (1, 2):
            source_path = grouped[match][period]
            frames, targets = load_half(source_path)
            coarse_frames, coarse_targets, anchors = coarsen_half(frames, targets)
            output_name = f"{match}_half{period}.npz"
            output_path = output_dir / output_name
            save_half(output_path, coarse_frames, coarse_targets)
            output_halves.append({"period": period, "cache": output_name,
                                  "cache_sha256": sha256_file(output_path)})
            mapping.append({"source_cache": source_path.name,
                            "source_sha256": source_hashes[source_path.name],
                            "derived_cache": output_name,
                            "derived_sha256": sha256_file(output_path),
                            "source_frames": len(frames.frame_id),
                            "coarse_frames": len(anchors)})
        output_matches[match] = {"match_id": match,
                                 "sources": provenance["sources"][match],
                                 "halves": output_halves}
    write_json(output_dir / "manifest.json", {"version": 1, "matches": output_matches})
    derived_paths = [output_dir / name for name in sorted(_expected_derived_names(grouped))]
    verify_prepared(output_dir, derived_paths)
    derived_hashes = {path.name: sha256_file(path) for path in derived_paths}
    record = {"schema": "coarse_context_derivation_v1", "processor": PROCESSOR,
              "source_prepared": provenance, "source_cache_sha256": source_hashes,
              "derived_cache_sha256": derived_hashes,
              "derived_manifest_sha256": sha256_file(output_dir / "manifest.json"),
              "mapping": mapping}
    return _publish_derivation(output_dir / _DERIVATION_FILENAME, record)


def _coarse_logits(features, model, center_size):
    if not isinstance(center_size, int) or isinstance(center_size, bool) or center_size < 1:
        raise ValueError("center_size must be a positive integer")
    radius = getattr(model, "radius", None)
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("coarse model must expose an integer temporal radius")
    parameters = list(model.parameters())
    device = parameters[0].device if parameters else torch.device("cpu")
    n = len(features.time_valid)
    values = {"live_logits": [], "owner_logits": []}
    previous_mode = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, n, center_size):
                stop = min(start + center_size, n)
                left, right = max(0, start - radius), min(n, stop + radius)
                scores = model(
                    _slice_tensor(features.player_tokens, left, right, device),
                    _slice_tensor(features.player_mask, left, right, device),
                    _slice_tensor(features.global_tokens, left, right, device),
                    _slice_tensor(features.time_valid, left, right, device),
                )
                for name, chunks in values.items():
                    chunks.append(scores[name][0, start - left:stop - left].cpu().numpy())
    finally:
        model.train(previous_mode)
    return {name: np.concatenate(chunks).astype(np.float64, copy=False)
            for name, chunks in values.items()}


def predict_coarse_native(frames: FrameTable, model: PossessionModel,
                          center_size: int = 64) -> dict[str, np.ndarray]:
    """Predict ball-free 1 Hz logits and interpolate them to native frame IDs."""
    coarse_frames, _, anchors = coarsen_half(frames)
    features = build_features(coarse_frames)
    logits = _coarse_logits(features, model, center_size)
    native = np.arange(len(frames.frame_id), dtype=np.float64)
    live = np.interp(native, anchors, logits["live_logits"])
    home = np.interp(native, anchors, logits["owner_logits"])

    def sigmoid(value):
        return 1. / (1. + np.exp(-np.clip(value, -60., 60.)))

    return {"live_probability": sigmoid(live), "home_probability": sigmoid(home),
            "native_anchor_indices": anchors.copy()}


def blend_coarse_context(fine_arrays, coarse_live, coarse_home):
    """Gated logit blend that keeps the fine branch's boundary and references.

    Coarse logits receive at most .25 weight. Weight rises with distance to an
    observed ball, and is exactly zero on frames with a visible ball.
    """
    if not isinstance(fine_arrays, dict):
        raise TypeError("fine_arrays must be a mapping")
    required = {"neural_live", "neural_home", "neural_boundary", "ball_valid",
                "nearest_ball_seconds"}
    missing = required - set(fine_arrays)
    if missing:
        raise ValueError(f"fine arrays are missing required values: {sorted(missing)}")
    live = np.asarray(fine_arrays["neural_live"], dtype=np.float64)
    home = np.asarray(fine_arrays["neural_home"], dtype=np.float64)
    coarse_live, coarse_home = np.asarray(coarse_live, dtype=np.float64), np.asarray(coarse_home, dtype=np.float64)
    ball_valid = np.asarray(fine_arrays["ball_valid"], dtype=bool)
    age = np.asarray(fine_arrays["nearest_ball_seconds"], dtype=np.float64)
    n = len(live)
    if (live.shape != (n,) or home.shape != (n,) or coarse_live.shape != (n,)
            or coarse_home.shape != (n,) or ball_valid.shape != (n,) or age.shape != (n,)
            or not np.isfinite(live).all() or not np.isfinite(home).all()
            or not np.isfinite(coarse_live).all() or not np.isfinite(coarse_home).all()
            or not np.isfinite(age).all() or np.any(age < 0)
            or np.any((live < 0) | (live > 1)) or np.any((home < 0) | (home > 1))
            or np.any((coarse_live < 0) | (coarse_live > 1))
            or np.any((coarse_home < 0) | (coarse_home > 1))):
        raise ValueError("fine and coarse probabilities or ball ages have invalid shapes or values")
    alpha = .25 * np.clip(age / 10., 0., 1.)
    alpha[ball_valid] = 0.
    result = dict(fine_arrays)
    for name, fine, coarse in (("neural_live", live, coarse_live),
                               ("neural_home", home, coarse_home)):
        blended = fine.copy()
        active = alpha > 0
        if np.any(active):
            p = np.clip(fine[active], 1e-7, 1. - 1e-7)
            q = np.clip(coarse[active], 1e-7, 1. - 1e-7)
            fine_logit = np.log(p) - np.log1p(-p)
            coarse_logit = np.log(q) - np.log1p(-q)
            mixed = (1. - alpha[active]) * fine_logit + alpha[active] * coarse_logit
            blended[active] = 1. / (1. + np.exp(-np.clip(mixed, -40., 40.)))
        result[name] = blended
    return result
