"""Exact, resumable TrainingSession trajectories for campaign recipes."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from possession.augmentation import normalize_augmentation
from possession.outages import OutageConfig
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache, TrainingSession


class _TranscriptHash:
    """Hash-compatible recorder whose complete transcript is checkpointable."""
    def __init__(self, transcript: bytes = b""):
        self.transcript = bytearray(transcript)

    def update(self, value: bytes) -> None:
        self.transcript.extend(value)

    def hexdigest(self) -> str:
        return hashlib.sha256(self.transcript).hexdigest()


def _schedule_hook(session: TrainingSession, spec: dict):
    schedule = spec["schedule"]
    kind = schedule["kind"]
    total = int(schedule["total_steps"])
    warmup = int(schedule["warmup_steps"])
    floor = float(schedule["min_lr_ratio"])
    base = float(spec["training_args"]["learning_rate"])
    if kind not in ("constant", "cosine") or total < 1 or not 0 <= warmup < total:
        raise ValueError("invalid campaign learning-rate schedule")
    if not 0 < floor <= 1 or (kind == "constant" and (warmup or floor != 1)):
        raise ValueError("invalid schedule warmup or minimum ratio")

    def before_step(optimizer, args, kwargs):
        step = session.steps
        if kind == "constant":
            ratio = 1.0
        elif warmup and step < warmup:
            ratio = (step + 1) / warmup
        else:
            span = max(1, total - warmup)
            progress = min(1.0, max(0.0, (step - warmup) / span))
            ratio = floor + (1.0 - floor) * .5 * (1.0 + np.cos(np.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = base * float(ratio)

    return session.optimizer.register_step_pre_hook(before_step)


def build_session(data_dir, spec, train_matches, seed, cache: HalfCache | None = None):
    """Create the repository trainer and attach the declared step schedule."""
    if not isinstance(spec, dict) or set(spec) != {
        "model_kwargs", "training_args", "outage_config", "augmentation_config", "schedule"
    }:
        raise ValueError("campaign spec fields differ")
    args = dict(spec["training_args"])
    session = TrainingSession(
        data_dir, train_matches=train_matches, seed=seed, device="cpu", cache=cache,
        model_kwargs=spec["model_kwargs"], learning_rate=args["learning_rate"],
        weight_decay=args["weight_decay"], boundary_weight=args["boundary_weight"],
        boundary_pos_weight=args["boundary_pos_weight"], augment=args["augment"],
        center_size=args["center_size"], outage_config=OutageConfig(**spec["outage_config"]),
        augmentation_config=normalize_augmentation(spec["augmentation_config"]),
    )
    session._training_window_hash = _TranscriptHash()
    session._campaign_spec = json.loads(json.dumps(spec, sort_keys=True, allow_nan=False))
    # TrainingSession.save_checkpoint serializes `config` as training_config;
    # this makes the exact LR schedule and sampled recipe visible to refit artifacts.
    session.config["campaign_spec"] = session._campaign_spec
    session._campaign_schedule_hook = _schedule_hook(session, spec)
    return session


def _json_identity(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("identity must be finite JSON data") from exc


def _signature(session, identity):
    config = {key: value for key, value in session.config.items() if key != "steps"}
    return {
        "identity": _json_identity(identity), "data_dir": str(session.data_dir),
        "train_matches": list(session.train_matches),
        "dataset_provenance": session.dataset_provenance,
        "model_kwargs": session.model_kwargs, "config": config,
        "campaign_spec": session._campaign_spec,
    }


def _rng_state(session):
    np_states = [session._rng.bit_generator.state,
                 session._augmentation_rng.bit_generator.state,
                 session._tracking_rng.bit_generator.state]
    legacy = np.random.get_state()
    return {
        "session_numpy": np_states,
        "numpy_legacy": {"kind": legacy[0], "keys": legacy[1].tolist(),
                         "position": int(legacy[2]), "has_gauss": int(legacy[3]),
                         "cached_gaussian": float(legacy[4])},
        "python": random.getstate(),
        "torch_cpu_internal": session._torch_cpu_state.cpu(),
        "torch_cuda_internal": (session._torch_cuda_state.cpu()
                                 if session._torch_cuda_state is not None else None),
        "torch_cpu_global": torch.get_rng_state().cpu(),
        "torch_cuda_global": ([state.cpu() for state in torch.cuda.get_rng_state_all()]
                              if torch.cuda.is_available() else []),
    }


_COUNTERS = (
    "_train_sums", "_train_counts", "_train_possible", "_boundary_positive_count",
    "_boundary_negative_count", "_augmentation_counts", "_center_ball_missing_frames",
    "_center_ball_frames", "_tracking_counts", "_tracking_selected_counts",
    "_tracking_applied_windows", "_tracking_strength_sum", "_tracking_strength_max",
    "_center_player_observations", "_center_player_slots", "_trained_halves",
    "_schedule_blocks", "_sampled_training_loss_trace", "_last_step_losses",
)


def _session_state(session):
    if not isinstance(session._training_window_hash, _TranscriptHash):
        raise TypeError("session must be created by build_session to preserve transcript")
    return {
        "steps": session.steps,
        "pending_paths": [path.name for path in session._pending_paths],
        "pending_index": session._pending_index,
        "counters": {key: (sorted(getattr(session, key)) if key == "_trained_halves"
                            else getattr(session, key)) for key in _COUNTERS},
        "training_transcript": bytes(session._training_window_hash.transcript),
        "model": {key: value.detach().cpu().clone()
                  for key, value in session.model.state_dict().items()},
        "optimizer": session.optimizer.state_dict(),
        "rng": _rng_state(session),
    }


def _atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_training_state(path, session: TrainingSession, identity: dict) -> Path:
    """Atomically save model, optimizer, data schedule, statistics, transcript and RNGs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "campaign_training_state_v1",
               "signature": _signature(session, identity), "state": _session_state(session)}
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent or Path("."))
    os.close(fd)
    try:
        torch.save(payload, temporary)
        raw = Path(temporary).read_bytes()
        _atomic_bytes(path, raw)
        write_json(path.with_suffix(path.suffix + ".sha256.json"), {"sha256": sha256_file(path)})
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def _validate_payload(payload, session, identity):
    if not isinstance(payload, dict) or payload.get("schema") != "campaign_training_state_v1":
        raise ValueError("unsupported campaign state schema")
    if payload.get("signature") != _signature(session, identity):
        raise ValueError("campaign state identity/config/data differs")
    state = payload.get("state")
    if not isinstance(state, dict) or not isinstance(state.get("steps"), int) or state["steps"] < 0:
        raise ValueError("campaign state progress is invalid")
    if set(state.get("model", {})) != set(session.model.state_dict()):
        raise ValueError("campaign model parameter names differ")
    for key, expected in session.model.state_dict().items():
        actual = state["model"][key]
        if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
                or actual.dtype != expected.dtype):
            raise ValueError(f"campaign model tensor differs: {key}")
    if not isinstance(state.get("optimizer"), dict) or not isinstance(state.get("rng"), dict):
        raise TypeError("campaign optimizer or RNG state missing")
    expected_optimizer = session.optimizer.state_dict()
    optimizer = state["optimizer"]
    if (len(optimizer.get("param_groups", ())) != len(expected_optimizer["param_groups"])
            or any(len(actual.get("params", ())) != len(expected["params"])
                   for actual, expected in zip(optimizer["param_groups"],
                                               expected_optimizer["param_groups"], strict=True))):
        raise ValueError("campaign optimizer parameter layout differs")
    if set(state.get("counters", {})) != set(_COUNTERS):
        raise ValueError("campaign training statistics differ")
    if not isinstance(state.get("training_transcript"), bytes):
        raise TypeError("campaign training transcript missing")
    if state["training_transcript"].count(b"\n") != state["steps"]:
        raise ValueError("campaign training transcript length differs")
    allowed_paths = {path.name for path in session.train_paths}
    if (not isinstance(state.get("pending_paths"), list)
            or any(path not in allowed_paths for path in state["pending_paths"])
            or not isinstance(state.get("pending_index"), int)
            or not 0 <= state["pending_index"] <= len(state["pending_paths"])):
        raise TypeError("campaign pending schedule is invalid")
    if len(state["pending_paths"]) not in (0, len(session.train_paths)):
        raise ValueError("campaign pending path schedule differs")
    rng = state["rng"]
    if (set(rng) != {"session_numpy", "numpy_legacy", "python", "torch_cpu_internal",
                     "torch_cuda_internal", "torch_cpu_global", "torch_cuda_global"}
            or len(rng["session_numpy"]) != 3
            or not isinstance(rng["torch_cpu_internal"], torch.Tensor)
            or not isinstance(rng["torch_cpu_global"], torch.Tensor)):
        raise ValueError("campaign RNG state is invalid")


def load_training_state(path, session: TrainingSession, identity: dict) -> dict:
    """Verify identity and hashes before mutating a fresh session, then restore exactly."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("campaign state or SHA256 sidecar is missing")
    digest = json.loads(sidecar.read_text()).get("sha256")
    if digest != sha256_file(path):
        raise ValueError("campaign state SHA256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_payload(payload, session, identity)
    state, rng = payload["state"], payload["state"]["rng"]
    session.model.load_state_dict(state["model"], strict=True)
    session.optimizer.load_state_dict(state["optimizer"])
    session.steps = state["steps"]
    session.config["steps"] = session.steps
    session._pending_paths = [session.data_dir / name for name in state["pending_paths"]]
    session._pending_index = state["pending_index"]
    for key, value in state["counters"].items():
        setattr(session, key, set(value) if key == "_trained_halves" else value)
    session._training_window_hash = _TranscriptHash(state["training_transcript"])
    for generator, generator_state in zip(
            (session._rng, session._augmentation_rng, session._tracking_rng),
            rng["session_numpy"], strict=True):
        generator.bit_generator.state = generator_state
    legacy = rng["numpy_legacy"]
    np.random.set_state((legacy["kind"], np.asarray(legacy["keys"], dtype=np.uint32),
                         legacy["position"], legacy["has_gauss"], legacy["cached_gaussian"]))
    random.setstate(tuple(rng["python"]))
    session._torch_cpu_state = rng["torch_cpu_internal"].clone()
    session._torch_cuda_state = rng["torch_cuda_internal"]
    torch.set_rng_state(rng["torch_cpu_global"])
    if rng["torch_cuda_global"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda_global"])
    return session._progress()
