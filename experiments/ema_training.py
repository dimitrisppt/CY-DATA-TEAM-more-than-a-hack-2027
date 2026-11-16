"""Replay an authenticated campaign tail while exporting only EMA weights."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from experiments import campaign_protocol, campaign_training
from possession.predict import load_model_checkpoint
from possession.storage import sha256_file
from possession.training_session import HalfCache


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _tensor_state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(left, right, equal_nan=True))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(_tree_equal(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True)))
    if isinstance(left, float) and isinstance(right, float):
        return (math.isnan(left) and math.isnan(right)) or left == right
    return left == right


def _update_ema(ema_state, raw_state, decay):
    if ema_state.keys() != raw_state.keys():
        raise ValueError("EMA and raw state tensor names differ")
    for name, raw in raw_state.items():
        value = raw.detach().to(device="cpu")
        target = ema_state[name]
        if target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError(f"EMA tensor layout differs: {name}")
        if target.is_floating_point() or target.is_complex():
            target.mul_(decay).add_(value, alpha=1. - decay)
        else:
            target.copy_(value)


def _state_hash(path):
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"campaign state or checksum sidecar missing: {path}")
    expected = json.loads(sidecar.read_text()).get("sha256")
    actual = sha256_file(path)
    if expected != actual:
        raise ValueError(f"campaign state SHA256 differs: {path}")
    return actual


def _expected_model_kwargs(spec):
    result = {"player_dim": 8, "global_dim": 9, "width": 64,
              "dilations": [1, 2, 4, 8, 16, 32, 64, 128]}
    result.update(spec.get("model_kwargs", {}))
    return result


def _identity(identity, spec, validation_matches, start_steps, end_steps, decay):
    if not isinstance(identity, dict):
        raise TypeError("identity must be a campaign fit identity mapping")
    if not isinstance(spec, dict):
        raise TypeError("spec must be a campaign model specification mapping")
    required = {"protocol_sha256", "spec_sha256", "train_matches", "seed", "name"}
    if set(identity) != required:
        raise ValueError("campaign fit identity fields differ")
    normalized = _canonical(identity)
    if normalized["spec_sha256"] != _digest(spec):
        raise ValueError("campaign identity spec hash differs")
    train = normalized["train_matches"]
    if (not isinstance(train, list) or not train or len(train) != len(set(train))
            or any(not isinstance(match, str) or not match for match in train)):
        raise ValueError("campaign train matches are invalid")
    if (isinstance(normalized["seed"], bool) or not isinstance(normalized["seed"], int)
            or normalized["seed"] < 0 or not isinstance(normalized["name"], str)
            or not normalized["name"]):
        raise ValueError("campaign identity seed or name is invalid")
    validation = list(validation_matches)
    if (not validation or len(validation) != len(set(validation))
            or any(not isinstance(match, str) or not match for match in validation)):
        raise ValueError("validation matches must be a nonempty unique list")
    if set(train) & set(validation):
        raise ValueError("training and validation matches overlap")
    if (not isinstance(start_steps, int) or isinstance(start_steps, bool)
            or not isinstance(end_steps, int) or isinstance(end_steps, bool)
            or start_steps < 0 or end_steps <= start_steps):
        raise ValueError("EMA tail steps must be increasing nonnegative integers")
    if spec.get("schedule", {}).get("total_steps") != end_steps:
        raise ValueError("spec schedule total_steps must equal end_steps")
    if (isinstance(decay, bool) or not isinstance(decay, (int, float))
            or not np.isfinite(decay) or not 0 <= decay < 1):
        raise ValueError("EMA decay must lie in [0, 1)")
    return normalized, validation


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _verify_existing(output_dir, identity, inputs, spec, validation, decay,
                     start_steps, end_steps):
    fit_path = Path(output_dir) / "fit.json"
    previous = campaign_protocol.read_record(fit_path)
    if previous is None:
        return None
    expected = {"identity": identity, "steps": end_steps, "inputs": inputs}
    if any(previous.get(key) != value for key, value in expected.items()):
        raise ValueError("completed EMA fit identity differs")
    metadata = previous.get("tail_ema", {})
    if any(metadata.get(key) != value for key, value in {
            "decay": float(decay), "start_steps": start_steps,
            "end_steps": end_steps, "updates": end_steps - start_steps}.items()):
        raise ValueError("completed EMA fit schedule differs")
    expected_parity = {"raw_weights_match_endpoint_state": True,
                       "raw_weights_match_baseline_checkpoint": True,
                       "raw_training_progress_matches_endpoint": True,
                       "compared_optimizer_rng_transcript_and_counters": True,
                       "start_steps": start_steps, "end_steps": end_steps}
    if previous.get("raw_parity") != expected_parity or metadata.get("raw_parity") != expected_parity:
        raise ValueError("completed EMA raw parity evidence differs")
    model_path = Path(previous["model_path"])
    if (model_path != Path(output_dir).resolve() / "model.pt"
            or sha256_file(model_path) != previous.get("checkpoint_sha256")):
        raise ValueError("completed EMA checkpoint hash or path differs")
    model, payload = load_model_checkpoint(model_path)
    metadata = payload.get("ema_metadata")
    if metadata != previous.get("tail_ema") or _tensor_state_digest(payload["state_dict"]) != previous.get("ema_sha256"):
        raise ValueError("completed EMA checkpoint metadata differs")
    config = payload.get("training_config", {})
    if (payload.get("model_kwargs") != _expected_model_kwargs(spec)
            or config.get("campaign_spec") != spec
            or config.get("seed") != identity["seed"]
            or config.get("steps") != end_steps
            or config.get("train_matches") != identity["train_matches"]
            or config.get("validation_matches") not in ([], validation)):
        raise ValueError("completed EMA checkpoint training provenance differs")
    del model
    return previous


def replay_tail_ema(*, data_dir, spec, identity, state_path, baseline_checkpoint,
                    baseline_sha256, output_dir, validation_matches, decay=.99,
                    start_steps=7000, end_steps=7200, cache: HalfCache | None = None,
                    progress_callback: Callable[[], None] | None = None) -> dict:
    """Replay the final training tail and export an EMA averaged model.

    The raw replay must exactly match both the authenticated end state and the
    supplied final checkpoint before EMA weights can be published. This is a
    tail-only EMA experiment, not a full-run EMA training recipe.
    """
    normalized_identity, validation = _identity(
        identity, spec, validation_matches, start_steps, end_steps, decay)
    data_dir, state_path = Path(data_dir).resolve(), Path(state_path).resolve()
    baseline_checkpoint = Path(baseline_checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    end_state_path = state_path.parent / f"state_{end_steps:06d}.pt"
    for source in (state_path, end_state_path, baseline_checkpoint):
        if output_dir == source or source.is_relative_to(output_dir):
            raise ValueError("EMA output cannot contain or replace a source state or checkpoint")
    start_hash = _state_hash(state_path)
    end_hash = _state_hash(end_state_path)
    checkpoint_hash = sha256_file(baseline_checkpoint)
    if checkpoint_hash != baseline_sha256:
        raise ValueError("baseline checkpoint SHA256 differs from supplied fit record")
    inputs = {"start_state_sha256": start_hash, "end_state_sha256": end_hash,
              "baseline_checkpoint_sha256": checkpoint_hash,
              "spec_sha256": _digest(spec), "validation_matches": validation}
    previous = _verify_existing(output_dir, normalized_identity, inputs, spec,
                                validation, decay, start_steps, end_steps)
    if previous is not None:
        return previous

    model, checkpoint_payload = load_model_checkpoint(baseline_checkpoint)
    del model
    config = checkpoint_payload["training_config"]
    if (checkpoint_payload["model_kwargs"] != _expected_model_kwargs(spec)
            or config.get("campaign_spec") != spec
            or config.get("seed") != normalized_identity["seed"]
            or config.get("steps") != end_steps
            or config.get("train_matches") != normalized_identity["train_matches"]
            or config.get("validation_matches") not in ([], validation)):
        raise ValueError("baseline checkpoint config, spec, or split differs")

    started = time.monotonic()
    if progress_callback is not None:
        progress_callback()
    session = campaign_training.build_session(
        data_dir, spec, normalized_identity["train_matches"], normalized_identity["seed"], cache)
    start_progress = campaign_training.load_training_state(
        state_path, session, normalized_identity)
    if start_progress["steps"] != start_steps:
        raise ValueError(f"start state must be exactly {start_steps} steps")
    ema_state = {name: tensor.detach().cpu().clone()
                 for name, tensor in session.model.state_dict().items()}
    for step in range(start_steps + 1, end_steps + 1):
        if progress_callback is not None:
            progress_callback()
        session.advance_to(step)
        _update_ema(ema_state, session.model.state_dict(), float(decay))

    # Validate the complete raw trajectory before changing or exporting weights.
    endpoint_payload = torch.load(end_state_path, map_location="cpu", weights_only=True)
    campaign_training._validate_payload(endpoint_payload, session, normalized_identity)
    raw_state = campaign_training._session_state(session)
    if (session.steps != end_steps
            or not _tree_equal(raw_state, endpoint_payload["state"])
            or not _tree_equal(session.model.state_dict(), checkpoint_payload["state_dict"])):
        raise ValueError("raw replay does not exactly match authenticated end state and checkpoint")
    parity = {"raw_weights_match_endpoint_state": True,
              "raw_weights_match_baseline_checkpoint": True,
              "raw_training_progress_matches_endpoint": True,
              "compared_optimizer_rng_transcript_and_counters": True,
              "start_steps": start_steps, "end_steps": end_steps}
    ema_sha = _tensor_state_digest(ema_state)
    metadata = {"decay": float(decay), "start_steps": start_steps,
                "end_steps": end_steps, "updates": end_steps - start_steps,
                "initialization": "model state at start_steps",
                "floating_tensors": "exponential average",
                "nonfloating_tensors": "copy latest raw value",
                "ema_sha256": ema_sha, "raw_parity": parity}
    checkpoint_payload["state_dict"] = ema_state
    checkpoint_payload["ema_metadata"] = metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    _atomic_torch_save(model_path, checkpoint_payload)
    ema_model, saved_payload = load_model_checkpoint(model_path)
    del ema_model
    if _tensor_state_digest(saved_payload["state_dict"]) != ema_sha:
        raise ValueError("saved EMA checkpoint does not match averaged weights")
    parameter_count = sum(parameter.numel() for parameter in session.model.parameters())
    fit = {"identity": normalized_identity, "steps": end_steps,
           "checkpoint_sha256": sha256_file(model_path), "ema_sha256": ema_sha,
           "parameters": parameter_count, "model_path": str(model_path),
           "tail_ema": metadata, "raw_parity": parity, "inputs": inputs,
           "validation_matches": validation, "invocation_seconds": time.monotonic() - started}
    return campaign_protocol.publish(output_dir / "fit.json", fit)
