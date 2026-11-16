"""Resumable continuation ablations, leaving the pinned trainer unchanged.

Load the authenticated raw campaign state before configure_refinement. Resuming
uses a newly built, configured campaign session and load_refinement_state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from experiments import campaign_training as campaign
from experiments.possession_objectives import (
    complete_spells,
    masked_teacher_loss,
    per_spell_loss,
    spell_center_start_range,
)
from possession.losses import masked_loss
from possession.outages import OutageConfig
from possession.storage import sha256_file
from possession.train import _window
from possession.training_session import TrainingSession

_CLEAN_OUTAGES = OutageConfig(mode="correlated", clean_probability=1., full_probability=0.)
_VARIANTS = ("control", "spell_balanced", "masked_teacher")


def _require_finite_tree(value, path="payload"):
    """Reject nonfinite numeric data throughout an otherwise valid state tree."""
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"nonfinite numeric state: {path}")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"nonfinite numeric state: {path}")
    elif isinstance(value, (float, complex, np.floating, np.complexfloating)):
        if not np.isfinite(value):
            raise ValueError(f"nonfinite numeric state: {path}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_tree(key, f"{path}.key")
            _require_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{path}[{index}]")


def _model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _clone_generator(generator, state):
    clone = copy.deepcopy(generator)
    clone.bit_generator.state = copy.deepcopy(state)
    return clone


def aligned_teacher_view(session, frames, targets, features, start, stop,
                         augmentation_before, tracking_before, student_targets,
                         student_choices):
    """Replay tracking transforms with no added ball outages and align team swap.

    Only cloned generators are advanced. A different outage draw count can
    change the team's swap draw, so explicitly reconcile the resulting view.
    """
    inputs, teacher_targets, choices = _window(
        frames, targets, features, start, stop, session.halo,
        _clone_generator(session._augmentation_rng, augmentation_before),
        session.augment, session.device, _CLEAN_OUTAGES,
        session.augmentation_config, session.steps + 1,
        _clone_generator(session._tracking_rng, tracking_before),
    )
    if choices["team_swap"] != student_choices["team_swap"]:
        inputs = (inputs[0].flip(2), inputs[1].flip(2), inputs[2], inputs[3])
        owner = teacher_targets["owner"]
        teacher_targets["owner"] = torch.where(teacher_targets["owner_valid"], 1 - owner, owner)
        state = teacher_targets["state"]
        teacher_targets["state"] = torch.where((state == 0) | (state == 1), 1 - state, state)
    if choices["tracking_augmentation"] != student_choices["tracking_augmentation"]:
        raise AssertionError("teacher tracking transformations differ")
    for name in student_targets:
        if not torch.equal(student_targets[name], teacher_targets[name]):
            raise AssertionError(f"teacher target alignment differs: {name}")
    if not torch.equal(inputs[3], student_targets["time_valid"]):
        raise AssertionError("teacher input time validity differs")
    return tuple(value.detach() for value in inputs)


def configure_refinement(session, *, variant, teacher=None, start_steps=7200,
                         total_extra_steps=1200, lr_scale=.2, spell_weight=.25,
                         short_sampling_probability=.25, distill_weight=.25,
                         temperature=2., confidence=.8):
    """Attach experimental behavior after raw-state authentication.

    A zero-step session may be configured solely to receive a refinement resume.
    Training cannot start until that restore brings it to start_steps.
    """
    if not hasattr(session, "_campaign_spec") or not isinstance(
            session._training_window_hash, campaign._TranscriptHash):
        raise TypeError("refinement requires campaign_training.build_session")
    if hasattr(session, "_refinement_config"):
        raise ValueError("session is already configured for refinement")
    if variant not in _VARIANTS:
        raise ValueError("unknown refinement variant")
    for name, value, positive in (("lr_scale", lr_scale, True),
                                   ("spell_weight", spell_weight, False),
                                   ("distill_weight", distill_weight, False),
                                   ("temperature", temperature, True)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or (value <= 0 if positive else value < 0)):
            raise ValueError(f"invalid {name}")
    for name, value in (("short_sampling_probability", short_sampling_probability),
                        ("confidence", confidence)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"invalid {name}")
    if not .5 <= confidence <= 1:
        raise ValueError("confidence must be between .5 and 1")
    for name, value, minimum in (("start_steps", start_steps, 0),
                                 ("total_extra_steps", total_extra_steps, 1)):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"invalid {name}")
    if session.steps not in (0, start_steps):
        raise ValueError("configure requires the raw start step or a fresh resume receiver")
    if variant == "masked_teacher":
        if not isinstance(teacher, torch.nn.Module):
            raise TypeError("masked_teacher requires a frozen same-fold model")
        _require_finite_tree(teacher.state_dict(), "teacher")
        if set(teacher.state_dict()) != set(session.model.state_dict()):
            raise ValueError("teacher architecture differs")
        for name, expected in session.model.state_dict().items():
            if teacher.state_dict()[name].shape != expected.shape:
                raise ValueError(f"teacher tensor shape differs: {name}")
        bound_teacher = copy.deepcopy(teacher).to(session.device).eval()
        bound_teacher.requires_grad_(False)
    elif teacher is not None:
        raise ValueError("only masked_teacher accepts a teacher")
    else:
        bound_teacher = None
    config = {
        "variant": variant, "start_steps": start_steps, "total_extra_steps": total_extra_steps,
        "lr_scale": float(lr_scale), "learning_rate": session.config["learning_rate"] * lr_scale,
        "spell_weight": float(spell_weight),
        "short_sampling_probability": float(short_sampling_probability),
        "distill_weight": float(distill_weight), "temperature": float(temperature),
        "confidence": float(confidence), "tail_ema_decay": .99,
        "tail_ema_start_steps": start_steps + max(0, total_extra_steps - 200),
        "teacher_state_sha256": _model_hash(bound_teacher) if bound_teacher is not None else None,
    }
    session.__class__ = RefinedTrainingSession
    session._refinement_config = config
    session._refinement_teacher = bound_teacher
    session.config["refinement"] = copy.deepcopy(config)
    session._refinement_counters = {
        "updates": 0, "original_samples": 0, "short_spell_draws": 0,
        "short_spell_samples": 0, "short_spell_unavailable": 0,
        "aux_steps": 0, "aux_loss_sum": 0., "aux_live_loss_sum": 0.,
        "aux_owner_loss_sum": 0., "spell_count": 0, "live_count": 0,
        "owner_count": 0, "newly_hidden_count": 0,
    }
    session._tail_ema = None
    session._tail_ema_updates = 0
    if session.steps == config["tail_ema_start_steps"]:
        session._tail_ema = session._copy_model_state()
    session._campaign_schedule_hook.remove()
    session._refinement_lr_hook = session.optimizer.register_step_pre_hook(session._constant_lr)
    session._refinement_ema_hook = session.optimizer.register_step_post_hook(session._after_update)
    session._constant_lr(session.optimizer, (), {})
    return session


def build_refinement_session(data_dir, spec, train_matches, seed, *, parent_state,
                             parent_identity, variant, teacher=None, cache=None, **kwargs):
    """Build and authenticate the shared raw state before attaching the ablation."""
    session = campaign.build_session(data_dir, spec, train_matches, seed, cache=cache)
    campaign.load_training_state(parent_state, session, parent_identity)
    return configure_refinement(session, variant=variant, teacher=teacher, **kwargs)


class RefinedTrainingSession(TrainingSession):
    """Original counters and losses plus isolated auxiliary training behavior."""

    def _constant_lr(self, optimizer, args, kwargs):
        for group in optimizer.param_groups:
            group["lr"] = self._refinement_config["learning_rate"]

    def _copy_model_state(self):
        return {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

    def _after_update(self, optimizer, args, kwargs):
        step = self.steps + 1
        self._refinement_counters["updates"] += 1
        if self._refinement_config["variant"] != "spell_balanced":
            self._refinement_counters["original_samples"] += 1
        ema_start = self._refinement_config["tail_ema_start_steps"]
        if step == ema_start:
            self._tail_ema = self._copy_model_state()
        elif step > ema_start:
            if self._tail_ema is None:
                raise RuntimeError("tail EMA initialization is missing")
            with torch.no_grad():
                for key, value in self.model.state_dict().items():
                    source = value.detach().cpu()
                    if source.is_floating_point() or source.is_complex():
                        self._tail_ema[key].mul_(.99).add_(source, alpha=.01)
                    else:
                        self._tail_ema[key].copy_(source)
            self._tail_ema_updates += 1

    def _validate_budget(self, total_steps):
        cfg = self._refinement_config
        if self.steps < cfg["start_steps"]:
            raise ValueError("restore the authenticated start state before refinement training")
        if total_steps > cfg["start_steps"] + cfg["total_extra_steps"]:
            raise ValueError("refinement budget exceeded")

    def _sample_start(self, frames, targets, features, transitions, size):
        cfg = self._refinement_config
        if cfg["variant"] == "spell_balanced":
            counters = self._refinement_counters
            if self._rng.random() < cfg["short_sampling_probability"]:
                counters["short_spell_draws"] += 1
                candidates = [
                    (spell, spell_center_start_range(spell, size, len(frames.frame_id)))
                    for spell in complete_spells(targets.state, targets.state_valid,
                                                 features.time_valid, frames.frame_id)
                    if spell.state in (0, 1) and spell.frames < 50
                ]
                candidates = [(spell, starts) for spell, starts in candidates if len(starts)]
                if candidates:
                    _, starts = candidates[int(self._rng.integers(len(candidates)))]
                    counters["short_spell_samples"] += 1
                    return int(starts[int(self._rng.integers(len(starts)))])
                counters["short_spell_unavailable"] += 1
            counters["original_samples"] += 1
        if len(transitions) and self._rng.random() < .5:
            transition = int(self._rng.choice(transitions))
            return int(np.clip(transition - int(self._rng.integers(size)),
                               0, len(frames.frame_id) - size))
        return int(self._rng.integers(len(frames.frame_id) - size + 1))

    def _record_aux(self, aux):
        counters = self._refinement_counters
        counters["aux_steps"] += 1
        for name in ("live", "owner"):
            counters[f"aux_{name}_loss_sum"] += float(aux[name].detach().item())
        counters["aux_loss_sum"] += float(aux["total"].detach().item())
        for name in ("spell_count", "live_count", "owner_count", "newly_hidden_count"):
            counters[name] += int(aux.get(name, 0))

    def _progress(self):
        return {**super()._progress(), "refinement": {
            "config": copy.deepcopy(self._refinement_config),
            "counters": self._refinement_counters.copy(),
            "tail_ema_initialized": self._tail_ema is not None,
            "tail_ema_updates": self._tail_ema_updates,
        }}

    def advance_to(self, total_steps: int) -> dict:
        """Train through an absolute step number; calls preserve optimizer/RNG state."""
        if not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps < 0:
            raise ValueError("total_steps must be a nonnegative integer")
        if total_steps < self.steps:
            raise ValueError("cannot advance backward")
        if total_steps == self.steps:
            return self._progress()
        self._validate_budget(total_steps)
        if self._refinement_config["variant"] == "control":
            return super().advance_to(total_steps)
        cuda_devices = [self._cuda_device] if self._cuda_device is not None else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.set_rng_state(self._torch_cpu_state)
            if self._cuda_device is not None:
                torch.cuda.set_rng_state(self._torch_cuda_state, self._cuda_device)
            self.model.train()
            while self.steps < total_steps:
                path = self._next_path()
                frames, targets, features, transitions = self.cache.get(path)
                size = min(self.center_size, len(frames.frame_id))
                start = self._sample_start(frames, targets, features, transitions, size)
                self._training_window_hash.update(f"{path.name}:{start}:{size}\n".encode())
                augmentation_before = copy.deepcopy(self._augmentation_rng.bit_generator.state)
                tracking_before = copy.deepcopy(self._tracking_rng.bit_generator.state)
                inputs, loss_targets, choices = _window(
                    frames, targets, features, start, start + size, self.halo,
                    self._augmentation_rng, self.augment, self.device,
                    self.outage_config, self.augmentation_config, self.steps + 1,
                    self._tracking_rng,
                )
                self.optimizer.zero_grad(set_to_none=True)
                scores = self.model(*inputs)
                result = masked_loss(scores, loss_targets,
                                     self.boundary_weight, self.boundary_pos_weight)
                aux = None
                if self._refinement_config["variant"] == "spell_balanced":
                    aux = per_spell_loss(scores, loss_targets)
                    weight = self._refinement_config["spell_weight"]
                elif self._refinement_config["variant"] == "masked_teacher":
                    teacher_inputs = aligned_teacher_view(
                        self, frames, targets, features, start, start + size,
                        augmentation_before, tracking_before, loss_targets, choices)
                    with torch.no_grad():
                        teacher_scores = self._refinement_teacher(*teacher_inputs)
                    aux = masked_teacher_loss(
                        scores, teacher_scores, loss_targets, inputs, teacher_inputs,
                        confidence=self._refinement_config["confidence"],
                        temperature=self._refinement_config["temperature"])
                    weight = self._refinement_config["distill_weight"]
                loss = result["total"]
                if aux is not None:
                    loss = loss + weight * aux["total"]
                    self._record_aux(aux)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0,
                                               error_if_nonfinite=True)
                self.optimizer.step()
                if not all(torch.isfinite(parameter).all().item()
                           for parameter in self.model.parameters()):
                    raise FloatingPointError("nonfinite model parameter")
                for head in self._train_sums:
                    count = int(result[f"{head}_count"])
                    self._train_sums[head] += float(result[head].detach().item()) * count
                    self._train_counts[head] += count
                    self._train_possible[head] += size - int(head == "boundary" and start == 0)
                step_losses = {"step": self.steps + 1, **{
                    name: float(result[name].detach().item())
                    for name in ("live", "owner", "boundary", "total")
                }}
                self._last_step_losses = step_losses
                if step_losses["step"] == 1 or step_losses["step"] % 100 == 0:
                    self._sampled_training_loss_trace.append(step_losses)
                self._boundary_positive_count += int(result["boundary_positive_count"])
                self._boundary_negative_count += int(result["boundary_negative_count"])
                for name in self._augmentation_counts:
                    self._augmentation_counts[name] += int(choices[name])
                self._center_ball_missing_frames += int(choices["center_ball_missing_frames"])
                self._center_ball_frames += int(choices["center_ball_frames"])
                tracking = choices["tracking_augmentation"]
                self._tracking_applied_windows += int(bool(tracking["applied"]))
                self._tracking_strength_sum += float(tracking["strength"])
                self._tracking_strength_max = max(self._tracking_strength_max, float(tracking["strength"]))
                for operation in tracking["applied"]:
                    self._tracking_counts[operation] = self._tracking_counts.get(operation, 0) + 1
                for operation in tracking["selected"]:
                    self._tracking_selected_counts[operation] = self._tracking_selected_counts.get(operation, 0) + 1
                self._center_player_observations += int(choices["center_player_observations"])
                self._center_player_slots += int(choices["center_player_slots"])
                self._trained_halves.add(path.name)
                self.steps += 1
                self.config["steps"] = self.steps
            self._torch_cpu_state = torch.get_rng_state().clone()
            if self._cuda_device is not None:
                self._torch_cuda_state = torch.cuda.get_rng_state(self._cuda_device).clone()
        return self._progress()


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        campaign._atomic_bytes(path.with_suffix(path.suffix + ".sha256.json"),
                               json.dumps({"sha256": digest}, sort_keys=True).encode() + b"\n")
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def _refinement_payload(session, identity):
    if not isinstance(session, RefinedTrainingSession):
        raise TypeError("configure the refinement session before saving")
    return {
        "schema": "refinement_training_state_v1",
        "campaign": {"schema": "campaign_training_state_v1",
                     "signature": campaign._signature(session, identity),
                     "state": campaign._session_state(session)},
        "refinement": {"config": copy.deepcopy(session._refinement_config),
                       "counters": session._refinement_counters.copy(),
                       "tail_ema": session._tail_ema,
                       "tail_ema_updates": session._tail_ema_updates},
    }


def save_refinement_state(path, session, identity):
    """Persist the base trajectory and all custom state with an atomic hash sidecar."""
    payload = _refinement_payload(session, identity)
    _validate_refinement_payload(payload, session, identity)
    return _atomic_torch_save(path, payload)


def _validate_refinement_payload(payload, session, identity):
    if not isinstance(session, RefinedTrainingSession):
        raise TypeError("configure the refinement session before restoring")
    if not isinstance(payload, dict) or payload.get("schema") != "refinement_training_state_v1":
        raise ValueError("unsupported refinement state schema")
    _require_finite_tree(payload)
    campaign._validate_payload(payload.get("campaign"), session, identity)
    custom = payload.get("refinement")
    if not isinstance(custom, dict) or set(custom) != {
            "config", "counters", "tail_ema", "tail_ema_updates"}:
        raise ValueError("refinement state fields differ")
    if custom["config"] != session._refinement_config:
        raise ValueError("refinement config or teacher binding differs")
    if session._refinement_teacher is not None:
        _require_finite_tree(session._refinement_teacher.state_dict(), "teacher")
        if _model_hash(session._refinement_teacher) != custom["config"]["teacher_state_sha256"]:
            raise ValueError("refinement teacher weights differ")
        if session._refinement_teacher.training or any(
                value.requires_grad for value in session._refinement_teacher.parameters()):
            raise ValueError("refinement teacher must remain frozen in evaluation mode")
    cfg = custom["config"]
    steps = payload["campaign"]["state"]["steps"]
    if not cfg["start_steps"] <= steps <= cfg["start_steps"] + cfg["total_extra_steps"]:
        raise ValueError("refinement progress is outside the declared budget")
    counters = custom["counters"]
    if not isinstance(counters, dict) or set(counters) != set(session._refinement_counters):
        raise ValueError("refinement counters differ")
    for name, value in counters.items():
        if name.endswith("loss_sum"):
            if (not isinstance(value, (float, int)) or isinstance(value, bool)
                    or not math.isfinite(value) or value < 0):
                raise ValueError("invalid refinement auxiliary loss sum")
        elif not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("invalid refinement count")
    if counters["updates"] != steps - cfg["start_steps"]:
        raise ValueError("refinement update counter differs")
    if counters["original_samples"] + counters["short_spell_samples"] != counters["updates"]:
        raise ValueError("refinement sample counters differ")
    expected_aux = counters["updates"] if cfg["variant"] != "control" else 0
    if counters["aux_steps"] != expected_aux:
        raise ValueError("refinement auxiliary counter differs")
    ema_start = cfg["tail_ema_start_steps"]
    updates = custom["tail_ema_updates"]
    if (not isinstance(updates, int) or isinstance(updates, bool)
            or updates != max(0, steps - ema_start)):
        raise ValueError("refinement tail EMA update counter differs")
    ema = custom["tail_ema"]
    if steps < ema_start:
        if ema is not None:
            raise ValueError("refinement tail EMA initialized too early")
    else:
        expected = session.model.state_dict()
        if not isinstance(ema, dict) or set(ema) != set(expected):
            raise ValueError("refinement tail EMA parameter names differ")
        for name, tensor in ema.items():
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != expected[name].dtype
                    or tensor.shape != expected[name].shape or not torch.isfinite(tensor).all()):
                raise ValueError(f"invalid refinement tail EMA tensor: {name}")


def load_refinement_state(path, session, identity):
    """Validate the hash, split identity, teacher binding and custom state before restore."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("refinement state or SHA256 sidecar is missing")
    if json.loads(sidecar.read_text()).get("sha256") != sha256_file(path):
        raise ValueError("refinement state SHA256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_refinement_payload(payload, session, identity)
    state = payload["campaign"]["state"]
    rng = state["rng"]
    session.model.load_state_dict(state["model"], strict=True)
    session.optimizer.load_state_dict(state["optimizer"])
    session.steps = state["steps"]
    session.config["steps"] = session.steps
    session._pending_paths = [session.data_dir / name for name in state["pending_paths"]]
    session._pending_index = state["pending_index"]
    for key, value in state["counters"].items():
        setattr(session, key, set(value) if key == "_trained_halves" else value)
    session._training_window_hash = campaign._TranscriptHash(state["training_transcript"])
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
    custom = payload["refinement"]
    session._refinement_counters = custom["counters"].copy()
    session._tail_ema = custom["tail_ema"]
    session._tail_ema_updates = custom["tail_ema_updates"]
    return session._progress()


def export_refinement_checkpoint(path, session, identity, *, use_tail_ema=True):
    """Export a prediction-compatible checkpoint with parent and teacher identities.

    Only the session's training split is read for prediction-checkpoint provenance.
    The caller supplies authenticated parent and teacher file hashes in identity.
    """
    if not isinstance(session, RefinedTrainingSession):
        raise TypeError("configure the refinement session before export")
    cfg = session._refinement_config
    if session.steps != cfg["start_steps"] + cfg["total_extra_steps"]:
        raise ValueError("refinement checkpoint export requires the complete declared budget")
    if use_tail_ema and session._tail_ema is None:
        raise ValueError("refinement tail EMA is not initialized")
    _validate_refinement_payload(_refinement_payload(session, identity), session, identity)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    original = session._copy_model_state() if use_tail_ema else None
    try:
        if use_tail_ema:
            session.model.load_state_dict(session._tail_ema, strict=True)
        session.save_checkpoint(temporary, extra_training_config={
            "refinement_identity": campaign._json_identity(identity),
            "refinement_export": "tail_ema" if use_tail_ema else "raw",
            "tail_ema_updates": session._tail_ema_updates,
        })
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        campaign._atomic_bytes(path.with_suffix(path.suffix + ".sha256.json"),
                               json.dumps({"sha256": digest}, sort_keys=True).encode() + b"\n")
    finally:
        if original is not None:
            session.model.load_state_dict(original, strict=True)
        Path(temporary).unlink(missing_ok=True)
        Path(temporary + ".tmp").unlink(missing_ok=True)
    return path
