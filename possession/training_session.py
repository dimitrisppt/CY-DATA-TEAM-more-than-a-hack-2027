"""Incremental, reproducible training with a shared bounded prepared-half cache."""

from __future__ import annotations

import hashlib
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from possession.augmentation import normalize_augmentation
from possession.losses import masked_loss
from possession.model import PossessionModel
from possession.outages import OutageConfig
from possession.prepared import verify_prepared
from possession.provenance import runtime_provenance
from possession.storage import sha256_file
from possession.train import _canonical_matches, _mean_losses, _prepared_paths, _read_half, _window


def _half_bytes(half) -> int:
    """Count each backing array once, including raw and derived features."""
    frames, targets, features, transitions = half
    arrays = [getattr(frames, name) for name in
              ("frame_id", "players_xy", "player_valid", "ball_xy", "ball_valid", "graded")]
    arrays += [getattr(targets, name) for name in
               ("live", "live_valid", "owner", "owner_valid", "state", "state_valid")]
    arrays += [getattr(features, name) for name in
               ("player_tokens", "player_mask", "global_tokens", "time_valid")]
    arrays.append(transitions)
    return sum(array.nbytes for array in arrays)


class HalfCache:
    """LRU of prepared halves, shared by fold trainers and bounded by array bytes.

    An individual half larger than the budget is read and returned without being
    retained. The byte count covers NumPy data buffers; Python object and model
    overhead are outside this cache's memory budget.
    """

    def __init__(self, data_dir, max_bytes: int = 1_500_000_000):
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self.data_dir = Path(data_dir).resolve()
        self.max_bytes = max_bytes
        self.bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self._entries: OrderedDict[Path, tuple[tuple, int]] = OrderedDict()

    def get(self, path):
        path = Path(path)
        path = (path if path.is_absolute() else self.data_dir / path).resolve()
        if path.parent != self.data_dir:
            raise ValueError(f"prepared half outside data directory: {path}")
        if path in self._entries:
            self.hits += 1
            self._entries.move_to_end(path)
            return self._entries[path][0]
        self.misses += 1
        half = _read_half(path)
        size = _half_bytes(half)
        if size <= self.max_bytes:
            while self.bytes + size > self.max_bytes:
                _, (_, removed_size) = self._entries.popitem(last=False)
                self.bytes -= removed_size
            self._entries[path] = (half, size)
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        return half


class TrainingSession:
    """Advance one seeded optimizer trajectory to successive absolute step budgets."""

    def __init__(
        self,
        data_dir,
        *,
        train_matches,
        seed: int = 0,
        device: str = "cpu",
        model_kwargs=None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        boundary_weight: float = 0.25,
        boundary_pos_weight: float = 1.0,
        augment: bool = True,
        outage_config: OutageConfig | dict | None = None,
        augmentation_config=None,
        center_size: int = 512,
        halo: int | None = None,
        cache: HalfCache | None = None,
        schedule_block_size: int = 4,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.train_matches = _canonical_matches(self.data_dir, list(train_matches))
        if not self.train_matches:
            raise ValueError("at least one train match is required")
        self.train_paths = _prepared_paths(self.data_dir, self.train_matches)
        self.dataset_provenance = verify_prepared(self.data_dir, self.train_paths)
        if (not isinstance(seed, int) or isinstance(seed, bool)
                or not isinstance(center_size, int) or center_size < 1):
            raise ValueError("seed must be an integer and center_size must be positive")
        for name, value, strictly_positive in (
            ("learning_rate", learning_rate, True),
            ("weight_decay", weight_decay, False),
            ("boundary_weight", boundary_weight, False),
            ("boundary_pos_weight", boundary_pos_weight, True),
        ):
            if not math.isfinite(value) or (value <= 0 if strictly_positive else value < 0):
                raise ValueError(f"{name} must be finite and {'positive' if strictly_positive else 'nonnegative'}")
        if cache is not None and cache.data_dir != self.data_dir:
            raise ValueError("cache data directory differs from training data directory")
        if (not isinstance(schedule_block_size, int) or isinstance(schedule_block_size, bool)
                or schedule_block_size < 1):
            raise ValueError("schedule_block_size must be a positive integer")
        self.cache = cache if cache is not None else HalfCache(self.data_dir)
        self.schedule_block_size = schedule_block_size
        self.seed = seed
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)
        self._augmentation_rng = np.random.default_rng(seed ^ 0xA5B3571D)
        self._tracking_rng = np.random.default_rng(seed ^ 0x63D49B27)
        self.augmentation_config = normalize_augmentation(augmentation_config)
        if outage_config is None:
            self.outage_config = OutageConfig()
        elif isinstance(outage_config, OutageConfig):
            self.outage_config = outage_config
        elif isinstance(outage_config, dict):
            self.outage_config = OutageConfig(**outage_config)
        else:
            raise ValueError("outage_config must be an OutageConfig or dictionary")
        self.model_kwargs = {"player_dim": 8, "global_dim": 9, "width": 64,
                             "dilations": [1, 2, 4, 8, 16, 32, 64, 128]}
        if model_kwargs is not None:
            self.model_kwargs.update(model_kwargs)
        self.model_kwargs["dilations"] = list(self.model_kwargs["dilations"])
        cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] \
            if self.device.type == "cuda" else []
        self._cuda_device = cuda_devices[0] if cuda_devices else None
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            self.model = PossessionModel(**self.model_kwargs).to(self.device)
            self._torch_cpu_state = torch.get_rng_state().clone()
            self._torch_cuda_state = (torch.cuda.get_rng_state(self._cuda_device).clone()
                                      if cuda_devices else None)
        self.halo = self.model.radius if halo is None else halo
        if not isinstance(self.halo, int) or self.halo < self.model.radius:
            raise ValueError(f"halo must be >= model radius {self.model.radius}")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate,
                                           weight_decay=weight_decay)
        self.boundary_weight = boundary_weight
        self.boundary_pos_weight = boundary_pos_weight
        self.augment = bool(augment)
        self.center_size = center_size
        self.steps = 0
        self._pending_paths: list[Path] = []
        self._pending_index = 0
        self._train_sums = {head: 0.0 for head in ("live", "owner", "boundary")}
        self._train_counts = {head: 0 for head in self._train_sums}
        self._train_possible = {head: 0 for head in self._train_sums}
        self._boundary_positive_count = 0
        self._boundary_negative_count = 0
        self._augmentation_counts = {"team_swap": 0, "ball_outage": 0}
        self._center_ball_missing_frames = 0
        self._center_ball_frames = 0
        self._tracking_counts: dict[str, int] = {}
        self._tracking_selected_counts: dict[str, int] = {}
        self._tracking_applied_windows = 0
        self._tracking_strength_sum = 0.0
        self._tracking_strength_max = 0.0
        self._center_player_observations = 0
        self._center_player_slots = 0
        self._trained_halves: set[str] = set()
        self._schedule_blocks: list[dict] = []
        self._training_window_hash = hashlib.sha256()
        self._sampled_training_loss_trace: list[dict] = []
        self._last_step_losses: dict | None = None
        self.config = {
            "train_matches": self.train_matches.copy(), "steps": self.steps,
            "seed": seed, "device": str(self.device), "center_size": center_size,
            "halo": self.halo, "learning_rate": learning_rate,
            "weight_decay": weight_decay, "boundary_weight": boundary_weight,
            "boundary_pos_weight": boundary_pos_weight, "augment": self.augment,
            "outage_config": self.outage_config.to_dict(),
            "augmentation_config": self.augmentation_config.to_dict(),
            "rng_scheme": "separate_data_and_augmentation_v1",
            "tracking_rng_scheme": "independent_tracking_augmentation_v1",
            "schedule_block_size": schedule_block_size,
        }

    def _next_path(self) -> Path:
        if self._pending_index >= len(self._pending_paths):
            self._pending_paths = self.train_paths.copy()
            self._rng.shuffle(self._pending_paths)
            self._pending_index = 0
        path = self._pending_paths[self._pending_index]
        if (self.steps % self.schedule_block_size == 0 or not self._schedule_blocks
                or self._schedule_blocks[-1]["half"] != path.name):
            self._schedule_blocks.append({"half": path.name, "steps": 0})
        self._schedule_blocks[-1]["steps"] += 1
        if self._schedule_blocks[-1]["steps"] == self.schedule_block_size:
            self._pending_index += 1
        return path

    def _progress(self) -> dict:
        losses = _mean_losses(self._train_sums, self._train_counts, self.boundary_weight)
        return {
            "steps": self.steps, "train_loss": losses["total"], "train_losses": losses,
            "train_counts": self._train_counts.copy(),
            "train_possible": self._train_possible.copy(),
            "boundary_positive_count": self._boundary_positive_count,
            "boundary_negative_count": self._boundary_negative_count,
            "augmentation_counts": self._augmentation_counts.copy(),
            "tracking_augmentation": {
                "operation_counts": self._tracking_counts.copy(),
                "selected_operation_counts": self._tracking_selected_counts.copy(),
                "applied_windows": self._tracking_applied_windows,
                "mean_strength": self._tracking_strength_sum / self.steps if self.steps else 0.0,
                "max_strength": self._tracking_strength_max,
            },
            "center_player_observed_fraction": (
                self._center_player_observations / self._center_player_slots
                if self._center_player_slots else 0.0),
            "center_ball_missing_frames": self._center_ball_missing_frames,
            "center_ball_frames": self._center_ball_frames,
            "center_ball_missing_fraction": (
                self._center_ball_missing_frames / self._center_ball_frames
                if self._center_ball_frames else 0.0
            ),
            "trained_halves": sorted(self._trained_halves),
            "training_schedule_blocks": [block.copy() for block in self._schedule_blocks],
            "training_window_sha256": self._training_window_hash.hexdigest(),
            "sampled_training_loss_trace": [entry.copy() for entry in self._sampled_training_loss_trace],
            "last_step_losses": None if self._last_step_losses is None else self._last_step_losses.copy(),
        }

    def advance_to(self, total_steps: int) -> dict:
        """Train through an absolute step number; calls preserve optimizer/RNG state."""
        if not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps < 0:
            raise ValueError("total_steps must be a nonnegative integer")
        if total_steps < self.steps:
            raise ValueError("cannot advance backward")
        if total_steps == self.steps:
            return self._progress()
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
                if len(transitions) and self._rng.random() < 0.5:
                    transition = int(self._rng.choice(transitions))
                    start = int(np.clip(transition - int(self._rng.integers(size)),
                                        0, len(frames.frame_id) - size))
                else:
                    start = int(self._rng.integers(len(frames.frame_id) - size + 1))
                self._training_window_hash.update(f"{path.name}:{start}:{size}\n".encode())
                inputs, loss_targets, choices = _window(
                    frames, targets, features, start, start + size, self.halo,
                    self._augmentation_rng, self.augment, self.device,
                    self.outage_config, self.augmentation_config, self.steps + 1,
                    self._tracking_rng,
                )
                self.optimizer.zero_grad(set_to_none=True)
                result = masked_loss(self.model(*inputs), loss_targets,
                                     self.boundary_weight, self.boundary_pos_weight)
                loss = result["total"]
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

    def save_checkpoint(
        self, path, *, validation_matches=(), extra_training_config=None
    ) -> Path:
        """Write a prediction-compatible checkpoint with explicit split provenance."""
        validation_matches = _canonical_matches(self.data_dir, list(validation_matches))
        if set(self.train_matches) & set(validation_matches):
            raise ValueError("train and validation matches must be disjoint")
        validation_paths = _prepared_paths(self.data_dir, validation_matches)
        selected_paths = sorted(self.train_paths + validation_paths)
        dataset_provenance = verify_prepared(self.data_dir, selected_paths)
        prepared_hashes = {source.name: sha256_file(source) for source in selected_paths}
        provenance = hashlib.sha256()
        for name, digest in sorted(prepared_hashes.items()):
            provenance.update(f"{name}:{digest}\n".encode())
        training_config = {**self.config, "validation_matches": validation_matches}
        if extra_training_config is not None:
            collisions = set(extra_training_config) & set(training_config)
            if collisions:
                raise ValueError(f"extra_training_config uses reserved keys: {sorted(collisions)}")
            training_config.update(extra_training_config)
        payload = {
            "state_dict": self.model.state_dict(),
            "model_kwargs": self.model_kwargs.copy(),
            "training_config": training_config,
            "prepared_sha256": prepared_hashes,
            "prepared_inputs_hash": provenance.hexdigest(),
            "runtime": runtime_provenance(),
            "dataset_provenance": dataset_provenance,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return path
