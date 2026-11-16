"""Bounded, reproducible training on prepared source halves."""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from possession.augmentation import AugmentationConfig, augment_tracking, normalize_augmentation
from possession.features import build_features
from possession.losses import masked_loss
from possession.outages import OutageConfig, generate_outage_mask
from possession.prepared import verify_prepared
from possession.schema import FeatureBatch, FrameTable, TrainingTargets
from possession.storage import load_half, sha256_file, write_json


def _loss_targets_for_window(
    targets: TrainingTargets,
    time_valid: np.ndarray,
    frame_id: np.ndarray,
    *,
    left: int,
    right: int,
    center_start: int,
    center_stop: int,
) -> dict[str, torch.Tensor]:
    """Mask context labels, preserving only the center's incoming boundary."""

    if not (0 <= left <= center_start < center_stop <= right <= len(targets.live)):
        raise ValueError("invalid center/context bounds")
    center = np.zeros(right - left, dtype=bool)
    center[center_start - left : center_stop - left] = True
    boundary = center.copy()
    if center_start > 0:
        boundary[center_start - left - 1] = True
    arrays = {
        "live": targets.live[left:right],
        "live_valid": targets.live_valid[left:right] & center,
        "owner": targets.owner[left:right],
        "owner_valid": targets.owner_valid[left:right] & center,
        "state": targets.state[left:right],
        "state_valid": targets.state_valid[left:right] & boundary,
        "time_valid": time_valid[left:right],
        "frame_id": frame_id[left:right],
    }
    return {name: torch.from_numpy(np.asarray(values).copy())[None] for name, values in arrays.items()}


def _selected_features(
    frames: FrameTable,
    base: FeatureBatch,
    left: int,
    right: int,
    center_start: int,
    center_stop: int,
    rng: np.random.Generator,
    augment: bool,
    outage_config: OutageConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Apply a ball outage, then recompute affected features."""

    if augment and outage_config is not None and outage_config.mode == "correlated":
        draw = rng.random()
        if draw < outage_config.clean_probability:
            return (base.player_tokens[left:right], base.player_mask[left:right],
                    base.global_tokens[left:right], False)
        margin = max(1, math.ceil(10 * frames.fps))
        feature_left = max(0, left - margin)
        feature_right = min(len(frames.frame_id), right + margin)
        altered = frames.subset(feature_left, feature_right)
        window = slice(left - feature_left, right - feature_left)
        if draw < outage_config.clean_probability + outage_config.full_probability:
            additional = altered.ball_valid[window].copy()
        else:
            fraction = rng.uniform(outage_config.fraction_min, outage_config.fraction_max)
            additional = generate_outage_mask(
                altered.ball_valid[window], frames.fps, rng,
                missing_fraction=fraction,
                min_seconds=outage_config.min_gap_seconds,
                max_seconds=outage_config.max_gap_seconds,
                long_probability=outage_config.long_gap_probability,
                long_min_seconds=outage_config.long_gap_min_seconds,
                long_max_seconds=outage_config.long_gap_max_seconds,
            )
        if not additional.any():
            return (base.player_tokens[left:right], base.player_mask[left:right],
                    base.global_tokens[left:right], False)
        altered.ball_valid[window] &= ~additional
        altered.ball_xy[window][additional] = 0
        rebuilt = build_features(altered)
        start, stop = left - feature_left, right - feature_left
        return (rebuilt.player_tokens[start:stop], rebuilt.player_mask[start:stop],
                rebuilt.global_tokens[start:stop], True)

    if augment and rng.random() < 0.5:
        # Ball-gap ages use up to ten seconds of context outside the model
        # window; include that margin before recomputing on the altered input.
        margin = max(1, math.ceil(10 * frames.fps))
        feature_left = max(0, left - margin)
        feature_right = min(len(frames.frame_id), right + margin)
        altered = frames.subset(feature_left, feature_right)
        if rng.random() < 0.1:
            outage_start, outage_stop = left, right
        else:
            length = min(center_stop - center_start, int(rng.integers(4, 26)))
            first = int(rng.integers(center_start, center_stop - length + 1))
            outage_start, outage_stop = first, first + length
        altered.ball_valid[outage_start - feature_left : outage_stop - feature_left] = False
        altered.ball_xy[outage_start - feature_left : outage_stop - feature_left] = 0
        rebuilt = build_features(altered)
        start, stop = left - feature_left, right - feature_left
        player = rebuilt.player_tokens[start:stop]
        mask = rebuilt.player_mask[start:stop]
        context = rebuilt.global_tokens[start:stop]
        return player, mask, context, True
    return (base.player_tokens[left:right], base.player_mask[left:right],
            base.global_tokens[left:right], False)


def _window(
    frames: FrameTable,
    targets: TrainingTargets,
    base: FeatureBatch,
    start: int,
    stop: int,
    halo: int,
    rng: np.random.Generator,
    augment: bool,
    device: torch.device,
    outage_config: OutageConfig | None = None,
    augmentation_config: AugmentationConfig | None = None,
    augmentation_step: int = 0,
    tracking_rng: np.random.Generator | None = None,
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor], dict]:
    left = max(0, start - halo)
    right = min(len(frames.frame_id), stop + halo)
    tracking_report = {"applied": [], "selected": [], "strength": 0.0}
    if augment and augmentation_config is not None and augmentation_config.mode != "none":
        # Copy a bounded half-local context before geometry/time transforms.
        # Extra context keeps gap ages and backward motion consistent at the
        # model's first frame; labels follow the same temporal resampling map.
        margin = max(1, math.ceil(10 * frames.fps),
                     math.ceil(augmentation_config.temporal_shift_max_seconds * frames.fps) + 1,
                     augmentation_config.blur_radius + 1)
        feature_left, feature_right = max(0, left - margin), min(len(frames.frame_id), right + margin)
        frames, targets, tracking_report = augment_tracking(
            frames.subset(feature_left, feature_right), targets.subset(feature_left, feature_right),
            tracking_rng if tracking_rng is not None else rng,
            augmentation_config, step=augmentation_step,
        )
        left, right = left - feature_left, right - feature_left
        start, stop = start - feature_left, stop - feature_left
        base = build_features(frames)
    player, mask, context, outage = _selected_features(
        frames, base, left, right, start, stop, rng, augment, outage_config
    )
    swapped = bool(augment and rng.random() < 0.5)
    if swapped:
        player = player[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
    inputs = (
        torch.from_numpy(np.ascontiguousarray(player))[None].to(device),
        torch.from_numpy(np.ascontiguousarray(mask))[None].to(device),
        torch.from_numpy(np.ascontiguousarray(context))[None].to(device),
        torch.from_numpy(base.time_valid[left:right].copy())[None].to(device),
    )
    loss_targets = _loss_targets_for_window(
        targets, base.time_valid, frames.frame_id, left=left, right=right,
        center_start=start, center_stop=stop,
    )
    if swapped:
        owner = loss_targets["owner"]
        loss_targets["owner"] = torch.where(loss_targets["owner_valid"], 1 - owner, owner)
        states = loss_targets["state"]
        loss_targets["state"] = torch.where((states == 0) | (states == 1), 1 - states, states)
    choices = {
        "team_swap": swapped, "ball_outage": outage,
    }
    if outage_config is not None:
        center_ball_valid = context[start - left:stop - left, 2]
        choices["center_ball_missing_frames"] = int(np.count_nonzero(center_ball_valid == 0))
        choices["center_ball_frames"] = stop - start
    if augmentation_config is not None:
        choices["tracking_augmentation"] = tracking_report
        center_players = mask[start - left:stop - left]
        choices["center_player_observations"] = int(np.count_nonzero(center_players))
        choices["center_player_slots"] = int(center_players.size)
    return inputs, {key: value.to(device) for key, value in loss_targets.items()}, choices


def _prepared_paths(data_dir: Path, matches: list[str]) -> list[Path]:
    paths: list[Path] = []
    found: set[str] = set()
    requested = set(matches)
    for path in sorted(data_dir.glob("*_half*.npz")):
        parsed = re.fullmatch(r"(.+)_half([12])\.npz", path.name)
        if parsed and parsed.group(1) in requested:
            paths.append(path)
            found.add(parsed.group(1))
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"missing prepared matches: {missing}")
    return paths


def _canonical_matches(data_dir: Path, requested: list[str]) -> list[str]:
    available = set()
    for path in data_dir.glob("*_half*.npz"):
        parsed = re.fullmatch(r"(.+)_half([12])\.npz", path.name)
        if parsed:
            available.add(parsed.group(1))
    canonical = []
    for name in requested:
        matches = [name] if name in available else sorted(
            candidate for candidate in available if candidate.endswith("-" + name)
        )
        if not matches:
            raise ValueError(f"missing prepared matches: {name}")
        if len(matches) > 1:
            raise ValueError(f"ambiguous prepared match ID {name}: {matches}")
        if matches[0] not in canonical:
            canonical.append(matches[0])
    return canonical


def _read_half(path: Path) -> tuple[FrameTable, TrainingTargets, FeatureBatch, np.ndarray]:
    frames, targets = load_half(path)
    parsed = re.fullmatch(r"(.+)_half([12])\.npz", path.name)
    if not parsed or (frames.match_id, frames.period) != (parsed.group(1), int(parsed.group(2))):
        raise ValueError(f"prepared cache metadata does not match {path.name}")
    if len(frames.frame_id) < 1:
        raise ValueError(f"empty prepared half: {path.name}")
    base = build_features(frames)
    transitions = np.flatnonzero(
        targets.state_valid[1:] & targets.state_valid[:-1]
        & (targets.state[1:] != targets.state[:-1])
    ) + 1
    return frames, targets, base, transitions


def _accumulate(
    sums: dict[str, float], counts: dict[str, int], result: dict[str, torch.Tensor | int]
) -> None:
    for head in ("live", "owner", "boundary"):
        count = int(result[f"{head}_count"])
        sums[head] += float(result[head].detach().item()) * count
        counts[head] += count


def _mean_losses(sums: dict[str, float], counts: dict[str, int], boundary_weight: float) -> dict[str, float]:
    means = {head: sums[head] / counts[head] if counts[head] else 0.0
             for head in ("live", "owner", "boundary")}
    means["total"] = means["live"] + means["owner"] + boundary_weight * means["boundary"]
    return means


def train_model(
    data_dir,
    output_dir,
    *,
    train_matches,
    validation_matches=(),
    steps=100,
    seed=0,
    device="cpu",
    width=64,
    dilations=(1, 2, 4, 8, 16, 32, 64, 128),
    center_size=512,
    halo=256,
    learning_rate=1e-3,
    weight_decay=1e-4,
    boundary_weight=0.25,
    boundary_pos_weight=1.0,
    pooling="mean",
    augment=True,
    outage_config=None,
    augmentation_config=None,
) -> dict:
    """Train on declared source matches, then report optional holdout loss.

    Prepared halves use the session's bounded cache. Windows have their true
    temporal halo, with loss restricted to the kept center.
    Validation is a read-only final-checkpoint loss; it selects no weights.
    """

    train_matches = list(train_matches)
    validation_matches = list(validation_matches)
    if not train_matches:
        raise ValueError("at least one train match is required")
    if steps < 1 or center_size < 1 or learning_rate <= 0:
        raise ValueError("steps, center_size, and learning_rate must be positive")
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    train_matches = _canonical_matches(data_dir, train_matches)
    validation_matches = _canonical_matches(data_dir, validation_matches)
    if set(train_matches) & set(validation_matches):
        raise ValueError("train and validation matches must be disjoint")
    train_paths = _prepared_paths(data_dir, train_matches)
    validation_paths = _prepared_paths(data_dir, validation_matches)
    verify_prepared(data_dir, train_paths + validation_paths)
    if steps < len(train_paths):
        raise ValueError("steps must cover every prepared train half")
    model_kwargs = {"player_dim": 8, "global_dim": 9, "width": int(width),
                    "dilations": list(dilations)}
    if pooling != "mean":
        model_kwargs["pooling"] = pooling
    # The session imports window helpers from this module, so import it only
    # after this module's definitions have been initialized.
    from possession.training_session import TrainingSession

    block_size = min(4, max(1, steps // len(train_paths)))
    session = TrainingSession(
        data_dir, train_matches=train_matches, seed=seed, device=device,
        model_kwargs=model_kwargs, learning_rate=learning_rate,
        weight_decay=weight_decay, boundary_weight=boundary_weight,
        boundary_pos_weight=boundary_pos_weight,
        augment=augment, outage_config=outage_config,
        augmentation_config=normalize_augmentation(augmentation_config), center_size=center_size,
        halo=halo, schedule_block_size=block_size,
    )
    started = time.perf_counter()
    progress = session.advance_to(steps)
    elapsed = time.perf_counter() - started

    validation_sums = {head: 0.0 for head in ("live", "owner", "boundary")}
    validation_counts = {head: 0 for head in validation_sums}
    validation_possible = {head: 0 for head in validation_sums}
    validation_rng = np.random.default_rng(0)
    session.model.eval()
    with torch.inference_mode():
        for path in validation_paths:
            frames, targets, base, _ = session.cache.get(path)
            for start in range(0, len(frames.frame_id), center_size):
                stop = min(start + center_size, len(frames.frame_id))
                inputs, loss_targets, _ = _window(
                    frames, targets, base, start, stop, session.halo,
                    validation_rng, False, session.device,
                )
                result = masked_loss(session.model(*inputs), loss_targets,
                                     boundary_weight, boundary_pos_weight)
                _accumulate(validation_sums, validation_counts, result)
                validation_possible["live"] += stop - start
                validation_possible["owner"] += stop - start
                validation_possible["boundary"] += stop - start - int(start == 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = session.save_checkpoint(output_dir / "last.pt",
                                              validation_matches=validation_matches)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    training_config = checkpoint["training_config"]
    train_counts = progress["train_counts"]
    train_possible = progress["train_possible"]
    train_losses = progress["train_losses"]
    val_losses = _mean_losses(validation_sums, validation_counts, boundary_weight)
    loss_trace = progress["sampled_training_loss_trace"]
    if loss_trace[-1]["step"] != steps:
        loss_trace.append(progress["last_step_losses"])
    report = {
        **training_config,
        "runtime": checkpoint["runtime"],
        "dataset_provenance": checkpoint["dataset_provenance"],
        "model_kwargs": checkpoint["model_kwargs"],
        "prepared_sha256": checkpoint["prepared_sha256"],
        "prepared_inputs_hash": checkpoint["prepared_inputs_hash"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_counts": train_counts,
        "train_excluded": {head: train_possible[head] - train_counts[head]
                           for head in train_counts},
        "trained_halves": progress["trained_halves"],
        "training_schedule_blocks": progress["training_schedule_blocks"],
        "training_window_sha256": progress["training_window_sha256"],
        "train_loss": train_losses["total"],
        "train_losses": train_losses,
        "validation_counts": validation_counts,
        "validation_excluded": {head: validation_possible[head] - validation_counts[head]
                                for head in validation_counts},
        "validation_loss": val_losses["total"] if validation_paths else None,
        "validation_losses": val_losses if validation_paths else None,
        "augmentation_counts": progress["augmentation_counts"],
        "tracking_augmentation": progress["tracking_augmentation"],
        "center_player_observed_fraction": progress["center_player_observed_fraction"],
        "boundary_positive_count": progress["boundary_positive_count"],
        "boundary_negative_count": progress["boundary_negative_count"],
        "center_ball_missing_frames": progress["center_ball_missing_frames"],
        "center_ball_frames": progress["center_ball_frames"],
        "center_ball_missing_fraction": progress["center_ball_missing_fraction"],
        "sampled_training_loss_trace": loss_trace,
        "elapsed_train_seconds": elapsed,
        "throughput_center_frames_per_second": train_possible["live"] / elapsed,
    }
    write_json(output_dir / "training-report.json", report)
    return report
