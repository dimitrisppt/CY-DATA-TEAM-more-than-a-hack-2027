"""Head-specific masked supervision for partial source labels."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


def _binary_loss(logits: Tensor, values: Tensor, valid: Tensor) -> tuple[Tensor, int]:
    selected = logits[valid]
    count = selected.numel()
    if count == 0:
        return selected.sum() * 0.0, 0
    return F.binary_cross_entropy_with_logits(selected, values[valid].to(selected.dtype)), count


def masked_loss(
    scores: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    boundary_weight: float = 0.25,
    boundary_pos_weight: float = 1.0,
) -> dict[str, Tensor | int]:
    """Return masked BCE components and their supervised-frame denominators.

    `owner` is H=1/A=0 and contributes only where known live play is valid.
    `state` is H=0/A=1/D=2, with -1 for unknown. Boundary at t represents
    state[t-1] -> state[t]; all six off-diagonal logits receive BCE at each
    eligible pair. Optional `frame_id` rejects discontinuities, and optional
    `time_valid` excludes padded time positions from every head.
    """

    if not math.isfinite(boundary_weight) or boundary_weight < 0:
        raise ValueError("boundary_weight must be nonnegative")
    if not math.isfinite(boundary_pos_weight) or boundary_pos_weight <= 0:
        raise ValueError("boundary_pos_weight must be finite and positive")
    live_logits = scores["live_logits"]
    owner_logits = scores["owner_logits"]
    boundaries = scores["boundary"]
    if live_logits.shape != owner_logits.shape or boundaries.shape != (*live_logits.shape, 3, 3):
        raise ValueError("score shape mismatch")
    batch, frames = live_logits.shape
    time_valid = targets.get("time_valid")
    if time_valid is None:
        time_valid = torch.ones(batch, frames, dtype=torch.bool, device=live_logits.device)
    elif time_valid.shape != (batch, frames):
        raise ValueError("time_valid shape mismatch")
    active = time_valid.bool()

    live_valid = targets["live_valid"].bool() & active
    live_loss, live_count = _binary_loss(live_logits, targets["live"], live_valid)

    # A source owner field without a valid live status cannot supervise the
    # conditional owner head. Dead frames likewise have no conditional owner.
    owner_valid = targets["owner_valid"].bool() & live_valid & targets["live"].bool()
    owner_loss, owner_count = _binary_loss(owner_logits, targets["owner"], owner_valid)

    states = targets["state"]
    state_valid = targets["state_valid"].bool() & active & states.ge(0) & states.le(2)
    pair_valid = state_valid[:, :-1] & state_valid[:, 1:]
    frame_id = targets.get("frame_id")
    if frame_id is not None:
        if frame_id.shape != (batch, frames):
            raise ValueError("frame_id shape mismatch")
        pair_valid = pair_valid & (frame_id[:, 1:] - frame_id[:, :-1] == 1)

    from_state = torch.arange(3, device=states.device).view(1, 1, 3, 1)
    to_state = torch.arange(3, device=states.device).view(1, 1, 1, 3)
    truth = (states[:, :-1, None, None] == from_state) & (
        states[:, 1:, None, None] == to_state
    )
    off_diagonal = ~torch.eye(3, dtype=torch.bool, device=boundaries.device)
    boundary_logits = boundaries[:, 1:, :, :][..., off_diagonal]
    selected_logits = boundary_logits[pair_valid]
    boundary_count = int(pair_valid.sum().item())
    if boundary_count:
        selected_truth = truth[..., off_diagonal][pair_valid].to(selected_logits.dtype)
        positive_count = int(selected_truth.sum().item())
        boundary_loss = F.binary_cross_entropy_with_logits(
            selected_logits, selected_truth,
            pos_weight=selected_logits.new_tensor(boundary_pos_weight),
        )
    else:
        positive_count = 0
        boundary_loss = selected_logits.sum() * 0.0

    return {
        "total": live_loss + owner_loss + boundary_weight * boundary_loss,
        "live": live_loss,
        "owner": owner_loss,
        "boundary": boundary_loss,
        "live_count": live_count,
        "owner_count": owner_count,
        "boundary_count": boundary_count,
        "boundary_positive_count": positive_count,
        "boundary_negative_count": boundary_count * 6 - positive_count,
    }
