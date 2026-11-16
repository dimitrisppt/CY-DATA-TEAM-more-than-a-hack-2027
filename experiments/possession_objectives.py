"""Independent auxiliary objectives for frozen-schedule possession ablations.

The ordinary framewise objective remains ``possession.losses.masked_loss``.
Event weighting here is an auxiliary surrogate, not the official spellF1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class CompleteSpell:
    """One known H=0/A=1/D=2 spell, indexed by half-open [start, stop).

    Both immediately adjacent frames are known, temporally consecutive and
    have a different state. Indices refer to the supplied array, not frame IDs.
    """

    start: int
    stop: int
    state: int

    @property
    def frames(self) -> int:
        return self.stop - self.start

    @property
    def seconds(self) -> float:
        return self.frames / 25.0


def _vector(value, name: str) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one dimensional")
    return result


def complete_spells(
    state,
    state_valid,
    time_valid,
    frame_id,
    center_start: int = 0,
    center_stop: int | None = None,
) -> list[CompleteSpell]:
    """Enumerate uncensored spells with both boundary neighbors in the center.

    Pass one half or one window at a time, using one dimensional NumPy arrays,
    tensors or sequences. Unknown states, invalid labels, invalid time and
    nonconsecutive frame IDs break observations. No run touching such a break,
    a half/window edge or a center edge is complete. D spells are enumerated
    for auditing, callers can filter ``spell.state in (0, 1)`` for possession.
    """

    values = _vector(state, "state")
    known = _vector(state_valid, "state_valid")
    active = _vector(time_valid, "time_valid")
    ids = _vector(frame_id, "frame_id")
    frames = len(values)
    if any(len(array) != frames for array in (known, active, ids)):
        raise ValueError("spell array lengths differ")
    stop = frames if center_stop is None else center_stop
    if not (0 <= center_start <= stop <= frames):
        raise ValueError("invalid center bounds")
    valid = known.astype(bool) & active.astype(bool) & np.isin(values, [0, 1, 2])
    adjacent = valid[:-1] & valid[1:] & (np.diff(ids) == 1)
    result = []
    start = 0
    while start < frames:
        if not valid[start]:
            start += 1
            continue
        end = start + 1
        while end < frames and adjacent[end - 1] and values[end] == values[start]:
            end += 1
        if (start > center_start and end < stop
                and adjacent[start - 1] and adjacent[end - 1]
                and values[start - 1] != values[start] and values[end] != values[start]):
            result.append(CompleteSpell(start, end, int(values[start])))
        start = end
    return result


def short_spells(
    state,
    state_valid,
    time_valid,
    frame_id,
    center_start: int = 0,
    center_stop: int | None = None,
) -> list[CompleteSpell]:
    """Return complete spells strictly shorter than two seconds at 25 Hz."""

    return [spell for spell in complete_spells(
        state, state_valid, time_valid, frame_id, center_start, center_stop
    ) if spell.frames < 50]


def spell_center_start_range(spell: CompleteSpell, center_size: int, frames: int) -> range:
    """Return all center starts containing a spell and both boundary neighbors.

    Centers have exactly ``center_size`` frames and stay within ``frames``.
    An empty range means the spell and its neighbors cannot fit. Sampling an
    event first, then uniformly from this range, gives equal event sampling.
    """

    if center_size < 1 or frames < 0:
        raise ValueError("center_size must be positive and frames nonnegative")
    if not 0 <= spell.start < spell.stop <= frames:
        raise ValueError("spell bounds exceed frame array")
    first = max(0, spell.stop + 1 - center_size)
    last = min(spell.start - 1, frames - center_size)
    return range(first, last + 1)


def _head_shape(scores: Mapping[str, Tensor]) -> tuple[int, int]:
    shape = scores["live_logits"].shape
    if len(shape) != 2 or scores["owner_logits"].shape != shape:
        raise ValueError("live and owner scores must have matching [B,T] shapes")
    return shape


def _target_time(targets: Mapping[str, Tensor], reference: Tensor) -> Tensor:
    for name in ("live", "live_valid", "owner", "owner_valid"):
        if targets[name].shape != reference.shape:
            raise ValueError(f"{name} shape mismatch")
    active = targets.get("time_valid", torch.ones_like(reference, dtype=torch.bool))
    if active.shape != reference.shape:
        raise ValueError("time_valid shape mismatch")
    return active.bool()


def _zero(logits: Tensor) -> Tensor:
    # Empty selection also stays finite if ignored positions contain NaNs.
    return logits.reshape(-1)[:0].sum()


def per_spell_loss(
    scores: Mapping[str, Tensor], targets: Mapping[str, Tensor]
) -> dict[str, Tensor | int]:
    """Average live and conditional-owner BCE within each complete H/A spell.

    Each spell receives equal weight, regardless of duration or team. Both
    boundary neighbors must be inside the supervised center. ``_window`` masks
    its predecessor as state-valid for boundary supervision, so live_valid
    additionally defines the eligible center here. Owner labels must be valid
    throughout a possession spell. D spells do not enter this auxiliary loss.
    No complete spells yields a differentiable zero on both prediction heads.
    """

    batch, _frames = _head_shape(scores)
    live, owner = scores["live_logits"], scores["owner_logits"]
    active = _target_time(targets, live)
    for name in ("state", "state_valid", "frame_id"):
        if targets[name].shape != live.shape:
            raise ValueError(f"{name} shape mismatch")
    eligible = targets["state_valid"].bool() & targets["live_valid"].bool() & active
    live_events, owner_events = [], []
    count = 0
    for row in range(batch):
        spells = complete_spells(targets["state"][row], eligible[row], active[row],
                                 targets["frame_id"][row])
        for spell in spells:
            if spell.state == 2:
                continue
            span = slice(spell.start, spell.stop)
            if not (targets["owner_valid"][row, span].bool()
                    & targets["live"][row, span].eq(1)).all().item():
                continue
            live_events.append(F.binary_cross_entropy_with_logits(
                live[row, span], targets["live"][row, span].to(live.dtype)))
            owner_events.append(F.binary_cross_entropy_with_logits(
                owner[row, span], targets["owner"][row, span].to(owner.dtype)))
            count += spell.frames
    live_loss = torch.stack(live_events).mean() if live_events else _zero(live)
    owner_loss = torch.stack(owner_events).mean() if owner_events else _zero(owner)
    return {"total": live_loss + owner_loss, "live": live_loss, "owner": owner_loss,
            "spell_count": len(live_events), "live_count": count, "owner_count": count}


def _input_visibility(inputs: Mapping[str, Tensor] | Sequence[Tensor], reference: Tensor):
    if isinstance(inputs, Mapping):
        context = inputs["global_tokens"]
        active = inputs.get("time_valid", torch.ones_like(reference, dtype=torch.bool))
    else:
        context = inputs[2]
        active = inputs[3] if len(inputs) > 3 else torch.ones_like(reference, dtype=torch.bool)
    if context.ndim != 3 or context.shape[:2] != reference.shape or context.shape[2] < 3:
        raise ValueError("global_tokens must have shape [B,T,F] with ball visibility column 2")
    if active.shape != reference.shape:
        raise ValueError("input time_valid shape mismatch")
    visible = context[..., 2].detach()
    return visible, active.detach().bool() & torch.isfinite(visible)


def _bernoulli_kl(student: Tensor, teacher: Tensor, mask: Tensor, temperature: float):
    student = student[mask] / temperature
    teacher = teacher[mask] / temperature
    if not student.numel():
        return student.sum(), 0
    probability = torch.sigmoid(teacher)
    divergence = (probability * (F.logsigmoid(teacher) - F.logsigmoid(student))
                  + (1 - probability) * (F.logsigmoid(-teacher) - F.logsigmoid(-student)))
    # Near equality, cancellation in float32 can put an otherwise nonnegative
    # KL a few ulps below zero. Keep event counters and checkpoint guards valid.
    return divergence.clamp_min(0).mean() * temperature**2, student.numel()


def masked_teacher_loss(
    student_scores: Mapping[str, Tensor],
    teacher_scores: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    student_inputs: Mapping[str, Tensor] | Sequence[Tensor],
    teacher_inputs: Mapping[str, Tensor] | Sequence[Tensor],
    confidence: float = .8,
    temperature: float = 2.,
) -> dict[str, Tensor | int]:
    """Distill a detached clean teacher onto newly hidden student ball frames.

    Views must share geometry, time and team ordering. Ball visibility is
    global_tokens column 2. Existing missing-ball frames never contribute.
    Masks use valid training gold in the center, input/target time validity,
    untempered teacher confidence and teacher agreement with that head's gold.
    Owner additionally requires gold live and a confident live teacher agreeing
    with gold. Selected heads use Bernoulli KL(teacher || student) at T, scaled
    by T squared. No boundary distillation is applied.
    """

    if not math.isfinite(confidence) or not .5 <= confidence <= 1:
        raise ValueError("confidence must be finite and lie in [0.5, 1]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if _head_shape(student_scores) != _head_shape(teacher_scores):
        raise ValueError("student and teacher score shapes differ")
    live, owner = student_scores["live_logits"], student_scores["owner_logits"]
    active = _target_time(targets, live)
    student_visible, student_active = _input_visibility(student_inputs, live)
    teacher_visible, teacher_active = _input_visibility(teacher_inputs, live)
    newly_hidden = (student_visible <= .5) & (teacher_visible > .5)
    newly_hidden &= active & student_active & teacher_active
    teacher_live = teacher_scores["live_logits"].detach()
    teacher_owner = teacher_scores["owner_logits"].detach()
    live_probability, owner_probability = torch.sigmoid(teacher_live), torch.sigmoid(teacher_owner)
    live_confident = torch.maximum(live_probability, 1 - live_probability) >= confidence
    owner_confident = torch.maximum(owner_probability, 1 - owner_probability) >= confidence
    live_agrees = live_probability.ge(.5) == targets["live"].bool()
    owner_agrees = owner_probability.ge(.5) == targets["owner"].bool()
    live_mask = (newly_hidden & targets["live_valid"].bool() & live_confident & live_agrees
                 & torch.isfinite(teacher_live) & torch.isfinite(live))
    owner_mask = (live_mask & targets["live"].eq(1) & targets["owner_valid"].bool()
                  & owner_confident & owner_agrees
                  & torch.isfinite(teacher_owner) & torch.isfinite(owner))
    live_loss, live_count = _bernoulli_kl(live, teacher_live, live_mask, temperature)
    owner_loss, owner_count = _bernoulli_kl(owner, teacher_owner, owner_mask, temperature)
    return {"total": live_loss + owner_loss, "live": live_loss, "owner": owner_loss,
            "newly_hidden_count": int(newly_hidden.sum().item()),
            "live_count": live_count, "owner_count": owner_count}
