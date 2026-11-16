"""Auxiliary objectives must respect source censoring and the kept center."""

import math

import numpy as np
import pytest
import torch

from experiments.possession_objectives import (
    CompleteSpell,
    complete_spells,
    masked_teacher_loss,
    per_spell_loss,
    short_spells,
    spell_center_start_range,
)
from possession.schema import TrainingTargets
from possession.train import _loss_targets_for_window


def _targets(state):
    state = torch.tensor([state])
    return {
        "state": state,
        "state_valid": state.ge(0),
        "live": state.ne(2).long(),
        "live_valid": state.ge(0),
        "owner": state.eq(0).long(),
        "owner_valid": state.ge(0) & state.ne(2),
        "time_valid": torch.ones_like(state, dtype=torch.bool),
        "frame_id": torch.arange(state.shape[1])[None],
    }


def _scores(n, live=0., owner=0.):
    return {
        "live_logits": torch.full((1, n), float(live), requires_grad=True),
        "owner_logits": torch.full((1, n), float(owner), requires_grad=True),
    }


def _inputs(visible):
    context = torch.zeros(1, len(visible), 9)
    context[0, :, 2] = torch.tensor(visible)
    return (torch.empty(0), torch.empty(0), context,
            torch.ones(1, len(visible), dtype=torch.bool))


def test_complete_spells_require_observable_boundaries_and_reject_edges():
    targets = _targets([0, 0, 1, 1, 2, 0, -1, 1, 0, 2])
    args = [targets[key][0] for key in ("state", "state_valid", "time_valid", "frame_id")]
    assert complete_spells(*args) == [CompleteSpell(2, 4, 1), CompleteSpell(4, 5, 2),
                                     CompleteSpell(8, 9, 0)]
    assert complete_spells(*args, center_start=2, center_stop=6) == [CompleteSpell(4, 5, 2)]


@pytest.mark.parametrize("gap", ["state_valid", "time_valid", "frame_id"])
def test_gap_censors_both_neighboring_spells(gap):
    targets = _targets([2, 0, 0, 1, 1, 2])
    if gap == "frame_id":
        targets[gap][0, 3:] += 4
    else:
        targets[gap][0, 3] = False
    args = [targets[key][0] for key in ("state", "state_valid", "time_valid", "frame_id")]
    assert complete_spells(*args) == []


def test_short_spells_use_strict_two_seconds_and_fit_both_neighbors_in_center():
    state = np.array([2] + [0] * 49 + [1] * 50 + [2])
    valid = np.ones(len(state), bool)
    spells = short_spells(state, valid, valid, np.arange(len(state)))
    assert spells == [CompleteSpell(1, 50, 0)]
    assert spells[0].seconds == 49 / 25
    assert list(spell_center_start_range(CompleteSpell(10, 15, 1), 10, 30)) == [6, 7, 8, 9]
    assert list(spell_center_start_range(CompleteSpell(10, 15, 1), 6, 30)) == []
    assert list(spell_center_start_range(CompleteSpell(1, 50, 0), 51, 101)) == [0]


def test_per_spell_loss_weights_events_equally_and_excludes_dead_spells():
    targets = _targets([2, 0, 1, 1, 1, 2])
    scores = _scores(6)
    with torch.no_grad():
        scores["live_logits"][0, 1] = -2
        scores["live_logits"][0, 2:5] = 2
    result = per_spell_loss(scores, targets)
    assert (result["spell_count"], result["live_count"], result["owner_count"]) == (2, 4, 4)
    torch.testing.assert_close(result["live"], torch.tensor((math.log1p(math.exp(2))
                                                           + math.log1p(math.exp(-2))) / 2))
    result["total"].backward()
    torch.testing.assert_close(scores["owner_logits"].grad,
                               torch.tensor([[0, -.25, 1 / 12, 1 / 12, 1 / 12, 0]]))
    assert torch.count_nonzero(scores["live_logits"].grad[:, [0, 5]]) == 0


def test_window_predecessor_cannot_make_clipped_spell_complete():
    arrays = np.array([2, 0, 0, 1, 1, 2, 0, 2], dtype=np.int8)
    native = TrainingTargets(arrays != 2, np.ones(8, bool), arrays == 0,
                             arrays != 2, arrays, np.ones(8, bool))
    targets = _loss_targets_for_window(native, np.ones(8, bool), np.arange(8),
                                       left=0, right=8, center_start=2, center_stop=7)
    result = per_spell_loss(_scores(8), targets)
    assert result["spell_count"] == 1
    assert result["live_count"] == 2


def test_missing_owner_label_censors_spell():
    targets = _targets([2, 0, 0, 1, 2])
    targets["owner_valid"][0, 2] = False
    result = per_spell_loss(_scores(5), targets)
    assert result["spell_count"] == 1
    assert result["live_count"] == 1


def test_no_complete_spells_returns_differentiable_zero():
    scores = _scores(3)
    result = per_spell_loss(scores, _targets([0, 0, 0]))
    assert result["total"].item() == 0
    result["total"].backward()
    assert all(value.grad is not None and not value.grad.any() for value in scores.values())


@pytest.mark.parametrize("gap", ["state", "state_valid", "time_valid", "frame_id"])
def test_per_spell_loss_does_not_turn_gaps_into_event_boundaries(gap):
    targets = _targets([2, 0, 0, 1, 1, 2])
    if gap == "frame_id":
        targets[gap][0, 3:] += 2
    elif gap == "state":
        targets[gap][0, 3] = -1
    else:
        targets[gap][0, 3] = False
    scores = _scores(6)
    result = per_spell_loss(scores, targets)
    assert result["spell_count"] == 0
    result["total"].backward()
    assert all(not value.grad.any() for value in scores.values())


def test_event_average_also_weights_events_equally_across_batch_rows():
    targets = _targets([2, 0, 1, 1, 1, 2])
    targets = {name: value.repeat(2, 1) for name, value in targets.items()}
    targets["state_valid"][1, 3] = False
    scores = {name: torch.zeros(2, 6, requires_grad=True)
              for name in ("live_logits", "owner_logits")}
    result = per_spell_loss(scores, targets)
    assert result["spell_count"] == 3
    result["total"].backward()
    # Each complete event gets mass 1/3, even when one row has fewer events.
    torch.testing.assert_close(scores["live_logits"].grad[0, 1], torch.tensor(-1 / 6))
    torch.testing.assert_close(scores["live_logits"].grad[1, 1], torch.tensor(-1 / 6))


def test_teacher_only_updates_newly_hidden_frames_and_has_no_gradients():
    targets = _targets([0, 0, 0, 0, 0])
    student, teacher = _scores(5), _scores(5, 4, 4)
    result = masked_teacher_loss(student, teacher, targets,
                                 _inputs([0, 1, 0, 0, 0]), _inputs([1, 1, 0, 1, 1]))
    assert (result["newly_hidden_count"], result["live_count"], result["owner_count"]) == (3, 3, 3)
    result["total"].backward()
    for value in student.values():
        assert torch.count_nonzero(value.grad[:, [0, 3, 4]]) == 3
        assert torch.count_nonzero(value.grad[:, [1, 2]]) == 0
    assert all(value.grad is None for value in teacher.values())


def test_teacher_respects_both_input_time_masks_and_owner_label_validity():
    targets = _targets([0, 0, 1, 1])
    targets["owner_valid"][0, 2] = False
    student_inputs, teacher_inputs = _inputs([0] * 4), _inputs([1] * 4)
    student_inputs[3][0, 0] = False
    teacher_inputs[3][0, 1] = False
    student, teacher = _scores(4), _scores(4, 4, -4)
    result = masked_teacher_loss(student, teacher, targets, student_inputs, teacher_inputs)
    assert result["live_count"] == 2
    assert result["owner_count"] == 1
    result["total"].backward()
    torch.testing.assert_close(student["owner_logits"].grad[:, :3], torch.zeros(1, 3))
    assert student["owner_logits"].grad[0, 3] > 0


def test_teacher_confidence_gold_agreement_and_owner_live_gates():
    targets = _targets([0, 0, 0, 2, 1, 0, 0, 0])
    targets["live_valid"][0, 6] = False
    targets["time_valid"][0, 7] = False
    student, teacher = _scores(8), _scores(8)
    with torch.no_grad():
        teacher["live_logits"].copy_(torch.tensor([[4., 0., -4., -4., 4., 4., 4., 4.]]))
        teacher["owner_logits"].copy_(torch.tensor([[4., 4., 4., 4., 4., 0., 4., 4.]]))
    result = masked_teacher_loss(student, teacher, targets, _inputs([0] * 8), _inputs([1] * 8))
    assert result["live_count"] == 4
    assert result["owner_count"] == 1
    result["total"].backward()
    assert torch.count_nonzero(student["live_logits"].grad) == 4
    torch.testing.assert_close(student["owner_logits"].grad[:, 1:], torch.zeros(1, 7))


def test_teacher_bernoulli_kl_has_temperature_squared_scaling():
    student, teacher = _scores(1, -1, -2), _scores(1, 3, 4)
    temperature = 2.
    result = masked_teacher_loss(student, teacher, _targets([0]),
                                 _inputs([0]), _inputs([1]), temperature=temperature)
    for name in ("live", "owner"):
        p = torch.sigmoid(teacher[f"{name}_logits"].detach() / temperature)
        q = torch.sigmoid(student[f"{name}_logits"] / temperature)
        expected = (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).mean() * 4
        torch.testing.assert_close(result[name], expected)


def test_nearly_identical_teacher_predictions_cannot_produce_negative_kl():
    teacher = _scores(1, 1.4318031072616577, 1.4318031072616577)
    student = _scores(1, 1.4319031238555908, 1.4319031238555908)
    result = masked_teacher_loss(student, teacher, _targets([0]), _inputs([0]), _inputs([1]))
    for name in ("live", "owner", "total"):
        assert 0 <= result[name].item() < 1e-6
    result["total"].backward()
    assert all(torch.isfinite(value.grad).all() and value.grad.abs().max() < 1e-4
               for value in student.values())


def test_empty_teacher_mask_is_finite_deterministic_and_differentiable():
    student, teacher = _scores(2), _scores(2, 4, 4)
    result = masked_teacher_loss(student, teacher, _targets([0, 1]),
                                 _inputs([1, 1]), _inputs([1, 1]))
    assert result["total"].item() == 0
    assert (result["live_count"], result["owner_count"]) == (0, 0)
    result["total"].backward()
    assert all(value.grad is not None and not value.grad.any() for value in student.values())
    assert all(value.grad is None for value in teacher.values())


def test_ignored_nan_logits_and_extreme_selected_logits_stay_finite():
    student, teacher = _scores(2, -1000, -1000), _scores(2, 1000, 1000)
    with torch.no_grad():
        student["live_logits"][0, 1] = float("nan")
        student["owner_logits"][0, 1] = float("nan")
        teacher["live_logits"][0, 1] = float("nan")
        teacher["owner_logits"][0, 1] = float("nan")
    result = masked_teacher_loss(student, teacher, _targets([0, -1]),
                                 _inputs([0, 0]), _inputs([1, 1]))
    assert (result["live_count"], result["owner_count"]) == (1, 1)
    assert torch.isfinite(result["total"])
    result["total"].backward()
    assert all(torch.isfinite(value.grad).all() for value in student.values())
    assert all(value.grad[0, 1] == 0 for value in student.values())


def test_empty_losses_remain_zero_when_every_ignored_logit_is_nan():
    student, teacher = _scores(2, float("nan"), float("nan")), _scores(2)
    result = masked_teacher_loss(student, teacher, _targets([-1, -1]),
                                 _inputs([0, 0]), _inputs([0, 0]))
    events = per_spell_loss(student, _targets([-1, -1]))
    assert result["total"].item() == events["total"].item() == 0
    (result["total"] + events["total"]).backward()
    assert all(torch.isfinite(value.grad).all() and not value.grad.any()
               for value in student.values())


def test_team_swap_preserves_both_auxiliary_objectives():
    torch.manual_seed(31)
    targets = _targets([2, 0, 0, 1, 2])
    student = {name: torch.randn(1, 5) for name in ("live_logits", "owner_logits")}
    teacher = {"live_logits": (targets["live"] * 2 - 1).float() * 4,
               "owner_logits": (targets["owner"] * 2 - 1).float() * 4}
    swapped_targets = dict(targets)
    swapped_targets["state"] = torch.where(targets["state"] == 2, 2, 1 - targets["state"])
    swapped_targets["owner"] = 1 - targets["owner"]
    swapped_student = dict(student, owner_logits=-student["owner_logits"])
    swapped_teacher = dict(teacher, owner_logits=-teacher["owner_logits"])
    torch.testing.assert_close(per_spell_loss(student, targets)["total"],
                               per_spell_loss(swapped_student, swapped_targets)["total"])
    inputs = (_inputs([0] * 5), _inputs([1] * 5))
    reference = masked_teacher_loss(student, teacher, targets, *inputs)
    swapped = masked_teacher_loss(swapped_student, swapped_teacher, swapped_targets, *inputs)
    torch.testing.assert_close(reference["total"], swapped["total"])


@pytest.mark.parametrize("argument,value", [("confidence", .49), ("confidence", 1.1),
                                           ("confidence", float("nan")),
                                           ("temperature", 0), ("temperature", float("inf"))])
def test_teacher_rejects_invalid_hyperparameters(argument, value):
    with pytest.raises(ValueError, match=argument):
        masked_teacher_loss(_scores(1), _scores(1), _targets([0]),
                            _inputs([0]), _inputs([1]), **{argument: value})
