"""Loss masks must follow source-label availability, not inferred labels."""

import pytest
import torch

from possession.losses import masked_loss
from possession.model import PossessionModel


def leaf_scores(frames):
    return {
        "live_logits": torch.zeros(1, frames, requires_grad=True),
        "owner_logits": torch.zeros(1, frames, requires_grad=True),
        "boundary": torch.zeros(1, frames, 3, 3, requires_grad=True),
    }


def test_partial_labels_only_update_their_valid_heads():
    scores = leaf_scores(3)
    targets = {
        "live": torch.tensor([[1, 0, 1]]),
        "live_valid": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        "owner": torch.tensor([[0, 1, 1]]),
        "owner_valid": torch.tensor([[0, 1, 1]], dtype=torch.bool),
        "state": torch.tensor([[-1, 2, -1]]),
        "state_valid": torch.tensor([[0, 1, 0]], dtype=torch.bool),
    }
    result = masked_loss(scores, targets)
    assert (result["live_count"], result["owner_count"], result["boundary_count"]) == (2, 0, 0)
    result["total"].backward()
    assert torch.count_nonzero(scores["live_logits"].grad) == 2
    assert torch.count_nonzero(scores["owner_logits"].grad) == 0
    assert torch.count_nonzero(scores["boundary"].grad) == 0
    assert torch.isfinite(result["total"])


def test_directed_boundary_uses_previous_to_current_state_at_current_frame():
    scores = leaf_scores(2)
    targets = {
        "live": torch.zeros(1, 2),
        "live_valid": torch.zeros(1, 2, dtype=torch.bool),
        "owner": torch.zeros(1, 2),
        "owner_valid": torch.zeros(1, 2, dtype=torch.bool),
        "state": torch.tensor([[0, 1]]),
        "state_valid": torch.ones(1, 2, dtype=torch.bool),
        "frame_id": torch.tensor([[40, 41]]),
    }
    with torch.no_grad():
        scores["boundary"][:, 1].fill_(-7)
        scores["boundary"][:, 1, 0, 1] = 7
    correct = masked_loss(scores, targets, boundary_weight=1)
    assert correct["boundary_count"] == 1
    with torch.no_grad():
        scores["boundary"][:, 1, 0, 1] = -7
        scores["boundary"][:, 1, 1, 0] = 7
    reversed_direction = masked_loss(scores, targets, boundary_weight=1)
    assert correct["boundary"] < reversed_direction["boundary"]
    reversed_direction["total"].backward()
    assert torch.count_nonzero(scores["boundary"].grad[:, 0]) == 0
    assert scores["boundary"].grad[0, 1, 0, 1] < 0
    assert scores["boundary"].grad[0, 1, 1, 0] > 0


def test_boundary_excludes_frame_gap_and_time_padding():
    scores = leaf_scores(4)
    targets = {
        "live": torch.ones(1, 4),
        "live_valid": torch.ones(1, 4, dtype=torch.bool),
        "owner": torch.ones(1, 4),
        "owner_valid": torch.ones(1, 4, dtype=torch.bool),
        "state": torch.tensor([[0, 1, 2, 0]]),
        "state_valid": torch.ones(1, 4, dtype=torch.bool),
        "frame_id": torch.tensor([[20, 21, 23, 24]]),
        "time_valid": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
    }
    result = masked_loss(scores, targets)
    assert (result["live_count"], result["owner_count"], result["boundary_count"]) == (3, 3, 1)
    result["total"].backward()
    assert torch.count_nonzero(scores["boundary"].grad[:, 1]) == 6
    assert torch.count_nonzero(scores["boundary"].grad[:, 2:]) == 0


def test_all_invalid_labels_have_finite_zero_loss():
    scores = leaf_scores(2)
    targets = {
        "live": torch.zeros(1, 2),
        "live_valid": torch.zeros(1, 2, dtype=torch.bool),
        "owner": torch.zeros(1, 2),
        "owner_valid": torch.zeros(1, 2, dtype=torch.bool),
        "state": torch.full((1, 2), -1),
        "state_valid": torch.zeros(1, 2, dtype=torch.bool),
    }
    result = masked_loss(scores, targets)
    assert result["total"].item() == 0
    result["total"].backward()
    assert all(torch.count_nonzero(value.grad) == 0 for value in scores.values())


def test_team_swap_preserves_total_loss():
    torch.manual_seed(7)
    scores = {
        "live_logits": torch.randn(1, 4),
        "owner_logits": torch.randn(1, 4),
        "boundary": torch.randn(1, 4, 3, 3),
    }
    targets = {
        "live": torch.tensor([[1, 1, 0, 1]]),
        "live_valid": torch.ones(1, 4, dtype=torch.bool),
        "owner": torch.tensor([[1, 0, 0, 1]]),
        "owner_valid": torch.tensor([[1, 1, 0, 1]], dtype=torch.bool),
        "state": torch.tensor([[0, 1, 2, 0]]),
        "state_valid": torch.ones(1, 4, dtype=torch.bool),
    }
    reference = masked_loss(scores, targets)
    order = torch.tensor([1, 0, 2])
    swapped_scores = {
        "live_logits": scores["live_logits"],
        "owner_logits": -scores["owner_logits"],
        "boundary": scores["boundary"][:, :, order][:, :, :, order],
    }
    swapped_targets = dict(targets)
    swapped_targets["owner"] = 1 - targets["owner"]
    swapped_targets["state"] = torch.where(targets["state"] == 2, 2, 1 - targets["state"])
    swapped = masked_loss(swapped_scores, swapped_targets)
    torch.testing.assert_close(swapped["total"], reference["total"])


def test_small_cpu_optimization_has_finite_gradients_and_lower_loss():
    torch.manual_seed(11)
    model = PossessionModel(width=8, dilations=(1,))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    tokens = torch.randn(1, 6, 2, 2, 8)
    mask = torch.ones(1, 6, 2, 2, dtype=torch.bool)
    context = torch.randn(1, 6, 9)
    targets = {
        "live": torch.tensor([[0, 1, 1, 1, 0, 1]]),
        "live_valid": torch.ones(1, 6, dtype=torch.bool),
        "owner": torch.tensor([[0, 1, 0, 1, 0, 0]]),
        "owner_valid": torch.tensor([[0, 1, 1, 1, 0, 1]], dtype=torch.bool),
        "state": torch.tensor([[2, 0, 1, 0, 2, 1]]),
        "state_valid": torch.ones(1, 6, dtype=torch.bool),
    }
    initial = masked_loss(model(tokens, mask, context), targets)["total"].item()
    for _ in range(12):
        optimizer.zero_grad()
        loss = masked_loss(model(tokens, mask, context), targets)["total"]
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters())
        optimizer.step()
    final = masked_loss(model(tokens, mask, context), targets)["total"].item()
    assert final < initial


def test_boundary_positive_weight_changes_only_valid_positive_pairs():
    scores = leaf_scores(3)
    targets = {
        "live": torch.zeros(1, 3),
        "live_valid": torch.zeros(1, 3, dtype=torch.bool),
        "owner": torch.zeros(1, 3),
        "owner_valid": torch.zeros(1, 3, dtype=torch.bool),
        "state": torch.tensor([[0, 1, -1]]),
        "state_valid": torch.tensor([[1, 1, 0]], dtype=torch.bool),
    }
    base = masked_loss(scores, targets, boundary_weight=1)
    weighted = masked_loss(scores, targets, boundary_weight=1, boundary_pos_weight=7)
    assert weighted["boundary_positive_count"] == 1
    assert weighted["boundary_negative_count"] == 5
    assert weighted["boundary_count"] == 1
    torch.testing.assert_close(weighted["boundary"], base["boundary"] * 2)
    weighted["total"].backward()
    assert torch.isfinite(scores["boundary"].grad).all()


@pytest.mark.parametrize("weight", [0, -1, float("inf"), float("nan")])
def test_boundary_positive_weight_must_be_finite_and_positive(weight):
    scores = leaf_scores(2)
    targets = {
        "live": torch.zeros(1, 2), "live_valid": torch.zeros(1, 2, dtype=torch.bool),
        "owner": torch.zeros(1, 2), "owner_valid": torch.zeros(1, 2, dtype=torch.bool),
        "state": torch.tensor([[0, 1]]), "state_valid": torch.ones(1, 2, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="boundary_pos_weight"):
        masked_loss(scores, targets, boundary_pos_weight=weight)
