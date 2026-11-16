"""Spatial pooling symmetry, masking, checkpoint, and chunk contracts."""

import pytest
import torch

from possession.interaction import PlayerInteractionPool
from possession.model import PossessionModel, predict_chunked
from possession.predict import load_model_checkpoint


def _model(pooling="interaction"):
    torch.manual_seed(27)
    return PossessionModel(width=32, dilations=(1, 2, 4), pooling=pooling)


def _inputs(*, batch=2, frames=17, players=4):
    generator = torch.Generator().manual_seed(57)
    tokens = torch.randn(batch, frames, 2, players, 8, generator=generator)
    mask = torch.ones(batch, frames, 2, players, dtype=torch.bool)
    global_tokens = torch.randn(batch, frames, 9, generator=generator)
    global_tokens[..., 2] = 1
    time_valid = torch.ones(batch, frames, dtype=torch.bool)
    return tokens, mask, global_tokens, time_valid


def _assert_scores_close(actual, expected, *, atol=1e-6, rtol=1e-6):
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=atol, rtol=rtol)


def test_independent_slot_permutations_and_invalid_padding():
    model = _model().eval()
    tokens, mask, global_tokens, time_valid = _inputs()
    mask[..., 3] = False
    tokens[..., 3, :] = float("nan")
    reference = model(tokens, mask, global_tokens, time_valid)
    shuffled = tokens.clone()
    shuffled_mask = mask.clone()
    shuffled[:, :, 0] = tokens[:, :, 0, [3, 1, 0, 2]]
    shuffled[:, :, 1] = tokens[:, :, 1, [2, 3, 1, 0]]
    shuffled_mask[:, :, 0] = mask[:, :, 0, [3, 1, 0, 2]]
    shuffled_mask[:, :, 1] = mask[:, :, 1, [2, 3, 1, 0]]
    _assert_scores_close(model(shuffled, shuffled_mask, global_tokens, time_valid), reference)

    padded = torch.cat((tokens, torch.full((*tokens.shape[:-2], 2, 8), float("nan"))), dim=3)
    padded_mask = torch.cat((mask, torch.zeros(*mask.shape[:-1], 2, dtype=torch.bool)), dim=3)
    _assert_scores_close(model(padded, padded_mask, global_tokens, time_valid), reference)


def test_team_swap_equivariance_for_all_heads():
    model = _model().eval()
    tokens, mask, global_tokens, time_valid = _inputs()
    original = model(tokens, mask, global_tokens, time_valid)
    swapped = model(tokens.flip(2), mask.flip(2), global_tokens, time_valid)
    torch.testing.assert_close(swapped["live_logits"], original["live_logits"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(swapped["owner_logits"], -original["owner_logits"], atol=1e-6, rtol=1e-6)
    order = torch.tensor([1, 0, 2])
    expected_boundary = original["boundary"][:, :, order][:, :, :, order]
    torch.testing.assert_close(swapped["boundary"], expected_boundary, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("players", [0, 1, 4])
def test_empty_teams_frames_and_time_padding_are_finite(players):
    model = _model().eval()
    tokens, mask, global_tokens, time_valid = _inputs(players=players)
    if players:
        mask[:, 0, 0] = False  # One team missing.
        mask[:, 1] = False  # Entire frame missing.
        tokens[:, 0, 0] = float("nan")
        tokens[:, 1] = float("nan")
        tokens[:, -1] = float("nan")
    time_valid[:, -1] = False
    scores = model(tokens, mask, global_tokens, time_valid)
    assert scores["live_logits"].shape == (2, 17)
    assert scores["boundary"].shape == (2, 17, 3, 3)
    assert all(torch.isfinite(value).all() for value in scores.values())


def test_gradients_reach_player_and_spatial_parameters():
    model = _model()
    tokens, mask, global_tokens, time_valid = _inputs(batch=1, frames=9, players=3)
    tokens.requires_grad_()
    scores = model(tokens, mask, global_tokens, time_valid)
    loss = scores["live_logits"].square().mean() + scores["owner_logits"].square().mean()
    loss = loss + scores["boundary"].square().mean()
    loss.backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    for parameter in model.player_interaction.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_chunked_interaction_matches_full_half_and_retains_radius():
    model = _model().eval()
    tokens, mask, global_tokens, time_valid = _inputs(batch=1, frames=39, players=3)
    mask[:, 18:22] = False
    tokens[:, 18:22] = float("nan")
    full = model(tokens, mask, global_tokens, time_valid)
    chunked = predict_chunked(model, tokens, mask, global_tokens, time_valid,
                              center_size=8, halo=model.radius)
    _assert_scores_close(chunked, full)
    assert model.radius == 7 and model.receptive_field == 15


@pytest.mark.parametrize("pooling", ["mean", "interaction"])
def test_checkpoint_round_trip(pooling, tmp_path):
    model = _model(pooling).eval()
    path = tmp_path / f"{pooling}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "model_kwargs": {"width": 32, "dilations": (1, 2, 4), "pooling": pooling},
        "training_config": {},
    }, path)
    loaded, _ = load_model_checkpoint(path)
    args = _inputs()
    _assert_scores_close(loaded(*args), model(*args), atol=0, rtol=0)
    if pooling == "mean":
        assert not any("interaction" in key for key in model.state_dict())


def test_interaction_requires_two_position_channels():
    with pytest.raises(ValueError, match="coordinates"):
        PossessionModel(player_dim=1, pooling="interaction")


def test_geometry_and_opponents_change_contextual_team_summary():
    torch.manual_seed(91)
    pool = PlayerInteractionPool(32).eval()
    embedded = torch.randn(1, 1, 2, 3, 32)
    coords = torch.randn(1, 1, 2, 3, 2)
    mask = torch.ones(1, 1, 2, 3, dtype=torch.bool)
    original = pool(embedded, coords, mask)
    moved = coords.clone()
    moved[..., 0, 0] += 2
    assert not torch.allclose(pool(embedded, moved, mask), original)
    changed_opponents = embedded.clone()
    changed_opponents[:, :, 1] += 2
    assert not torch.allclose(pool(changed_opponents, coords, mask)[:, :, 0], original[:, :, 0])
