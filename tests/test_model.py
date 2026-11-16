"""Observable symmetry and temporal-context contracts for the learned encoder."""

import pytest
import torch

from possession.model import PossessionModel, predict_chunked


def inputs(batch=2, frames=13, players=3):
    generator = torch.Generator().manual_seed(31)
    tokens = torch.randn(batch, frames, 2, players, 8, generator=generator)
    mask = torch.ones(batch, frames, 2, players, dtype=torch.bool)
    global_tokens = torch.randn(batch, frames, 9, generator=generator)
    global_tokens[..., 2] = 1  # Exercise visible-ball attention by default.
    return tokens, mask, global_tokens


def small_model(pooling="mean"):
    torch.manual_seed(19)
    return PossessionModel(width=16, dilations=(1, 2, 4), pooling=pooling).eval()


@pytest.mark.parametrize("pooling", ["mean", "ball_attention"])
def test_player_permutation_and_invalid_padding_do_not_change_scores(pooling):
    model = small_model(pooling)
    tokens, mask, global_tokens = inputs()
    mask[..., 2] = False
    reference = model(tokens, mask, global_tokens)
    permutation = torch.tensor([2, 0, 1])
    permuted = model(tokens[..., permutation, :], mask[..., permutation], global_tokens)
    padded_tokens = torch.cat((tokens, torch.randn(*tokens.shape[:-2], 2, 8) * 100), dim=-2)
    padded_mask = torch.cat((mask, torch.zeros(*mask.shape[:-1], 2, dtype=torch.bool)), dim=-1)
    padded = model(padded_tokens, padded_mask, global_tokens)
    for key in reference:
        torch.testing.assert_close(permuted[key], reference[key], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(padded[key], reference[key], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("pooling", ["mean", "ball_attention"])
def test_team_swap_transforms_every_directed_boundary_score(pooling):
    model = small_model(pooling)
    tokens, mask, global_tokens = inputs()
    original = model(tokens, mask, global_tokens)
    swapped = model(tokens.flip(2), mask.flip(2), global_tokens)
    torch.testing.assert_close(swapped["live_logits"], original["live_logits"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(swapped["owner_logits"], -original["owner_logits"], atol=1e-6, rtol=1e-6)
    order = torch.tensor([1, 0, 2])
    expected = original["boundary"][:, :, order][:, :, :, order]
    torch.testing.assert_close(swapped["boundary"], expected, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(original["boundary"]).all()
    assert torch.all(torch.diagonal(original["boundary"], dim1=-2, dim2=-1) == 0)


@pytest.mark.parametrize("pooling", ["mean", "ball_attention"])
def test_absent_ball_and_players_produce_finite_scores(pooling):
    model = small_model(pooling)
    tokens, mask, global_tokens = inputs(players=1)
    tokens.fill_(float("nan"))
    global_tokens.zero_()
    mask.zero_()
    scores = model(tokens, mask, global_tokens)
    assert scores["live_logits"].shape == (2, 13)
    assert scores["owner_logits"].shape == (2, 13)
    assert scores["boundary"].shape == (2, 13, 3, 3)
    assert all(torch.isfinite(value).all() for value in scores.values())


@pytest.mark.parametrize("pooling", ["mean", "ball_attention"])
def test_chunked_predictions_match_full_sequence_at_seams_and_edges(pooling):
    model = small_model(pooling)
    tokens, mask, global_tokens = inputs(batch=1, frames=39, players=2)
    time_valid = torch.ones(1, 39, dtype=torch.bool)
    full = model(tokens, mask, global_tokens, time_valid)
    chunked = predict_chunked(model, tokens, mask, global_tokens, time_valid, center_size=8, halo=7)
    for key in full:
        torch.testing.assert_close(chunked[key], full[key], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="halo"):
        predict_chunked(model, tokens, mask, global_tokens, time_valid, center_size=8, halo=6)


def test_ball_attention_ignores_unobserved_relative_channels():
    model = small_model("ball_attention")
    tokens, mask, global_tokens = inputs()
    global_tokens[..., 2] = 0
    clean = tokens.clone()
    clean[..., 5:8] = 0
    stale = tokens.clone()
    stale[..., 5] = float("nan")
    stale[..., 6] = 1000
    stale[..., 7] = -1000
    expected = model(clean, mask, global_tokens)
    actual = model(stale, mask, global_tokens)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        assert torch.isfinite(actual[key]).all()


def test_ball_attention_handles_empty_team_with_visible_ball():
    model = small_model("ball_attention")
    tokens, mask, global_tokens = inputs(players=2)
    mask[:, :, 0] = False
    tokens[:, :, 0] = float("nan")
    scores = model(tokens, mask, global_tokens)
    assert all(torch.isfinite(value).all() for value in scores.values())


def test_ball_attention_responds_to_visible_relative_evidence():
    model = small_model("ball_attention")
    tokens, mask, global_tokens = inputs()
    changed = tokens.clone()
    changed[..., 5:7] = changed[..., 5:7] + 2
    original_scores = model(tokens, mask, global_tokens)
    changed_scores = model(changed, mask, global_tokens)
    assert not torch.allclose(changed_scores["owner_logits"], original_scores["owner_logits"])


def test_default_mean_pooling_loads_legacy_state_dict():
    legacy = small_model()
    state = legacy.state_dict()
    assert state["input_projection.weight"].shape == (16, 2 * 16 + 9)
    assert not any("attention" in name for name in state)
    restored = PossessionModel(width=16, dilations=(1, 2, 4)).eval()
    restored.load_state_dict(state, strict=True)
    args = inputs()
    expected = legacy(*args)
    actual = restored(*args)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def test_full_receptive_field_and_half_calls_are_isolated():
    model = PossessionModel()
    assert model.receptive_field == 511
    assert model.radius == 255
    short = small_model()
    first = inputs(batch=1, frames=11, players=2)
    second = inputs(batch=1, frames=11, players=2)
    a = short(*first)
    changed_second = tuple(item * 17 if item.dtype.is_floating_point else item for item in second)
    _ = short(*changed_second)
    again = short(*first)
    for key in a:
        torch.testing.assert_close(a[key], again[key])
