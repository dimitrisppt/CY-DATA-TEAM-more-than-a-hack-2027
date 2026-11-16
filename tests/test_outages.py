"""Correlated input outages preserve source observations and exact coverage."""

import numpy as np
import pytest

from possession.features import build_features
from possession.outages import OutageConfig, gap_statistics, generate_outage_mask
from possession.schema import FrameTable
from possession.train import _selected_features


def _frames(n):
    players = np.zeros((n, 2, 2, 2), dtype=np.float32)
    players[:, 0, :, 0] = 20
    players[:, 1, :, 0] = 80
    ball = np.column_stack((25 + np.arange(n) * 0.1, np.full(n, 30)))
    return FrameTable(np.arange(n), players, np.ones((n, 2, 2), dtype=bool),
                      ball, np.ones(n, dtype=bool), np.ones(n, dtype=bool),
                      match_id="a", period=1)


def test_config_is_frozen_serializable_and_validated():
    config = OutageConfig(mode="correlated")
    assert OutageConfig(**config.to_dict()) == config
    with pytest.raises(AttributeError):
        config.mode = "legacy"
    with pytest.raises(ValueError, match="mode"):
        OutageConfig(mode="unknown")
    with pytest.raises(ValueError, match="probabilities"):
        OutageConfig(clean_probability=0.9, full_probability=0.2)
    with pytest.raises(ValueError, match="fraction"):
        OutageConfig(fraction_min=0.8, fraction_max=0.2)


@pytest.mark.parametrize("n,fraction", [(0, 0.4), (1, 1), (2, 0.5), (100, 0.41)])
def test_mask_exact_deterministic_and_does_not_change_source(n, fraction):
    valid = np.ones(n, dtype=bool)
    if n > 5:
        valid[3:8] = False
    original = valid.copy()
    kwargs = {"missing_fraction": fraction, "min_seconds": 0.04, "max_seconds": 0.2}
    first = generate_outage_mask(valid, 25, np.random.default_rng(11), **kwargs)
    second = generate_outage_mask(valid, 25, np.random.default_rng(11), **kwargs)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(valid, original)
    assert not (first & ~valid).any()
    assert int((~valid | first).sum()) == max(int((~valid).sum()), int(np.floor(fraction * n)))


def test_all_missing_and_short_input_are_safe():
    valid = np.zeros(3, dtype=bool)
    assert not generate_outage_mask(valid, 25, np.random.default_rng(0), missing_fraction=0.8).any()
    mask = generate_outage_mask(np.ones(1, dtype=bool), 25, np.random.default_rng(0),
                                missing_fraction=1, min_seconds=10, max_seconds=30)
    assert mask.tolist() == [True]


@pytest.mark.parametrize("kwargs", [
    {"missing_fraction": float("nan")},
    {"missing_fraction": -0.1},
    {"missing_fraction": 0.5, "min_seconds": 0},
    {"missing_fraction": 0.5, "long_probability": 2},
])
def test_mask_rejects_bad_numeric_arguments(kwargs):
    with pytest.raises(ValueError):
        generate_outage_mask(np.ones(10, dtype=bool), 25, np.random.default_rng(0), **kwargs)


def test_gap_statistics_includes_edge_runs_and_empty_input():
    result = gap_statistics(np.array([True, True, False, True, False, True, True, True]), 2)
    assert result["count"] == 3
    assert result["missing_frames"] == 6
    assert result["fraction"] == 0.75
    assert result["max_frames"] == 3
    assert result["max_seconds"] == 1.5
    assert gap_statistics(np.zeros(0, dtype=bool), 25)["count"] == 0


def test_correlated_features_rebuilt_with_existing_missing_and_source_untouched():
    frames = _frames(100)
    frames.ball_valid[5:10] = False
    frames.ball_xy[5:10] = 0
    original_valid = frames.ball_valid.copy()
    original_ball = frames.ball_xy.copy()
    base = build_features(frames)
    player, mask, context, outage = _selected_features(
        frames, base, 10, 90, 25, 75, np.random.default_rng(31), True,
        OutageConfig(mode="correlated", clean_probability=0, full_probability=0,
                     fraction_min=0.41, fraction_max=0.41,
                     min_gap_seconds=0.04, max_gap_seconds=0.2),
    )
    assert outage
    assert int(np.count_nonzero(context[:, 2] == 0)) == 32
    assert not player[context[:, 2] == 0, ..., 7].any()
    np.testing.assert_array_equal(mask, base.player_mask[10:90])
    np.testing.assert_array_equal(frames.ball_valid, original_valid)
    np.testing.assert_array_equal(frames.ball_xy, original_ball)
    np.testing.assert_array_equal(base.global_tokens[:, 2], original_valid)
    # Visible-ball motion and gap ages must follow the changed visibility.
    assert not context[context[:, 2] == 0, 5].any()
    assert (context[context[:, 2] == 0, 6:8] > 0).any()
    rebuilt_frames = frames.subset(0, len(frames.frame_id))
    rebuilt_frames.ball_valid[10:90] = context[:, 2].astype(bool)
    rebuilt_frames.ball_xy[~rebuilt_frames.ball_valid] = 0
    rebuilt = build_features(rebuilt_frames)
    np.testing.assert_array_equal(player, rebuilt.player_tokens[10:90])
    np.testing.assert_array_equal(context, rebuilt.global_tokens[10:90])
