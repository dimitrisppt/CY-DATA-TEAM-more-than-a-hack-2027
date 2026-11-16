"""Tracking augmentation preserves schemas, masks, and target meaning."""

import json

import numpy as np
import pytest

from possession.augmentation import AugmentationConfig, augment_tracking, normalize_augmentation
from possession.schema import FrameTable, TrainingTargets


def sample_window(n=80):
    time = np.arange(n, dtype=np.float32)
    players = np.zeros((n, 2, 2, 2), np.float32)
    players[:, 0, 0] = np.stack((20 + 0.1 * time, 12 + 0.02 * time), axis=1)
    players[:, 0, 1] = np.stack((30 + 0.1 * time, 25 + 0.02 * time), axis=1)
    players[:, 1, 0] = np.stack((50 + 0.1 * time, 40 + 0.02 * time), axis=1)
    players[:, 1, 1] = np.stack((65 + 0.1 * time, 55 + 0.02 * time), axis=1)
    player_valid = np.ones((n, 2, 2), bool)
    player_valid[10:14, 0, 1] = False
    ball = np.stack((27 + 0.1 * time, 20 + 0.02 * time), axis=1)
    ball_valid = np.ones(n, bool)
    ball_valid[40:44] = False
    graded = np.ones(n, bool)
    graded[60:65] = False
    frames = FrameTable(np.arange(n), players, player_valid, ball, ball_valid, graded)
    targets = TrainingTargets(np.ones(n, np.int8), np.ones(n, bool),
                              np.ones(n, np.int8), np.ones(n, bool),
                              np.zeros(n, np.int8), np.ones(n, bool))
    return frames, targets


def run_one(operation, *, seed=4, step=400, **config_overrides):
    frames, targets = sample_window()
    config = AugmentationConfig(mode="spatial", operations=(operation,),
                                clean_probability=0, warmup_steps=400,
                                **config_overrides)
    augmented, labels, report = augment_tracking(frames, targets,
                                                 np.random.default_rng(seed), config,
                                                 step=step)
    return frames, targets, augmented, labels, report


def test_none_is_identity_and_does_not_consume_randomness():
    frames, targets = sample_window()
    rng = np.random.default_rng(123)
    state = json.dumps(rng.bit_generator.state, sort_keys=True)
    augmented, labels, report = augment_tracking(frames, targets, rng, None)
    assert augmented is frames and labels is targets
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == state
    assert report["selected"] == report["applied"] == []
    json.dumps(report, allow_nan=False)


def test_config_normalization_and_validation():
    config = normalize_augmentation({"mode": "spatial", "operations": ["reflection"]})
    assert config.operations == ("reflection",)
    assert AugmentationConfig.from_dict(config.to_dict()) == config
    for kwargs in ({"mode": "bad"}, {"mode": "spatial", "operations": ("time_warp",)},
                   {"mode": "spatial", "operations": ("rotation", "rotation")},
                   {"strength": float("nan")}, {"clean_probability": -0.1},
                   {"max_transforms": 0}, {"noise_correlation": 1.01},
                   {"warmup_steps": -1}, {"time_warp_knots": 1}):
        with pytest.raises((ValueError, TypeError)):
            AugmentationConfig(**kwargs)


def test_policy_bounded_and_warmup():
    frames, targets = sample_window()
    config = AugmentationConfig(mode="spatial", clean_probability=0,
                                operations=("reflection", "rotation", "translation"),
                                max_transforms=2)
    for seed in range(30):
        _, _, report = augment_tracking(frames, targets, np.random.default_rng(seed),
                                        config, step=200)
        assert 1 <= len(report["selected"]) <= 2
        assert len(report["selected"]) == len(set(report["selected"]))
        assert set(report["applied"]) <= set(report["selected"])
        assert report["strength"] == 0.5
        json.dumps(report, allow_nan=False)
    config = AugmentationConfig(mode="spatial", operations=("translation",),
                                clean_probability=0)
    augmented, _, report = augment_tracking(frames, targets, np.random.default_rng(1), config)
    assert np.array_equal(augmented.players_xy, frames.players_xy)
    assert report["selected"] == report["applied"] == []


def test_zero_strength_disables_even_reflections_without_rng_use():
    frames, targets = sample_window()
    config = AugmentationConfig(mode="spatial", operations=("reflection",),
                                clean_probability=0, warmup_steps=0, strength=0)
    rng = np.random.default_rng(7)
    before = json.dumps(rng.bit_generator.state)
    changed, _, report = augment_tracking(frames, targets, rng, config)
    np.testing.assert_array_equal(changed.players_xy, frames.players_xy)
    assert report["applied"] == []
    assert json.dumps(rng.bit_generator.state) == before


def test_coordinate_noise_state_resets_across_observation_gap():
    from possession.augmentation import _coordinate_noise

    frames, _ = sample_window(3)
    frames.player_valid[:] = False
    frames.players_xy[:] = 0
    frames.ball_valid[1] = False
    frames.ball_xy[1] = 0
    before = frames.ball_xy.copy()
    config = AugmentationConfig(noise_ball_std=1, noise_player_std=0, noise_correlation=1)
    expected_rng = np.random.default_rng(12)
    expected = [np.clip(expected_rng.normal(0, 1, 2), -3, 3) for _ in range(2)]
    _coordinate_noise(frames, np.random.default_rng(12), config, 1.0)
    np.testing.assert_allclose(frames.ball_xy[[0, 2]] - before[[0, 2]], expected, atol=2e-6)


@pytest.mark.parametrize("operation", ["translation", "rotation", "reflection"])
def test_geometry_is_coherent_and_labels_unchanged(operation):
    frames, targets, augmented, labels, report = run_one(operation)
    assert augmented is not frames and labels is not targets
    assert np.array_equal(labels.state, targets.state)
    assert np.array_equal(augmented.player_valid, frames.player_valid)
    assert np.array_equal(augmented.ball_valid, frames.ball_valid)
    assert np.all(augmented.players_xy[~augmented.player_valid] == 0)
    assert np.all(augmented.ball_xy[~augmented.ball_valid] == 0)
    before = frames.players_xy[20, 0, 0] - frames.players_xy[20, 0, 1]
    after = augmented.players_xy[20, 0, 0] - augmented.players_xy[20, 0, 1]
    np.testing.assert_allclose(np.linalg.norm(after), np.linalg.norm(before), atol=2e-5)
    np.testing.assert_allclose(augmented.players_xy[20, 0, 0] - augmented.players_xy[21, 0, 0],
                               augmented.players_xy[30, 0, 0] - augmented.players_xy[31, 0, 0],
                               atol=2e-5)
    assert report["applied"] == [operation]


def test_correlated_noise_respects_gaps_and_does_not_mutate_source():
    baseline, _ = sample_window()
    frames, targets, augmented, labels, report = run_one("coordinate_noise", seed=12)
    np.testing.assert_array_equal(frames.players_xy, baseline.players_xy)
    np.testing.assert_array_equal(frames.ball_xy, baseline.ball_xy)
    assert np.all(frames.players_xy[10:14, 0, 1] == 0)
    assert np.array_equal(augmented.player_valid, frames.player_valid)
    assert np.array_equal(augmented.ball_valid, frames.ball_valid)
    assert np.array_equal(augmented.players_xy[~augmented.player_valid],
                          frames.players_xy[~frames.player_valid])
    assert np.array_equal(augmented.ball_xy[40:44], np.zeros((4, 2)))
    assert np.array_equal(augmented.players_xy[60:65], frames.players_xy[60:65])
    assert np.array_equal(labels.owner, targets.owner)
    noise = augmented.players_xy[:10, 0, 0] - frames.players_xy[:10, 0, 0]
    assert np.linalg.norm(noise) > 0
    assert np.mean(np.linalg.norm(np.diff(noise, axis=0), axis=1)) < 0.15
    assert report["details"]["coordinate_noise"]["observations"] > 0
    zero_config = AugmentationConfig(mode="spatial", operations=("coordinate_noise",),
                                     clean_probability=0, warmup_steps=0,
                                     noise_player_std=0, noise_ball_std=0)
    zero, _, zero_report = augment_tracking(frames, targets, np.random.default_rng(2), zero_config)
    np.testing.assert_array_equal(zero.players_xy, frames.players_xy)
    np.testing.assert_array_equal(zero.ball_xy, frames.ball_xy)
    assert zero_report["selected"] == ["coordinate_noise"]
    assert zero_report["applied"] == []
    assert zero_report["details"]["coordinate_noise"]["changed"] is False


def test_reflection_maps_observed_coordinates_across_pitch_dimensions():
    frames, _, augmented, _, report = run_one("reflection")
    axis = report["details"]["reflection"]["axis"]
    expected_players = frames.players_xy.copy()
    expected_ball = frames.ball_xy.copy()
    for coordinate, dimension, included in ((0, frames.pitch[0], axis in ("x", "both")),
                                             (1, frames.pitch[1], axis in ("y", "both"))):
        if included:
            expected_players[..., coordinate][frames.player_valid] = (
                dimension - expected_players[..., coordinate][frames.player_valid])
            expected_ball[..., coordinate][frames.ball_valid] = (
                dimension - expected_ball[..., coordinate][frames.ball_valid])
    np.testing.assert_array_equal(augmented.players_xy, expected_players)
    np.testing.assert_array_equal(augmented.ball_xy, expected_ball)


@pytest.mark.parametrize("operation", ["player_dropout", "spatial_occlusion"])
def test_missingness_is_masked_and_labels_stay_intact(operation):
    frames, targets, augmented, labels, report = run_one(operation)
    assert not np.any(augmented.player_valid & ~frames.player_valid)
    assert not np.any(augmented.ball_valid & ~frames.ball_valid)
    assert np.all(augmented.players_xy[~augmented.player_valid] == 0)
    assert np.all(augmented.ball_xy[~augmented.ball_valid] == 0)
    assert np.array_equal(labels.live, targets.live)
    assert np.array_equal(labels.state, targets.state)
    assert np.array_equal(frames.player_valid[10:14, 0, 1], np.zeros(4, bool))
    assert report["details"][operation]["observations_hidden"] > 0


def test_all_missing_noise_and_patch_are_safe():
    frames, targets = sample_window()
    frames.player_valid[:] = False
    frames.ball_valid[:] = False
    frames.players_xy[:] = 0
    frames.ball_xy[:] = 0
    config = AugmentationConfig(mode="spatial", operations=("coordinate_noise", "spatial_occlusion"),
                                clean_probability=0, max_transforms=2, warmup_steps=0)
    augmented, _, report = augment_tracking(frames, targets, np.random.default_rng(0), config)
    assert not augmented.player_valid.any() and not augmented.ball_valid.any()
    assert not augmented.players_xy.any() and not augmented.ball_xy.any()
    assert report["applied"] == []
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("operation", ["time_warp", "time_shift", "frame_dropout",
                                       "frame_hold", "temporal_blur"])
def test_temporal_policy_dispatch_preserves_schema(operation):
    frames, targets = sample_window()
    config = AugmentationConfig(mode="temporal", operations=(operation,),
                                clean_probability=0, warmup_steps=0)
    augmented, labels, report = augment_tracking(frames, targets,
                                                 np.random.default_rng(9), config)
    assert augmented is not frames and labels is not targets
    assert len(augmented.frame_id) == len(frames.frame_id)
    assert len(labels.live) == len(targets.live)
    assert report["applied"] == [operation]
    json.dumps(report, allow_nan=False)


def test_combined_policy_applies_temporal_first():
    frames, targets = sample_window()
    config = AugmentationConfig(mode="combined", operations=("rotation", "time_warp"),
                                clean_probability=0, max_transforms=2, warmup_steps=0)
    for seed in range(20):
        _, _, report = augment_tracking(frames, targets, np.random.default_rng(seed), config)
        if len(report["selected"]) == 2:
            assert report["applied"] == ["time_warp", "rotation"]
            break
    else:
        pytest.fail("policy never selected both operations")
