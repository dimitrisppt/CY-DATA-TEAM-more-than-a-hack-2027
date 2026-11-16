"""Spatial/temporal corruption is aligned, reproducible and isolated from sampling."""

import numpy as np
import torch

from possession.augmentation import AugmentationConfig, augment_tracking
from possession.features import build_features
from possession.outages import OutageConfig
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half
from possession.train import _window, train_model
from possession.training_session import TrainingSession


def _half(match="a", n=80):
    t = np.arange(n)
    xy = np.zeros((n, 2, 2, 2), np.float32)
    xy[:, 0, :, 0] = 20 + t[:, None] * .05
    xy[:, 1, :, 0] = 70 - t[:, None] * .05
    xy[..., 1] = 30
    state = np.resize(np.repeat([0, 2, 1], [9, 3, 8]), n).astype(np.int8)
    live = (state != 2).astype(np.int8)
    frames = FrameTable(t, xy, np.ones((n, 2, 2), bool),
                        np.column_stack((25 + t * .1, np.full(n, 31))),
                        np.ones(n, bool), np.ones(n, bool), match_id=match)
    targets = TrainingTargets(live, np.ones(n, bool), (state == 0).astype(np.int8),
                              live.astype(bool), state, np.ones(n, bool))
    return frames, targets


def test_window_rebuilds_features_and_supervision_on_same_time_map():
    frames, targets = _half()
    base = build_features(frames)
    config = AugmentationConfig(mode="temporal", operations=("time_shift",),
                                clean_probability=0, warmup_steps=0)
    expected, labels, _ = augment_tracking(frames, targets, np.random.default_rng(17),
                                           config, step=1)
    expected_features = build_features(expected)
    # Seed 5 skips the original ball dropout and team swap, isolating the new policy.
    inputs, loss, choices = _window(
        frames, targets, base, 12, 28, 4, np.random.default_rng(5), True,
        torch.device("cpu"), OutageConfig(), config, 1, np.random.default_rng(17))
    for actual, wanted in zip(inputs, (
        expected_features.player_tokens[8:32], expected_features.player_mask[8:32],
        expected_features.global_tokens[8:32], expected_features.time_valid[8:32],
    )):
        np.testing.assert_array_equal(actual.numpy()[0], wanted)
    np.testing.assert_array_equal(loss["state"].numpy()[0], labels.state[8:32])
    np.testing.assert_array_equal(loss["owner_valid"].numpy()[0, 4:20], labels.owner_valid[12:28])
    assert choices["tracking_augmentation"]["applied"] == ["time_shift"]
    fresh_frames, fresh_targets = _half()
    np.testing.assert_array_equal(frames.ball_xy, fresh_frames.ball_xy)
    np.testing.assert_array_equal(targets.state, fresh_targets.state)


def test_disabled_policy_preserves_original_window_tensors_and_rng():
    frames, targets = _half()
    base = build_features(frames)
    legacy_rng, disabled_rng = np.random.default_rng(18), np.random.default_rng(18)
    old = _window(frames, targets, base, 12, 28, 4, legacy_rng, True,
                  torch.device("cpu"), OutageConfig())
    disabled = _window(frames, targets, base, 12, 28, 4, disabled_rng, True,
                       torch.device("cpu"), OutageConfig(), AugmentationConfig(), 1,
                       np.random.default_rng(19))
    for before, after in zip(old[0], disabled[0]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    for name in old[1]:
        torch.testing.assert_close(old[1][name], disabled[1][name], rtol=0, atol=0)
    assert legacy_rng.bit_generator.state == disabled_rng.bit_generator.state


def test_combined_training_matches_staged_session_and_preserves_window_schedule(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for match in ("a", "b", "c"):
        save_half(data / f"{match}_half1.npz", *_half(match))
    config = AugmentationConfig(mode="combined", clean_probability=0, warmup_steps=6,
                                operations=("time_warp", "coordinate_noise", "reflection"))
    kwargs = {"train_matches": ["a", "b"], "seed": 16, "center_size": 16,
              "halo": 1, "learning_rate": .005, "boundary_pos_weight": 3.0}
    session = TrainingSession(data, model_kwargs={"width": 8, "dilations": [1]},
                              augmentation_config=config, **kwargs)
    session.advance_to(3)
    progress = session.advance_to(8)
    report = train_model(data, tmp_path / "fit", steps=8, validation_matches=["c"],
                         width=8, dilations=[1], augmentation_config=config.to_dict(), **kwargs)
    checkpoint = torch.load(tmp_path / "fit" / "last.pt", weights_only=True)
    for name, tensor in session.model.state_dict().items():
        torch.testing.assert_close(tensor, checkpoint["state_dict"][name], rtol=0, atol=0)
    assert report["training_window_sha256"] == progress["training_window_sha256"]
    assert report["tracking_augmentation"] == progress["tracking_augmentation"]
    assert checkpoint["training_config"]["augmentation_config"] == config.to_dict()
    assert progress["tracking_augmentation"]["applied_windows"] == 8
    assert 0 < progress["tracking_augmentation"]["mean_strength"] < 1
    control = TrainingSession(data, model_kwargs={"width": 8, "dilations": [1]}, **kwargs)
    control_progress = control.advance_to(8)
    assert control_progress["training_window_sha256"] == progress["training_window_sha256"]
    assert control_progress["tracking_augmentation"]["applied_windows"] == 0
