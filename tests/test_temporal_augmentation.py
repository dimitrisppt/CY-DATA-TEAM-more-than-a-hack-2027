import json

import numpy as np
import pytest

from possession.schema import FrameTable, TrainingTargets
from possession.temporal_augmentation import apply_temporal, resample_tracking


def make_case(n=8):
    t = np.arange(n, dtype=np.float32)
    players = np.stack((t, t + 100), axis=1)[:, :, None, None]
    players = np.concatenate((players, players + 0.5), axis=-1)
    live = np.ones(n, dtype=np.int8)
    owner = (np.arange(n) % 2).astype(np.int8)
    state = (1 - owner).astype(np.int8)
    frames = FrameTable(
        frame_id=np.arange(100, 100 + n), players_xy=players,
        player_valid=np.ones((n, 2, 1), dtype=bool),
        ball_xy=np.stack((t, t + 0.25), axis=1), ball_valid=np.ones(n, dtype=bool),
        graded=np.ones(n, dtype=bool), fps=25.0,
    )
    targets = TrainingTargets(live, np.ones(n, dtype=bool), owner,
                              np.ones(n, dtype=bool), state, np.ones(n, dtype=bool))
    return frames, targets


def test_resample_aligns_coordinates_and_nearest_labels_at_boundaries():
    frames, targets = make_case(6)
    positions = np.array([0.0, 0.49, 1.51, 2.5, 4.0, 5.0])
    out, labels = resample_tracking(frames, targets, positions)
    np.testing.assert_allclose(out.ball_xy[:, 0], positions, atol=1e-6)
    np.testing.assert_array_equal(labels.owner, [0, 0, 0, 1, 0, 1])
    np.testing.assert_array_equal(out.frame_id, frames.frame_id)
    assert out.fps == frames.fps
    assert np.all(out.player_valid) and np.all(out.ball_valid)


def test_resample_never_bridges_missing_coordinates_or_unknown_labels():
    frames, targets = make_case(6)
    frames.player_valid[2, 0, 0] = False
    frames.players_xy[2, 0, 0] = 0
    frames.ball_valid[2] = False
    frames.ball_xy[2] = 0
    targets.live_valid[3] = False
    targets.owner_valid[3] = False
    targets.state_valid[3] = False
    frames.graded[4] = False
    positions = np.array([0.0, 1.5, 2.0, 2.5, 3.5, 5.0])
    out, labels = resample_tracking(frames, targets, positions)
    assert not out.player_valid[1, 0, 0]
    assert out.player_valid[1, 1, 0]
    assert not out.ball_valid[1] and not out.ball_valid[2]
    assert not labels.live_valid[3] and not labels.owner_valid[3]
    assert not labels.state_valid[4] and not out.graded[4]
    assert out.ball_xy[1].tolist() == [0.0, 0.0]


def transition_case():
    frames, _ = make_case(6)
    # H, D, A, H, A, D: all three meaningful boundary types occur.
    live = np.array([1, 0, 1, 1, 1, 0], dtype=np.int8)
    owner = np.array([1, 0, 0, 1, 0, 0], dtype=np.int8)
    owner_valid = live == 1
    state = np.array([0, 2, 1, 0, 1, 2], dtype=np.int8)
    targets = TrainingTargets(live, np.ones(6, dtype=bool), owner, owner_valid,
                              state, np.ones(6, dtype=bool))
    return frames, targets


def test_fractional_samples_keep_known_h_d_a_and_h_a_transitions_consistent():
    frames, targets = transition_case()
    positions = np.array([0.49, 0.51, 1.49, 1.51, 2.49, 2.51])
    out, labels = resample_tracking(frames, targets, positions)
    np.testing.assert_allclose(out.ball_xy[:, 0], positions, atol=1e-6)
    np.testing.assert_array_equal(labels.live, [1, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(labels.owner_valid, [1, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(labels.state, [0, 2, 2, 1, 1, 0])
    assert labels.live_valid.all() and labels.state_valid.all()

    # The A/H transition has no dead endpoint; it remains fully labeled too.
    _, active_labels = resample_tracking(frames, targets,
                                         np.array([2.49, 2.51, 3.49, 3.51, 4.49, 4.51]))
    assert active_labels.owner_valid[:5].all()
    assert not active_labels.owner_valid[-1]  # Nearest source frame is dead.
    assert active_labels.state_valid.all()


def test_fractional_samples_mask_actually_unknown_owner_or_live_endpoints():
    frames, targets = transition_case()
    targets.owner_valid[2] = False
    targets.state_valid[2] = False
    positions = np.array([0, 0.5, 1.49, 1.51, 2.49, 5.0])
    _, labels = resample_tracking(frames, targets, positions)
    assert not labels.owner_valid[2:5].any()
    assert not labels.state_valid[2:5].any()
    assert labels.live_valid[2:5].all()

    frames, targets = transition_case()
    targets.live_valid[1] = False
    targets.state_valid[1] = False
    positions = np.array([0.49, 0.51, 1.49, 1.51, 4, 5])
    _, labels = resample_tracking(frames, targets, positions)
    assert not labels.live_valid[:4].any()
    assert not labels.owner_valid[:4].any()
    assert not labels.state_valid[:4].any()


def test_time_warp_keeps_endpoints_bounded_rate_and_joint_alignment():
    frames, targets = make_case(80)
    out, labels, metadata = apply_temporal(
        frames, targets, np.random.default_rng(8), operation="time_warp",
        time_warp_max_rate=0.1,
    )
    source = out.ball_xy[:, 0]
    assert source[0] == pytest.approx(0)
    assert source[-1] == pytest.approx(79)
    assert np.all(np.diff(source) > 0)
    assert np.max(np.abs(np.diff(source) - 1)) <= 0.10001
    np.testing.assert_array_equal(labels.owner, targets.owner[np.floor(source + 0.5).astype(int)])
    assert metadata["max_source_rate_deviation"] <= 0.10001


def test_time_shift_masks_outside_half_without_wraparound():
    frames, targets = make_case(8)
    out, labels, metadata = apply_temporal(
        frames, targets, np.random.default_rng(3), operation="time_shift",
        temporal_shift_max_seconds=0.12,
    )
    shift = metadata["shift_frames"]
    positions = np.arange(8) + shift
    outside = (positions < 0) | (positions > 7)
    assert np.any(outside)
    assert not np.any(out.ball_valid[outside])
    assert not np.any(labels.live_valid[outside])
    assert not np.any(out.graded[outside])
    assert np.all(out.ball_xy[outside] == 0)
    np.testing.assert_array_equal(out.frame_id, frames.frame_id)


def test_dropout_masks_one_synchronized_block_but_keeps_labels():
    frames, targets = make_case(40)
    out, labels, metadata = apply_temporal(
        frames, targets, np.random.default_rng(4), operation="frame_dropout",
        frame_dropout_fraction=0.1,
    )
    start = metadata["dropout_start"]
    stop = start + metadata["dropout_frames"]
    assert stop - start == 4
    assert not out.player_valid[start:stop].any()
    assert not out.ball_valid[start:stop].any()
    np.testing.assert_array_equal(labels.owner, targets.owner)
    np.testing.assert_array_equal(labels.owner_valid, targets.owner_valid)
    assert np.all(out.player_valid[:start]) and np.all(out.ball_valid[stop:])


@pytest.mark.parametrize("operation", ["frame_hold", "temporal_blur"])
def test_sensor_operations_keep_missing_masks_and_do_not_cross_ungraded_gaps(operation):
    frames, targets = make_case(12)
    frames.graded[5] = False
    frames.ball_valid[8] = False
    frames.ball_xy[8] = 0
    out, labels, metadata = apply_temporal(
        frames, targets, np.random.default_rng(12), operation=operation,
    )
    np.testing.assert_array_equal(out.ball_valid, frames.ball_valid)
    np.testing.assert_array_equal(out.player_valid, frames.player_valid)
    np.testing.assert_array_equal(out.ball_xy[5], frames.ball_xy[5])
    np.testing.assert_array_equal(out.ball_xy[8], frames.ball_xy[8])
    np.testing.assert_array_equal(labels.owner, targets.owner)
    assert metadata["coordinate_samples_changed"] > 0
    json.dumps(metadata)


def test_short_inputs_are_deterministic_and_inputs_remain_unchanged():
    for n in (0, 1, 2):
        for operation in ("time_warp", "time_shift", "frame_dropout", "frame_hold",
                          "temporal_blur"):
            frames, targets = make_case(n)
            original_xy = frames.players_xy.copy()
            original_owner = targets.owner.copy()
            a = apply_temporal(frames, targets, np.random.default_rng(5), operation=operation)
            b = apply_temporal(frames, targets, np.random.default_rng(5), operation=operation)
            np.testing.assert_array_equal(a[0].players_xy, b[0].players_xy)
            np.testing.assert_array_equal(a[1].owner, b[1].owner)
            assert a[2] == b[2]
            np.testing.assert_array_equal(frames.players_xy, original_xy)
            np.testing.assert_array_equal(targets.owner, original_owner)
            assert a[0] is not frames and a[1] is not targets


def test_rejects_bad_positions_and_operation():
    frames, targets = make_case(3)
    with pytest.raises(ValueError, match="source_positions"):
        resample_tracking(frames, targets, [0, np.nan, 2])
    with pytest.raises(ValueError, match="monotone"):
        resample_tracking(frames, targets, [0, 2, 1])
    with pytest.raises(ValueError, match="operation"):
        apply_temporal(frames, targets, np.random.default_rng(1), operation="reverse")
