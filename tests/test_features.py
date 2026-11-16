import numpy as np

from possession.features import build_features
from possession.schema import FrameTable


def frames(frame_id=(10, 11, 12), fps=1.0):
    players = np.array([
        [[[10., 10.], [20., 10.]], [[80., 10.], [90., 10.]]],
        [[[12., 10.], [20., 10.]], [[80., 10.], [90., 10.]]],
        [[[14., 10.], [20., 10.]], [[80., 10.], [90., 10.]]],
    ])
    return FrameTable(
        frame_id=np.asarray(frame_id), players_xy=players,
        player_valid=np.ones((3, 2, 2), dtype=bool),
        ball_xy=np.array([[11., 11.], [13., 11.], [15., 11.]]),
        ball_valid=np.ones(3, dtype=bool), graded=np.array([True, False, True]),
        fps=fps, pitch=(105.3, 68.0),
    )


def test_motion_uses_adjacent_observed_same_slot_and_not_grade():
    sample = frames()
    batch = build_features(sample)
    assert batch.player_tokens.shape == (3, 2, 2, 8)
    assert batch.global_tokens.shape == (3, 9)
    assert batch.time_valid.tolist() == [True, False, True]
    assert batch.player_tokens[0, 0, 0, 4] == 0
    np.testing.assert_allclose(batch.player_tokens[1:, 0, 0, 2], 2 / 15)
    np.testing.assert_allclose(batch.global_tokens[1:, 3], 2 / 50)
    assert batch.global_tokens[1:, 5].tolist() == [1, 1]


def test_gaps_bad_jumps_and_missing_coordinates_are_masked():
    sample = frames()
    sample.player_valid[1, 0, 0] = False
    sample.players_xy[1, 0, 0] = np.nan
    sample.players_xy[2, 0, 1] = [100., 10.]  # implausible speed
    sample.ball_valid[1] = False
    sample.ball_xy[1] = np.nan
    sample.validate()
    batch = build_features(sample)
    assert np.isfinite(batch.player_tokens).all()
    assert np.isfinite(batch.global_tokens).all()
    assert not batch.player_mask[1, 0, 0]
    assert not batch.player_tokens[2, 0, 0, 4]
    assert not batch.player_tokens[2, 0, 1, 4]
    assert not batch.global_tokens[2, 5]
    assert batch.global_tokens[1, 6] == 0.1
    assert batch.global_tokens[1, 7] == 0.1
    assert not batch.player_tokens[1, 0, 0].any()


def test_slot_permutation_and_team_swap_are_exact_tensor_permutations():
    sample = frames()
    original = build_features(sample)
    sample.players_xy = sample.players_xy[:, ::-1, ::-1].copy()
    sample.player_valid = sample.player_valid[:, ::-1, ::-1].copy()
    swapped = build_features(sample)
    np.testing.assert_allclose(swapped.player_tokens, original.player_tokens[:, ::-1, ::-1])
    np.testing.assert_allclose(swapped.global_tokens, original.global_tokens)


def test_no_observations_are_finite_and_frame_jump_resets_motion():
    sample = frames()
    sample.frame_id = np.array([10, 12, 13])
    sample.player_valid[:] = False
    sample.ball_valid[:] = False
    sample.players_xy[:] = np.nan
    sample.ball_xy[:] = np.nan
    # Feature code should be robust to a segmented input even if a validated
    # FrameTable normally enforces contiguous frame IDs.
    batch = build_features(sample)
    assert not batch.player_mask.any()
    assert np.isfinite(batch.player_tokens).all()
    assert np.isfinite(batch.global_tokens).all()
    assert not batch.global_tokens[:, 5].any()
