import numpy as np
import pytest

from possession.schema import FeatureBatch, FrameTable, TrainingTargets


def make_frames():
    return FrameTable(
        frame_id=np.array([10, 11]),
        players_xy=np.array([[[[1.0, 2.0]], [[np.nan, np.nan]]], [[[3.0, 4.0]], [[5.0, 6.0]]]]),
        player_valid=np.array([[[True], [False]], [[True], [True]]]),
        ball_xy=np.array([[np.nan, np.nan], [7.0, 8.0]]),
        ball_valid=np.array([False, True]),
        graded=np.array([True, False]),
    )


def test_invalid_coordinates_are_zero_filled_after_masking():
    frames = make_frames()
    assert frames.players_xy[0, 1, 0].tolist() == [0.0, 0.0]
    assert frames.ball_xy[0].tolist() == [0.0, 0.0]
    assert frames.players_xy[1, 0, 0].tolist() == [3.0, 4.0]


@pytest.mark.parametrize("ids", [[10, 10], [10, 12], [11, 10]])
def test_frame_ids_must_be_contiguous_and_ordered(ids):
    frames = make_frames()
    frames.frame_id = np.array(ids)
    with pytest.raises(ValueError, match="frame_id"):
        frames.validate()


def test_observed_nonfinite_coordinate_is_rejected():
    frames = make_frames()
    frames.ball_xy[1, 0] = np.nan
    with pytest.raises(ValueError, match="ball_xy"):
        frames.validate()


def test_fractional_frame_id_is_not_silently_truncated():
    kwargs = vars(make_frames()).copy()
    kwargs["frame_id"] = np.array([10.5, 11.5])
    with pytest.raises(ValueError, match="frame_id"):
        FrameTable(**kwargs)


def test_training_targets_reject_valid_unknown_state():
    with pytest.raises(ValueError, match="state"):
        TrainingTargets(
            live=np.array([1]), live_valid=np.array([True]),
            owner=np.array([-1]), owner_valid=np.array([False]),
            state=np.array([-1]), state_valid=np.array([True]),
        )


def test_training_targets_reject_inconsistent_dead_state():
    with pytest.raises(ValueError, match="state|live"):
        TrainingTargets(
            live=np.array([1]), live_valid=np.array([True]),
            owner=np.array([-1]), owner_valid=np.array([False]),
            state=np.array([2]), state_valid=np.array([True]),
        )


def test_feature_batch_has_no_target_fields_and_validates_shapes():
    batch = FeatureBatch(
        player_tokens=np.zeros((2, 2, 1, 8)),
        player_mask=np.ones((2, 2, 1), dtype=bool),
        global_tokens=np.zeros((2, 9)),
        time_valid=np.ones(2, dtype=bool),
    )
    assert batch.player_tokens.shape == (2, 2, 1, 8)
    assert not hasattr(batch, "owner")
