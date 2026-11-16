"""Framewise quality gating stays symmetric and preserves ensemble endpoints."""

import numpy as np
import pytest

from possession.ensemble import combine_scores
from possession.quality_blend import QualityBlendConfig, combine_quality_scores


def _scores(live, home, boundary_value=1.):
    n = len(live)
    return {"live_probability": np.asarray(live, dtype=float),
            "home_probability": np.asarray(home, dtype=float),
            "boundary": np.full((n, 3, 3), boundary_value, dtype=float)}


def test_config_round_trip_and_strict_validation():
    config = QualityBlendConfig(.2, .8, 3.)
    assert QualityBlendConfig.from_dict(config.to_dict()) == config
    assert QualityBlendConfig.from_dict({}) == QualityBlendConfig()
    with pytest.raises(ValueError, match="unknown"):
        QualityBlendConfig.from_dict({"neural_weight": .5})
    with pytest.raises(TypeError, match="mapping"):
        QualityBlendConfig.from_dict(None)
    for field, value in (("visible_neural_weight", -1),
                         ("missing_neural_weight", 1.1),
                         ("visible_neural_weight", float("nan")),
                         ("missing_neural_weight", True),
                         ("transition_seconds", 0),
                         ("transition_seconds", float("inf"))):
        with pytest.raises((TypeError, ValueError), match=field):
            QualityBlendConfig(**{field: value})


@pytest.mark.parametrize("weight", [0., .25, .5, 1.])
def test_constant_weight_matches_existing_ensemble_including_endpoints(weight):
    neural = _scores([0., 1., .7], [1., 0., .9], boundary_value=3.)
    reference = _scores([.2, .8, .4], [.7, .3, .6], boundary_value=2.)
    actual = combine_quality_scores(
        neural, reference, np.array([True, False, False]), np.array([0., 4., 10.]),
        QualityBlendConfig(weight, weight, 2.))
    expected = combine_scores(neural, reference, weight)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_long_gap_saturates_observed_ball_resets_and_endpoint_frames_are_exact():
    neural = _scores([0., 1., .7, .8], [1., 0., .9, .8], boundary_value=3.)
    reference = _scores([.2, .8, .4, .5], [.7, .3, .6, .5], boundary_value=2.)
    result = combine_quality_scores(
        neural, reference, np.array([True, False, False, False]),
        np.array([99., 1., 2., 20.]), QualityBlendConfig(0., 1., 2.))
    # A visible ball forces the low weight even if distance is inconsistent.
    for key in result:
        np.testing.assert_array_equal(result[key][0], reference[key][0])
        np.testing.assert_array_equal(result[key][2:], neural[key][2:])
    midpoint = combine_scores(
        {key: value[1:2] for key, value in neural.items()},
        {key: value[1:2] for key, value in reference.items()}, .5)
    for key in midpoint:
        np.testing.assert_allclose(result[key][1:2], midpoint[key], rtol=0, atol=1e-15)


def test_owner_symmetry_uses_same_frame_weight_for_live_and_owner():
    neural = _scores([.7, .3, .9], [.9, .2, .6], boundary_value=.4)
    reference = _scores([.4, .8, .2], [.6, .7, .3], boundary_value=.2)
    visible = np.array([True, False, False])
    distance = np.array([0., .7, 3.])
    config = QualityBlendConfig(.2, .8, 2.)
    original = combine_quality_scores(neural, reference, visible, distance, config)
    swapped_neural = _scores(neural["live_probability"],
                             1 - neural["home_probability"])
    swapped_reference = _scores(reference["live_probability"],
                                1 - reference["home_probability"])
    swapped_neural["boundary"] = neural["boundary"][:, [1, 0, 2]][:, :, [1, 0, 2]]
    swapped_reference["boundary"] = reference["boundary"][:, [1, 0, 2]][:, :, [1, 0, 2]]
    swapped = combine_quality_scores(
        swapped_neural, swapped_reference, visible, distance, config)
    np.testing.assert_allclose(swapped["live_probability"], original["live_probability"],
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(swapped["home_probability"], 1 - original["home_probability"],
                               rtol=0, atol=1e-15)
    np.testing.assert_array_equal(swapped["boundary"],
                                  original["boundary"][:, [1, 0, 2]][:, :, [1, 0, 2]])


def test_empty_timeline_and_all_missing_ball_need_no_player_inputs():
    empty = _scores([], [])
    output = combine_quality_scores(empty, empty, np.array([], dtype=bool),
                                    np.array([]), QualityBlendConfig())
    assert output["live_probability"].shape == (0,)
    assert output["boundary"].shape == (0, 3, 3)
    neural = _scores([.8, .9], [.6, .7])
    reference = _scores([.2, .3], [.4, .5])
    output = combine_quality_scores(neural, reference, np.array([False, False]),
                                    np.array([10., 10.]), QualityBlendConfig(0., 1., 2.))
    for key in output:
        np.testing.assert_array_equal(output[key], neural[key])


@pytest.mark.parametrize("visible,seconds,error", [
    (None, [0., 1.], "ball_visible"),
    ([0, 1], [0., 1.], "ball_visible"),
    ([True], [0., 1.], "ball_visible"),
    ([True, False], None, "nearest_ball_seconds"),
    ([True, False], [0.], "nearest_ball_seconds"),
    ([True, False], [-1., 0.], "nearest_ball_seconds"),
    ([True, False], [float("inf"), 0.], "nearest_ball_seconds"),
    ([True, False], ["bad", "data"], "nearest_ball_seconds"),
])
def test_invalid_quality_arrays_rejected(visible, seconds, error):
    scores = _scores([.5, .5], [.5, .5])
    with pytest.raises(ValueError, match=error):
        combine_quality_scores(scores, scores, visible, seconds, QualityBlendConfig())


def test_score_shape_validation_applies_at_endpoints():
    neural = _scores([.5], [.5])
    reference = _scores([.4, .6], [.5, .5])
    with pytest.raises(ValueError, match="shape"):
        combine_quality_scores(neural, reference, [True], [0.],
                               QualityBlendConfig(0., 0.))
