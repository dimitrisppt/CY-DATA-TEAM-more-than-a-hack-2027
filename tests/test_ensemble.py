"""A single shared blend weight preserves team exchange symmetry."""

import numpy as np
import pytest

from possession.ensemble import combine_scores


def _scores(live, home, boundary=None):
    n = len(live)
    return {"live_probability": np.asarray(live, dtype=float),
            "home_probability": np.asarray(home, dtype=float),
            "boundary": np.zeros((n, 3, 3)) if boundary is None else boundary}


def test_endpoints_copy_exact_probabilities_and_boundaries():
    neural = _scores([0, 1], [1, 0], np.ones((2, 3, 3)))
    reference = _scores([.2, .8], [.7, .3], np.full((2, 3, 3), 2.))
    for weight, expected in ((0, reference), (1, neural)):
        actual = combine_scores(neural, reference, neural_weight=weight)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
            assert not np.shares_memory(actual[key], expected[key])


def test_logit_blend_is_symmetric_and_scales_neural_boundary():
    neural = _scores([.7, .3], [.9, .2], np.ones((2, 3, 3)))
    reference = _scores([.4, .8], [.6, .7])
    blended = combine_scores(neural, reference, neural_weight=.25)
    swapped_neural = _scores(neural["live_probability"], 1-neural["home_probability"],
                             neural["boundary"][:, [1, 0, 2]][:, :, [1, 0, 2]])
    swapped_reference = _scores(reference["live_probability"],
                                1-reference["home_probability"])
    swapped = combine_scores(swapped_neural, swapped_reference, neural_weight=.25)
    np.testing.assert_allclose(swapped["live_probability"], blended["live_probability"])
    np.testing.assert_allclose(swapped["home_probability"], 1-blended["home_probability"])
    np.testing.assert_allclose(swapped["boundary"],
                               blended["boundary"][:, [1, 0, 2]][:, :, [1, 0, 2]])
    np.testing.assert_array_equal(blended["boundary"], np.full((2, 3, 3), .25))


@pytest.mark.parametrize("weight", [-1, 1.1, float("nan"), float("inf")])
def test_invalid_weight_rejected(weight):
    scores = _scores([.5], [.5])
    with pytest.raises(ValueError, match="neural_weight"):
        combine_scores(scores, scores, weight)


def test_nonfinite_or_misaligned_scores_rejected_even_at_endpoint():
    neural = _scores([.5], [.5])
    reference = _scores([.4, .6], [.5, .5])
    with pytest.raises(ValueError, match="shape"):
        combine_scores(neural, reference, 0)
    reference = _scores([.4], [float("nan")])
    with pytest.raises(ValueError, match="finite"):
        combine_scores(neural, reference, 1)
