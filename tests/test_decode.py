import itertools

import numpy as np
import pytest

from possession._viterbi import viterbi_compiled, viterbi_python
from possession.calibrate import CalibrationConfig, apply_temperature_calibration
from possession.decode import decode_half, decode_half_details, emissions_from_probabilities


def oracle(log_emit, boundary, cost):
    paths = itertools.product(range(3), repeat=len(log_emit))
    best = None
    for path in paths:
        score = sum(log_emit[t, s] for t, s in enumerate(path))
        score += sum(boundary[t, path[t - 1], path[t]] - cost
                     if path[t - 1] != path[t] else 0
                     for t in range(1, len(path)))
        key = (score, -sum(path[t - 1] != path[t] for t in range(1, len(path))))
        if best is None or key > best[0]:
            best = key, path
    return best[0]


def test_emissions_are_finite_and_factor_live_owner_probabilities():
    emit = emissions_from_probabilities([1, 0, .8], [.75, .2, .5])
    assert emit.shape == (3, 3) and np.isfinite(emit).all()
    np.testing.assert_allclose(np.exp(emit[2]), [.4, .4, .2])


@pytest.mark.parametrize("length", [1, 2, 3, 5])
def test_decoder_matches_exhaustive_optimum_and_switch_count(length):
    rng = np.random.default_rng(length)
    emit = rng.normal(size=(length, 3))
    boundary = rng.normal(size=(length, 3, 3))
    cost = .35
    path = decode_half(emit, boundary, switch_cost=cost)
    score = sum(emit[t, s] for t, s in enumerate(path))
    score += sum(boundary[t, path[t - 1], path[t]] - cost
                 if path[t - 1] != path[t] else 0 for t in range(1, length))
    switches = sum(path[t - 1] != path[t] for t in range(1, length))
    optimum, neg_switches = oracle(emit, boundary, cost)
    np.testing.assert_allclose(score, optimum)
    assert switches == -neg_switches


def test_short_direct_switches_and_dead_split_survive_finite_cost():
    emit = np.full((7, 3), -10.)
    emit[np.arange(7), [0, 1, 0, 2, 2, 0, 1]] = 0
    np.testing.assert_array_equal(decode_half(emit, switch_cost=1), [0, 1, 0, 2, 2, 0, 1])


def test_boundary_at_arrival_frame_and_half_reset():
    emit = np.zeros((3, 3))
    emit[:, 2] = -5
    boundary = np.zeros((3, 3, 3))
    boundary[1, 0, 1] = 4
    np.testing.assert_array_equal(decode_half(emit, boundary, switch_cost=1), [0, 1, 1])
    np.testing.assert_array_equal(decode_half(np.array([[-5., -5., 0.]])), [2])


def test_unique_path_team_swap_and_degenerate_tie_preference():
    emit = np.array([[3., 0., -2.], [0., 5., -2.], [0., 4., -2.]])
    path = decode_half(emit, switch_cost=.2)
    swapped = decode_half(emit[:, [1, 0, 2]], switch_cost=.2)
    np.testing.assert_array_equal(swapped, np.where(path == 2, 2, 1 - path))
    tied = decode_half_details(np.zeros((3, 3)), switch_cost=0)
    np.testing.assert_array_equal(tied.path, [0, 0, 0])
    assert tied.switches == 0 and tied.tie_count > 0


def test_invalid_scores_fail_instead_of_leaking_nan_into_path():
    with pytest.raises(ValueError):
        decode_half(np.array([[0., np.nan, 0.]]))
    with pytest.raises(ValueError):
        decode_half(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        decode_half(np.zeros((2, 3)), switch_cost=np.inf)


def test_temperature_transform_preserves_team_swap_and_finite_boundaries():
    live = np.array([.2, .8])
    home = np.array([.15, .75])
    boundary = np.arange(18, dtype=float).reshape(2, 3, 3)
    cfg = CalibrationConfig(1.4, .7, 2.)
    scaled_live, scaled_home, scaled_boundary = apply_temperature_calibration(
        live, home, boundary, cfg)
    flipped_live, flipped_home, flipped_boundary = apply_temperature_calibration(
        live, 1 - home, boundary[:, [1, 0, 2]][:, :, [1, 0, 2]], cfg)
    np.testing.assert_allclose(flipped_live, scaled_live)
    np.testing.assert_allclose(flipped_home, 1 - scaled_home)
    np.testing.assert_allclose(flipped_boundary,
                               scaled_boundary[:, [1, 0, 2]][:, :, [1, 0, 2]])


@pytest.mark.parametrize("seed,length", [(0, 0), (1, 1), (2, 2), (3, 7), (4, 41)])
def test_python_kernel_matches_public_decoder_with_finite_matrix_cost(seed, length):
    rng = np.random.default_rng(seed)
    emit = rng.normal(size=(length, 3))
    boundary = rng.normal(size=(length, 3, 3))
    cost = rng.normal(size=(3, 3))  # Negative switching costs are legal.
    details = decode_half_details(emit, boundary, cost)
    path, score, switches, ties = viterbi_python(emit, boundary, cost)
    np.testing.assert_array_equal(path, details.path)
    assert (score, switches, ties) == (details.score, details.switches, details.tie_count)


@pytest.mark.parametrize("length", [0, 1, 2, 5, 20])
def test_optional_compiled_kernel_exactly_matches_python_on_ties_and_random_inputs(length):
    if viterbi_compiled is None:
        pytest.skip("Numba optional dependency is not installed")
    rng = np.random.default_rng(length)
    cases = [
        (np.zeros((length, 3)), np.zeros((length, 3, 3)), np.zeros((3, 3))),
        (rng.normal(size=(length, 3)), rng.normal(size=(length, 3, 3)),
         rng.normal(size=(3, 3))),
    ]
    for emit, boundary, cost in cases:
        expected = viterbi_python(emit, boundary, cost)
        actual = viterbi_compiled(emit, boundary, cost)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]


def test_decoder_empty_half_and_negative_matrix_cost():
    empty = decode_half_details(np.empty((0, 3)), switch_cost=-2)
    assert empty.path.dtype == np.int8 and len(empty.path) == 0
    assert (empty.score, empty.switches, empty.tie_count) == (0.0, 0, 0)
    emit = np.zeros((3, 3))
    cost = np.full((3, 3), 10.0)
    cost[0, 1] = cost[1, 0] = -1.0
    result = decode_half_details(emit, switch_cost=cost)
    assert result.switches == 2 and result.score == 2.0
