import numpy as np
import pytest

from experiments.campaign_decode import (
    _smooth_basis,
    baseline_decoder,
    decode_arrays,
    reference_basis,
    reference_from_basis,
    sample_decoder,
)
from experiments.robustness import _decode
from possession.features import build_features
from possession.reference import ReferenceConfig, _smooth_observed, reference_scores
from possession.schema import FrameTable


def _frames(seed=3, n=96):
    rng = np.random.default_rng(seed)
    players = rng.uniform([0, 0], [105.3, 68.0], (n, 2, 5, 2)).astype(np.float32)
    player_valid = rng.random((n, 2, 5)) > .12
    ball = rng.uniform([0, 0], [105.3, 68.0], (n, 2)).astype(np.float32)
    ball_valid = rng.random(n) > .3
    graded = rng.random(n) > .03
    return FrameTable(np.arange(n), players, player_valid, ball, ball_valid, graded)


def _arrays(frames):
    rng = np.random.default_rng(91)
    n = len(frames.frame_id)
    boundary = rng.normal(0, .15, (n, 3, 3))
    boundary[:, np.arange(3), np.arange(3)] = 0
    return {
        "neural_live": rng.uniform(.05, .95, n),
        "neural_home": rng.uniform(.05, .95, n),
        "neural_boundary": boundary,
        "ball_valid": frames.ball_valid.copy(),
        "nearest_ball_seconds": rng.uniform(0, 30, n),
        **reference_basis(frames),
    }


def test_reference_basis_matches_reference_scores_for_multiple_configs():
    frames = _frames()
    basis = reference_basis(frames, build_features(frames))
    configs = [None, {"owner_strength": 2.7, "second_weight": .72},
               {"owner_smoothing_seconds": 1.3, "activity_smoothing_seconds": .31,
                "live_bias": -.2, "outside_penalty": 1.2, "ball_motion_bonus": .61}]
    for values in configs:
        actual = reference_from_basis(basis, values)
        expected = reference_scores(frames, None if values is None else ReferenceConfig(**values))
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("n", [0, 1, 17, 2500])
def test_compiled_smoother_matches_original_recurrence_tightly(n):
    rng = np.random.default_rng(100 + n)
    values = rng.normal(size=n)
    observed = rng.random(n) > .37
    for seconds in (.2, .8, 2.0):
        actual = _smooth_basis(values, observed, 25.0, seconds)
        expected = _smooth_observed(values, observed, 25.0, seconds)
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_allclose(left, right, rtol=1e-14, atol=1e-14)


def test_reference_basis_uses_float64_for_threshold_evidence():
    basis = reference_basis(_frames())
    for key in ("ref_nearest_m", "ref_second_m", "ref_motion_factor", "ref_activity",
                "ref_ball_speed"):
        assert basis[key].dtype == np.float64


def test_cached_basis_preserves_empty_player_slots_and_all_missing_ball():
    n = 11
    frames = FrameTable(np.arange(n), np.zeros((n, 2, 0, 2), np.float32),
                        np.zeros((n, 2, 0), bool), np.zeros((n, 2), np.float32),
                        np.zeros(n, bool), np.ones(n, bool))
    basis = reference_basis(frames)
    actual = reference_from_basis(basis)
    expected = reference_scores(frames)
    np.testing.assert_allclose(actual["live_probability"], expected["live_probability"])
    np.testing.assert_allclose(actual["home_probability"], expected["home_probability"])


def test_baseline_config_decodes_identically_to_robustness_helper():
    frames = _frames(seed=5)
    arrays = _arrays(frames)
    neural = {"live_probability": arrays["neural_live"],
              "home_probability": arrays["neural_home"],
              "boundary": arrays["neural_boundary"]}
    reference = reference_from_basis({key: value for key, value in arrays.items()
                                     if key.startswith("ref_")})
    frozen = {
        "ensemble": {"neural_weight": .41},
        "calibration": {"live_temperature": 1.73, "owner_temperature": .68,
                        "boundary_temperature": 1.0},
        "decoder": {"switch_cost": 2.3, "boundary_scale": 0.0},
    }
    actual = decode_arrays(arrays, baseline_decoder(frozen))
    expected = _decode(neural, reference, frozen, neural_only=False)
    np.testing.assert_array_equal(actual, expected)


def test_quality_equal_weights_are_static_equivalent():
    frames = _frames(seed=8)
    arrays = _arrays(frames)
    static = baseline_decoder({"ensemble": {"neural_weight": .63},
                               "calibration": {}, "decoder": {"switch_cost": 1.7,
                                                                   "boundary_scale": 0}})
    quality = {**static, "blend_mode": "quality_shared",
               "visible_neural_weight": .63, "missing_neural_weight": .63}
    np.testing.assert_array_equal(decode_arrays(arrays, static), decode_arrays(arrays, quality))


def test_typed_cost_is_team_symmetric_and_path_equivariant():
    frames = _frames(seed=9)
    arrays = _arrays(frames)
    cfg = baseline_decoder({"ensemble": {"neural_weight": .77}, "calibration": {},
                            "decoder": {"switch_cost": 1.5, "boundary_scale": .7}})
    cfg["switch_cost_type"] = "typed"
    cfg["switch_cost"] = {"owner_switch": 3.1, "to_dead": 1.8, "from_dead": .6}
    original = decode_arrays(arrays, cfg)
    swapped = dict(arrays)
    swapped["neural_home"] = 1 - arrays["neural_home"]
    permutation = np.array([1, 0, 2])
    swapped["neural_boundary"] = arrays["neural_boundary"][:, permutation][:, :, permutation]
    for key in ("ref_nearest_m", "ref_second_m", "ref_motion_factor"):
        swapped[key] = arrays[key][:, ::-1]
    reversed_path = decode_arrays(swapped, cfg)
    np.testing.assert_array_equal(reversed_path, permutation[original])


class _Trial:
    def __init__(self, choices=None):
        self.choices = choices or {}
        self.calls = []

    def suggest_categorical(self, name, choices):
        self.calls.append(name)
        return self.choices.get(name, choices[0])

    def suggest_float(self, name, low, high, log=False):
        self.calls.append(name)
        return (low + high) / 2 if not log else np.sqrt(low * high)


def test_sampler_omits_inactive_reference_boundary_and_blend_dimensions():
    trial = _Trial()
    config = sample_decoder(trial)
    assert config["reference_mode"] == "default"
    assert config["blend_mode"] == "static"
    assert config["boundary_temperature"] == 1
    assert "ref_owner_strength" not in trial.calls
    assert "boundary_temperature" not in trial.calls
    assert "missing_neural_weight" not in trial.calls


def test_reference_only_sampler_omits_blend_gate_and_boundary_parameters():
    trial = _Trial({"reference_mode": "tuned", "switch_cost_type": "typed"})
    config = sample_decoder(trial, reference_only=True)
    assert config["blend_mode"] == "static"
    assert config["neural_weight"] == config["visible_neural_weight"] == 0
    assert config["boundary_scale"] == 0
    assert config["live_temperature"] > 0 and config["owner_temperature"] > 0
    assert config["switch_cost_type"] == "typed"
    assert "blend_mode" not in trial.calls
    assert "transition_seconds" not in trial.calls
    assert "boundary_scale" not in trial.calls
    assert "neural_weight" not in trial.calls


@pytest.mark.parametrize("mutate", [
    lambda a: a.update(neural_live=np.ones((2, 2))),
    lambda a: a.update(neural_boundary=np.full((96, 3, 3), np.nan)),
    lambda a: a.update(nearest_ball_seconds=np.full(96, np.inf)),
    lambda a: a.update(ball_valid=np.ones(96, dtype=np.int8)),
])
def test_decode_rejects_bad_shapes_nonfinite_and_mask_types(mutate):
    frames = _frames()
    arrays = _arrays(frames)
    mutate(arrays)
    cfg = baseline_decoder({"ensemble": {"neural_weight": .5}, "calibration": {},
                            "decoder": {"switch_cost": 2, "boundary_scale": 0}})
    with pytest.raises(ValueError):
        decode_arrays(arrays, cfg)


def test_reference_basis_rejects_missing_and_malformed_arrays():
    with pytest.raises(ValueError, match="lacks keys"):
        reference_from_basis({})
    basis = reference_basis(_frames())
    basis["ref_activity"] = np.zeros(1)
    with pytest.raises(ValueError, match="shape"):
        reference_from_basis(basis)
