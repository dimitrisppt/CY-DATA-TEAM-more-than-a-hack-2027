import numpy as np
import pytest

from experiments import campaign_decode
from experiments.spell_evidence_audit import (
    _spells,
    aggregate_stage,
    duration_bin,
    insertion_delta,
    reconstruct_evidence,
    spell_counts,
)
from possession.decode import decode_half


def _fixture(mode):
    n = 11
    rng = np.random.default_rng(401)
    arrays = {
        "neural_live": np.array([0., 1., .9, .2, .6, .8, .5, .4, .8, .1, .9]),
        "neural_home": rng.uniform(.03, .97, n),
        "neural_boundary": rng.normal(0, 2, (n, 3, 3)),
        "ball_valid": np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0], dtype=bool),
        "nearest_ball_seconds": np.array([5, 0, .2, 1, 10, 3, .1, 4, 0, 1, 10]),
        "ref_nearest_m": rng.uniform(.2, 15, (n, 2)),
        "ref_second_m": rng.uniform(1, 20, (n, 2)),
        "ref_motion_factor": rng.uniform(.85, 1, (n, 2)),
        "ref_owner_observed": rng.random(n) > .2,
        "ref_activity": rng.random(n), "ref_activity_observed": rng.random(n) > .2,
        "ref_ball_motion": rng.random(n) > .2, "ref_ball_speed": rng.uniform(0, 8, n),
        "ref_outside": rng.random(n) > .8, "ref_fps": np.asarray(25.),
        "ref_pitch": np.array([105., 68.]),
    }
    cfg = campaign_decode.baseline_decoder({
        "ensemble": {"neural_weight": .51},
        "calibration": {"live_temperature": 3.5, "owner_temperature": .91,
                        "boundary_temperature": 1.4},
        "decoder": {"boundary_scale": 1.2, "switch_cost": 2.},
    })
    cfg.update(blend_mode=mode, visible_neural_weight=0., missing_neural_weight=1.,
               visible_live_weight=.9, missing_live_weight=.1,
               visible_owner_weight=.3, missing_owner_weight=.8,
               boundary_temperature=.7)
    return arrays, cfg


@pytest.mark.parametrize("mode", ["static", "quality_shared", "quality_separate"])
@pytest.mark.parametrize("tuned", [False, True])
def test_reconstructed_inputs_are_exact_decoder_inputs(monkeypatch, mode, tuned):
    arrays, cfg = _fixture(mode)
    if tuned:
        cfg.update(reference_mode="tuned", reference={"owner_smoothing_seconds": .4,
                                                      "live_bias": -.1})
        cfg.update(switch_cost_type="typed", switch_cost={
            "owner_switch": .01, "to_dead": 3.9, "from_dead": 3.1})
    captured = {}

    def capture(emit, boundary, switch_cost):
        captured.update(emissions=emit, boundary=boundary)
        return decode_half(emit, boundary, switch_cost)

    monkeypatch.setattr(campaign_decode, "decode_half", capture)
    expected = campaign_decode.decode_arrays(arrays, cfg)
    actual = reconstruct_evidence(arrays, cfg)
    np.testing.assert_array_equal(actual["emissions"], captured["emissions"])
    np.testing.assert_array_equal(actual["boundary"], captured["boundary"])
    np.testing.assert_array_equal(
        decode_half(actual["emissions"], actual["boundary"], campaign_decode._switch_matrix(cfg)),
        expected,
    )
    # Visible observations always zero the outage gate even for stale seconds.
    np.testing.assert_array_equal(actual["gate"][arrays["ball_valid"]], 0.)
    assert actual["gate"][4] == 1.
    if mode == "quality_shared":
        assert actual["weight_live"][1] == 0.
        assert actual["weight_live"][4] == 1.


def _score(emit, boundary, cost, path):
    score = emit[np.arange(len(path)), path].sum()
    for t in range(1, len(path)):
        a, b = path[t - 1:t + 1]
        if a != b:
            score += boundary[t, a, b] - cost[a, b]
    return score


@pytest.mark.parametrize("interval", [(0, 2), (2, 5), (5, 7), (0, 7)])
def test_insertion_delta_equals_complete_path_score_difference(interval):
    rng = np.random.default_rng(2)
    emit = rng.normal(size=(7, 3))
    boundary = rng.normal(size=(7, 3, 3))
    cost = rng.uniform(0, 4, (3, 3))
    path = np.array([0, 1, 2, 0, 2, 1, 0])
    start, stop = interval
    for state in (0, 1, 2):
        modified = path.copy()
        modified[start:stop] = state
        delta = insertion_delta(emit, boundary, cost, path, start, stop, state)
        assert delta["total"] == pytest.approx(
            _score(emit, boundary, cost, modified) - _score(emit, boundary, cost, path),
            abs=1e-12,
        )


def test_emission_supported_short_spell_can_be_suppressed_by_switch_cost():
    emit = np.tile([0., -10., -10.], (10, 1))
    emit[4:6] = [-1., 0., -10.]
    boundary = np.zeros((10, 3, 3))
    path = decode_half(emit, boundary, switch_cost=2.)
    assert np.all(path == 0)
    assert np.all(np.argmax(emit[4:6], axis=1) == 1)
    delta = insertion_delta(emit, boundary, 2., path, 4, 6, 1)
    assert delta == {"emission": 2., "signed_boundary": 0., "switch_cost": -4., "total": -2.}


def test_fragmented_argmax_is_not_spell_matching_oracle():
    gt = np.zeros(10, dtype=np.int8)
    argmax = np.array([0, 1] * 5)
    gold, pred, matched, overlap_count = _spells(gt, argmax)
    assert len(gold) == 1 and len(pred) == 10
    assert not matched[0] and overlap_count[0] == 5
    summary = aggregate_stage([{"matched": False, "overlaps": 5}])
    assert summary["gt_with_multiple_same_team_overlaps"] == 1


def test_unknown_runs_and_exact_duration_bin_boundaries():
    gt = np.array([0, 0, 3, 0, 0])
    gold, _, matched, _ = _spells(gt, np.zeros(5, dtype=np.int8))
    assert gold == [(0, 2, 0), (3, 5, 0)]
    assert matched.all()
    assert [duration_bin(n, 25) for n in (24, 25, 49, 50, 124, 125)] == [
        "under_1s", "1_to_2s", "1_to_2s", "2_to_5s", "2_to_5s", "5s_and_above"]


def test_reusable_counts_fill_abstentions_per_half_without_crossing_x_separator():
    gt = np.array([0, 0, 0, 3, 0, 0, 0])
    pred = np.array([3, 0, 3, 3, 0, 3, 3])
    result = spell_counts(gt, pred)
    assert result["counts"] == {"gt_spells": 2, "pred_spells": 2,
                                "matched_at_0_5": 2, "missed_at_0_5": 0,
                                "spurious_at_0_5": 0}
    assert result["duration_bins"]["under_1s"]["matched_at_0_5"] == 2
    # A completely abstained run remains X, it cannot borrow a neighbor's state.
    pred[4:] = 3
    assert spell_counts(gt, pred)["counts"]["matched_at_0_5"] == 1


@pytest.mark.parametrize("gt,pred", [([0, -1], [0, 0]), ([0, 0], [0]),
                                     ([0., 0.], [0, 0]), ([0, 0], [0, 4])])
def test_reusable_counts_reject_unknown_encoding_or_misaligned_timelines(gt, pred):
    with pytest.raises(ValueError):
        spell_counts(gt, pred)
