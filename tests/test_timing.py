"""Timing calibration tests use synthetic states and model evidence only."""

import numpy as np
import pytest

from evaluation.possession_metrics import A, D, H, X


def _timing():
    from possession import timing
    return timing


def _fixture(lengths=(30, 40, 35, 45), lag=4, seed=0):
    rng = np.random.default_rng(seed)
    states = [D, H, A, D]
    truth = np.repeat(states, lengths).astype(np.int8)
    times = np.cumsum(lengths)[:-1]
    pred = truth.copy()
    for index, time in enumerate(times):
        pred[time:time + lag] = states[index]
    live = np.where(truth == D, .03, .97)
    home = np.where(truth == H, .97, np.where(truth == A, .03, .5))
    boundary = np.full((len(truth), 3, 3), .01)
    for index, time in enumerate(times):
        boundary[time, states[index], states[index + 1]] = .99
    arrays = {"neural_live": np.clip(live + rng.normal(0, .003, len(truth)), 0, 1),
              "neural_home": np.clip(home + rng.normal(0, .003, len(truth)), 0, 1),
              "neural_boundary": boundary, "ball_valid": np.ones(len(truth), bool),
              "nearest_ball_seconds": np.zeros(len(truth))}
    return pred, truth, arrays


def test_fixed_shift_sign_and_exact_identity():
    timing = _timing()
    labels = np.array([D, D, D, H, H, H, A, A, D, D], np.int8)
    np.testing.assert_array_equal(timing.shift_labels(labels, 0), labels)
    np.testing.assert_array_equal(timing.shift_labels(labels, 1),
                                  [D, D, D, D, H, H, H, A, A, D])
    np.testing.assert_array_equal(timing.shift_labels(labels,
                                  {"restart": -1, "team_switch": 0, "stoppage": 1}),
                                  [D, D, H, H, H, H, A, A, A, D])


def test_ordered_collisions_preserve_segments_and_exclusions():
    timing = _timing()
    labels = np.array([X, D, H, A, D, X, H, H, A, A, D, D, X], np.int8)
    run_ids = np.array([0] * 8 + [1] * 5)
    shifted = timing.shift_labels(labels, {"restart": 25, "team_switch": -25,
                                           "stoppage": -25}, run_ids=run_ids)
    np.testing.assert_array_equal(shifted[:6], labels[:6])
    np.testing.assert_array_equal(shifted[6:8], [H, H])
    assert np.array_equal(shifted == X, labels == X)
    assert len(shifted) == len(labels)
    assert sorted(set(shifted[8:12])) == [A, D]


def test_target_association_is_exact_type_ordered_and_explicitly_ambiguous():
    timing = _timing()
    pred, truth, arrays = _fixture()
    batch = timing.make_boundary_batch(pred, arrays, radius=8)
    targets = timing.associate_targets(batch, truth)
    np.testing.assert_array_equal(targets.offsets, [-4, -4, -4])
    assert targets.matched.all()
    ambiguous_pred = np.array([D] * 10 + [H] * 3 + [D] * 3 + [H] * 14, np.int8)
    ambiguous_gt = np.array([D] * 13 + [H] * 17, np.int8)
    dummy = {key: np.resize(value, (30,) + value.shape[1:])
             for key, value in arrays.items()}
    targets = timing.associate_targets(
        timing.make_boundary_batch(ambiguous_pred, dummy, radius=8), ambiguous_gt)
    np.testing.assert_array_equal(targets.matched, [False, False, False])
    np.testing.assert_array_equal(targets.ambiguous, [True, False, True])


def test_contextual_fit_recovers_lag_on_distinct_synthetic_matches(tmp_path):
    timing = _timing()
    batches, targets = [], []
    for match, lengths in enumerate([(31, 38, 42, 39), (37, 43, 36, 44), (40, 32, 48, 35)]):
        pred, truth, arrays = _fixture(lengths, lag=4, seed=match)
        batch = timing.make_boundary_batch(pred, arrays, radius=8)
        batches.append(batch)
        targets.append(timing.associate_targets(batch, truth))
    config = timing.TimingConfig(radius=8, l2=.01, family_l2=.1)
    model = timing.fit_timing_model(batches, targets, match_ids=["one", "two", "three"],
                                    config=config)
    pred, truth, arrays = _fixture((35, 45, 37, 43), lag=6, seed=9)
    original_arrays = {key: value.copy() for key, value in arrays.items()}
    original_pred = pred.copy()
    heldout = timing.make_boundary_batch(pred, arrays, radius=8)
    prediction = model.predict(heldout)
    refined = timing.refine_labels(pred, heldout, prediction, config)
    np.testing.assert_array_equal(refined, truth)
    np.testing.assert_array_equal(pred, original_pred)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], original_arrays[key])
    for extension in ("npz", "json"):
        path = tmp_path / ("timing." + extension)
        model.save(path)
        loaded = timing.TimingModel.load(path)
        np.testing.assert_array_equal(loaded.predict(heldout).probabilities,
                                      prediction.probabilities)
        np.testing.assert_array_equal(loaded.predict(heldout).reliability,
                                      prediction.reliability)
    assert np.isfinite(model.fit_diagnostics["offset_loss"])


def test_team_swap_symmetry():
    timing = _timing()
    pred, truth, arrays = _fixture()
    batch = timing.make_boundary_batch(pred, arrays, radius=8)
    config = timing.TimingConfig(radius=8)
    model = timing.fit_timing_model([batch], [timing.associate_targets(batch, truth)],
                                    config=config)
    swap = np.array([A, H, D, X])
    swapped_arrays = {key: value.copy() for key, value in arrays.items()}
    swapped_arrays["neural_home"] = 1 - arrays["neural_home"]
    swapped_arrays["neural_boundary"] = arrays["neural_boundary"][:, [1, 0, 2]][:, :, [1, 0, 2]]
    swapped = timing.make_boundary_batch(swap[pred], swapped_arrays, radius=8)
    np.testing.assert_allclose(batch.features, swapped.features, atol=1e-12)
    np.testing.assert_allclose(model.predict(batch).probabilities,
                               model.predict(swapped).probabilities, atol=1e-12)
    np.testing.assert_array_equal(
        timing.refine_labels(swap[pred], swapped, model.predict(swapped), config),
        swap[timing.refine_labels(pred, batch, model.predict(batch), config)])


def test_nonfinite_evidence_falls_back_to_identity_and_predict_does_not_rescale():
    timing = _timing()
    pred, truth, arrays = _fixture()
    batch = timing.make_boundary_batch(pred, arrays, radius=8)
    config = timing.TimingConfig(radius=8)
    model = timing.fit_timing_model([batch], [timing.associate_targets(batch, truth)],
                                    config=config)
    means = model.feature_mean.copy()
    for key in ("neural_live", "neural_home", "neural_boundary"):
        arrays[key][:] = np.nan
    missing = timing.make_boundary_batch(pred, arrays, radius=8)
    prediction = model.predict(missing)
    assert np.isfinite(prediction.probabilities).all()
    np.testing.assert_array_equal(prediction.reliability, 0)
    np.testing.assert_array_equal(timing.refine_labels(pred, missing, prediction, config), pred)
    np.testing.assert_array_equal(model.feature_mean, means)


def test_point_mass_identity_and_empty_boundaries():
    timing = _timing()
    pred, _, arrays = _fixture()
    batch = timing.make_boundary_batch(pred, arrays, radius=8)
    distribution = np.zeros(batch.valid.shape)
    distribution[:, 8] = 1
    np.testing.assert_array_equal(timing.refine_labels(pred, batch, distribution), pred)
    constant = np.full(len(pred), H, np.int8)
    batch = timing.make_boundary_batch(constant, arrays)
    model = timing.fit_timing_model([batch], [timing.associate_targets(batch, constant)])
    assert model.predict(batch).probabilities.shape == (0, 51)
    np.testing.assert_array_equal(timing.refine_labels(constant, batch, model.predict(batch)), constant)


def test_builder_rejects_invalid_grid_or_run_identifiers():
    timing = _timing()
    pred, _, arrays = _fixture()
    with pytest.raises(ValueError):
        timing.make_boundary_batch(pred, arrays, radius=-1)
    with pytest.raises(ValueError):
        timing.make_boundary_batch(pred, arrays, run_ids=np.zeros(len(pred) - 1))
    with pytest.raises(ValueError):
        timing.make_boundary_batch(pred.astype(float) + .1, arrays)


def test_repeated_scenarios_keep_equal_match_influence():
    timing = _timing()
    pred, truth, arrays = _fixture()
    one = timing.make_boundary_batch(pred, arrays, radius=8)
    pred_two, truth_two, arrays_two = _fixture((45, 35, 40, 30), lag=2)
    two = timing.make_boundary_batch(pred_two, arrays_two, radius=8)
    targets = [timing.associate_targets(one, truth), timing.associate_targets(two, truth_two)]
    config = timing.TimingConfig(radius=8)
    reference = timing.fit_timing_model([one, two], targets, match_ids=["a", "b"], config=config)
    repeated = timing.fit_timing_model([one, one, two], [targets[0], targets[0], targets[1]],
                                       match_ids=["a", "a", "b"], config=config)
    np.testing.assert_allclose(repeated.feature_mean, reference.feature_mean, atol=1e-12)
    np.testing.assert_allclose(repeated.predict(two).probabilities,
                               reference.predict(two).probabilities, atol=1e-6)


def test_features_and_associations_cannot_cross_explicit_half_or_truth_x():
    timing = _timing()
    pred, truth, arrays = _fixture()
    run_ids = np.zeros(len(pred), int)
    run_ids[80:] = 1
    batch = timing.make_boundary_batch(pred, arrays, radius=25, run_ids=run_ids)
    before_first_half = batch.features[batch.run_ids == 0].copy()
    arrays["neural_live"][80:] = 0
    arrays["neural_home"][80:] = 1
    arrays["neural_boundary"][80:] = 1
    changed = timing.make_boundary_batch(pred, arrays, radius=25, run_ids=run_ids)
    np.testing.assert_array_equal(changed.features[changed.run_ids == 0], before_first_half)
    assert np.all(batch.candidate_times[batch.run_ids == 0][batch.valid[batch.run_ids == 0]] < 80)
    assert np.all(batch.candidate_times[batch.run_ids == 1][batch.valid[batch.run_ids == 1]] > 80)
    truth[31:33] = X
    targets = timing.associate_targets(batch, truth)
    assert not targets.matched[0]


def test_truth_looking_extra_arrays_do_not_change_prediction_features():
    timing = _timing()
    pred, truth, arrays = _fixture()
    baseline = timing.make_boundary_batch(pred, arrays)
    arrays["gt"] = truth
    arrays["truth"] = np.roll(truth, 20)
    np.testing.assert_array_equal(timing.make_boundary_batch(pred, arrays).features,
                                  baseline.features)


def test_matchability_head_distinguishes_unmatched_boundary_context():
    timing = _timing()
    pred, truth, arrays = _fixture()
    matchable = timing.make_boundary_batch(pred, arrays, radius=8)
    missing_arrays = {key: value.copy() for key, value in arrays.items()}
    missing_arrays["ball_valid"][:] = 0
    missing_arrays["nearest_ball_seconds"][:] = 10
    unmatched = timing.make_boundary_batch(pred, missing_arrays, radius=8)
    config = timing.TimingConfig(radius=8, l2=.01)
    model = timing.fit_timing_model(
        [matchable, unmatched],
        [timing.associate_targets(matchable, truth),
         timing.associate_targets(unmatched, np.full(len(truth), D, np.int8))],
        match_ids=["a", "b"], config=config)
    assert np.min(model.predict(matchable).reliability) > .9
    assert np.max(model.predict(unmatched).reliability) < .1


def test_optimizer_diagnostics_make_iteration_limit_visible():
    timing = _timing()
    pred, truth, arrays = _fixture()
    batch = timing.make_boundary_batch(pred, arrays, radius=8)
    model = timing.fit_timing_model([batch], [timing.associate_targets(batch, truth)],
                                    config=timing.TimingConfig(radius=8, max_iter=1))
    assert not model.fit_diagnostics["offset_converged"]
    assert model.fit_diagnostics["offset_iterations"] == 1
    assert model.fit_diagnostics["offset_gradient_norm"] > 0
    assert model.fit_diagnostics["offset_message"]


def test_fixed_shift_projection_matches_brute_force_optimum_and_movement_ties():
    from itertools import product

    timing = _timing()
    labels = np.array([D, D, H, A, A, D, D], np.int8)
    original = np.array([2, 3, 5])
    for offsets in ((2, -2, -1), (-2, 2, -2), (1, 1, 1), (0, 0, 0)):
        # Tuple order here follows boundaries: restart, switch, stoppage.
        requested = original + offsets
        candidates = [range(max(1, time - 2), min(6, time + 2) + 1)
                      for time in original]
        objectives = []
        for times in product(*candidates):
            if times[0] < times[1] < times[2]:
                objectives.append((sum((np.array(times) - requested) ** 2),
                                   sum(abs(np.array(times) - original))))
        best = min(objectives)
        shifted = timing.shift_labels(labels, {"restart": offsets[0], "team_switch": offsets[1],
                                               "stoppage": offsets[2]}, radius=2)
        times = np.flatnonzero(shifted[1:] != shifted[:-1]) + 1
        assert len(times) == 3
        assert (sum((times - requested) ** 2), sum(abs(times - original))) == best
        np.testing.assert_array_equal(
            shifted, timing.shift_labels(labels, {"restart": offsets[0],
                                                 "team_switch": offsets[1],
                                                 "stoppage": offsets[2]}, radius=2))


def test_frame_evidence_resolves_uniform_distribution_without_changing_states():
    timing = _timing()
    labels = np.array([D, D, H, H, H, D, D], np.int8)
    arrays = {"neural_live": np.array([.01, .99, .99, .99, .01, .01, .01]),
              "neural_home": np.full(7, .99), "neural_boundary": np.zeros((7, 3, 3)),
              "ball_valid": np.ones(7), "nearest_ball_seconds": np.zeros(7)}
    batch = timing.make_boundary_batch(labels, arrays, radius=2)
    uniform = np.ones(batch.valid.shape)
    refined = timing.refine_labels(
        labels, batch, uniform,
        timing.TimingConfig(radius=2, movement_penalty=0, evidence_weight=1))
    np.testing.assert_array_equal(refined, [D, H, H, H, D, D, D])


def test_random_boundary_movements_preserve_all_run_state_sequences():
    timing = _timing()
    rng = np.random.default_rng(112)
    for _ in range(15):
        labels = rng.integers(0, 3, 100, dtype=np.int8)
        labels[45:50] = X
        run_ids = np.zeros(100, int)
        run_ids[75:] = 1
        result = timing.shift_labels(labels, dict(zip(timing.FAMILIES, rng.integers(-5, 6, 3))),
                                     radius=5, run_ids=run_ids)
        np.testing.assert_array_equal(result == X, labels == X)
        for start, stop in ((0, 45), (50, 75), (75, 100)):
            before = labels[start:stop]
            after = result[start:stop]
            before_states = before[np.r_[True, before[1:] != before[:-1]]]
            after_states = after[np.r_[True, after[1:] != after[:-1]]]
            np.testing.assert_array_equal(after_states, before_states)


def test_negative_boundary_scores_preserve_peaks_and_team_swap_symmetry():
    timing = _timing()
    pred, _, arrays = _fixture()
    arrays["neural_boundary"] = arrays["neural_boundary"] * 10 - 12
    batch = timing.make_boundary_batch(pred, arrays, radius=8)
    row = batch.features[0, :, 3]
    assert np.min(row) == pytest.approx(-11.9)
    assert np.max(row) == pytest.approx(-2.1)
    assert np.argmax(row) == 4  # True boundary is four frames before the prediction.
    swap = np.array([A, H, D, X])
    arrays["neural_home"] = 1 - arrays["neural_home"]
    arrays["neural_boundary"] = arrays["neural_boundary"][:, [1, 0, 2]][:, :, [1, 0, 2]]
    swapped = timing.make_boundary_batch(swap[pred], arrays, radius=8)
    np.testing.assert_allclose(swapped.features, batch.features, atol=1e-12)


def test_negative_score_peaks_recover_unseen_lag_without_state_evidence():
    timing = _timing()
    batches, targets = [], []
    for index in range(3):
        pred, truth, arrays = _fixture((30 + index, 35, 40, 45), lag=4, seed=index)
        arrays["neural_live"][:] = .8
        arrays["neural_home"][:] = .5
        arrays["neural_boundary"] = arrays["neural_boundary"] * 10 - 12
        batch = timing.make_boundary_batch(pred, arrays, radius=8)
        batches.append(batch)
        targets.append(timing.associate_targets(batch, truth))
    config = timing.TimingConfig(radius=8)
    model = timing.fit_timing_model(batches, targets, match_ids=["a", "b", "c"], config=config)
    pred, truth, arrays = _fixture((40, 45, 35, 30), lag=6, seed=9)
    arrays["neural_live"][:] = .8
    arrays["neural_home"][:] = .5
    arrays["neural_boundary"] = arrays["neural_boundary"] * 10 - 12
    heldout = timing.make_boundary_batch(pred, arrays, radius=8)
    np.testing.assert_array_equal(
        timing.refine_labels(pred, heldout, model.predict(heldout), config), truth)


def test_old_probability_feature_artifacts_are_rejected(tmp_path):
    import json

    timing = _timing()
    pred, truth, arrays = _fixture()
    batch = timing.make_boundary_batch(pred, arrays, radius=8)
    model = timing.fit_timing_model([batch], [timing.associate_targets(batch, truth)])
    path = tmp_path / "legacy.json"
    model.save(path)
    metadata = json.loads(path.read_text())
    metadata["version"] = 1
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="schema"):
        timing.TimingModel.load(path)
