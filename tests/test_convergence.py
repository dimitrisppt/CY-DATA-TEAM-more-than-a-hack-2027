"""Small real trajectories and immutable learning-curve bookkeeping."""

import json

import numpy as np
import pytest
import torch

from experiments import convergence
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half, sha256_file, write_json

MATCHES = ["DFL-MAT-J03W" + suffix for suffix in ("MX", "N1", "OH", "OY", "PY", "QQ", "R9")]


def _half(match, period, n=30):
    state = np.resize(np.array([0] * 10 + [1] * 10 + [2] * 10, np.int8), n)
    live = (state != 2).astype(np.int8)
    players = np.zeros((n, 2, 1, 2), np.float32)
    players[:, 0, 0] = [20, 30]
    players[:, 1, 0] = [80, 30]
    frames = FrameTable(np.arange(period * 1000, period * 1000 + n), players,
                        np.ones((n, 2, 1), bool),
                        np.column_stack((30 + np.arange(n) * .01, np.full(n, 30))).astype(np.float32),
                        np.ones(n, bool), np.ones(n, bool), match_id=match, period=period)
    targets = TrainingTargets(live, np.ones(n, bool), (state == 0).astype(np.int8),
                              live.astype(bool), state, np.ones(n, bool))
    return frames, targets


def _fixture(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    for match in MATCHES:
        for period in (1, 2):
            save_half(data / f"{match}_half{period}.npz", *_half(match, period))
    recipe = tmp_path / "recipe.json"
    write_json(recipe, {
        "outer_test": MATCHES[0], "outer_train": sorted(MATCHES[1:]),
        "ensemble": {"neural_weight": .5},
        "calibration": {"live_temperature": 1., "owner_temperature": 1.},
        "decoder": {"switch_cost": 2., "boundary_scale": 0.},
        "selected_model_recipe": {
            "model_kwargs": {"width": 8, "dilations": [1]},
            "training_args": {"learning_rate": .005, "weight_decay": .0001,
                              "boundary_weight": .25, "boundary_pos_weight": 1.,
                              "augment": True, "center_size": 12},
        },
    })
    return data, recipe


def _fast_score(data_dir, match, model, frozen, cache, *, scenarios,
                inference_center_size, include_diagnostics):
    assert include_diagnostics is True
    assert tuple(scenarios) == convergence.SCENARIO_NAMES
    total = float(next(model.parameters()).detach().sum())
    offset = (sum(match.encode()) % 7) / 1000
    return {scenario: {"scores": {
        diagnostic: {"spell_f1_mean": total / 100 + offset + index / 1000}
        for diagnostic in convergence.DIAGNOSTICS}, "masks": {}}
        for index, scenario in enumerate(scenarios)}


def test_staged_replay_matches_uninterrupted_and_cached_resume_does_not_refit(tmp_path, monkeypatch):
    data, recipe = _fixture(tmp_path)
    monkeypatch.setattr(convergence, "evaluate_augmented_model", _fast_score)
    options = {"seeds": (10,), "rungs": (2, 4), "cache_bytes": 2_000_000,
               "inference_center_size": 16}
    staged = tmp_path / "staged"
    actual_advance = convergence.TrainingSession.advance_to
    failed = False

    def interrupt(self, steps):
        nonlocal failed
        if steps == 4 and not failed:
            failed = True
            raise RuntimeError("simulated interruption")
        return actual_advance(self, steps)

    with monkeypatch.context() as patch:
        patch.setattr(convergence.TrainingSession, "advance_to", interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            convergence.run_convergence(data, recipe, staged, **options)
    assert len(list(staged.rglob("rung_2/result.json"))) == 1
    resumed = convergence.run_convergence(data, recipe, staged, **options)
    uninterrupted = convergence.run_convergence(data, recipe, tmp_path / "whole", **options)
    assert resumed["rungs"] == uninterrupted["rungs"]
    assert resumed["selection"] == uninterrupted["selection"]
    for index in range(3):
        left = torch.load(staged / f"seed_10/fold_{index}/rung_4/model.pt",
                          weights_only=True)
        right = torch.load(tmp_path / f"whole/seed_10/fold_{index}/rung_4/model.pt",
                           weights_only=True)
        assert all(torch.equal(left["state_dict"][key], right["state_dict"][key])
                   for key in left["state_dict"])
        fit = json.loads((staged / f"seed_10/fold_{index}/rung_4/fit.json").read_text())
        assert fit["progress"]["steps"] == 4
        assert fit["progress"]["training_window_sha256"]
        assert fit["progress"]["trained_halves"]
        assert fit["fit_seconds"] >= 0

    checkpoints = sorted(staged.rglob("model.pt"))
    hashes = {path: sha256_file(path) for path in checkpoints}
    mtimes = {path: path.stat().st_mtime_ns for path in checkpoints}
    with monkeypatch.context() as patch:
        patch.setattr(convergence, "TrainingSession", lambda *a, **k: pytest.fail("refit"))
        assert convergence.run_convergence(data, recipe, staged, **options) == resumed
    assert {path: sha256_file(path) for path in checkpoints} == hashes
    assert {path: path.stat().st_mtime_ns for path in checkpoints} == mtimes
    checkpoints[0].write_bytes(checkpoints[0].read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint hash differs"):
        convergence.run_convergence(data, recipe, staged, **options)


def test_changed_protocol_inputs_rejected_before_refit(tmp_path, monkeypatch):
    data, recipe = _fixture(tmp_path)
    monkeypatch.setattr(convergence, "evaluate_augmented_model", _fast_score)
    output = tmp_path / "study"
    options = {"seeds": (10,), "rungs": (1,), "cache_bytes": 2_000_000,
               "inference_center_size": 16}
    convergence.run_convergence(data, recipe, output, **options)
    for changed in ({"seeds": (11,)}, {"rungs": (1, 2)},
                    {"inference_center_size": 32}):
        with pytest.raises(ValueError, match="completed record differs"):
            convergence.run_convergence(data, recipe, output, **{**options, **changed})
    real_plan = convergence._plan
    with monkeypatch.context() as patch:
        def changed_source(*args, **kwargs):
            original = real_plan(*args, **kwargs)
            return {**original, "payload": {**original["payload"],
                    "source_sha256": {"changed.py": "different"}}}
        patch.setattr(convergence, "_plan", changed_source)
        with pytest.raises(ValueError, match="completed record differs"):
            convergence.run_convergence(data, recipe, output, **options)
    original_recipe = recipe.read_bytes()
    edited_recipe = json.loads(recipe.read_text())
    edited_recipe["decoder"]["switch_cost"] = 3.
    write_json(recipe, edited_recipe)
    with pytest.raises(ValueError, match="completed record differs"):
        convergence.run_convergence(data, recipe, output, **options)
    recipe.write_bytes(original_recipe)
    source = next(data.glob("*_half1.npz"))
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="completed record differs"):
        convergence.run_convergence(data, recipe, output, **options)


def test_equal_match_and_seed_weighting_and_smallest_near_best_rung():
    plan = {"payload": {"resources": {"seeds": [10, 11], "rungs": [100, 200, 300]},
                        "fold": {"outer_train": ["a", "b", "c"]}}}
    # Fold one has two matches. A fold-average would overweight c.
    by_seed_rung = {
        10: {100: [.1, .2, .9], 200: [.3, .4, .9], 300: [.301, .401, .9]},
        11: {100: [.7, .8, .9], 200: [.7, .8, .9], 300: [.701, .801, .9]},
    }
    raw = {}
    for seed in (10, 11):
        folds = [{}, {}]
        for rung in (100, 200, 300):
            values = by_seed_rung[seed][rung]
            for index, match_group in enumerate((("a", "b"), ("c",))):
                by_match = {match: {scenario: {"scores": {
                    diagnostic: {"spell_f1_mean": values[("a", "b", "c").index(match)]}
                    for diagnostic in convergence.DIAGNOSTICS}}
                    for scenario in convergence.SCENARIO_NAMES} for match in match_group}
                folds[index][str(rung)] = {"result": {"by_match": by_match}}
        raw[str(seed)] = folds
    aggregate = convergence._aggregate(raw, plan)
    first = aggregate["rungs"]["100"]
    assert first["by_seed"]["10"]["blend"]["clean"] == pytest.approx(.4)
    assert first["mean"]["blend"]["clean"] == pytest.approx(.6)
    assert first["std_across_seeds"]["blend"]["clean"] == pytest.approx(np.std([.4, .8], ddof=1))
    assert aggregate["rungs"]["200"]["paired_gain_from_previous"]["blend"]["objective"]["mean"] == pytest.approx(1 / 15)
    assert aggregate["selection"]["rung"] == 200
    assert aggregate["selection"]["global_default_promoted"] is False
