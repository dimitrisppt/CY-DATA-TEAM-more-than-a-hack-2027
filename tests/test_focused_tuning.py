"""Frozen convergence ancestry, gates, and a tiny real focused pipeline."""

import copy
import json

import numpy as np
import pytest

from experiments import convergence, focused_tuning
from experiments.oof_selection import score_oof
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half, sha256_file, write_json
from possession.training_session import HalfCache

MATCHES = ["DFL-MAT-J03W" + suffix for suffix in ("MX", "N1", "OH", "OY", "PY", "QQ", "R9")]


def _half(match, period, n=30):
    state = np.resize(np.array([0] * 10 + [1] * 10 + [2] * 10, np.int8), n)
    live = (state != 2).astype(np.int8)
    players = np.zeros((n, 2, 1, 2), np.float32)
    players[:, 0, 0] = [20, 30]
    players[:, 1, 0] = [80, 30]
    frames = FrameTable(np.arange(period * 1000, period * 1000 + n), players,
                        np.ones((n, 2, 1), bool),
                        np.column_stack((30 + np.arange(n) * .01,
                                         np.full(n, 30))).astype(np.float32),
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
            "model_kwargs": {"width": 32, "dilations": [1], "pooling": "mean"},
            "training_args": {"learning_rate": .001, "weight_decay": .0001,
                              "boundary_weight": .25, "boundary_pos_weight": 1.,
                              "augment": True, "center_size": 12},
        },
    })
    return data, recipe


def _convergence(tmp_path):
    data, recipe = _fixture(tmp_path)
    directory = tmp_path / "convergence"
    convergence.run_convergence(data, recipe, directory, seeds=(10,), rungs=(1,),
                                cache_bytes=2_000_000, inference_center_size=16)
    return data, recipe, directory


def _scores(clean, short, long, transition, match_gains=None):
    if match_gains is None:
        match_gains = [0.] * 6
    names = MATCHES[1:]
    by_match = {}
    for match, match_gain in zip(names, match_gains, strict=True):
        by_match[match] = {
            scenario: {"spell_f1_mean": value + (match_gain if scenario != "clean" else 0.)}
            for scenario, value in (("clean", clean), ("block41_seed3101", short),
                                    ("block41_seed3102", short), ("long41_seed4101", long))}
    return {"means": {"clean": clean, "block41_seed3101": short,
                      "block41_seed3102": short, "long41_seed4101": long,
                      "short_mean": short, "objective": .5 * (clean + short),
                      "transition_f1_at_12": {name: transition for name in
                                              focused_tuning.ALL_SCENARIOS}},
            "by_seed": {"10": {"by_match": by_match}}}


def test_gate_requires_four_matches_and_all_declared_limits():
    control = _scores(.5, .5, .5, .5)
    candidate = _scores(.495, .51, .495, .495,
                        [.01, .01, .01, .01, -.03, -.03])
    gate = focused_tuning._gate(control, candidate, MATCHES[1:], (10,))
    assert gate["eligible"]
    assert gate["positive_short_matches"] == 4
    weak = _scores(.495, .51, .495, .495,
                   [.01, .01, .01, -.03, -.03, -.03])
    assert not focused_tuning._gate(control, weak, MATCHES[1:], (10,))["eligible"]
    for changes in ((.489, .51, .5, .5), (.5, .49, .5, .5),
                    (.5, .51, .489, .5), (.5, .51, .5, .489)):
        assert not focused_tuning._gate(control, _scores(*changes), MATCHES[1:],
                                       (10,))["eligible"]


def test_three_optuna_proposals_follow_published_anchor_history():
    optuna = focused_tuning._optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    for index in range(3):
        params = focused_tuning._anchor_params(index, .001)
        record = {"params": params, "reused_from_trial": None,
                  "scores": {"objective": .2 + .01 * index}}
        focused_tuning._append_trial(study, index, record, .001)
    for index in range(3, 6):
        params = focused_tuning._suggest_params(study, index, 10)
        canonical = focused_tuning._canonical(params)
        assert canonical["policy"] in focused_tuning.POLICIES
        assert .0005 <= canonical["learning_rate"] <= .003
        assert params["strength"] in focused_tuning.STRENGTHS
        assert params["skip"] in focused_tuning.SKIPS
        record = {"params": params, "reused_from_trial": None,
                  "scores": {"objective": .2 + .01 * index}}
        focused_tuning._append_trial(study, index, record, .001)
    assert len(study.trials) == 6


def test_convergence_ancestry_rejects_changed_source_and_bundle(tmp_path, monkeypatch):
    data, recipe, directory = _convergence(tmp_path)
    kwargs = {"n_trials": 3, "decoder_trials": 4, "seeds": (10,),
              "confirmation_seed": 13, "cache_bytes": 2_000_000}
    plan = focused_tuning._protocol(data, recipe, directory, **kwargs)
    assert plan["payload"]["selected_rung"] == 1
    real_plan = focused_tuning._plan
    with monkeypatch.context() as patch:
        def changed_source(*args, **kw):
            original = real_plan(*args, **kw)
            return {**original, "payload": {**original["payload"],
                    "source_sha256": {"changed.py": "hash"}}}
        patch.setattr(focused_tuning, "_plan", changed_source)
        with pytest.raises(ValueError, match="source/data/recipe"):
            focused_tuning._protocol(data, recipe, directory, **kwargs)
    checkpoint = directory / "seed_10/fold_0/rung_1/model.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint ancestry"):
        focused_tuning._protocol(data, recipe, directory, **kwargs)


def test_orphan_checkpoint_replays_before_fit_record(tmp_path, monkeypatch):
    data, recipe, convergence_dir = _convergence(tmp_path)
    plan = focused_tuning._protocol(
        data, recipe, convergence_dir, n_trials=3, decoder_trials=4,
        seeds=(10,), confirmation_seed=13, cache_bytes=2_000_000)
    canonical = focused_tuning._canonical(focused_tuning._anchor_params(0, .001))
    spec = focused_tuning._model_spec(canonical)
    directory = tmp_path / "orphan"
    inner = plan["payload"]["fold"]["inner"][0]
    publish = focused_tuning._publish

    def fail_fit(path, value):
        if path.name == "fit.json":
            raise RuntimeError("interrupted after checkpoint")
        return publish(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(focused_tuning, "_publish", fail_fit)
        with pytest.raises(RuntimeError, match="interrupted after checkpoint"):
            focused_tuning._fit(data, directory, plan, HalfCache(data, 2_000_000),
                                spec, 10, 0, inner)
    checkpoint = directory / "model.pt"
    assert checkpoint.is_file() and not (directory / "fit.json").exists()
    digest, mtime = sha256_file(checkpoint), checkpoint.stat().st_mtime_ns
    completed = focused_tuning._fit(data, directory, plan, HalfCache(data, 2_000_000),
                                    spec, 10, 0, inner)
    assert completed["fit"]["replayed_orphan_checkpoint"] is True
    assert sha256_file(checkpoint) == digest and checkpoint.stat().st_mtime_ns == mtime


def test_short_spell_summary_uses_the_supplied_frozen_decoder(monkeypatch):
    gt = np.array([0] * 12 + [1] * 12, dtype=np.int8)
    arrays = {"gt": gt, "ball_valid": np.ones(len(gt), bool)}
    monkeypatch.setattr(focused_tuning, "_oof_records", lambda entries, scenarios: [
        {"scenario": name, "prediction": {"scenario": name}}
        for name in focused_tuning.ALL_SCENARIOS])
    monkeypatch.setattr(focused_tuning, "load_prediction_bundle", lambda record: arrays)
    monkeypatch.setattr(focused_tuning, "decode_oof_half",
                        lambda stored, frozen: gt if frozen["correct"] else 1 - gt)
    correct = focused_tuning._short_spell_recall({}, {"correct": True})
    wrong = focused_tuning._short_spell_recall({}, {"correct": False})
    assert correct["clean"]["n"] == wrong["clean"]["n"] == 2
    assert correct["clean"]["recall_at_iou_0_5"] == 1
    assert wrong["clean"]["recall_at_iou_0_5"] == 0


def test_tiny_real_staged_pipeline_freezes_selection_and_reuses_verified_fits(tmp_path, monkeypatch):
    data, recipe, convergence_dir = _convergence(tmp_path)
    output = tmp_path / "focused"
    actual_advance = focused_tuning.TrainingSession.advance_to
    calls = 0

    def interrupt(self, steps):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return actual_advance(self, steps)

    quality_sources = []

    def fast_selector(records, *, mode, baseline_frozen, output_dir, **_):
        frozen = copy.deepcopy(baseline_frozen)
        if mode == "quality":
            frozen["ensemble"] = {"mode": "quality", "visible_neural_weight": .5,
                                  "missing_neural_weight": .5, "transition_seconds": 2.}
            if output_dir == output / "quality_decoder_search":
                quality_sources.append({record["checkpoint_sha256"] for record in records})
        return {"mode": mode, "frozen": frozen,
                "best_scores": score_oof(records, frozen)}

    monkeypatch.setattr(focused_tuning, "select_oof_decoder", fast_selector)
    options = {"n_trials": 3, "decoder_trials": 4, "seeds": (10,),
               "confirmation_seed": 13, "cache_bytes": 2_000_000}
    with monkeypatch.context() as patch:
        patch.setattr(focused_tuning.TrainingSession, "advance_to", interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            focused_tuning.run_focused_study(data, recipe, convergence_dir, output, **options)
    completed = output / "trials/trial_000/models/seed_10/fold_0/model.pt"
    digest, mtime = sha256_file(completed), completed.stat().st_mtime_ns
    summary = focused_tuning.run_focused_study(
        data, recipe, convergence_dir, output, **options)
    assert summary["status"] == "complete"
    assert summary["selected_rung"] == 1
    assert summary["global_defaults_changed"] is False
    assert summary["external_proxy_scored"] is False
    assert sha256_file(completed) == digest and completed.stat().st_mtime_ns == mtime
    selection = json.loads((output / "selection.json").read_text())
    assert selection["locked_before_confirmation"]
    assert (output / "confirmation.json").is_file()
    assert (output / "baseline-recipe.json").is_file()
    assert (output / "selected-recipe.json").is_file()
    policy = json.loads((output / "policy-selection.json").read_text())
    chosen_trial = json.loads((output / f"trials/trial_{policy['selected_trial']:03d}/trial.json").read_text())
    expected_quality_sources = {entry["fit"]["checkpoint_sha256"]
                                for folds in chosen_trial["entries"].values() for entry in folds}
    assert quality_sources and quality_sources[0] == expected_quality_sources
    checkpoints = sorted(output.rglob("model.pt"))
    hashes = {path: sha256_file(path) for path in checkpoints}
    with monkeypatch.context() as patch:
        patch.setattr(focused_tuning, "TrainingSession", lambda *a, **k: pytest.fail("refit"))
        assert focused_tuning.run_focused_study(
            data, recipe, convergence_dir, output, **options) == summary
    assert {path: sha256_file(path) for path in checkpoints} == hashes
