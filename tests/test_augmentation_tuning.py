"""Small real run and policy-selection invariants for augmentation tuning."""

import json

import numpy as np
import pytest

from experiments import augmentation_tuning as tuning
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half, sha256_file, write_json


def _scores(clean, short1, short2):
    return {
        scenario: {"scores": {diagnostic: {"spell_f1_mean": value}
                              for diagnostic in ("blend", "neural_only")}}
        for scenario, value in zip(tuning.SEARCH_SCENARIOS, (clean, short1, short2))
    }


def test_means_weight_matches_equally_and_comparison_enforces_both_constraints():
    by_match = {"b": _scores(.8, .5, .7), "c": _scores(.2, .9, .3)}
    means = tuning._means(by_match)
    assert means["blend"]["clean"] == pytest.approx(.5)
    assert means["blend"]["block41_seed3101"] == pytest.approx(.7)
    assert means["blend"]["block41_seed3102"] == pytest.approx(.5)
    assert means["blend"]["short_mean"] == pytest.approx(.6)
    assert means["blend"]["objective"] == pytest.approx(.55)
    assert means["neural_only"] == means["blend"]

    control = {"means": {"blend": {"clean": .50, "short_mean": .50, "objective": .50}}}
    eligible = {"means": {"blend": {"clean": .495, "short_mean": .53, "objective": .5125}}}
    assert tuning._comparison(control, eligible)["eligible"]
    exact_floor = {"means": {"blend": {"clean": .49, "short_mean": .53, "objective": .51}}}
    assert tuning._comparison(control, exact_floor)["eligible"]
    clean_loss = {"means": {"blend": {"clean": .489, "short_mean": .60, "objective": .5445}}}
    assert not tuning._comparison(control, clean_loss)["eligible"]
    short_loss = {"means": {"blend": {"clean": .60, "short_mean": .499,
                                      "objective": .5495}}}
    assert not tuning._comparison(control, short_loss)["eligible"]
    no_objective_gain = {"means": {"blend": {"clean": .49, "short_mean": .50,
                                             "objective": .495}}}
    assert not tuning._comparison(control, no_objective_gain)["eligible"]


def _half(match, period, n=60):
    frame_id = np.arange(period * 1000, period * 1000 + n)
    state = np.resize(np.array([0] * 15 + [1] * 15 + [2] * 10 + [0] * 20, np.int8), n)
    live = (state != 2).astype(np.int8)
    players = np.zeros((n, 2, 1, 2), np.float32)
    players[:, 0, 0] = [20, 30]
    players[:, 1, 0] = [80, 30]
    ball = np.column_stack((30 + np.arange(n) * .01, np.full(n, 30))).astype(np.float32)
    visible = np.ones(n, bool)
    visible[5:8] = False
    frames = FrameTable(frame_id, players, np.ones((n, 2, 1), bool), ball,
                        visible, np.ones(n, bool), match_id=match, period=period)
    targets = TrainingTargets(live, np.ones(n, bool), (state == 0).astype(np.int8),
                              live.astype(bool), state, np.ones(n, bool))
    return frames, targets


def _fixture(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    for match in "abc":
        for period in (1, 2):
            save_half(data / f"{match}_half{period}.npz", *_half(match, period))
    recipe = tmp_path / "frozen_recipe.json"
    write_json(recipe, {
        "outer_test": "a", "outer_train": ["b", "c"],
        "ensemble": {"neural_weight": .5},
        "calibration": {"live_temperature": 1., "owner_temperature": 1.},
        "decoder": {"switch_cost": 2., "boundary_scale": 0.},
        "selected_model_recipe": {
            "model_kwargs": {"width": 8, "dilations": [1]},
            "training_args": {"learning_rate": .005, "weight_decay": .0001,
                              "boundary_weight": .25, "boundary_pos_weight": 1.,
                              "augment": True, "center_size": 20},
        },
    })
    return data, recipe


def test_real_tiny_study_pairs_windows_masks_and_reuses_verified_artifacts(tmp_path, monkeypatch):
    data, recipe = _fixture(tmp_path)
    output = tmp_path / "study"
    real_fit_score = tuning._fit_score
    outer_calls = []

    def observe_selection_before_outer(*args, **kwargs):
        if kwargs["validation_matches"] == ["a"]:
            selection = json.loads((output / "selection.json").read_text())
            candidate = json.loads((output / "candidate-config.json").read_text())
            assert selection["locked_before_confirmation_and_outer"]
            assert selection["augmentation_config"] == candidate
            assert (output / "confirmation.json").is_file()
            outer_calls.append(kwargs["seed"])
        return real_fit_score(*args, **kwargs)

    monkeypatch.setattr(tuning, "_fit_score", observe_selection_before_outer)
    options = {"steps": 4, "n_trials": 7, "seed": 0, "confirmation_seed": 1,
               "outer_seeds": (20,), "warmup_steps": 0, "cache_bytes": 2_000_000,
               "inference_center_size": 64}
    summary = tuning.run_augmentation_study(data, recipe, output, **options)
    assert summary["status"] == "complete"
    assert summary["completed_trials"] == 7
    assert summary["global_default_changed"] is False
    assert len(summary["outer"]) == 1
    assert len(outer_calls) == 2
    assert summary["selection"]["augmentation_config"] == json.loads(
        (output / "candidate-config.json").read_text())
    assert (output / "study.db").is_file()
    optuna = tuning._optuna()
    study = optuna.load_study(study_name="augmentation_policy_v1",
                              storage=f"sqlite:///{output / 'study.db'}")
    assert len(study.trials) == 7
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    for frozen_trial in study.trials:
        assert frozen_trial.constraints == dict(zip(
            ("clean_decline", "short_decline"), frozen_trial.user_attrs["constraints"]))

    control = json.loads((output / "control.json").read_text())
    trial = json.loads((output / "trials/trial_000/trial.json").read_text())
    assert trial["cv"]["augmentation_config"]["mode"] == "combined"
    assert trial["cv"]["augmentation_config"]["operations"] == [
        "translation", "rotation", "reflection"]
    for left, right in zip(control["fits"], trial["cv"]["fits"]):
        for field in ("training_window_sha256", "training_schedule_blocks", "trained_halves"):
            assert left["progress"][field] == right["progress"][field]
    for match in ("b", "c"):
        for scenario in tuning.SEARCH_SCENARIOS:
            assert (control["by_match"][match][scenario]["masks"]
                    == trial["cv"]["by_match"][match][scenario]["masks"])

    checkpoint_paths = sorted(output.rglob("model.pt"))
    assert checkpoint_paths
    hashes = {path: sha256_file(path) for path in checkpoint_paths}
    mtimes = {path: path.stat().st_mtime_ns for path in checkpoint_paths}
    again = tuning.run_augmentation_study(data, recipe, output, **options)
    assert again == summary
    assert {path: sha256_file(path) for path in checkpoint_paths} == hashes
    assert {path: path.stat().st_mtime_ns for path in checkpoint_paths} == mtimes

    with pytest.raises(ValueError, match="configuration changed"):
        tuning.run_augmentation_study(data, recipe, output, **{**options, "steps": 5})
    with monkeypatch.context() as source_patch:
        source_patch.setattr(tuning, "_source_hashes", lambda: {"changed.py": "new-source"})
        with pytest.raises(ValueError, match="configuration changed"):
            tuning.run_augmentation_study(data, recipe, output, **options)
    checkpoint_paths[0].write_bytes(checkpoint_paths[0].read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint|result hash|ancestry"):
        tuning.run_augmentation_study(data, recipe, output, **options)
