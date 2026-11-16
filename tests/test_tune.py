"""Fast checks of nested HPO selection and durable trial mechanics."""

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import optuna
import pytest

from experiments import tune
from experiments.cross_validation import nested_splits
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half, sha256_file
from possession.training_session import HalfCache


class FakeSession:
    instances: ClassVar[list] = []

    def __init__(self, data_dir, *, train_matches, seed, model_kwargs, **kwargs):
        self.train_matches = list(train_matches)
        self.seed = seed
        self.model_kwargs = model_kwargs
        self.augmentation_config = kwargs.get("augmentation_config")
        self.model = self
        self.halo = 255
        self.steps = 0
        self.advances = []
        self.instances.append(self)

    def advance_to(self, steps):
        assert steps > self.steps
        self.steps = steps
        self.advances.append(steps)
        return {"steps": steps, "train_loss": .5,
                "train_losses": {"live": .2, "owner": .2, "boundary": .1},
                "train_counts": {"live": steps, "owner": steps,
                                 "boundary": steps},
                "boundary_positive_count": 1, "boundary_negative_count": steps,
                "augmentation_counts": {"team_swap": 0, "ball_outage": 0},
                "trained_halves": ["fake_half1.npz"],
                "training_schedule_blocks": [{"half": "fake_half1.npz", "steps": steps}]}

    def save_checkpoint(self, path, *, validation_matches, extra_training_config):
        Path(path).write_text("fake checkpoint")


class FakeTrial:
    number = 0

    def __init__(self, prune=False):
        self.attrs = {}
        self.reports = []
        self.prune = prune

    def set_user_attr(self, key, value):
        self.attrs[key] = value

    def report(self, value, step):
        self.reports.append((value, step))

    def should_prune(self):
        return self.prune


def _fake_score(data_dir, match, session, cache, **kwargs):
    assert match not in session.train_matches
    return float(ord(match[-1]) - ord("a") + 1), {"official": {"spell_f1_mean": 0}}, []


def _fixture(tmp_path, monkeypatch):
    FakeSession.instances.clear()
    monkeypatch.setattr(tune, "TrainingSession", FakeSession)
    monkeypatch.setattr(tune, "_score_match", _fake_score)
    data = tmp_path / "data"
    data.mkdir()
    for match in "abcd":
        (data / f"{match}_half1.npz").write_bytes(match.encode())
    return data


def test_trial_reports_only_after_all_folds_and_scores_unweighted_match_mean(tmp_path, monkeypatch):
    data = _fixture(tmp_path, monkeypatch)
    fold = nested_splits(list("abcd"))[0]
    trial = FakeTrial()
    score, sessions, records, rungs = tune._run_trial(
        trial, fold, data, object(), params=tune._baseline("mean"),
        rungs=(2, 4), seed=7, device="cpu", center_size=8, pruning=True)
    expected = np.mean([float(ord(match) - ord("a") + 1) for match in fold["outer_train"]])
    assert score == expected
    assert trial.reports == [(expected, 2), (expected, 4)]
    assert all(session.advances == [2, 4] for session in sessions)
    assert len(rungs[0]["by_match"]) == 3
    assert len(rungs[0]["fold_progress"]) == 3
    assert rungs[0]["fold_progress"][0]["steps"] == 2
    assert "training_schedule_blocks" not in json.dumps(rungs)
    assert records == []


def test_pruning_waits_for_every_fold_at_same_rung(tmp_path, monkeypatch):
    data = _fixture(tmp_path, monkeypatch)
    fold = nested_splits(list("abcd"))[0]
    trial = FakeTrial(prune=True)
    with pytest.raises(optuna.TrialPruned):
        tune._run_trial(trial, fold, data, object(), params=tune._baseline("mean"),
                        rungs=(2, 4), seed=7, device="cpu", center_size=8, pruning=True)
    assert len(FakeSession.instances) == 3
    assert all(session.advances == [2] for session in FakeSession.instances)
    assert trial.reports == [(trial.reports[0][0], 2)]


def test_resume_preserves_trial_params_and_fingerprint_rejects_data_change(tmp_path, monkeypatch):
    data = _fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    uninterrupted = tune.run_study(data, tmp_path / "full", outer_test="a",
                                    matches=list("abcd"), n_trials=4,
                                    rungs=(1, 2), poolings=("mean", "ball_attention"))
    first = tune.run_study(data, tmp_path / "staged", outer_test="a",
                           matches=list("abcd"), n_trials=2,
                           rungs=(1, 2), poolings=("mean", "ball_attention"))
    assert len(first["trials"]) == 2
    resumed = tune.run_study(data, tmp_path / "staged", outer_test="a",
                             matches=list("abcd"), n_trials=4,
                             rungs=(1, 2), poolings=("mean", "ball_attention"))
    assert [trial["params"] for trial in uninterrupted["trials"]] == [
        trial["params"] for trial in resumed["trials"]]
    assert set(resumed["best_by_pooling"]) == {"mean", "ball_attention"}
    assert resumed["best"]["training_args"]["steps"] == 2
    assert all(trial["baseline_full_budget"] for trial in resumed["trials"][:2])
    mean_pointer = tmp_path / "staged" / "a" / "best_by_pooling" / "mean" / "recipe.json"
    mean_pointer.unlink()
    recovered = tune.run_study(data, tmp_path / "staged", outer_test="a",
                               matches=list("abcd"), n_trials=4,
                               rungs=(1, 2), poolings=("mean", "ball_attention"))
    assert mean_pointer.is_file()
    assert recovered["best_by_pooling"]["mean"] == json.loads(mean_pointer.read_text())
    assert (tmp_path / "staged" / "a" / "splits.json").is_file()
    assert (tmp_path / "staged" / "a" / "study.sqlite3").is_file()
    fingerprint = json.loads((tmp_path / "staged" / "a" / "fingerprint.json").read_text())
    assert fingerprint["payload"]["versions"]["python"]
    assert fingerprint["payload"]["versions"]["numba"]
    assert fingerprint["payload"]["uv_lock_sha256"] == sha256_file(
        Path(tune.__file__).resolve().parents[1] / "uv.lock")
    assert fingerprint["payload"]["thread_environment"]["OMP_NUM_THREADS"] == "3"
    (data / "b_half1.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        tune.run_study(data, tmp_path / "staged", outer_test="a",
                       matches=list("abcd"), n_trials=4,
                       rungs=(1, 2), poolings=("mean", "ball_attention"))


def test_outage_policy_is_frozen_in_recipe_and_resume(tmp_path, monkeypatch):
    data = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "out"
    report = tune.run_study(data, output, outer_test="a", matches=list("abcd"),
                            n_trials=1, rungs=(1,), poolings=("mean",),
                            outage_config={"mode": "correlated"})
    config = report["best"]["training_args"]["outage_config"]
    assert config["mode"] == "correlated"
    fingerprint = json.loads((output / "a" / "fingerprint.json").read_text())
    assert fingerprint["payload"]["fixed"]["outage_config"] == config
    with pytest.raises(ValueError, match="fingerprint"):
        tune.run_study(data, output, outer_test="a", matches=list("abcd"),
                       n_trials=1, rungs=(1,), poolings=("mean",),
                       outage_config={"mode": "legacy"})


def test_augmentation_config_is_frozen_in_recipe_and_resume(tmp_path, monkeypatch):
    data = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "out"
    requested = {"mode": "combined", "operations": ["translation", "time_shift"],
                 "clean_probability": 0.35, "max_transforms": 2,
                 "translation_max_meters": 0.7, "temporal_shift_max_seconds": 0.08,
                 "warmup_steps": 0}
    report = tune.run_study(data, output, outer_test="a", matches=list("abcd"),
                            n_trials=1, rungs=(1,), poolings=("mean",),
                            augmentation_config=requested)
    config = report["best"]["training_args"]["augmentation_config"]
    assert config == tune._augmentation_dict(requested)
    assert all(session.augmentation_config == config for session in FakeSession.instances)
    fingerprint = json.loads((output / "a" / "fingerprint.json").read_text())
    assert fingerprint["payload"]["fixed"]["augmentation_config"] == config
    recipe = json.loads((output / "a" / "trials" / "trial_0" / "recipe.json").read_text())
    assert recipe["training_args"]["augmentation_config"] == config
    with pytest.raises(ValueError, match="fingerprint"):
        tune.run_study(data, output, outer_test="a", matches=list("abcd"),
                       n_trials=1, rungs=(1,), poolings=("mean",),
                       augmentation_config={**requested, "clean_probability": 0.4})


def test_frozen_recipe_allows_readonly_summary_but_no_new_trial(tmp_path, monkeypatch):
    data = _fixture(tmp_path, monkeypatch)
    tune.run_study(data, tmp_path / "out", outer_test="a", matches=list("abcd"), n_trials=1,
                   rungs=(1,), poolings=("mean",))
    frozen = tmp_path / "out" / "a" / "frozen_recipe.json"
    frozen.write_text("{}")
    existing = tune.run_study(data, tmp_path / "out", outer_test="a",
                              matches=list("abcd"), n_trials=1,
                              rungs=(1,), poolings=("mean",))
    assert len(existing["trials"]) == 1
    with pytest.raises(RuntimeError, match="frozen"):
        tune.run_study(data, tmp_path / "out", outer_test="a",
                       matches=list("abcd"), n_trials=2,
                       rungs=(1,), poolings=("mean",))


def test_both_enqueued_architecture_baselines_bypass_pruning(tmp_path, monkeypatch):
    data = _fixture(tmp_path, monkeypatch)
    calls = []
    original = tune._run_trial

    def record_pruning(*args, **kwargs):
        calls.append(kwargs["pruning"])
        return original(*args, **kwargs)

    monkeypatch.setattr(tune, "_run_trial", record_pruning)
    report = tune.run_study(data, tmp_path / "out", outer_test="a",
                            matches=list("abcd"), n_trials=2, rungs=(1, 2),
                            poolings=("mean", "ball_attention"), pruning=True)
    assert calls == [False, False]
    assert all(trial["state"] == "COMPLETE" for trial in report["trials"])


def test_best_artifacts_have_documented_native_half_arrays(tmp_path):
    fold = nested_splits(list("abcd"))[0]
    sessions = [FakeSession(tmp_path, train_matches=inner["train"], seed=0,
                            model_kwargs={}) for inner in fold["inner"]]
    fold_index = next(index for index, inner in enumerate(fold["inner"])
                      if "b" in inner["validation"])
    record = {"match": "b", "period": 1, "inner_fold": fold_index,
              "train_matches": ["c", "d"],
              "frame_id": np.array([9, 10]), "gt": np.array([0, 2]),
              "live_probability": np.array([.9, .1]),
              "home_probability": np.array([.8, .2]),
              "boundary": np.zeros((2, 3, 3))}
    directory = tmp_path / "trials" / "trial_0"
    manifest = tune._save_trial_artifacts(
        directory, trial=FakeTrial(), fold=fold, sessions=sessions,
        records=[record], params=tune._baseline("mean"), score=.5,
        rungs=(1, 2), center_size=8, seed=0, fingerprint_sha256="studyhash")
    with np.load(directory / "b_half1.npz") as arrays:
        assert set(arrays.files) == {"gt", "live_probability", "home_probability",
                                      "boundary", "frame_id"}
        np.testing.assert_array_equal(arrays["frame_id"], [9, 10])
    sidecar = json.loads((directory / "b_half1.json").read_text())
    assert sidecar["prediction_match"] == "b"
    assert sidecar["train_matches"] == ["c", "d"]
    assert sidecar["study_fingerprint_sha256"] == "studyhash"
    assert sidecar["oof_sha256"] == sha256_file(directory / "b_half1.npz")
    assert sidecar["checkpoint_sha256"] == sha256_file(directory / f"fold_{fold_index}.pt")
    assert manifest["artifact_sha256"]["b_half1.npz"] == sidecar["oof_sha256"]
    assert manifest["artifact_sha256"]["b_half1.json"] == sha256_file(
        directory / "b_half1.json")


def test_failed_artifact_save_keeps_previous_family_pointer(tmp_path):
    fold = nested_splits(list("abcd"))[0]
    pointer = tmp_path / "best_by_pooling" / "mean" / "recipe.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text('{"trial_number": 0}')
    sessions = [FakeSession(tmp_path, train_matches=inner["train"], seed=0,
                            model_kwargs={}) for inner in fold["inner"]]

    def fail_save(*args, **kwargs):
        raise OSError("simulated checkpoint write failure")

    sessions[1].save_checkpoint = fail_save
    directory = tmp_path / "trials" / "trial_1"
    with pytest.raises(OSError, match="simulated"):
        tune._save_trial_artifacts(
            directory, trial=FakeTrial(), fold=fold, sessions=sessions,
            records=[], params=tune._baseline("mean"), score=.8,
            rungs=(1, 2), center_size=8, seed=0, fingerprint_sha256="studyhash")
    assert not directory.exists()
    assert list(directory.parent.iterdir()) == []
    assert json.loads(pointer.read_text()) == {"trial_number": 0}


def test_score_match_keeps_half_reset_and_native_frame_ids(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    for period in (1, 2):
        n = 4
        frames = FrameTable(np.arange(period * 100, period * 100 + n),
                            np.zeros((n, 2, 1, 2), np.float32),
                            np.ones((n, 2, 1), bool), np.zeros((n, 2), np.float32),
                            np.ones(n, bool), np.ones(n, bool), match_id="b", period=period)
        states = np.array([0, 0, 0, 0], np.int8)
        targets = TrainingTargets(np.ones(n, np.int8), np.ones(n, bool),
                                  np.ones(n, np.int8), np.ones(n, bool),
                                  states, np.ones(n, bool))
        save_half(data / f"b_half{period}.npz", frames, targets)

    from possession import predict

    monkeypatch.setattr(predict, "predict_features", lambda features, model, **kwargs: {
        "live_probability": np.full(len(features.time_valid), .99),
        "home_probability": np.full(len(features.time_valid), .99),
        "boundary": np.zeros((len(features.time_valid), 3, 3)),
    })
    session = type("Session", (), {"model": object(), "halo": 255})()
    score, detail, records = tune._score_match(
        data, "b", session, HalfCache(data), switch_cost=2.0, center_size=4,
        device="cpu", train_matches=["a"])
    assert score == 1.0
    assert detail["official"]["n_gt_spells"] == 2
    assert [record["period"] for record in records] == [1, 2]
    np.testing.assert_array_equal(records[1]["frame_id"], [200, 201, 202, 203])


def test_summary_ranks_only_completed_full_budget_trials(tmp_path):
    fold = nested_splits(list("abcd"))[0]
    directory = tmp_path / "outer"
    recipe_dir = directory / "trials" / "trial_0"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "recipe.json").write_text(json.dumps({
        "score": .4, "params": {"pooling": "mean"}, "trial_number": 0,
        "study_fingerprint_sha256": "abc", "artifact_dir": str(recipe_dir)}))

    class StoredTrial:
        def __init__(self, number, state, value, steps):
            self.number = number
            self.state = state
            self.value = value
            self.params = {"pooling": "mean"}
            self.user_attrs = {"full_budget_steps": steps}

    study = type("Study", (), {"trials": [
        StoredTrial(0, optuna.trial.TrialState.COMPLETE, .4, 2),
        StoredTrial(1, optuna.trial.TrialState.PRUNED, .9, 1),
        StoredTrial(2, optuna.trial.TrialState.COMPLETE, .8, 1),
    ]})()
    summary = tune._summary(study, fold, directory, fingerprint={"sha256": "abc"},
                            rungs=(1, 2), center_size=4, seed=0)
    assert summary["n_complete_full_budget"] == 1
    assert summary["best"]["score"] == .4


def test_default_discovery_requires_exact_seven_dfl_ids(tmp_path):
    for match in ("a", "b", "c"):
        (tmp_path / f"{match}_half1.npz").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="seven known DFL"):
        tune._discover_matches(tmp_path)
    with pytest.raises(ValueError, match="pooling baseline"):
        tune.run_study(tmp_path, tmp_path / "out", matches=["a", "b", "c"],
                       n_trials=1, poolings=("mean", "ball_attention"))


def test_cli_parses_outer_and_resource_paths(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_run(data_dir, output_dir, **kwargs):
        captured.update(data_dir=data_dir, output_dir=output_dir, **kwargs)
        return {"ok": True}

    monkeypatch.setattr(tune, "run_study", fake_run)
    tune.main([str(tmp_path / "data"), str(tmp_path / "out"), "--outer-test", "a",
               "--n-trials", "3", "--rungs", "2", "4", "--poolings", "mean"])
    assert captured["data_dir"] == tmp_path / "data"
    assert captured["output_dir"] == tmp_path / "out"
    assert captured["rungs"] == (2, 4)
    assert captured["poolings"] == ("mean",)
    assert json.loads(capsys.readouterr().out) == {"ok": True}
