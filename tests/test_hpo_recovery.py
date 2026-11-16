"""Small synthetic checks for the explicit interrupted-trial recovery path."""

import hashlib
import json
import os
from pathlib import Path

import optuna
import pytest

from experiments import tune
from scripts import recover_interrupted_trial as recovery


class FakeSession:
    def __init__(self, data_dir, *, train_matches, seed, model_kwargs, **kwargs):
        self.train_matches = train_matches
        self.model_kwargs = model_kwargs
        self.steps = 0

    def advance_to(self, steps):
        assert steps > self.steps
        self.steps = steps
        return {
            "steps": steps, "train_loss": 0.5,
            "train_losses": {"live": 0.2, "owner": 0.2, "boundary": 0.1},
            "train_counts": {"live": steps, "owner": steps, "boundary": steps},
            "boundary_positive_count": 1, "boundary_negative_count": steps,
            "augmentation_counts": {"team_swap": 0, "ball_outage": 0},
            "trained_halves": ["synthetic_half1.npz"],
        }

    def save_checkpoint(self, path, *, validation_matches, extra_training_config):
        Path(path).write_bytes(b"synthetic checkpoint")


def _fake_score(data_dir, match, session, cache, **kwargs):
    assert match not in session.train_matches
    score = (0.2 + 0.01 * sorted(tune.DEFAULT_MATCHES).index(match)
             + 0.02 * (session.model_kwargs["width"] == 64))
    return score, {"official": {"spell_f1_mean": score}}, []


@pytest.fixture
def synthetic_data(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, "TrainingSession", FakeSession)
    monkeypatch.setattr(tune, "_score_match", _fake_score)
    data = tmp_path / "data"
    data.mkdir()
    for match in tune.DEFAULT_MATCHES:
        (data / f"{match}_half1.npz").write_bytes(match.encode())
    return data


class _InterruptAfterReport:
    def __init__(self, trial):
        self.trial = trial

    def set_user_attr(self, name, value):
        self.trial.set_user_attr(name, value)

    def report(self, value, step):
        self.trial.report(value, step)
        raise RuntimeError("synthetic worker interruption after persisted 600 rung")


def test_replay_same_trial_and_next_proposal(synthetic_data, tmp_path):
    outer = tune.DEFAULT_MATCHES[0]
    full = tune.run_study(synthetic_data, tmp_path / "full", outer_test=outer,
                          n_trials=4)
    staged_dir = tmp_path / "staged"
    tune.run_study(synthetic_data, staged_dir, outer_test=outer, n_trials=2)
    outer_dir = staged_dir / outer
    study = optuna.load_study(
        study_name=f"nested-{outer}", storage=f"sqlite:///{outer_dir / 'study.sqlite3'}",
        sampler=optuna.samplers.TPESampler(seed=2, multivariate=True, n_startup_trials=3),
        pruner=optuna.pruners.NopPruner())
    trial = study.ask()
    assert trial.number == 2
    params = tune._suggest(trial, ("mean", "ball_attention"))
    trial.set_user_attr("baseline_full_budget", False)
    fold = next(fold for fold in json.loads((outer_dir / "splits.json").read_text())[
        "outer_folds"] if fold["outer_test"] == [outer])
    with pytest.raises(RuntimeError, match="synthetic worker interruption"):
        tune._run_trial(
            _InterruptAfterReport(trial), fold, synthetic_data,
            tune.HalfCache(synthetic_data), params=params, rungs=(600, 1200),
            seed=0, device="cpu", center_size=512, pruning=False)
    assert study.get_trials(deepcopy=True)[2].state == optuna.trial.TrialState.RUNNING
    assert set(study.get_trials(deepcopy=True)[2].intermediate_values) == {600}
    (outer_dir / "run.lock").write_text("pid=2147483647\n")

    plan = recovery.recover(synthetic_data, outer_dir)
    assert plan["trial_number"] == 2 and plan["reported_steps"] == [600]
    assert plan["execute"] is False
    result = recovery.recover(synthetic_data, outer_dir, execute=True)
    assert result["trial_number"] == 2
    assert result["checked_rungs"] == [600, 1200]
    record = json.loads(Path(result["recovery_record"]).read_text())
    assert record["status"] == "complete"
    assert record["recovery_script_sha256"] == hashlib.sha256(
        Path(recovery.__file__).read_bytes()).hexdigest()
    assert record["original_trial"]["state"] == "RUNNING"

    resumed = tune.run_study(synthetic_data, staged_dir, outer_test=outer, n_trials=4)
    assert [(item["number"], item["params"], item["value"])
            for item in resumed["trials"]] == [
                (item["number"], item["params"], item["value"])
                for item in full["trials"]]


def test_changed_replayed_score_leaves_trial_running(tmp_path):
    study = optuna.create_study(study_name="mismatch",
                                storage=f"sqlite:///{tmp_path / 'study.sqlite3'}")
    original = study.ask()
    original.report(0.4, 600)
    old_rungs = [{"steps_per_fold": 600,
                  "by_match": {"match": {"spell_f1_mean": 0.4}}}]
    original.set_user_attr("rungs", old_rungs)
    frozen = study.get_trials(deepcopy=True)[0]
    trial = optuna.trial.Trial(study, frozen._trial_id)
    replay = recovery._ReplayTrial(trial, dict(frozen.intermediate_values), old_rungs)
    replay.set_user_attr("rungs", [{"steps_per_fold": 600,
                                    "by_match": {"match": {"spell_f1_mean": 0.5}}}])
    with pytest.raises(RuntimeError, match="differs from stored score"):
        replay.report(0.5, 600)
    stored = study.get_trials(deepcopy=True)[0]
    assert stored.state == optuna.trial.TrialState.RUNNING
    assert stored.intermediate_values == {600: 0.4}
    assert stored.user_attrs["rungs"] == old_rungs


def test_live_pid_lock_refused(tmp_path):
    lock = tmp_path / "run.lock"
    lock.write_text(f"pid={os.getpid()}\n")
    with pytest.raises(RuntimeError, match="still alive"):
        recovery._dead_lock_pid(lock)


@pytest.mark.parametrize("count", [2, 7, 9, True])
def test_only_eight_trial_protocol_allowed(tmp_path, count):
    with pytest.raises(ValueError, match="exactly eight"):
        recovery._preflight(tmp_path, tmp_path, count)
