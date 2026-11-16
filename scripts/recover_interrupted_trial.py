"""Replay exactly one orphaned Optuna HPO trial from optimizer step zero.

Default mode is read-only preflight. ``--execute`` requires a dead PID in the
existing run.lock, preserves the RUNNING trial number and parameters, and never
creates or fails a trial. Only Optuna 5.0.0 is supported: the documented Trial
constructor needs the stored trial ID, which FrozenTrial exposes as _trial_id.
No SQLite rows are edited directly. Run only one recovery or normal study
worker per outer directory; the dead-lock handoff is not a concurrency mutex.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import optuna

from experiments import tune
from possession.storage import write_json
from possession.training_session import HalfCache

SUPPORTED_OPTUNA = "5.0.0"
PARAMETERS = frozenset(("pooling", "width", "learning_rate", "weight_decay",
                        "boundary_pos_weight", "switch_cost"))


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _dead_lock_pid(path: Path) -> tuple[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"orphaned study must retain run.lock: {path}")
    content = path.read_text(encoding="utf-8")
    found = re.fullmatch(r"pid=([1-9]\d*)\n?", content)
    _require(found is not None, f"malformed study PID lock: {path}")
    pid = int(found.group(1))
    try:
        os.kill(pid, 0)  # Existence check only. Never sends a terminating signal.
    except ProcessLookupError:
        return pid, content
    except PermissionError as exc:
        raise RuntimeError(f"lock PID {pid} exists but is not inspectable") from exc
    raise RuntimeError(f"lock PID {pid} is still alive; stop it outside this script")


def _validate_params(params: dict, poolings: list[str], search_space: dict) -> None:
    _require(isinstance(params, dict) and set(params) == PARAMETERS,
             "RUNNING trial lacks the complete persisted HPO parameter set")
    _require(params["pooling"] in poolings
             and params["width"] in search_space["width"],
             "RUNNING trial architecture differs from fingerprint search space")
    for name in ("learning_rate", "weight_decay", "boundary_pos_weight", "switch_cost"):
        value = params[name]
        bounds = search_space[name]
        _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(value) and bounds[0] <= value <= bounds[1],
                 f"RUNNING trial {name} differs from fingerprint search space")


def _preflight(data_dir: Path, outer_dir: Path, total_trials: int) -> dict:
    if optuna.__version__ != SUPPORTED_OPTUNA:
        raise RuntimeError(f"recovery is pinned to Optuna {SUPPORTED_OPTUNA}; "
                           f"found {optuna.__version__}")
    if (outer_dir / "frozen_recipe.json").exists():
        raise RuntimeError("outer recipe is already frozen; refusing recovery")
    if not isinstance(total_trials, int) or isinstance(total_trials, bool) or total_trials != 8:
        raise ValueError("recovery requires exactly eight total trials")
    outer = outer_dir.name
    fingerprint = _read(outer_dir / "fingerprint.json")
    split = _read(outer_dir / "splits.json")
    summary_path = outer_dir / "summary.json"
    summary = _read(summary_path) if summary_path.is_file() else None
    payload = fingerprint.get("payload", {})
    _require(payload.get("outer_test") == [outer]
             and payload.get("split") == split,
             "outer split, summary, or HPO fingerprint differs")
    matches = split.get("matches")
    _require(matches == sorted(tune.DEFAULT_MATCHES)
             and len(split.get("outer_folds", [])) == len(matches),
             "recovery requires the fixed seven-match audit split")
    folds = [fold for fold in split["outer_folds"] if fold.get("outer_test") == [outer]]
    _require(len(folds) == 1, "outer fold is missing or duplicated")
    fold = folds[0]
    _require(sorted(fold["outer_train"]) == sorted(set(matches) - {outer}),
             "outer train or inner fold differs from study summary")
    if summary is not None:
        _require(summary.get("outer_test") == [outer]
                 and summary.get("fingerprint_sha256") == fingerprint.get("sha256")
                 and summary.get("outer_train") == fold.get("outer_train")
                 and summary.get("inner") == fold.get("inner"),
                 "existing summary split or fingerprint differs")
    resources = payload.get("resources", {})
    rungs = resources.get("rungs")
    poolings = payload.get("search_space", {}).get("pooling")
    _require(rungs == [600, 1200] and poolings == ["mean", "ball_attention"]
             and payload.get("seed") == 0 and payload.get("pruning") is False,
             "recovery requires the frozen eight-trial 600/1200 no-pruning audit protocol")
    if summary is not None:
        _require(summary.get("full_budget_steps") == rungs[-1],
                 "summary full budget differs from fingerprint")
    current = tune._fingerprint(
        data_dir, fold, split, seed=payload["seed"], rungs=rungs,
        device=resources["device"], poolings=poolings, pruning=payload["pruning"],
        center_size=resources["center_size"], cache_bytes=resources["cache_bytes"])
    _require(current == fingerprint, "current source, data, versions, or thread settings differ")
    lock_pid, lock_content = _dead_lock_pid(outer_dir / "run.lock")
    database = outer_dir / "study.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(database)
    study = optuna.load_study(
        study_name=f"nested-{outer}", storage=f"sqlite:///{database}",
        sampler=optuna.samplers.TPESampler(seed=payload["seed"], multivariate=True,
                                            n_startup_trials=3),
        pruner=optuna.pruners.NopPruner())
    _require(study.user_attrs.get("fingerprint_sha256") == fingerprint["sha256"],
             "stored study fingerprint differs")
    trials = study.get_trials(deepcopy=False)
    running = [trial for trial in trials if trial.state == optuna.trial.TrialState.RUNNING]
    _require(len(running) == 1, "study must contain exactly one RUNNING trial")
    orphan = running[0]
    _require(orphan.number < total_trials and len(trials) <= total_trials,
             "RUNNING trial exceeds fixed total trial budget")
    _validate_params(orphan.params, poolings, payload["search_space"])
    baseline = (orphan.number < len(poolings)
                and orphan.params == tune._baseline(poolings[orphan.number]))
    _require(orphan.user_attrs.get("baseline_full_budget") == baseline,
             "RUNNING trial baseline provenance differs")
    old_intermediate = dict(orphan.intermediate_values)
    _require(set(old_intermediate).issubset(set(rungs))
             and all(math.isfinite(score) for score in old_intermediate.values()),
             "RUNNING trial has unexpected or nonfinite reported steps")
    old_reports = orphan.user_attrs.get("rungs", [])
    _require(isinstance(old_reports, list)
             and all(report.get("steps_per_fold") in rungs for report in old_reports),
             "RUNNING trial diagnostics have invalid rungs")
    _require(len(old_reports) <= len(rungs), "RUNNING trial has duplicate rung diagnostics")
    completed = [trial for trial in trials
                 if trial.state == optuna.trial.TrialState.COMPLETE]
    _require(all(trial.state in (optuna.trial.TrialState.COMPLETE,
                                 optuna.trial.TrialState.WAITING,
                                 optuna.trial.TrialState.RUNNING) for trial in trials),
             "study has failed or pruned trials outside this recovery protocol")
    if summary is None:
        _require(not completed, "completed trials lack a durable study summary")
    else:
        prior = summary.get("trials", [])
        _require(summary.get("n_trials_total") == len(completed)
                 and {item.get("number") for item in prior}
                    == {item.number for item in completed}
                 and all(any(item.get("number") == trial.number
                             and item.get("state") == "COMPLETE"
                             and item.get("value") == trial.value
                             and item.get("params") == trial.params
                             for item in prior) for trial in completed),
                 "summary does not match prior completed trials")
    artifact = outer_dir / "trials" / f"trial_{orphan.number}"
    if artifact.exists():
        raise FileExistsError(f"orphan trial artifact collision: {artifact}")
    running_id = getattr(orphan, "_trial_id", None)
    _require(isinstance(running_id, int) and running_id > 0,
             "Optuna FrozenTrial storage ID is unavailable")
    return {
        "outer": outer, "fold": fold, "split": split, "fingerprint": fingerprint,
        "payload": payload, "resources": resources, "rungs": rungs,
        "study": study, "orphan": orphan, "running_id": running_id,
        "baseline": baseline, "old_intermediate": old_intermediate,
        "old_reports": old_reports, "lock_pid": lock_pid, "lock_content": lock_content,
        "artifact": artifact, "completed": completed, "total_trials": total_trials,
    }


class _ReplayTrial:
    """Compare old rung scores before allowing a persisted diagnostic update."""

    def __init__(self, trial, old_intermediate, old_reports):
        self._trial = trial
        self._old_intermediate = old_intermediate
        self._old_reports = {report["steps_per_fold"]: report for report in old_reports}
        self._pending_rungs = None
        self.checked_steps = []

    @property
    def number(self):
        return self._trial.number

    def set_user_attr(self, name, value):
        if name != "rungs":
            raise ValueError(f"unexpected replay trial attribute: {name}")
        self._pending_rungs = value

    def report(self, value, step):
        if self._pending_rungs is None or self._pending_rungs[-1]["steps_per_fold"] != step:
            raise AssertionError("rung diagnostic and reported step differ")
        old_score = self._old_intermediate.get(step)
        if old_score is not None and not math.isclose(
                float(value), old_score, rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError(f"replayed step {step} score {value} differs from "
                               f"stored score {old_score}; trial stays RUNNING")
        old_report = self._old_reports.get(step)
        if old_report is not None:
            new_report = self._pending_rungs[-1]
            old_by_match = old_report.get("by_match", {})
            new_by_match = new_report.get("by_match", {})
            if set(old_by_match) != set(new_by_match) or any(
                    not math.isclose(old_by_match[match]["spell_f1_mean"],
                                     new_by_match[match]["spell_f1_mean"],
                                     rel_tol=1e-10, abs_tol=1e-12)
                    for match in old_by_match):
                raise RuntimeError(f"replayed step {step} match scores differ; "
                                   "trial stays RUNNING")
        if old_score is None:
            self._trial.report(value, step)
        self._trial.set_user_attr("rungs", self._pending_rungs)
        self.checked_steps.append(step)
        self._pending_rungs = None

    def should_prune(self):
        return self._trial.should_prune()


def _replay(context: dict, data_dir: Path, outer_dir: Path) -> dict:
    """Train from zero, compare old rungs, then complete this same trial ID."""
    payload, resources = context["payload"], context["resources"]
    study, orphan = context["study"], context["orphan"]
    trial = optuna.trial.Trial(study, context["running_id"])
    _require(trial.number == orphan.number and trial.params == orphan.params,
             "reopened Optuna trial ID does not identify the orphan")
    replay = _ReplayTrial(trial, context["old_intermediate"], context["old_reports"])
    cache = HalfCache(data_dir, max_bytes=resources["cache_bytes"])
    params = orphan.params
    score, sessions, records, _reports = tune._run_trial(
        replay, context["fold"], data_dir, cache, params=params,
        rungs=context["rungs"], seed=payload["seed"], device=resources["device"],
        center_size=resources["center_size"], pruning=False)
    _require(replay.checked_steps == context["rungs"],
             "replay did not reach every fixed resource rung")
    current = tune._fingerprint(
        data_dir, context["fold"], context["split"], seed=payload["seed"],
        rungs=context["rungs"], device=resources["device"],
        poolings=payload["search_space"]["pooling"], pruning=False,
        center_size=resources["center_size"], cache_bytes=resources["cache_bytes"])
    _require(current == context["fingerprint"],
             "source, data, versions, or thread settings changed during replay")
    trial.set_user_attr("full_budget_steps", context["rungs"][-1])
    old_family = [item for item in study.trials
                  if item.state == optuna.trial.TrialState.COMPLETE
                  and item.user_attrs.get("full_budget_steps") == context["rungs"][-1]
                  and item.params.get("pooling") == params["pooling"]]
    saved = False
    if not old_family or score > max(item.value for item in old_family):
        tune._save_trial_artifacts(
            context["artifact"], trial=trial, fold=context["fold"],
            sessions=sessions, records=records, params=params, score=score,
            rungs=context["rungs"], center_size=resources["center_size"],
            seed=payload["seed"], fingerprint_sha256=context["fingerprint"]["sha256"])
        saved = True
    study.tell(trial, score)
    summary = tune._summary(study, context["fold"], outer_dir,
                            fingerprint=context["fingerprint"],
                            rungs=context["rungs"], center_size=resources["center_size"],
                            seed=payload["seed"])
    _require(summary["trials"][-1]["number"] == orphan.number
             and summary["trials"][-1]["state"] == "COMPLETE"
             and summary["trials"][-1]["value"] == score,
             "completed trial is missing from regenerated summary")
    return {"trial_number": orphan.number, "score": score,
            "checked_rungs": replay.checked_steps,
            "winning_artifact_saved": saved,
            "completed_trials": summary["n_complete_full_budget"]}


def recover(data_dir, outer_dir, *, execute=False, total_trials=8) -> dict:
    """Inspect an orphan read-only, or explicitly replay it in-place."""
    data_dir, outer_dir = Path(data_dir).resolve(), Path(outer_dir).resolve()
    context = _preflight(data_dir, outer_dir, total_trials)
    orphan = context["orphan"]
    plan = {"outer_test": context["outer"], "trial_number": orphan.number,
            "trial_id": context["running_id"], "params": orphan.params,
            "reported_steps": sorted(context["old_intermediate"]),
            "discarded_partial_resource": "optimizer and model state are replayed from step zero",
            "dead_lock_pid": context["lock_pid"],
            "fingerprint_sha256": context["fingerprint"]["sha256"],
            "remaining_total_trials": total_trials - len(context["completed"]) - 1,
            "execute": bool(execute)}
    if not execute:
        return plan
    lock = outer_dir / "run.lock"
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archive = outer_dir / f"run.lock.orphan.trial_{orphan.number}.{now}"
    if archive.exists():
        raise FileExistsError(f"orphan lock archive already exists: {archive}")
    checked_pid, checked_content = _dead_lock_pid(lock)
    _require(checked_pid == context["lock_pid"]
             and checked_content == context["lock_content"],
             "orphan PID lock changed since preflight")
    os.replace(lock, archive)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except Exception:
        if not lock.exists():
            os.replace(archive, lock)
        raise
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\n")
    record_path = outer_dir / f"recovery_trial_{orphan.number}_{now}.json"
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record = {"status": "replaying", "plan": plan,
              "recovery_script_sha256": script_sha256,
              "original_lock": context["lock_content"],
              "original_trial": {
                  "state": orphan.state.name, "number": orphan.number,
                  "params": orphan.params, "intermediate_values": context["old_intermediate"],
                  "user_attrs": orphan.user_attrs,
                  "datetime_start": (orphan.datetime_start.isoformat()
                                     if orphan.datetime_start is not None else None),
              }}
    write_json(record_path, record)
    try:
        result = _replay(context, data_dir, outer_dir)
        record.update({"status": "complete", "result": result})
        write_json(record_path, record)
        lock.unlink()
        return {**plan, **result, "recovery_record": str(record_path)}
    except Exception as exc:
        record.update({"status": "failed_trial_still_running_or_needs_review",
                       "error_type": type(exc).__name__, "error": str(exc)})
        write_json(record_path, record)
        # Keep run.lock for an explicit, later review. The current PID becomes
        # dead when this process exits; normal run_study still refuses the trial.
        raise


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("outer_dir", type=Path)
    parser.add_argument("--total-trials", type=int, default=8)
    parser.add_argument("--execute", action="store_true",
                        help="replay and complete the existing RUNNING trial")
    args = parser.parse_args(argv)
    print(json.dumps(recover(args.data_dir, args.outer_dir, execute=args.execute,
                             total_trials=args.total_trials),
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
