"""Serial, resumable joint Optuna development campaign with a 12-hour ceiling.

Run from a frozen source snapshot. Heavy caches/checkpoints belong on T7; the
single-writer Optuna SQLite file belongs on the Mac's internal filesystem.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import time
import traceback
from pathlib import Path

import optuna
import torch
from optuna.distributions import distribution_to_json, json_to_distribution

from experiments.campaign_decode import baseline_decoder
from experiments.campaign_evaluation import (
    load_records,
    predict_records,
    prepare_inputs,
    score_records,
)
from experiments.campaign_protocol import (
    CONFIRMATION_SEEDS,
    OUTER_SEEDS,
    POOLINGS,
    SEARCH_SEED,
    WEIGHTS,
    Budget,
    BudgetExceeded,
    make_protocol,
    paired_gate,
    publish,
    read_record,
)
from experiments.campaign_search import blend_records, tune_decoder
from experiments.campaign_space import baseline_spec, sample_spec
from experiments.campaign_training import build_session, load_training_state, save_training_state
from experiments.robustness import _digest, _lock
from experiments.tune import _seed_for_fold
from possession.predict import load_model_checkpoint
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache

ANCHORS = {
    "mean": ("none", "translation", "coordinate_noise", "time_warp", "frame_hold", "geometry"),
    "ball_attention": ("none", "reflection", "player_dropout", "time_shift", "temporal_blur", "time"),
    "interaction": ("none", "rotation", "spatial_occlusion", "frame_dropout", "sensor", "combined"),
}


def _anchor(round_index, pooling, baseline):
    args = baseline["training_args"]
    return {"depth": 8, "width": 32, "learning_rate": args["learning_rate"],
            "weight_decay": args["weight_decay"], "schedule_kind": "constant",
            "boundary_weight": args["boundary_weight"],
            "boundary_pos_weight": args["boundary_pos_weight"],
            "outage_mode": "correlated" if round_index == 5 else "legacy",
            "augmentation_policy": ANCHORS[pooling][round_index],
            "augmentation_strength": .5, "augmentation_clean_probability": .5,
            "augmentation_max_transforms": 1}


def _proposal(study, pooling, index, baseline, path, total_steps):
    previous = read_record(path)
    if previous is not None:
        if previous["pooling"] != pooling or previous["index"] != index:
            raise ValueError("proposal identity differs")
        return previous
    local = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=SEARCH_SEED + index * 101 + POOLINGS.index(pooling),
        multivariate=True, group=True, n_startup_trials=6))
    local.add_trials(study.get_trials(deepcopy=True))
    if index < 6:
        local.enqueue_trial(_anchor(index, pooling, baseline))
    trial = local.ask()
    spec = sample_spec(trial, pooling)
    spec["schedule"]["total_steps"] = total_steps
    return publish(path, {
        "pooling": pooling, "index": index, "id": f"{pooling}_{index:03d}", "spec": spec,
        "params": trial.params, "distributions": {
            key: distribution_to_json(value) for key, value in trial.distributions.items()}})


def _append(study, proposal, result):
    matching = [trial for trial in study.trials if trial.user_attrs.get("candidate") == proposal["id"]]
    attrs = {"candidate": proposal["id"], "proposal_sha256": _digest(proposal),
             "result_sha256": _digest(result)}
    if matching:
        trial = matching[0]
        if (len(matching) != 1 or trial.user_attrs != attrs or trial.params != proposal["params"]
                or trial.value != result["scores"]["objective"]):
            raise ValueError("Optuna journal disagrees with immutable completed trial")
        return
    study.add_trial(optuna.trial.create_trial(
        params=proposal["params"], distributions={key: json_to_distribution(value)
                                                for key, value in proposal["distributions"].items()},
        value=result["scores"]["objective"], user_attrs=attrs))


def _best(rows, count):
    return sorted(rows, key=lambda row: (-row["scores"]["objective"], row["id"], row["steps"]))[:count]


class Campaign:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(args.data_dir).resolve()
        self.target_dir = Path(args.target_dir).resolve()
        self.baseline = json.loads(Path(args.baseline).read_text())
        self.base_spec = baseline_spec(self.baseline)
        self.base_decoder = baseline_decoder(self.baseline)
        self.protocol = make_protocol(
            self.data_dir, self.target_dir, args.baseline, hours=args.hours,
            per_family=args.per_family, decoder_trials=tuple(args.decoder_trials))
        publish(self.output / "protocol.json", self.protocol)
        self.protocol_sha = _digest(self.protocol)
        self.budget = Budget(self.output, args.hours)
        self.cache = HalfCache(self.data_dir, self.protocol["resources"]["cache_bytes"])
        self.fold = self.protocol["fold"]
        database = Path(args.database).resolve()
        database.parent.mkdir(parents=True, exist_ok=True)
        self.studies = {pooling: optuna.create_study(
            storage=f"sqlite:///{database}", study_name=f"{self.protocol_sha[:16]}-{pooling}",
            direction="maximize", load_if_exists=True) for pooling in POOLINGS}

    def tick(self, fraction=1., **detail):
        self.budget.check(fraction)
        self.budget.status(self.budget.stage, **detail)

    def prepare(self, name, matches, confirmation=False, fraction=1.):
        self.budget.status(f"prepare_{name}")
        return prepare_inputs(
            self.data_dir, self.target_dir, self.output / "evaluation" / name,
            matches, self.protocol["confirmation_scenarios" if confirmation else "search_scenarios"],
            self.cache, progress_callback=lambda: self.tick(fraction))

    def fit(self, name, spec, train_matches, seed, steps, directory, fraction):
        directory = Path(directory)
        identity = {"protocol_sha256": self.protocol_sha, "spec_sha256": _digest(spec),
                    "train_matches": train_matches, "seed": seed, "name": name}
        destination = directory / f"rung_{steps}"
        fit_path, model_path = destination / "fit.json", destination / "model.pt"
        fitted = read_record(fit_path)
        if fitted is not None:
            if fitted["identity"] != identity or fitted["steps"] != steps:
                raise ValueError("fit identity or requested steps differ")
            if sha256_file(model_path) != fitted["checkpoint_sha256"]:
                raise ValueError("completed model checkpoint hash differs")
            return load_model_checkpoint(model_path)[0], fitted
        self.tick(fraction, task=name, seed=seed, steps=0, target_steps=steps)
        started = time.monotonic()
        session = build_session(self.data_dir, spec, train_matches, seed, self.cache)
        states = sorted(directory.glob("state_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
        committed = [path for path in states if path.with_suffix(".pt.sha256.json").is_file()]
        if committed:
            load_training_state(committed[-1], session, identity)
        if session.steps > steps:
            raise ValueError("requested earlier rung lacks its completed checkpoint")
        while session.steps < steps:
            self.tick(fraction, task=name, seed=seed, steps=session.steps, target_steps=steps)
            session.advance_to(min(steps, session.steps + 200))
            state = directory / f"state_{session.steps:06d}.pt"
            save_training_state(state, session, identity)
            # Keep two generations: interrupted publication cannot destroy the previous checkpoint.
            states = sorted(directory.glob("state_*.pt"))
            for obsolete in states[:-2]:
                obsolete.unlink()
                obsolete.with_suffix(".pt.sha256.json").unlink(missing_ok=True)
        destination.mkdir(parents=True, exist_ok=True)
        session.save_checkpoint(model_path)
        fitted = publish(fit_path, {
            "identity": identity, "steps": steps, "checkpoint_sha256": sha256_file(model_path),
            "parameters": sum(parameter.numel() for parameter in session.model.parameters()),
            "invocation_seconds": time.monotonic() - started,
            "progress": session._progress(), "model_path": str(model_path)})
        model = session.model.eval()
        del session
        gc.collect()
        return model, fitted

    def predictions(self, member, seed, inputs, *, scope="inner", fraction=1.):
        folds = self.fold["inner"] if scope == "inner" else [
            {"train": self.fold["outer_train"], "validation": self.fold["outer_test"]}]
        expected_grid = {(match, period, scenario) for fold in folds for match in fold["validation"]
                         for period in (1, 2) for scenario in WEIGHTS}
        actual_grid = [(record["match"], record["period"], record["scenario"]) for record in inputs]
        if len(actual_grid) != len(set(actual_grid)) or set(actual_grid) != expected_grid:
            raise ValueError("validation input grid must contain both halves and all scenarios exactly once")
        records = []
        for index, fold in enumerate(folds):
            if set(fold["train"]) & set(fold["validation"]):
                raise ValueError("train/validation match overlap")
            selected_inputs = [record for record in inputs if record["match"] in fold["validation"]]
            if {record["match"] for record in selected_inputs} != set(fold["validation"]):
                raise ValueError("validation input grid differs from fold")
            fit_dir = self.output / "fits" / scope / member["id"] / f"seed_{seed}" / f"fold_{index}"
            actual_seed = _seed_for_fold(seed, "DFL-MAT-J03WMX", index)
            model, fitted = self.fit(member["id"], member["spec"], fold["train"], actual_seed,
                                     member["steps"], fit_dir, fraction)
            identity = {"protocol_sha256": self.protocol_sha,
                        "checkpoint_sha256": fitted["checkpoint_sha256"],
                        "train_matches": fold["train"], "validation_matches": fold["validation"],
                        "scope": scope, "fold": index, "seed": seed}
            self.budget.status(f"predict_{scope}", task=member["id"], seed=seed, fold=index)
            records.extend(predict_records(
                model, selected_inputs, fit_dir / f"rung_{member['steps']}", identity,
                progress_callback=lambda: self.tick(fraction)))
            del model
        return records

    @property
    def control(self):
        return {"id": "frozen_baseline", "spec": self.base_spec, "steps": self.baseline["steps"]}

    def evaluate_trial(self, proposal, steps, trials, inputs, fraction, reference_decoder=None):
        member = {"id": proposal["id"], "spec": proposal["spec"], "steps": steps}
        path = self.output / "trials" / member["id"] / f"rung_{steps}"
        previous = read_record(path / "result.json")
        if previous is not None:
            if previous["proposal_sha256"] != _digest(proposal):
                raise ValueError("trial proposal changed")
            return previous
        records = self.predictions(member, SEARCH_SEED, inputs, fraction=fraction)
        self.budget.status("decoder_search", task=member["id"], steps=steps, trials=trials)
        anchors = () if reference_decoder is None else (reference_decoder,)
        tuned = tune_decoder(
            records, self.base_decoder, path / "decoder", trials,
            SEARCH_SEED + proposal["index"], extra_anchors=anchors,
            check=lambda: self.tick(fraction))
        return publish(path / "result.json", {
            **member, **tuned, "proposal_sha256": _digest(proposal), "records": records})

    def screening(self, inputs, reference_decoder):
        path = self.output / "screening.json"
        previous = read_record(path)
        if previous is not None:
            return previous
        rows = []
        closed_reason = "configured_rounds_complete"
        rungs = self.protocol["resources"]["rungs"]
        for index in range(self.args.per_family):
            completed_round = []
            try:
                for pooling in POOLINGS:
                    proposal = _proposal(
                        self.studies[pooling], pooling, index, self.baseline,
                        self.output / "proposals" / f"{pooling}_{index:03d}.json", rungs[-1])
                    result = self.evaluate_trial(proposal, rungs[0], self.args.decoder_trials[0],
                                                 inputs, .45, reference_decoder)
                    _append(self.studies[pooling], proposal, result)
                    completed_round.append(result)
            except BudgetExceeded:
                closed_reason = "screen_wall_clock_cutoff"
                break
            rows.extend(completed_round)
        return publish(path, {"rows": rows, "complete_rounds": len(rows) // 3,
                              "requested_rounds": self.args.per_family,
                              "closed_reason": closed_reason,
                              "incomplete_round_excluded": True})

    def promote(self, rows, inputs, reference_decoder, *, stage, top_count, rung_index, fraction):
        path = self.output / f"{stage}.json"
        previous = read_record(path)
        if previous is not None:
            return previous["rows"]
        groups = {pooling: _best([row for row in rows if row["spec"]["model_kwargs"]["pooling"] == pooling],
                                 top_count) for pooling in POOLINGS}
        if stage == "middle":
            for pooling in POOLINGS:
                anchor = next((row for row in rows if row["id"] == f"{pooling}_000"), None)
                if anchor is not None:
                    others = [row for row in rows if row["spec"]["model_kwargs"]["pooling"] == pooling
                              and row["id"] != anchor["id"]]
                    groups[pooling] = [anchor] + _best(others, top_count - 1)
        ordered = [groups[pooling][index] for index in range(top_count) for pooling in POOLINGS
                   if index < len(groups[pooling])]
        results = []
        closed_reason = "configured_promotions_complete"
        for rank in range(top_count):
            if any(len(groups[pooling]) <= rank for pooling in POOLINGS):
                break
            complete_round = []
            try:
                for pooling in POOLINGS:
                    row = groups[pooling][rank]
                    proposal = read_record(self.output / "proposals" / f"{row['id']}.json")
                    result = self.evaluate_trial(
                        proposal, self.protocol["resources"]["rungs"][rung_index],
                        self.args.decoder_trials[rung_index], inputs, fraction, reference_decoder)
                    complete_round.append(result)
            except BudgetExceeded:
                closed_reason = "stage_wall_clock_cutoff"
                break
            results.extend(complete_round)
        publish(path, {"rows": results, "requested_ids": [row["id"] for row in ordered],
                       "closed_reason": closed_reason, "complete_rounds": len(results) // 3,
                       "planned_stage_completed": len(results) == len(ordered),
                       "incomplete_round_excluded": True})
        return results

    def select(self, rows, reference_result):
        path = self.output / "selection.json"
        previous = read_record(path)
        if previous is not None:
            return previous
        if not rows:
            raise BudgetExceeded("no middle-rung configuration completed; no selection permitted")
        ordered = _best(rows, len(rows))
        winner = ordered[0]
        selection = {"kind": "single", "members": [{key: winner[key] for key in ("id", "spec", "steps")}],
                     "decoder": winner["decoder"], "scores": winner["scores"], "blend_weight": None,
                     "frozen_before_confirmation": True, "automatic_deployment": False}
        # A simple cross-fitted convex blend has few degrees of freedom. Never fit a
        # flexible stacking network to these six selection matches.
        runner_up = next((row for row in ordered if row["id"] != winner["id"]), None)
        blend_trials = []
        if runner_up is not None and self.budget.seconds_left(.75) > 300:
            left, right = load_records(winner["records"]), load_records(runner_up["records"])
            for weight in self.protocol["selection"]["pair_blend_weights"]:
                if self.budget.seconds_left(.75) <= 0:
                    break
                score = score_records(blend_records(left, right, weight), winner["decoder"], WEIGHTS)
                blend_trials.append({"weight": weight, "scores": score})
            best = max(blend_trials, key=lambda row: row["scores"]["objective"]) if blend_trials else None
            if (len(blend_trials) == 3
                    and best["scores"]["objective"] > winner["scores"]["objective"] + .005):
                selection.update(kind="convex_pair", blend_weight=best["weight"], scores=best["scores"],
                                 members=selection["members"] + [{key: runner_up[key]
                                                                for key in ("id", "spec", "steps")}])
            del left, right
            gc.collect()
        if reference_result["scores"]["objective"] >= selection["scores"]["objective"] - 1e-12:
            selection.update(kind="reference_only", members=[], blend_weight=None,
                             decoder=reference_result["decoder"], scores=reference_result["scores"])
        publish(self.output / "convex_blend_search.json", {
            "trials": blend_trials, "minimum_gain": .005,
            "baseline_id": winner["id"], "second_id": None if runner_up is None else runner_up["id"]})
        return publish(path, selection)

    def selected_score(self, selection, seed, inputs, scope):
        members = selection["members"] or [self.control]
        records = [load_records(self.predictions(member, seed, inputs, scope=scope)) for member in members]
        combined = records[0] if len(records) == 1 else blend_records(
            records[0], records[1], selection["blend_weight"])
        return score_records(combined, selection["decoder"], WEIGHTS)

    def confirmation(self, selection):
        path = self.output / "confirmation.json"
        previous = read_record(path)
        if previous is not None:
            return previous
        inputs = self.prepare("confirmation", self.fold["outer_train"], confirmation=True)
        baselines, candidates = [], []
        for seed in CONFIRMATION_SEEDS:
            self.budget.status("confirmation", seed=seed)
            for role in ("baseline", "candidate"):
                score_path = self.output / "confirmation" / f"{role}_{seed}.json"
                score = read_record(score_path)
                if score is None:
                    config = selection if role == "candidate" else {
                        "members": [self.control], "decoder": self.base_decoder}
                    score = self.selected_score(config, seed, inputs, "inner")
                    publish(score_path, score)
                (baselines if role == "baseline" else candidates).append(score)
        return publish(path, {"seeds": list(CONFIRMATION_SEEDS), "baseline": baselines,
                              "candidate": candidates, "gate": paired_gate(baselines, candidates),
                              "selection_sha256": _digest(selection)})

    def outer(self, selection):
        path = self.output / "outer.json"
        previous = read_record(path)
        if previous is not None:
            return previous
        inputs = self.prepare("outer", self.fold["outer_test"], confirmation=True)
        baselines, candidates = [], []
        for seed in OUTER_SEEDS:
            for role in ("baseline", "candidate"):
                score_path = self.output / "outer" / f"{role}_{seed}.json"
                score = read_record(score_path)
                if score is None:
                    config = selection if role == "candidate" else {
                        "members": [self.control], "decoder": self.base_decoder}
                    score = self.selected_score(config, seed, inputs, "outer")
                    publish(score_path, score)
                (baselines if role == "baseline" else candidates).append(score)
        delta = sum(c["objective"] - b["objective"] for b, c in zip(baselines, candidates, strict=True)) / 2
        return publish(path, {"seeds": list(OUTER_SEEDS), "baseline": baselines,
                              "candidate": candidates, "objective_delta": delta,
                              "gate_passed": delta >= -.01, "used_for_tuning": False,
                              "historically_reused_match": True})

    def run(self):
        done = read_record(self.output / "summary.json")
        if done is not None:
            return done
        inputs = self.prepare("search", self.fold["outer_train"], fraction=.45)
        self.budget.status("frozen_baseline_control")
        control_records = self.predictions(self.control, SEARCH_SEED, inputs, fraction=.45)
        baseline_path = self.output / "baseline.json"
        baseline_scores = read_record(baseline_path)
        if baseline_scores is None:
            baseline_scores = publish(baseline_path, score_records(control_records, self.base_decoder, WEIGHTS))
        self.budget.status("reference_only_search")
        reference = tune_decoder(
            control_records, self.base_decoder, self.output / "reference_only", 64, SEARCH_SEED,
            reference_only=True, check=lambda: self.tick(.45))
        screen = self.screening(inputs, reference["decoder"])
        if screen["complete_rounds"] < 6:
            raise BudgetExceeded("fewer than six complete architecture rounds; breadth requirement unmet")
        middle = self.promote(screen["rows"], inputs, reference["decoder"], stage="middle",
                              top_count=3, rung_index=1, fraction=.65)
        final = self.promote(middle, inputs, reference["decoder"], stage="final",
                             top_count=1, rung_index=2, fraction=.75)
        selection = self.select(middle + final, reference)
        confirmation = self.confirmation(selection)
        outer = self.outer(selection)
        eligible = confirmation["gate"]["passed"] and outer["gate_passed"]
        summary = publish(self.output / "summary.json", {
            "status": "complete", "development_only": True, "protocol_sha256": self.protocol_sha,
            "screen_rounds": screen["complete_rounds"], "middle_completed": len(middle),
            "final_completed": len(final), "baseline_search_scores": baseline_scores,
            "reference_only": reference, "selection": selection,
            "confirmation": confirmation, "outer": outer,
            "candidate_eligible_for_review": eligible, "existing_deployment_changed": False,
            "recommended": "candidate" if eligible else "frozen_baseline",
            "elapsed_seconds": time.time() - self.budget.clock["started_unix"]})
        self.report(summary)
        write_json(self.output / "status.json", {
            "status": "complete", "summary_path": str(self.output / "summary.json"),
            "recommended": summary["recommended"], **self.budget.clock, "updated_unix": time.time()})
        return summary

    def report(self, summary):
        gate = summary["confirmation"]["gate"]
        lines = ["# Overnight joint tuning results", "", "Development evidence; all public matches were reused.", "",
                 f"Recommended configuration: **{summary['recommended']}**.",
                 (f"Completed screening: {summary['screen_rounds'] * 3} configurations; "
                  f"middle rung: {summary['middle_completed']}; final rung: {summary['final_completed']}."), "",
                 "| Confirmation comparison | Candidate minus frozen baseline |", "|---|---:|",
                 f"| Weighted spell F1 | {gate['objective_delta']:+.5f} |",
                 f"| Missing-ball spell F1 | {gate['missing_delta']:+.5f} |"]
        lines += [f"| {name} spell F1 | {value:+.5f} |" for name, value in gate["scenario_deltas"].items()]
        lines += ["", f"Confirmation gates: {gate['checks']}", "",
                  f"WMX development holdout objective delta: {summary['outer']['objective_delta']:+.5f}.", "",
                  "The frozen baseline and submission remain unchanged. Candidate settings are in selection.json.",
                  "A single Mac campaign cannot establish a global optimum or state-of-the-art accuracy."]
        (self.output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--hours", type=float, default=12.)
    parser.add_argument("--per-family", type=int, default=12)
    parser.add_argument("--decoder-trials", type=int, nargs=3, default=(12, 32, 64))
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _lock(output):
        campaign = Campaign(args)
        # The OS alarm remains active during native operations. Training publishes
        # two checkpoint generations every 200 steps so interruption is recoverable.
        def alarm_handler(_signum, _frame):
            raise BudgetExceeded("12-hour wall-clock deadline reached")
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, max(.01, campaign.budget.seconds_left()))
        try:
            campaign.run()
        except (BudgetExceeded, KeyboardInterrupt) as exc:
            write_json(output / "status.json", {
                "status": "budget_exhausted" if isinstance(exc, BudgetExceeded) else "interrupted",
                "stage": campaign.budget.stage, "reason": str(exc), "pid": os.getpid(),
                **campaign.budget.clock, "updated_unix": time.time(),
                "existing_deployment_changed": False, "candidate_eligible_for_review": False})
            return 2
        except Exception as exc:
            write_json(output / "status.json", {
                "status": "failed", "stage": campaign.budget.stage, "reason": repr(exc),
                "traceback": traceback.format_exc(), "pid": os.getpid(),
                **campaign.budget.clock, "updated_unix": time.time()})
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
