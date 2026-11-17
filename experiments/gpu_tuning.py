"""Eight-hour, resumable GPU search with a frozen final confirmation recipe.

Only this campaign's own rolling checkpoints may be pruned. Existing datasets,
experiments and release artifacts are read-only inputs.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import json
import os
import shutil
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.distributions import distribution_to_json, json_to_distribution

from experiments.campaign_protocol import Budget, BudgetExceeded, publish, read_record
from experiments.gpu_runtime import accelerated_preprocessing, configure_device
from experiments.gpu_tuning_evaluation import (
    Bank,
    _blend,
    average_scores,
    comparison,
    evaluate,
    score_labels,
)
from experiments.gpu_tuning_space import (
    anchor_recipe,
    pipeline_anchor,
    sample_pipeline,
    sample_recipe,
)
from experiments.gpu_tuning_training import check_file, fit
from experiments.possession_refinement import CAVEAT, validate_folds
from experiments.robustness import _digest
from experiments.tune import _seed_for_fold
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache

SEARCH_SEED = 9301
CONFIRMATION_SEEDS = (9401, 9402)
GUARD_SEEDS = (9501, 9502)
STOP = False


def stop_requested(signum, frame):
    global STOP
    STOP = True


def freeze(repo, output, hours):
    previous = read_record(output / "protocol.json")
    if previous:
        if previous["hours"] != hours:
            raise ValueError("resume cannot extend the frozen budget")
        for name, digest in previous["upstream_sha256"].items():
            check_file(repo / name, digest)
        return previous
    followup_path = repo / "artifacts/gpu-iou-followup-v1/protocol.json"
    followup = read_record(followup_path)
    plan = followup["arms"]["confirmation"]
    validate_folds(plan["folds"])
    receipt_path = repo / "artifacts/gpu-iou-followup-v1/completion-verification.json"
    if not read_record(receipt_path):
        raise ValueError("verified completed follow-up required")
    files = ["models/v6/manifest.json", str(followup_path.relative_to(repo)),
             str(receipt_path.relative_to(repo)), "data/processed/dfl/manifest.json"]
    files = [p for p in files if (repo / p).exists()]
    for arm in ("confirmation", "guard"):
        files.extend(f"artifacts/gpu-iou-followup-v1/{arm}/{name}"
                     for name in ("training.json", "result.json"))
    sources = {}
    runtime = Path(__file__).resolve().parents[1]
    for directory in ("possession", "experiments", "evaluation", "scripts"):
        sources.update({str(p.relative_to(runtime)): sha256_file(p)
                        for p in (runtime / directory).rglob("*.py")})
    sources["games.py"] = sha256_file(runtime / "games.py")
    frozen = {
        "schema": "holistic_gpu_tuning_v1", "hours": hours, "created_unix": time.time(),
        "repo": str(repo), "runtime": str(runtime), "source_sha256": sources,
        "upstream_sha256": {p: sha256_file(repo / p) for p in files},
        "data_dir": plan["data_dir"], "folds": plan["folds"],
        "incumbent": "verified combined fine-seed ensemble 8201/8202 with existing coarse models",
        "guard_incumbent": "verified WMX ensemble 9201/9202",
        "members": plan["selection"]["members"], "decoder": plan["selection"]["decoder"],
        "search_seed": SEARCH_SEED, "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "guard_seeds": list(GUARD_SEEDS), "rungs": [1800, 3600, "recipe total: 7200 or 9600"],
        "cutoffs": {"screen": .28, "promote": .52, "polish": .62,
                    "confirmation": .90, "guard": .98, "stability": .98, "report": 1.},
        "search": "seeded TPE; whole-match OOF on all three fixed folds; batch_1 only",
        "selection": "best completed full-budget recipe/pipeline, with decoder-only incumbent eligible; "
                     "incomplete fits never win; no selection using fresh seeds, batch_2 or WMX",
        "confirmation": "same selected recipe, two fresh candidate seeds and both existing mask batches; "
                        "incumbent components remain fixed; same-six-match conditional replication",
        "gate": {"ensemble_gain_gt": .002, "both_single_seed_gains_gt": 0.,
                 "both_batch_gains_gt": 0., "positive_matches_at_least": 4,
                 "missing_gain_gt": 0., "scenario_loss_at_most": .01,
                 "guard_objective_and_scenario_loss_at_most": .01},
        "diagnostics": "IoUs .1-.9; short-possession recall and spurious counts at clean IoU .5; "
                       "decoder-only decomposition; whole-match descriptive bootstrap, never a search p-value",
        "remaining_budget": "After confirmation and WMX, run additional candidate seeds starting at 9601 "
                            "until the 98% deadline. These diagnostics cannot change selection or the gate. "
                            "Skip when the winner uses no new neural model. Seed checks are inapplicable "
                            "to a decoder-only winner; its mask and match checks still apply.",
        "resources": {"device": "cuda:0", "precision": "float32", "deterministic": True,
                      "workers": 1, "cpu_threads": 1, "cache_bytes": 1_600_000_000,
                      "minimum_free_bytes": 650_000_000,
                      "retention": "one committed resumable state per completed fit; two during updates; "
                                   "inference in memory, final decoded labels retained"},
        "environment": {"torch": torch.__version__, "numpy": np.__version__,
                        "optuna": optuna.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0)},
        "caveat": CAVEAT, "automatic_promotion": False, "target_ground_truth_used": False,
    }
    return publish(output / "protocol.json", frozen)


def study_from_rows(rows, seed):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=8, multivariate=True, group=True))
    for row in rows:
        if not row.get("params"):
            continue
        study.add_trial(optuna.trial.create_trial(
            params=row["params"], distributions={k: json_to_distribution(v)
                                                for k, v in row["distributions"].items()},
            value=row["value"]))
    return study


def ranking(rows):
    return sorted(rows, key=lambda r: (-r["scores"]["objective"], r["id"]))


class Campaign:
    def __init__(self, repo, output, hours):
        self.repo, self.output = repo, output
        self.plan = freeze(repo, output, hours)
        self.sha = _digest(self.plan)
        self.budget = Budget(output, hours)
        self.cache = HalfCache(Path(self.plan["data_dir"]), 1_600_000_000)
        self.bank = None
        self.stage = "starting"
        self.fraction = 1.
        self.detail = {}
        self.last_status = 0.
        self.base_pipeline = pipeline_anchor(self.plan["decoder"], 0.)

    def tick(self, **detail):
        if STOP:
            raise InterruptedError("termination requested; committed checkpoint retained")
        self.budget.check(self.fraction)
        if shutil.disk_usage(self.output).free < self.plan["resources"]["minimum_free_bytes"]:
            raise OSError("campaign disk reserve reached; existing artifacts preserved")
        self.detail.update(detail)
        if time.time() - self.last_status >= 5:
            self.budget.status(self.stage, **self.detail)
            self.last_status = time.time()

    def phase(self, stage):
        self.stage = stage
        self.fraction = self.plan["cutoffs"][stage]
        self.detail = {}
        self.last_status = 0

    def source_check(self):
        for name, digest in self.plan["source_sha256"].items():
            check_file(Path(self.plan["runtime"]) / name, digest)
        for name, digest in self.plan["upstream_sha256"].items():
            check_file(self.repo / name, digest)

    def get_bank(self, scope="inner", batch="batch_1"):
        if self.bank is None or (self.bank.scope, self.bank.batch) != (scope, batch):
            self.detail = {"scope": scope, "batch": batch, "task": "load_assessment_inputs"}
            self.bank = None
            gc.collect()
            self.bank = Bank(self.repo, self.output, scope, batch, self.tick)
        return self.bank

    def proposal(self, index):
        path = self.output / "proposals" / f"trial_{index:04d}.json"
        old = read_record(path)
        if old:
            return old
        if index < 4:
            recipe = anchor_recipe(self.plan["members"][index % 2], balanced=index < 2)
            params, distributions = {}, {}
        else:
            history = []
            for file in sorted((self.output / "screen").glob("trial_*/result.json")):
                row = read_record(file)
                proposal = read_record(self.output / "proposals" / f"{row['id']}.json")
                history.append({**proposal, "value": row["scores"]["objective"]})
            study = study_from_rows(history, 20261117 + index * 101)
            trial = study.ask()
            recipe = sample_recipe(trial)
            params = trial.params
            distributions = {k: distribution_to_json(v) for k, v in trial.distributions.items()}
        return publish(path, {"id": f"trial_{index:04d}", "index": index, "recipe": recipe,
                              "params": params, "distributions": distributions,
                              "protocol_sha256": self.sha})

    def fits(self, proposal, steps, seed, scope="inner"):
        self.source_check()
        folds = self.plan["folds"] if scope == "inner" else [{
            "train": sorted({m for f in self.plan["folds"] for m in f["validation"]}),
            "validation": ["DFL-MAT-J03WMX"]}]
        result = []
        for index, fold in enumerate(folds):
            self.detail = {"candidate": proposal["id"], "seed": seed, "fold": index, "scope": scope}
            directory = self.output / "fits" / scope / proposal["id"] / f"seed_{seed}" / f"fold_{index}"
            result.append(fit(
                self.plan["data_dir"], proposal["recipe"], fold["train"], fold["validation"],
                _seed_for_fold(seed, "DFL-MAT-J03WMX", index), steps, directory,
                {"protocol_sha256": self.sha, "proposal_sha256": _digest(proposal),
                 "scope": scope, "fold": index, "base_seed": seed}, self.cache, self.tick))
        return result

    def quick_evaluate(self, proposal, steps, destination):
        path = destination / "result.json"
        old = read_record(path)
        if old:
            return old
        fitted = self.fits(proposal, steps, SEARCH_SEED)
        bank = self.get_bank()
        predictions = bank.predict(fitted, self.tick)
        candidates = []
        for weight in (.25, .5, .75, 1.):
            self.tick(decoder_weight=weight)
            pipeline = pipeline_anchor(self.plan["decoder"], weight)
            candidates.append({"pipeline": pipeline, "scores": evaluate(bank, predictions, pipeline)})
        best = max(candidates, key=lambda r: r["scores"]["objective"])
        result = publish(path, {"id": proposal["id"], "steps": steps, "proposal_sha256": _digest(proposal),
                                "fits": fitted, **best, "pipeline_trials": candidates})
        del predictions
        gc.collect()
        return result

    def screen(self):
        self.phase("screen")
        if read_record(self.output / "screening.json"):
            return
        reason = "deadline"
        try:
            bank = self.get_bank()
            scores = evaluate(bank, None, self.base_pipeline, self.output / "incumbent/search")
            expected = read_record(self.repo / "artifacts/gpu-iou-followup-v1/confirmation/result.json")
            # Verify numerical equivalence against the completed study before any search result.
            expected_value = expected["groups"]["ensemble"]["summary"]["spell_balanced"]["by_batch"]["batch_1"]
            if not np.isclose(scores["objective"], expected_value, atol=1e-12, rtol=0):
                raise ValueError("incumbent does not reproduce the verified original objective")
            publish(self.output / "incumbent-parity.json", {"actual": scores["objective"],
                    "expected": expected_value, "passed": True})
            for index in range(10000):
                self.tick(screen_index=index)
                proposal = self.proposal(index)
                self.quick_evaluate(proposal, 1800, self.output / "screen" / proposal["id"])
        except BudgetExceeded:
            pass
        rows = [read_record(p) for p in sorted((self.output / "screen").glob("trial_*/result.json"))]
        publish(self.output / "screening.json", {"closed_reason": reason, "completed": rows})

    def promote(self):
        self.phase("promote")
        if read_record(self.output / "promotion.json"):
            return
        rows = read_record(self.output / "screening.json")["completed"]
        # Protect architecture diversity, then fill with the overall leaders.
        ordered = ranking(rows)
        chosen = []
        for pooling in ("mean", "ball_attention", "interaction"):
            chosen += [r for r in ordered if self.proposal(int(r["id"].split("_")[-1]))[
                "recipe"]["spec"]["model_kwargs"]["pooling"] == pooling][:2]
        chosen = ranking({r["id"]: r for r in chosen + ordered[:4]}.values())
        try:
            for row in chosen:
                proposal = self.proposal(int(row["id"].split("_")[-1]))
                self.quick_evaluate(proposal, 3600, self.output / "middle" / proposal["id"])
                total = proposal["recipe"]["spec"]["schedule"]["total_steps"]
                self.quick_evaluate(proposal, total, self.output / "full" / proposal["id"])
            # Use remaining neural-search time for additional full-budget TPE
            # proposals, rather than stopping just because the shortlist ended.
            index = max([int(p.stem.split("_")[-1]) for p in (self.output / "proposals").glob("*.json")], default=-1) + 1
            while True:
                self.tick(full_search_index=index)
                proposal = self.proposal(index)
                self.quick_evaluate(proposal, 1800, self.output / "screen" / proposal["id"])
                total = proposal["recipe"]["spec"]["schedule"]["total_steps"]
                self.quick_evaluate(proposal, total, self.output / "full" / proposal["id"])
                index += 1
        except BudgetExceeded:
            pass
        completed = [read_record(p) for p in sorted((self.output / "full").glob("trial_*/result.json"))]
        publish(self.output / "promotion.json", {"completed": completed, "closed_reason": "deadline"})

    def polish(self):
        self.phase("polish")
        if read_record(self.output / "selection.json"):
            return
        rows = ranking(read_record(self.output / "promotion.json")["completed"])[:3]
        bank = self.get_bank()
        incumbent = read_record(self.output / "incumbent/search/result.json")
        baseline = {"id": "incumbent", "steps": 0, "pipeline": self.base_pipeline,
                    "scores": incumbent["scores"], "fits": []}
        rows = [baseline, *rows]
        candidates = list(rows)
        # Round-robin batches prevent the first model from consuming the entire
        # decoder budget. Probabilities exist only for the current model in RAM.
        try:
            for block in range(1000):
                for row in rows:
                    self.tick(polish_candidate=row["id"], polish_block=block)
                    predictions = bank.predict(row["fits"], self.tick) if row["fits"] else None
                    trial_dir = self.output / "polish" / row["id"]
                    history = [read_record(p) for p in sorted(trial_dir.glob("trial_*.json"))]
                    for offset in range(12):
                        index = block * 12 + offset
                        path = trial_dir / f"trial_{index:05d}.json"
                        old = read_record(path)
                        if old:
                            candidates.append({**row, **old})
                            continue
                        self.tick(polish_candidate=row["id"], decoder_trial=index)
                        study = study_from_rows(history, 610000 + index * 37 + rows.index(row))
                        trial = study.ask()
                        pipeline = sample_pipeline(trial, self.plan["decoder"])
                        if predictions is None:
                            pipeline["candidate_weight"] = 0.
                        scores = evaluate(bank, predictions, pipeline)
                        result = publish(path, {"pipeline": pipeline, "scores": scores,
                            "params": trial.params, "distributions": {
                                k: distribution_to_json(v) for k, v in trial.distributions.items()},
                            "value": scores["objective"], "index": index})
                        history.append(result)
                        candidates.append({**row, **result})
                    del predictions
                    gc.collect()
        except BudgetExceeded:
            pass
        winner = ranking(candidates)[0]
        # This immutable file precedes every fresh-seed or WMX assessment.
        publish(self.output / "selection.json", {
            "id": winner["id"], "steps": winner["steps"], "pipeline": winner["pipeline"],
            "search_scores": winner["scores"], "selected_unix": time.time(),
            "proposal": (None if winner["id"] == "incumbent" else
                         read_record(self.output / "proposals" / f"{winner['id']}.json")),
            "selection_inputs": "completed search-seed whole-match OOF scores on batch_1 only",
            "completed_full_candidates": len(rows) - 1,
            "decoder_evaluations": sum(len(list((self.output / "polish" / r["id"]).glob("trial_*.json"))) for r in rows)})

    def assess(self, scope):
        stage = "guard" if scope == "guard" else "confirmation"
        self.phase(stage)
        path = self.output / f"{stage}.json"
        if read_record(path):
            return
        selection = read_record(self.output / "selection.json")
        pipeline, proposal = selection["pipeline"], selection["proposal"]
        seeds = GUARD_SEEDS if scope == "guard" else CONFIRMATION_SEEDS
        fitted = {}
        if proposal is not None and pipeline["candidate_weight"] > 0:
            for seed in seeds:
                fitted[seed] = self.fits(proposal, selection["steps"], seed, scope)
        base_rows, ensemble_rows, per_seed, batches = [], [], {str(s): [] for s in seeds}, {}
        for batch in ("batch_1", "batch_2"):
            bank = self.get_bank(scope, batch)
            destination = self.output / stage / batch
            baseline = evaluate(bank, None, self.base_pipeline, destination / "incumbent")
            previous_arm = "guard" if scope == "guard" else "confirmation"
            expected = read_record(self.repo / f"artifacts/gpu-iou-followup-v1/{previous_arm}/result.json")
            expected_value = expected["groups"]["ensemble"]["summary"]["spell_balanced"]["by_batch"][batch]
            if not np.isclose(baseline["objective"], expected_value, atol=1e-12, rtol=0):
                raise ValueError("assessment incumbent does not reproduce the verified study")
            base_rows.append(baseline)
            predictions = {}
            for seed in seeds:
                self.tick(assessment_seed=seed, batch=batch)
                pred = bank.predict(fitted[seed], self.tick) if fitted else None
                per_seed[str(seed)].append(evaluate(bank, pred, pipeline, destination / f"seed_{seed}"))
                predictions[seed] = pred
            ensemble = ({k: _blend(predictions[seeds[0]][k], predictions[seeds[1]][k], .5)
                         for k in bank.records} if fitted else None)
            score = evaluate(bank, ensemble, pipeline, destination / "ensemble")
            ensemble_rows.append(score)
            decoder_only = copy.deepcopy(pipeline)
            decoder_only["candidate_weight"] = 0.
            decoder_score = evaluate(bank, None, decoder_only, destination / "decoder_only")
            batches[batch] = {"incumbent": baseline, "ensemble": score, "decoder_only": decoder_score}
            del predictions, ensemble
            gc.collect()
        baseline, candidate = average_scores(base_rows), average_scores(ensemble_rows)
        gate = comparison(candidate, baseline, scope == "guard")
        seed_gains = {s: average_scores(rows)["objective"] - baseline["objective"] for s, rows in per_seed.items()}
        batch_gains = [c["objective"] - b["objective"] for c, b in zip(ensemble_rows, base_rows, strict=True)]
        if scope != "guard":
            gate["checks"].update(both_batches_improved=min(batch_gains) > 0)
            if fitted:
                gate["checks"]["both_seeds_improved"] = min(seed_gains.values()) > 0
            gate["passed"] = all(gate["checks"].values())
        publish(path, {"selection_sha256": _digest(selection), "candidate": candidate,
                       "incumbent": baseline, "comparison": gate, "by_batch": batches,
                       "single_seed_gains": seed_gains, "fits": fitted,
                       "fresh_candidate_training": bool(fitted),
                       "seed_checks_applicable": bool(fitted),
                       "completed_unix": time.time()})

    def stability(self):
        """Use spare GPU time for diagnostics; never change the frozen winner."""
        self.phase("stability")
        path = self.output / "stability.json"
        if read_record(path):
            return
        selection = read_record(self.output / "selection.json")
        if selection["proposal"] is None or selection["pipeline"]["candidate_weight"] == 0:
            publish(path, {"completed": [], "reason": "no new neural model; fresh candidate seeds inapplicable"})
            return
        baseline = read_record(self.output / "confirmation.json")["incumbent"]
        completed = []
        try:
            for seed in range(9601, 10000):
                seed_path = self.output / "stability" / f"seed_{seed}.json"
                old = read_record(seed_path)
                if old:
                    completed.append(old)
                    continue
                fitted = self.fits(selection["proposal"], selection["steps"], seed)
                rows = []
                for batch in ("batch_1", "batch_2"):
                    self.tick(stability_seed=seed, batch=batch)
                    bank = self.get_bank("inner", batch)
                    predictions = bank.predict(fitted, self.tick)
                    rows.append(evaluate(bank, predictions, selection["pipeline"],
                                         self.output / "stability" / batch / f"seed_{seed}"))
                    del predictions
                scores = average_scores(rows)
                completed.append(publish(seed_path, {
                    "seed": seed, "scores": scores, "delta": scores["objective"] - baseline["objective"],
                    "selection_sha256": _digest(selection), "fits": fitted,
                    "diagnostic_only": True, "changes_selection_or_gate": False}))
        except BudgetExceeded:
            pass
        publish(path, {"completed": completed, "reason": "remaining budget consumed",
                       "selection_sha256": _digest(selection), "changes_selection_or_gate": False})

    def verify(self):
        self.source_check()
        verified = []
        for path in sorted(self.output.glob("*/batch_*/*/result.json")):
            record = read_record(path)
            check_file(path.parent / "labels.npz", record["labels_sha256"])
            with np.load(path.parent / "labels.npz", allow_pickle=False) as archive:
                labels = {name: archive[name] for name in archive.files}
            if score_labels(labels, record["matches"]) != record["scores"]:
                raise ValueError(f"saved-label scores do not reproduce: {path}")
            verified.append(str(path.relative_to(self.output)))
        fits = []
        for path in sorted((self.output / "fits").rglob("fit.json")):
            record = read_record(path)
            check_file(record["model_path"], record["model_sha256"])
            fits.append({"path": str(path.relative_to(self.output)), "steps": record["steps"]})
        states = []
        for path in sorted((self.output / "fits").rglob("state_*.pt")):
            check_file(path, json.loads(path.with_suffix(".pt.sha256.json").read_text())["sha256"])
            payload = torch.load(path, map_location="cpu", weights_only=True)
            identity = payload["campaign"]["signature"]["identity"]
            state = payload["campaign"]["state"]
            lines = state["training_transcript"].decode().splitlines()
            used = {line.split(":", 1)[0].rsplit("_half", 1)[0] for line in lines}
            if (len(lines) != state["steps"] or not used <= set(identity["train_matches"])
                    or used & set(identity["validation_matches"])):
                raise ValueError("saved training transcript violates assessment exclusion")
            states.append({"path": str(path.relative_to(self.output)), "steps": state["steps"]})
        if len([p for p in verified if p.startswith(("confirmation/", "guard/"))]) != 20:
            raise ValueError("incomplete final assessment label grid")
        return {"saved_label_results_reproduced": verified, "checkpoints_verified": fits,
                "optimizer_states_and_split_transcripts_verified": states,
                "source_and_upstream_hashes_preserved": True, "automatic_promotion": False}

    def run(self):
        try:
            self.screen()
            self.promote()
            self.polish()
            self.assess("inner")
            self.assess("guard")
            self.stability()
            self.phase("report")
            receipt = self.verify()
            publish(self.output / "verification.json", receipt)
            status = "complete"
        except (BudgetExceeded, InterruptedError) as exc:
            status = "budget_exhausted" if isinstance(exc, BudgetExceeded) else "interrupted"
            write_json(self.output / "stop.json", {"status": status, "reason": str(exc), "unix": time.time()})
        except Exception as exc:
            status = "failed"
            write_json(self.output / "error.json", {"error": repr(exc), "traceback": traceback.format_exc(), "unix": time.time()})
            raise
        finally:
            try:
                generate_report(self.output)
            except Exception as exc:  # noqa: BLE001 -- preserve failure details and terminal status
                write_json(self.output / "report-error.json", {"error": repr(exc), "traceback": traceback.format_exc()})
                status = "failed"
            write_json(self.output / "status.json", {"status": locals().get("status", "failed"),
                "stage": "finished", "pid": os.getpid(), "updated_unix": time.time(), **self.budget.clock,
                "report": str(self.output / "report.md")})


def generate_report(output):
    """Always leave a readable account, including failed or time-limited runs."""
    selection = read_record(output / "selection.json")
    screen = read_record(output / "screening.json")
    full = read_record(output / "promotion.json")
    confirmation, guard = (read_record(output / f"{name}.json") for name in ("confirmation", "guard"))
    text = ["# Eight-hour GPU tuning campaign", "", CAVEAT, "",
            ("This is adaptive development tuning. A fresh seed is not a new match. "
            "The incumbent components are fixed; confirmation tests the selected candidate addition and decoder."), "",
            f"Completed screening recipes: {len(screen['completed']) if screen else 0}.",
            f"Completed full-budget recipes: {len(full['completed']) if full else 0}.", ""]
    if selection:
        text += [f"Frozen winner: `{selection['id']}`, {selection['steps']} training updates per new model.",
                 (f"Search objective: {selection['search_scores']['objective']:.6f}. "
                 "This selected development score is optimistic."), ""]
    text += ["| Assessment | Incumbent | Candidate | Change | Gate |",
             "|---|---:|---:|---:|---|"]
    confirmation_label = ("Fresh candidate seeds" if confirmation and confirmation.get("fresh_candidate_training", True)
                          else "Fixed-model mask check")
    for name, result in ((confirmation_label, confirmation), ("WMX guard", guard)):
        if result:
            text.append(f"| {name} | {result['incumbent']['objective']:.6f} | "
                        f"{result['candidate']['objective']:.6f} | {result['comparison']['delta']:+.6f} | "
                        f"{'PASS' if result['comparison']['passed'] else 'FAIL'} |")
        else:
            text.append(f"| {name} | — | — | — | Incomplete |")
    text += ["", ("The objective averages official temporal-IoU possession F1 at 0.3–0.7, "
             "with equal match weights and clean/short/long/replay weights 0.50/0.20/0.15/0.15. "
             "No release model is changed automatically."), ""]
    if confirmation:
        seed_detail = (f"Single candidate-seed gains: `{confirmation['single_seed_gains']}`."
                       if confirmation.get("fresh_candidate_training", True) else
                       "The selected pipeline uses fixed incumbent models; fresh candidate-seed checks are inapplicable.")
        text += ["## Confirmation details", "", seed_detail,
                 f"Gate checks: `{confirmation['comparison']['checks']}`.",
                 f"Descriptive paired-match 95% interval: `{confirmation['comparison'].get('descriptive_95_interval')}`.", "",
                 "| Clean batch 1, IoU 0.5 | Short matched / true | Spurious |", "|---|---:|---:|"]
        for label, key in (("Incumbent", "incumbent"), ("Candidate", "ensemble"), ("Decoder only", "decoder_only")):
            d = confirmation["by_batch"]["batch_1"][key]["diagnostics"]
            text.append(f"| {label} | {d['short_matched']}/{d['short_true']} | {d['spurious']} |")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for label, key in (("Incumbent", "incumbent"), ("Candidate", "ensemble"), ("Decoder only", "decoder_only")):
            rows = [r[key]["iou_curve"] for r in confirmation["by_batch"].values()]
            thresholds = sorted(rows[0], key=float)
            ax.plot([float(t) for t in thresholds], [np.mean([r[t] for r in rows]) for t in thresholds],
                    marker="o", label=label)
        ax.set(xlabel="Temporal IoU threshold", ylabel="Scenario-weighted possession F1",
               title=f"Frozen winner: {confirmation_label.lower()}, six reused matches")
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "confirmation-iou.png", dpi=180)
        fig.savefig(output / "confirmation-iou.pdf")
        plt.close(fig)
        text += ["", "![Confirmation IoU curves](confirmation-iou.png)", ""]
    stability = read_record(output / "stability.json")
    if stability:
        text += ["", "## Additional fixed-recipe seed checks", "",
                 "These use the remaining budget after the main assessments and cannot change selection or its gate.", ""]
        if stability["completed"]:
            text += ["| Candidate seed | Objective | Change from incumbent |", "|---|---:|---:|"]
            text.extend(f"| {r['seed']} | {r['scores']['objective']:.6f} | {r['delta']:+.6f} |"
                        for r in stability["completed"])
        else:
            text.append(stability["reason"])
    text += ["", "## Reproduction", "", ("See [protocol](protocol.json), [selection](selection.json), "
             "[confirmation](confirmation.json), [guard](guard.json), and [verification](verification.json). "
             "Missing files indicate an unfinished stage. `status.json`, `stdout.log`, and `error.json` "
             "record progress or failure. The original datasets and prior experiments remain in place."), "",
             ("The statistical-analysis and scientific-critical-thinking skills informed this design. "
             "Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). "
             "[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents]"
             "(https://doi.org/10.48550/arXiv.2609.00065)."), ""]
    (output / "report.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hours", type=float, default=8.)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (args.output or repo / "artifacts/gpu-tuning-8h-v1").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not 0 < args.hours <= 8:
        raise ValueError("budget must be positive and at most eight hours")
    configure_device("cuda:0")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_requested)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with accelerated_preprocessing():
            campaign = Campaign(repo, output, args.hours)
            if args.preflight:
                campaign.source_check()
                bank = campaign.get_bank()
                scores = evaluate(bank, None, campaign.base_pipeline, output / "incumbent/search")
                print(json.dumps({"objective": scores["objective"], "diagnostics": scores["diagnostics"]}), flush=True)
            else:
                campaign.run()


if __name__ == "__main__":
    main()
