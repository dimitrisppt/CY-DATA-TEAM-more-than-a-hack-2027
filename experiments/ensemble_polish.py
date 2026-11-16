"""Bounded joint decoder search and paired confirmation using frozen networks."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import time
from pathlib import Path

import torch

from experiments.campaign_protocol import Budget, BudgetExceeded, publish, read_record
from experiments.ensemble_polish_confirm import FRESH_SCENARIOS, confirm_pair
from experiments.ensemble_polish_search import tune_joint
from experiments.robustness import _digest, _execution_environment, _source_hashes
from possession.prepared import verify_prepared
from possession.storage import sha256_file, write_json

MEMBERS = ("mean_007", "ball_attention_002")


def _required(path):
    value = read_record(path)
    if value is None:
        raise FileNotFoundError(path)
    return value


def preflight(campaign_dir, data_dir, target_dir):
    """Authenticate the selected OOF data and old campaign without reading target truth."""
    campaign_dir, data_dir, target_dir = (Path(path).resolve()
                                          for path in (campaign_dir, data_dir, target_dir))
    protocol = _required(campaign_dir / "protocol.json")
    selection = _required(campaign_dir / "selection.json")
    summary = _required(campaign_dir / "summary.json")
    final = _required(campaign_dir / "final.json")
    if (summary.get("status") != "complete"
            or summary.get("candidate_eligible_for_review") is not True
            or summary.get("selection") != selection
            or summary.get("protocol_sha256") != _digest(protocol)
            or selection.get("kind") != "convex_pair"
            or [member["id"] for member in selection["members"]] != list(MEMBERS)):
        raise ValueError("campaign does not contain the confirmed frozen pair")
    current_source = _source_hashes()
    changed = [name for name, digest in protocol["source_sha256"].items()
               if current_source.get(name) != digest]
    if changed:
        raise ValueError(f"measured campaign sources changed: {changed}")
    if _execution_environment() != protocol["environment"]:
        raise ValueError("campaign runtime differs")
    paths = [data_dir / row["cache"] for row in protocol["prepared"]["caches"]]
    if verify_prepared(data_dir, paths) != protocol["prepared"]:
        raise ValueError("prepared public data differs")
    target_manifest = target_dir / "input-manifest.json"
    if sha256_file(target_manifest) != protocol["target_input_manifest_sha256"]:
        raise ValueError("target automatic-input manifest differs")
    rows = {row["id"]: row for row in final["rows"]}
    record_sets = []
    for member in selection["members"]:
        row = rows[member["id"]]
        if row["spec"] != member["spec"] or row["steps"] != member["steps"]:
            raise ValueError("final-rung specification differs from selection")
        records = row["records"]
        expected = {(match, period, scenario) for match in protocol["fold"]["outer_train"]
                    for period in (1, 2) for scenario in protocol["scenario_weights"]}
        actual = [(record["match"], record["period"], record["scenario"]) for record in records]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("OOF search grid is incomplete or duplicated")
        fits = {}
        for index, fold in enumerate(protocol["fold"]["inner"]):
            directory = (campaign_dir / "fits/inner" / member["id"] /
                         f"seed_{protocol['search_seed']}" / f"fold_{index}" / "rung_7200")
            fitted = _required(directory / "fit.json")
            if (fitted["identity"]["protocol_sha256"] != _digest(protocol)
                    or fitted["identity"]["spec_sha256"] != _digest(member["spec"])
                    or fitted["identity"]["train_matches"] != fold["train"]
                    or fitted["steps"] != member["steps"]
                    or sha256_file(directory / "model.pt") != fitted["checkpoint_sha256"]):
                raise ValueError("OOF model fit ancestry differs")
            fits[index] = fitted
        for record in records:
            identity = record["model_identity"]
            index = identity["fold"]
            fold = protocol["fold"]["inner"][index]
            if (identity["train_matches"] != fold["train"]
                    or identity["validation_matches"] != fold["validation"]
                    or identity["protocol_sha256"] != _digest(protocol)
                    or identity["checkpoint_sha256"] != fits[index]["checkpoint_sha256"]
                    or identity["seed"] != protocol["search_seed"]
                    or record["match"] not in fold["validation"]
                    or record["match"] in fold["train"]):
                raise ValueError("OOF record leaks a training match or differs from its fold")
        record_sets.append(records)
    return protocol, selection, record_sets


def run_polish(campaign_dir, data_dir, target_dir, output_dir, *, trials=64,
               seed=20261116, hours=1.5):
    """Tune only on search OOF, then freeze before fresh-mask confirmation."""
    campaign_dir, data_dir, target_dir, output_dir = (Path(path).resolve() for path in
                                                     (campaign_dir, data_dir, target_dir, output_dir))
    for base in (campaign_dir, data_dir, target_dir):
        if output_dir.is_relative_to(base) or base.is_relative_to(output_dir):
            raise ValueError("polish output must be separate from all inputs and the old campaign")
    protocol, baseline, records = preflight(campaign_dir, data_dir, target_dir)
    plan = {
        "schema": "ensemble_polish_protocol_v1", "development_only": True,
        "campaign_dir": str(campaign_dir), "data_dir": str(data_dir), "target_dir": str(target_dir),
        "campaign_protocol_sha256": _digest(protocol), "baseline_selection": baseline,
        "campaign_records_sha256": {name: sha256_file(campaign_dir / name)
                                    for name in ("protocol.json", "selection.json", "summary.json",
                                                 "final.json", "confirmation.json", "outer.json")},
        "source_sha256": _source_hashes(), "runtime": _execution_environment(),
        "trials": trials, "proposal_seed": seed, "hours": hours,
        "search_training_seed": 7100, "confirmation_training_seeds_reused": [8101, 8102],
        "outer_training_seeds_reused": [9101, 9102], "confirmation_scenarios": FRESH_SCENARIOS,
        "candidate_count_entering_confirmation": 1,
        "gate": {"paired_campaign_gate_required": True, "outer_loss_at_most": .01,
                 "automatic_baseline_replacement": False},
        "seed_bagging": {"weights": [.5, .5], "diagnostic_only": True,
                         "note": "One bagged predictor is not two independent confirmation seeds."},
        "longer_training": {"performed": False,
                             "decision": "Assess cached decoder changes and seed averaging first."},
        "interpretation": "Fresh synthetic masks, reused networks and matches, not independent test data.",
        "target_possession_labels_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".polish.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish(output_dir / "protocol.json", plan)
        completed = read_record(output_dir / "summary.json")
        if completed is not None:
            if completed["protocol_sha256"] != _digest(plan):
                raise ValueError("completed polish protocol differs")
            return completed
        budget = Budget(output_dir, hours)
        try:
            budget.status("search")
            result = tune_joint(*records, baseline, output_dir / "search", n_trials=trials,
                                seed=seed, check=lambda: budget.check(.7))
            candidate = copy.deepcopy(baseline)
            candidate.update(decoder=result["decoder"], blend_weight=result["blend_weight"],
                             scores=result["scores"], frozen_before_confirmation=True,
                             automatic_deployment=False)
            selected = {"schema": "ensemble_polish_selection_v1", "selection": candidate,
                        "baseline_selection_sha256": _digest(baseline), "search_best_trial": result["best_trial"],
                        "protocol_sha256": _digest(plan), "frozen_unix": time.time()}
            previous = read_record(output_dir / "selection.json")
            if previous is not None:
                selected["frozen_unix"] = previous["frozen_unix"]
            publish(output_dir / "selection.json", selected)
            budget.status("confirmation")
            confirmation = confirm_pair(campaign_dir, data_dir, target_dir,
                                        output_dir / "confirmation", baseline, candidate,
                                        check=budget.check)
            summary = {"schema": "ensemble_polish_summary_v1", "status": "complete",
                       "protocol_sha256": _digest(plan), "search": result,
                       "selection": candidate, "confirmation": confirmation,
                       "development_only": True, "baseline_replaced": False,
                       "target_possession_labels_used": False}
            publish(output_dir / "summary.json", summary)
            write_json(output_dir / "status.json", {"status": "complete", "pid": os.getpid(),
                                                     "updated_unix": time.time(),
                                                     "summary_path": str(output_dir / "summary.json")})
            return summary
        except BudgetExceeded as exc:
            write_json(output_dir / "status.json", {"status": "budget_reached", "reason": str(exc),
                                                     "updated_unix": time.time(),
                                                     "baseline_replaced": False})
            return {"status": "budget_reached", "reason": str(exc)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--hours", type=float, default=1.5)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    result = run_polish(args.campaign_dir, args.data_dir, args.target_dir, args.output_dir,
                        trials=args.trials, seed=args.seed, hours=args.hours)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
