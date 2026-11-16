"""Verify and summarize seven frozen outer matches without changing selection.

The JSON and CSV contain match-level descriptive results. The synthetic ball
outage is a separate post-selection diagnostic; neither its score nor any outer
score is used to choose a model, decoder, or recipe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

MATCHES = (
    "DFL-MAT-J03WMX", "DFL-MAT-J03WN1", "DFL-MAT-J03WOH",
    "DFL-MAT-J03WOY", "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ",
    "DFL-MAT-J03WR9",
)
BUNDESLIGA = frozenset(MATCHES[:2])
METRICS = ("spell_f1_mean", "macro_f1", "alive_dead_f1")
FINGERPRINT_COMMON = (
    "protocol", "source_sha256", "manifest_sha256", "uv_lock_sha256",
    "versions", "thread_environment", "resources", "search_space",
    "fixed", "seed", "pruning", "split",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required audit file missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _metric_values(scores, label: str) -> dict[str, float]:
    _require(isinstance(scores, dict), f"{label} official scores are missing")
    result = {}
    for metric in METRICS:
        value = scores.get(metric)
        _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(value), f"{label} {metric} must be finite")
        result[metric] = float(value)
    return result


def _source_matches_current(source_hashes: dict, root: Path) -> None:
    expected = set(source_hashes)
    current = {str(path.relative_to(root))
               for directory in ("possession", "experiments", "evaluation")
               for path in (root / directory).rglob("*.py")}
    current.update(("games.py", "pyproject.toml"))
    _require(current == expected, "current core source file set differs from study fingerprint")
    for name, digest in source_hashes.items():
        _require(_sha(root / name) == digest,
                 f"current core source differs from study fingerprint: {name}")


def _validate_splits(outer: str, summary: dict, fingerprint: dict, split: dict) -> None:
    ids = sorted(MATCHES)
    _require(split.get("matches") == ids and len(split.get("outer_folds", [])) == 7,
             "split manifest must cover the seven known matches")
    _require(fingerprint.get("payload", {}).get("split") == split,
             "study fingerprint and split manifest differ")
    folds = split["outer_folds"]
    _require(sorted(fold.get("outer_test", [None])[0] for fold in folds) == ids,
             "split manifest outer fold coverage differs")
    fold = next((item for item in folds if item.get("outer_test") == [outer]), None)
    _require(fold is not None, f"missing outer split for {outer}")
    other = sorted(set(MATCHES) - {outer})
    _require(fold.get("outer_train") == other and summary.get("outer_test") == [outer]
             and summary.get("outer_train") == other
             and summary.get("inner") == fold.get("inner"),
             f"outer split and HPO summary differ for {outer}")
    seen = []
    for inner in fold["inner"]:
        train, validation = inner.get("train"), inner.get("validation")
        _require(isinstance(validation, list) and validation
                 and sorted(train) == sorted(set(other) - set(validation))
                 and not set(train) & set(validation),
                 f"invalid inner split for {outer}")
        seen.extend(validation)
    _require(sorted(seen) == other, f"inner validation does not partition {outer} training")


def _validate_hpo(outer: str, directory: Path, fingerprint: dict,
                  summary: dict, split: dict) -> tuple[int, int]:
    payload = fingerprint.get("payload")
    _require(isinstance(payload, dict) and fingerprint.get("sha256") == _digest(payload),
             f"invalid HPO fingerprint for {outer}")
    _require(payload.get("outer_test") == [outer], f"HPO fingerprint outer ID differs: {outer}")
    _validate_splits(outer, summary, fingerprint, split)
    rungs = payload.get("resources", {}).get("rungs")
    _require(isinstance(rungs, list) and rungs and all(
        isinstance(step, int) and step > 0 for step in rungs),
        f"invalid HPO resource rungs: {outer}")
    _require(summary.get("fingerprint_sha256") == fingerprint["sha256"]
             and summary.get("full_budget_steps") == rungs[-1],
             f"HPO summary fingerprint or full budget differs: {outer}")
    trials = summary.get("trials")
    _require(isinstance(trials, list) and trials
             and summary.get("n_trials_total") == len(trials)
             and summary.get("n_complete_full_budget") == len(trials),
             f"HPO trial count or completion differs: {outer}")
    poolings = payload.get("search_space", {}).get("pooling")
    _require(poolings == ["mean", "ball_attention"]
             and set(summary.get("best_by_pooling", {})) == set(poolings),
             f"HPO pooling families differ: {outer}")
    for trial in trials:
        reports = trial.get("rungs", [])
        _require(trial.get("state") == "COMPLETE"
                 and [rung.get("steps_per_fold") for rung in reports] == rungs
                 and trial.get("params", {}).get("pooling") in poolings
                 and math.isfinite(trial.get("value", float("nan"))),
                 f"HPO trial did not complete every rung: {outer}")
    for pooling in poolings:
        candidates = [trial for trial in trials if trial["params"]["pooling"] == pooling]
        _require(candidates, f"HPO pooling has no complete trial: {outer} {pooling}")
        winner = max(candidates, key=lambda trial: (trial["value"], -trial["number"]))
        recipe = summary["best_by_pooling"][pooling]
        _require(recipe.get("trial_number") == winner["number"]
                 and recipe.get("score") == winner["value"]
                 and recipe.get("params") == winner["params"]
                 and recipe.get("study_fingerprint_sha256") == fingerprint["sha256"],
                 f"HPO family winner differs: {outer} {pooling}")
        _validate_recipe(outer, directory, pooling, recipe, split)
    return len(trials), len(trials) * len(summary["inner"]) * rungs[-1]


def _validate_recipe(outer: str, directory: Path, pooling: str,
                     recipe: dict, split: dict) -> None:
    number = recipe["trial_number"]
    artifact_dir = (directory / "trials" / f"trial_{number}").resolve()
    _require(Path(recipe.get("artifact_dir", "")).resolve() == artifact_dir
             and _read(directory / "best_by_pooling" / pooling / "recipe.json") == recipe
             and _read(artifact_dir / "recipe.json") == recipe,
             f"immutable HPO recipe or family pointer differs: {outer} {pooling}")
    hashes = recipe.get("artifact_sha256")
    checkpoints, oof_files = recipe.get("fold_checkpoints"), recipe.get("oof_files")
    _require(isinstance(hashes, dict) and isinstance(checkpoints, list)
             and isinstance(oof_files, list), f"HPO artifact manifest missing: {outer}")
    fold = next(item for item in split["outer_folds"] if item["outer_test"] == [outer])
    _require(len(checkpoints) == len(fold["inner"]), f"fold checkpoint count differs: {outer}")
    expected_names = set()
    for index, item in enumerate(checkpoints):
        name = f"fold_{index}.pt"
        _require(item.get("inner_fold") == index and item.get("file") == name
                 and item.get("train_matches") == fold["inner"][index]["train"]
                 and item.get("validation_matches") == fold["inner"][index]["validation"]
                 and item.get("sha256") == hashes.get(name),
                 f"fold checkpoint manifest differs: {outer} {pooling} {index}")
        expected_names.add(name)
    expected_oof = {(match, period) for match in fold["outer_train"] for period in (1, 2)}
    actual_oof = set()
    for item in oof_files:
        match, period, index = item.get("match"), item.get("period"), item.get("inner_fold")
        _require((match, period) in expected_oof and (match, period) not in actual_oof
                 and isinstance(index, int) and 0 <= index < len(checkpoints)
                 and match in fold["inner"][index]["validation"],
                 f"OOF ancestry differs: {outer} {pooling}")
        actual_oof.add((match, period))
        name, sidecar_name = f"{match}_half{period}.npz", f"{match}_half{period}.json"
        _require(item.get("file") == name and item.get("sidecar") == sidecar_name
                 and item.get("sha256") == hashes.get(name)
                 and item.get("sidecar_sha256") == hashes.get(sidecar_name),
                 f"OOF manifest hash differs: {outer} {pooling} {name}")
        sidecar = _read(artifact_dir / sidecar_name)
        _require(sidecar.get("prediction_match") == match
                 and sidecar.get("period") == period
                 and sidecar.get("inner_fold") == index
                 and sidecar.get("train_matches") == fold["inner"][index]["train"]
                 and sidecar.get("outer_test") == outer
                 and sidecar.get("trial_number") == number
                 and sidecar.get("checkpoint_file") == f"fold_{index}.pt"
                 and sidecar.get("checkpoint_sha256") == hashes.get(f"fold_{index}.pt")
                 and sidecar.get("oof_file") == name
                 and sidecar.get("oof_sha256") == hashes.get(name)
                 and sidecar.get("study_fingerprint_sha256")
                    == recipe["study_fingerprint_sha256"],
                 f"OOF sidecar ancestry differs: {outer} {pooling} {name}")
        expected_names.update((name, sidecar_name))
    _require(actual_oof == expected_oof and set(hashes) == expected_names,
             f"OOF artifact coverage differs: {outer} {pooling}")
    for name, expected in hashes.items():
        _require(Path(name).name == name and _sha(artifact_dir / name) == expected,
                 f"HPO artifact bytes differ: {outer} {pooling} {name}")


def _decoder_winner(decoder: dict) -> tuple[str, dict]:
    candidates = []
    for pooling, family in decoder["by_pooling"].items():
        for trial in family["trials"]:
            if trial["state"] == "COMPLETE":
                reference_priority = int(trial["params"]["neural_weight"] != 0)
                family_priority = 0 if pooling == "mean" else 1
                candidates.append((trial["value"], -reference_priority,
                                   -family_priority, -trial["number"], pooling, trial))
    _require(candidates, "decoder summary has no complete trial")
    _, _, _, _, pooling, trial = max(candidates)
    return pooling, trial


def _validate_decoder(outer: str, directory: Path, summary: dict,
                      hpo_fp: dict, decoder_fp: dict, decoder: dict,
                      frozen: dict) -> int:
    payload = decoder_fp.get("payload")
    _require(isinstance(payload, dict) and decoder_fp.get("sha256") == _digest(payload)
             and payload.get("hpo_study_fingerprint_sha256") == hpo_fp["sha256"]
             and payload.get("hpo_summary_sha256") == _digest(summary),
             f"decoder fingerprint differs: {outer}")
    expected_oof = {
        pooling: {
            "recipe_sha256": _sha(Path(recipe["artifact_dir"]) / "recipe.json"),
            "artifact_sha256": recipe["artifact_sha256"],
        }
        for pooling, recipe in summary["best_by_pooling"].items()
    }
    _require(payload.get("oof_artifacts") == expected_oof,
             f"decoder fingerprint OOF artifact ancestry differs: {outer}")
    _require(decoder.get("fingerprint_sha256") == decoder_fp["sha256"]
             and decoder.get("outer_test") == outer
             and decoder.get("outer_train") == summary["outer_train"]
             and set(decoder.get("by_pooling", {})) == {"mean", "ball_attention"},
             f"decoder summary ancestry differs: {outer}")
    counts = []
    for pooling, family in decoder["by_pooling"].items():
        trials = family.get("trials")
        _require(isinstance(trials, list) and trials
                 and family.get("n_trials_total") == len(trials)
                 and all(trial.get("state") == "COMPLETE" for trial in trials),
                 f"decoder trial completion differs: {outer} {pooling}")
        winner = max(trials, key=lambda trial: (trial["value"], -trial["number"]))
        _require(family.get("best") == winner, f"decoder family winner differs: {outer}")
        counts.append(len(trials))
    _require(len(set(counts)) == 1 and frozen.get("resources", {}).get(
        "n_trials_per_pooling") == counts[0],
        f"decoder trial counts differ: {outer}")
    selected_pooling, selected_trial = _decoder_winner(decoder)
    weight = selected_trial["params"]["neural_weight"]
    selected_recipe = (None if weight == 0 else summary["best_by_pooling"][selected_pooling])
    selected_label = "reference" if weight == 0 else selected_pooling
    _require(frozen.get("selected_decoder_trial") == {"pooling": selected_pooling,
                                                        **selected_trial}
             and frozen.get("selection_score") == selected_trial["value"]
             and frozen.get("selected_pooling") == selected_label
             and frozen.get("selected_model_recipe") == selected_recipe
             and frozen.get("ensemble") == {"neural_weight": weight}
             and frozen.get("decoder") == {"switch_cost": selected_trial["params"]["switch_cost"],
                                           "boundary_scale": 0.0}
             and frozen.get("calibration", {}).get("live_temperature")
                == selected_trial["params"]["live_temperature"]
             and frozen.get("calibration", {}).get("owner_temperature")
                == selected_trial["params"]["owner_temperature"],
             f"frozen decoder selection differs: {outer}")
    _require(frozen.get("selection_fingerprint") == decoder_fp["sha256"]
             and frozen.get("hpo_fingerprint_sha256") == hpo_fp["sha256"]
             and frozen.get("decoder_fingerprint_sha256") == decoder_fp["sha256"]
             and frozen.get("outer_test") == outer
             and frozen.get("outer_train") == summary["outer_train"]
             and frozen.get("resources", {}).get("full_budget_steps")
                == summary["full_budget_steps"]
             and frozen.get("resources", {}).get("device") == payload.get("device")
             and frozen.get("resources", {}).get("cache_bytes") == payload.get("cache_bytes")
             and frozen.get("refit_seed") == int.from_bytes(hashlib.sha256(
                 f"{payload['seed']}:{outer}:refit".encode()).digest()[:4], "big"),
             f"frozen recipe fingerprint or split differs: {outer}")
    return sum(counts)


def _validate_report_and_stress(outer: str, directory: Path, frozen: dict,
                                summary: dict, hpo_fp: dict, decoder_fp: dict,
                                report: dict, stress: dict,
                                data_dir: Path | None) -> tuple[dict, dict]:
    frozen_path = directory / "frozen_recipe.json"
    report_path = directory / "outer_report.json"
    frozen_sha, report_sha = _sha(frozen_path), _sha(report_path)
    _require(_read(directory / "outer_report.sha256.json").get("sha256") == report_sha,
             f"outer report SHA sidecar differs: {outer}")
    _require(report.get("outer_test") == outer
             and report.get("outer_train") == summary["outer_train"]
             and report.get("frozen_recipe_sha256") == frozen_sha
             and report.get("selection_fingerprint") == frozen["selection_fingerprint"]
             and report.get("predictor") == frozen["selected_pooling"]
             and set(report.get("by_match", {})) == {outer},
             f"outer report freeze or split differs: {outer}")
    by_match = report["by_match"][outer]
    _require(by_match.get("official") == report.get("official")
             and by_match.get("reference_official") == report.get("reference_official"),
             f"outer match metrics differ within report: {outer}")
    selected_path = directory / "selected.pt"
    progress = _read(directory / "refit_progress.json")
    if frozen["selected_pooling"] == "reference":
        _require(not selected_path.exists() and report.get("checkpoint_sha256") is None
                 and progress.get("steps") == 0 and progress.get("predictor") == "reference"
                 and progress.get("train_matches") == summary["outer_train"],
                 f"reference-only checkpoint or refit differs: {outer}")
        refit_steps = 0
    else:
        checkpoint_sha = _sha(selected_path)
        refit_steps = summary["full_budget_steps"]
        _require(checkpoint_sha == report.get("checkpoint_sha256")
                 and checkpoint_sha == progress.get("checkpoint_sha256")
                 and progress.get("steps") == refit_steps
                 and progress.get("train_matches") == summary["outer_train"],
                 f"selected checkpoint SHA or refit steps differ: {outer}")
    _require(stress.get("protocol") == "post-selection synthetic 41-percent ball-outage diagnostic v1"
             and stress.get("selection_use") == "diagnostic_only_no_model_change_or_promotion"
             and stress.get("outer_test") == outer
             and stress.get("mask_rule", {}).get("fraction_numerator") == 41
             and stress.get("mask_rule", {}).get("fraction_denominator") == 100
             and stress.get("clean_selected_official_from_saved_outer_report")
                == report["official"]
             and stress.get("clean_fixed_reference_official_from_saved_outer_report")
                == report["reference_official"],
             f"stress audit does not preserve fixed clean selection: {outer}")
    provenance = stress.get("provenance", {})
    expected = {
        "outer_report_sha256": report_sha,
        "frozen_recipe_sha256": frozen_sha,
        "hpo_summary_sha256": _sha(directory / "summary.json"),
        "hpo_fingerprint_sha256": hpo_fp["sha256"],
        "hpo_study_sha256": _sha(directory / "study.sqlite3"),
        "decoder_summary_sha256": _sha(directory / "decoder_summary.json"),
        "decoder_fingerprint_sha256": decoder_fp["sha256"],
        "selected_decoder_study_sha256": _sha(directory / f"decoder_{frozen['selected_decoder_trial']['pooling']}.sqlite3"),
        "selected_checkpoint_sha256": report.get("checkpoint_sha256"),
        "prepared_sha256": report.get("prepared_sha256"),
        "audit_script_sha256": _sha(Path(__file__).with_name("audit_ball_outages.py")),
    }
    for key, value in expected.items():
        _require(provenance.get(key) == value,
                 f"stress provenance differs: {outer} {key}")
    prepared = report.get("prepared_sha256")
    expected_names = {f"{outer}_half1.npz", f"{outer}_half2.npz"}
    _require(isinstance(prepared, dict) and set(prepared) == expected_names,
             f"outer prepared half hashes are incomplete: {outer}")
    if data_dir is not None:
        for name, digest in prepared.items():
            _require(_sha(data_dir / name) == digest,
                     f"current prepared data differs from outer report: {name}")
    clean = {"selected": _metric_values(report["official"], f"clean selected {outer}"),
             "reference": _metric_values(report["reference_official"],
                                          f"clean reference {outer}")}
    stressed = {"selected": _metric_values(stress.get("stressed_selected_official"),
                                             f"stress selected {outer}"),
                "reference": _metric_values(stress.get("stressed_fixed_reference_official"),
                                              f"stress reference {outer}")}
    return {"clean": clean, "stress": stressed}, {"refit_steps": refit_steps,
                                                   "prepared_sha256": prepared}


def _mean(values) -> float:
    return sum(values) / len(values)


def _group_summary(rows: list[dict]) -> dict:
    result = {"n_matches": len(rows),
              "winner_counts": dict(sorted(Counter(row["selected_pooling"] for row in rows).items())),
              "neural_weights": {"by_match": {row["match"]: row["neural_weight"] for row in rows},
                                 "mean": _mean([row["neural_weight"] for row in rows])},
              "clean": {}, "synthetic_ball_outage_41_percent": {}}
    for source, key in (("clean", "clean"),
                        ("stress", "synthetic_ball_outage_41_percent")):
        for metric in METRICS:
            selected = [row[source]["selected"][metric] for row in rows]
            reference = [row[source]["reference"][metric] for row in rows]
            result[key][metric] = {
                "selected_match_mean": _mean(selected),
                "reference_match_mean": _mean(reference),
                "selected_minus_reference_match_mean":
                    _mean([a - b for a, b in zip(selected, reference)]),
            }
    result["stress_minus_clean_selected_match_mean"] = {
        metric: _mean([row["stress"]["selected"][metric]
                       - row["clean"]["selected"][metric] for row in rows])
        for metric in METRICS
    }
    return result


def summarize_nested(study_dir, *, data_dir=None) -> tuple[dict, list[dict]]:
    """Validate all seven completed folds, then return aggregate JSON and CSV rows."""
    study_dir = Path(study_dir).resolve()
    data_dir = None if data_dir is None else Path(data_dir).resolve()
    _require(study_dir.is_dir(), f"study directory missing: {study_dir}")
    actual_dirs = {path.name for path in study_dir.iterdir()
                   if path.is_dir() and path.name.startswith("DFL-MAT-")}
    _require(actual_dirs == set(MATCHES),
             f"expected exactly seven outer directories; missing={sorted(set(MATCHES)-actual_dirs)}, "
             f"extra={sorted(actual_dirs-set(MATCHES))}")
    required = ("outer_report.json", "outer_report.sha256.json", "ball-outage-audit.json",
                "frozen_recipe.json", "refit_progress.json", "summary.json",
                "fingerprint.json", "splits.json", "decoder_summary.json",
                "decoder_fingerprint.json", "study.sqlite3")
    missing = [str(study_dir / match / name) for match in MATCHES for name in required
               if not (study_dir / match / name).is_file()]
    if missing:
        raise FileNotFoundError("seven-fold audit is incomplete: " + ", ".join(missing))
    root = Path(__file__).resolve().parents[1]
    rows = []
    first_hpo, first_decoder, first_split = None, None, None
    shared_prepared = {}
    common_trial_counts = None
    hashes_by_match = {}
    for outer in MATCHES:
        directory = study_dir / outer
        summary = _read(directory / "summary.json")
        hpo_fp = _read(directory / "fingerprint.json")
        split = _read(directory / "splits.json")
        decoder = _read(directory / "decoder_summary.json")
        decoder_fp = _read(directory / "decoder_fingerprint.json")
        frozen = _read(directory / "frozen_recipe.json")
        report = _read(directory / "outer_report.json")
        stress = _read(directory / "ball-outage-audit.json")
        hpo_count, hpo_steps = _validate_hpo(outer, directory, hpo_fp, summary, split)
        decoder_count = _validate_decoder(outer, directory, summary, hpo_fp,
                                          decoder_fp, decoder, frozen)
        scores, more = _validate_report_and_stress(outer, directory, frozen, summary,
                                                    hpo_fp, decoder_fp, report, stress,
                                                    data_dir)
        payload = hpo_fp["payload"]
        if first_hpo is None:
            first_hpo = {key: payload.get(key) for key in FINGERPRINT_COMMON}
            first_decoder = {key: decoder_fp["payload"].get(key) for key in
                             ("protocol", "search_space", "seed", "device", "cache_bytes")}
            first_split = split
            common_trial_counts = (hpo_count, decoder_count)
            _source_matches_current(payload["source_sha256"], root)
            if data_dir is not None:
                manifest = data_dir / "manifest.json"
                _require(_sha(manifest) == payload["manifest_sha256"],
                         "current prepared manifest differs from study fingerprint")
        else:
            _require({key: payload.get(key) for key in FINGERPRINT_COMMON} == first_hpo,
                     f"HPO source, versions, resources, search, or seed differ: {outer}")
            _require({key: decoder_fp["payload"].get(key) for key in
                      ("protocol", "search_space", "seed", "device", "cache_bytes")}
                     == first_decoder,
                     f"decoder search configuration differs: {outer}")
            _require(split == first_split, f"nested split manifest differs: {outer}")
            _require((hpo_count, decoder_count) == common_trial_counts,
                     f"HPO or decoder trial count differs: {outer}")
        for name, digest in payload.get("prepared_sha256", {}).items():
            _require(name not in shared_prepared or shared_prepared[name] == digest,
                     f"prepared half fingerprint differs across folds: {name}")
            shared_prepared[name] = digest
        for name, digest in more["prepared_sha256"].items():
            _require(name not in shared_prepared or shared_prepared[name] == digest,
                     f"outer prepared hash differs across folds: {name}")
            shared_prepared[name] = digest
        weight = frozen["ensemble"]["neural_weight"]
        rows.append({
            "match": outer,
            "group": "Bundesliga" if outer in BUNDESLIGA else "Düsseldorf",
            "selected_pooling": frozen["selected_pooling"],
            "neural_weight": float(weight),
            "selection_score_inner_oof": frozen["selection_score"],
            "n_frames_native": report["by_match"][outer]["n_frames_native"],
            "clean": scores["clean"], "stress": scores["stress"],
            "delta_selected_minus_reference": {
                source: {metric: scores[source]["selected"][metric]
                         - scores[source]["reference"][metric] for metric in METRICS}
                for source in ("clean", "stress")
            },
            "stress_minus_clean_selected": {
                metric: scores["stress"]["selected"][metric]
                        - scores["clean"]["selected"][metric] for metric in METRICS
            },
            "hpo_trials": hpo_count, "decoder_trials": decoder_count,
            "hpo_optimizer_steps": hpo_steps, "refit_optimizer_steps": more["refit_steps"],
        })
        hashes_by_match[outer] = {
            "hpo_fingerprint_sha256": hpo_fp["sha256"],
            "hpo_summary_sha256": _sha(directory / "summary.json"),
            "decoder_fingerprint_sha256": decoder_fp["sha256"],
            "decoder_summary_sha256": _sha(directory / "decoder_summary.json"),
            "frozen_recipe_sha256": _sha(directory / "frozen_recipe.json"),
            "outer_report_sha256": _sha(directory / "outer_report.json"),
            "ball_outage_audit_sha256": _sha(directory / "ball-outage-audit.json"),
            "selected_checkpoint_sha256": report["checkpoint_sha256"],
        }
    expected_halves = {f"{match}_half{period}.npz" for match in MATCHES for period in (1, 2)}
    _require(set(shared_prepared) == expected_halves,
             "prepared half hashes do not cover all seven matches")
    if data_dir is not None:
        for name, digest in shared_prepared.items():
            _require(_sha(data_dir / name) == digest,
                     f"current prepared half differs from study fingerprint: {name}")
    result = {
        "protocol": "seven-match nested leave-one-match-out fixed-selection evaluation",
        "interpretation": "descriptive match-level results; no IID-frame CI or paired significance test",
        "selection_rule": "one frozen inner-selected recipe per outer fold; no outer-model selection",
        "synthetic_stress_use": "41-percent ball-outage diagnostic only; no model change or promotion",
        "study_dir": str(study_dir),
        "prepared_verification": ("current prepared files and manifest SHA verified"
                                  if data_dir is not None else
                                  "cross-fold SHA consistency only; pass --data-dir for current-file check"),
        "matches": rows,
        "groups": {
            "all_seven": _group_summary(rows),
            "bundesliga_two": _group_summary([row for row in rows if row["match"] in BUNDESLIGA]),
            "duesseldorf_five": _group_summary([row for row in rows if row["match"] not in BUNDESLIGA]),
        },
        "resources": {
            "hpo_trials_total": sum(row["hpo_trials"] for row in rows),
            "decoder_trials_total": sum(row["decoder_trials"] for row in rows),
            "hpo_optimizer_steps_total": sum(row["hpo_optimizer_steps"] for row in rows),
            "refit_optimizer_steps_total": sum(row["refit_optimizer_steps"] for row in rows),
            "hpo_trials_per_outer": common_trial_counts[0],
            "decoder_trials_per_outer_both_poolings": common_trial_counts[1],
            "hpo_full_budget_steps_per_fold": first_hpo["resources"]["rungs"][-1],
        },
        "provenance": {"script_sha256": _sha(Path(__file__)),
                       "shared_hpo_configuration": first_hpo,
                       "shared_decoder_configuration": first_decoder,
                       "prepared_sha256": dict(sorted(shared_prepared.items())),
                       "by_match": hashes_by_match},
    }
    return result, rows


def _csv_text(rows: list[dict], groups: dict) -> str:
    columns = ["row_type", "match", "group", "selected_pooling", "neural_weight",
               "selection_score_inner_oof", "n_frames_native", "hpo_trials",
               "decoder_trials", "hpo_optimizer_steps", "refit_optimizer_steps"]
    for source in ("clean", "stress"):
        for metric in METRICS:
            columns.extend((f"{source}_{metric}_selected", f"{source}_{metric}_reference",
                            f"{source}_{metric}_selected_minus_reference"))
    columns.extend(f"stress_minus_clean_{metric}_selected" for metric in METRICS)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        flat = {"row_type": "match", **{key: row[key] for key in columns if key in row}}
        for source in ("clean", "stress"):
            for metric in METRICS:
                selected = row[source]["selected"][metric]
                reference = row[source]["reference"][metric]
                flat[f"{source}_{metric}_selected"] = selected
                flat[f"{source}_{metric}_reference"] = reference
                flat[f"{source}_{metric}_selected_minus_reference"] = selected - reference
        for metric in METRICS:
            flat[f"stress_minus_clean_{metric}_selected"] = (
                row["stress"]["selected"][metric] - row["clean"]["selected"][metric])
        writer.writerow(flat)
    labels = {"all_seven": "All seven match mean",
              "bundesliga_two": "Bundesliga two match mean",
              "duesseldorf_five": "Düsseldorf five match mean"}
    for key, group in groups.items():
        flat = {"row_type": "match_mean", "match": "", "group": labels[key],
                "selected_pooling": "", "neural_weight": group["neural_weights"]["mean"]}
        for source, group_key in (("clean", "clean"),
                                  ("stress", "synthetic_ball_outage_41_percent")):
            for metric in METRICS:
                values = group[group_key][metric]
                flat[f"{source}_{metric}_selected"] = values["selected_match_mean"]
                flat[f"{source}_{metric}_reference"] = values["reference_match_mean"]
                flat[f"{source}_{metric}_selected_minus_reference"] = (
                    values["selected_minus_reference_match_mean"])
        for metric in METRICS:
            flat[f"stress_minus_clean_{metric}_selected"] = (
                group["stress_minus_clean_selected_match_mean"][metric])
        writer.writerow(flat)
    return output.getvalue()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv=None) -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", nargs="?", type=Path,
                        default=root / "artifacts" / "optuna-audit")
    parser.add_argument("--data-dir", type=Path,
                        help="verify current prepared NPZ and manifest bytes against saved SHA")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args(argv)
    study_dir = args.study_dir.resolve()
    output_json = (args.output_json or study_dir / "nested-summary.json").resolve()
    output_csv = (args.output_csv or study_dir / "nested-summary.csv").resolve()
    _require(output_json != output_csv, "JSON and CSV output paths must differ")
    for output in (output_json, output_csv):
        _require(not any(output.is_relative_to(study_dir / match) for match in MATCHES),
                 "summary output cannot overwrite an outer fold artifact")
    result, rows = summarize_nested(study_dir, data_dir=args.data_dir)
    _atomic_text(output_json, json.dumps(result, indent=2, sort_keys=True,
                                         allow_nan=False) + "\n")
    _atomic_text(output_csv, _csv_text(rows, result["groups"]))
    print(json.dumps({"json": str(output_json), "csv": str(output_csv),
                      "matches": len(rows), "hpo_trials": result["resources"]["hpo_trials_total"],
                      "decoder_trials": result["resources"]["decoder_trials_total"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
