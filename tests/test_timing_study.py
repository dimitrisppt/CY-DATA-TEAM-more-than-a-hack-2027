"""Independent synthetic protocol and scoring tests, without public or target data."""

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest

from evaluation import possession_metrics as official
from experiments import timing_study as study
from experiments.timing_validation import (
    BasePredictionProvenance,
    ComponentProvenance,
    ConfigLineage,
    canonical_sha256,
)

HASH = "a" * 64
UNIVERSE = list("ABCDEFG")
COMPONENTS = {"inner": ["fine"], "outer": ["fine"]}


def records():
    result = []
    for batch in ("batch_1", "batch_2"):
        for index, match in enumerate(UNIVERSE):
            scope = "inner" if index < 6 else "outer"
            fold = index // 2 if scope == "inner" else 0
            excluded = set(UNIVERSE[2 * fold:2 * fold + 2]) if scope == "inner" else {"G"}
            train = sorted(set(UNIVERSE[:6]) - excluded)
            for scenario in study.WEIGHTS:
                for half in (1, 2):
                    provenance = asdict(BasePredictionProvenance(
                        match, half, scenario, f"{scope}_fold_{fold}", HASH,
                        [ComponentProvenance("fine", "fine_model", train, [], HASH, HASH)],
                        [ConfigLineage("v6_recipe", HASH, [], UNIVERSE, True)]))
                    result.append({"batch": batch, "match": match, "half": half,
                                   "scenario": scenario, "fold": fold, "scope": scope,
                                   "sha256": HASH, "path": "synthetic-unused.npz",
                                   "provenance": provenance})
    return result


def test_paired_three_way_splits_and_historical_guard_are_explicit():
    splits = study._make_splits(records())
    paired = [row for row in splits if row["kind"] == "paired_holdout"]
    assert len(paired) == 6
    assert sorted(row["assessment"][0] for row in paired) == list("ABCDEF")
    for split in paired:
        assert len(split["base_train"]) == 4
        assert len(split["calibration"]) == len(split["assessment"]) == 1
        assert not set(split["base_train"]) & set(split["calibration"] + split["assessment"])
        assert split["calibration"] != split["assessment"]
        assert split["plan"]["mode"] == "holdout_fixed"
    guard = splits[-1]
    assert guard["kind"] == "oof_training_guard"
    assert guard["calibration"] == list("ABCDEF")
    assert guard["assessment"] == ["G"]
    assert guard["plan"] is None


@pytest.mark.parametrize("corruption", ["overlap", "missing_pair", "guard_overlap", "unknown_scope"])
def test_make_splits_rejects_invalid_protocol(corruption):
    rows = records()
    if corruption == "overlap":
        for row in rows:
            if row["match"] == "C":
                row["match"] = "A"
    elif corruption == "missing_pair":
        rows = [row for row in rows if row["match"] != "F"]
    elif corruption == "guard_overlap":
        for row in rows:
            if row["scope"] == "outer":
                row["match"] = "A"
    else:
        rows[0]["scope"] = "unrecognized"
    with pytest.raises(ValueError):
        study._make_splits(rows)


def test_audit_accepts_both_disjoint_paired_and_guard_designs():
    rows = records()
    paired = [row for row in rows if row["match"] in ("A", "B")]
    assert len(study._audit(paired, ["B"], UNIVERSE, COMPONENTS)) == len(paired)
    training = [row for row in rows if row["scope"] == "inner"]
    assessment = [row for row in rows if row["scope"] == "outer"]
    assert study._audit(training, ["G"], UNIVERSE, COMPONENTS)
    assert study._audit(assessment, ["G"], UNIVERSE, COMPONENTS)


@pytest.mark.parametrize("corruption", ["fit_leak", "ancestor_leak", "missing_scenario", "identity",
                                         "artifact_binding", "split_binding"])
def test_audit_rejects_unsafe_cache(corruption):
    rows = [row for row in records() if row["match"] == "A"]
    if corruption == "fit_leak":
        rows[0]["provenance"]["components"][0]["train_matches"].append("B")
    elif corruption == "ancestor_leak":
        rows[0]["provenance"]["components"][0]["ancestors"].append(asdict(ComponentProvenance(
            "scale", "normalization", ["B"], [], HASH, HASH)))
    elif corruption == "missing_scenario":
        rows.pop()
    elif corruption == "identity":
        rows[0]["provenance"]["half"] = 2
    elif corruption == "artifact_binding":
        rows[0]["sha256"] = "b" * 64
    else:
        rows[0]["provenance"]["split_id"] = "inner_fold_999"
    with pytest.raises(ValueError):
        study._audit(rows, ["B"], UNIVERSE, COMPONENTS)


def test_score_matches_official_spell_metric_and_predeclared_weighting():
    truth, prediction = {}, {}
    for batch in ("batch_1", "batch_2"):
        for match in ("A", "B"):
            for scenario in study.WEIGHTS:
                for half in (1, 2):
                    key = (batch, match, scenario, half)
                    # Different match lengths make frame pooling incorrect.
                    repeats = 1 if match == "A" else 3
                    gt = np.tile(np.array([official.H] * 8 + [official.A] * 8 + [official.D] * 4,
                                          np.uint8), repeats)
                    truth[key] = gt
                    prediction[key] = gt.copy() if scenario == "clean" else np.full_like(gt, official.D)
    score = study._score(prediction, truth)
    fast = study._score(prediction, truth, full=False)
    expected = {}
    for scenario in study.WEIGHTS:
        values = []
        for batch in ("batch_1", "batch_2"):
            for match in ("A", "B"):
                keys = [(batch, match, scenario, half) for half in (1, 2)]
                gt = study._join([truth[key] for key in keys])
                pred = study._join([prediction[key] for key in keys])
                values.append(official.evaluate(gt, pred, bootstrap_n=0)["spell_f1_mean"])
        expected[scenario] = np.mean(values)
    assert score["means"] == pytest.approx(expected)
    assert score["objective"] == pytest.approx(sum(study.WEIGHTS[k] * expected[k] for k in expected))
    assert score["missing_ball"] == pytest.approx(
        sum(study.WEIGHTS[k] * expected[k] for k in ("short", "long", "replay")) / .5)
    assert fast["objective"] == pytest.approx(score["objective"])
    assert len(score["records"]) == 16


def summary_rows():
    result = []
    for index, match in enumerate("ABCDEFG"):
        kind = "paired_holdout" if index < 6 else "oof_training_guard"
        scores = {}
        for method in study.METHODS:
            value = .5 if method == "identity" else .51
            scores[method] = {"objective": value, "missing_ball": value,
                              "means": {scenario: value for scenario in study.WEIGHTS}}
        result.append({"split": {"kind": kind, "assessment": [match]}, "scores": scores})
    return result


def test_summary_equal_match_deltas_guard_separation_and_development_interpretation():
    rows = summary_rows()
    rows[-1]["scores"]["contextual"]["objective"] = .49
    result = study.summarize(rows)
    contextual = result["paired_holdout"]["contextual"]
    assert contextual["objective"] == pytest.approx(.51)
    assert contextual["objective_delta"] == pytest.approx(.01)
    assert contextual["matches_improved"] == 6
    assert set(contextual["by_match_objective_delta"]) == set("ABCDEF")
    assert contextual["descriptive_match_bootstrap_95pct"] == pytest.approx([.01, .01])
    assert result["historical_guard"]["contextual"]["objective"] == .49
    assert result["promotion_authorized_by_this_study"] is False
    assert "not the full nested" in result["interpretation"]
    assert "historical reuse" in result["interpretation"]
    assert study.summarize(rows) == result


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "two_guards", "overlap_guard"])
def test_summary_rejects_incomplete_or_duplicated_assessments(corruption):
    rows = summary_rows()
    if corruption == "missing":
        rows.pop(0)
    elif corruption == "duplicate":
        rows.insert(0, copy.deepcopy(rows[0]))
    elif corruption == "two_guards":
        rows.append(copy.deepcopy(rows[-1]))
    else:
        rows[-1]["split"]["assessment"] = ["A"]
    with pytest.raises(ValueError):
        study.summarize(rows)


def test_runner_reads_verified_cache_manifest_envelope_before_planning(tmp_path, monkeypatch):
    payload = {"schema": "public_dfl_timing_cache_v1", "records": records()}
    manifest = {"sha256": canonical_sha256(payload), "payload": payload}
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(json.dumps(manifest))
    def stop_after_manifest(rows):
        assert len(rows) == len(payload["records"])
        raise RuntimeError("manifest was verified and loaded")
    monkeypatch.setattr(study, "_make_splits", stop_after_manifest)
    with pytest.raises(RuntimeError, match="manifest was verified"):
        study.run_study(cache, tmp_path / "output")


@pytest.mark.parametrize("corruption", ["checksum", "schema"])
def test_runner_rejects_unverified_manifest_before_planning(tmp_path, monkeypatch, corruption):
    payload = {"schema": "public_dfl_timing_cache_v1", "records": records()}
    if corruption == "schema":
        payload["schema"] = "unrecognized"
    manifest = {"sha256": canonical_sha256(payload), "payload": payload}
    if corruption == "checksum":
        manifest["sha256"] = HASH
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(study, "_make_splits", lambda _: pytest.fail("unsafe manifest reached planning"))
    with pytest.raises(ValueError):
        study.run_study(cache, tmp_path / "output")


def test_study_protocol_schema_freezes_and_rejects_configuration_drift(tmp_path):
    protocol = {"schema": "timing_paired_study_v1", "config": asdict(study.TimingConfig()),
                "splits": study._make_splits(records()), "scenario_weights": study.WEIGHTS}
    path = tmp_path / "protocol.json"
    assert study.freeze_protocol(path, protocol) == protocol
    assert study.freeze_protocol(path, protocol) == protocol
    changed = copy.deepcopy(protocol)
    changed["config"]["radius"] += 1
    with pytest.raises(ValueError, match="differs from requested"):
        study.freeze_protocol(path, changed)


def test_training_receives_boundary_weight_vectors(tmp_path, monkeypatch):
    rows = records()
    split = study._make_splits(rows)[0]
    monkeypatch.setattr(study, "fit_offsets", lambda _: {"global": 0, "family": [0, 0, 0]})
    labels = np.array([official.D] * 4 + [official.H] * 8 + [official.A] * 8, np.uint8)
    arrays = {"labels": labels, "gt": labels,
              "neural_live": np.full(len(labels), .8),
              "neural_home": np.full(len(labels), .5),
              "neural_boundary": np.zeros((len(labels), 3, 3)),
              "ball_valid": np.ones(len(labels), bool),
              "nearest_ball_seconds": np.zeros(len(labels)),
              "fine_coarse_disagreement": np.zeros(len(labels))}
    monkeypatch.setattr(study, "_load", lambda row, **kwargs: arrays)
    def inspect_fit(batches, targets, *, match_ids, weights, config):
        assert len(weights) == len(batches)
        for batch, weight in zip(batches, weights):
            assert np.shape(weight) == (len(batch.boundaries),)
            assert np.all(np.asarray(weight) >= 0)
        train = [row for row in rows if row["match"] in split["calibration"]
                 and row["scope"] == "inner"]
        for scenario, expected in study.WEIGHTS.items():
            mass = sum(np.sum(weight) for row, weight in zip(train, weights)
                       if row["scenario"] == scenario)
            assert mass == pytest.approx(expected)
        raise RuntimeError("weights inspected")
    monkeypatch.setattr(study, "fit_timing_model", inspect_fit)
    with pytest.raises(RuntimeError, match="weights inspected"):
        study._run_split(split, rows, tmp_path, study.TimingConfig(), HASH,
                         UNIVERSE, COMPONENTS, lambda *args, **kwargs: None)
