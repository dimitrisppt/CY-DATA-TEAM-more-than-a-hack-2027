"""Synthetic metadata tests only, no match or participant data is opened."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from threading import Barrier

import pytest

from experiments.timing_validation import (
    BasePredictionProvenance,
    ComponentProvenance,
    ConfigLineage,
    build_holdout_plan,
    build_nested_plan,
    canonical_sha256,
    freeze_protocol,
    load_frozen_protocol,
    validate_group_assignments,
    validate_prediction_provenance,
    validate_prediction_records,
)

HASH = "a" * 64
MATCHES = list("ABCDEF")


def prediction(match="A", trained=("C", "D", "E", "F")):
    return asdict(BasePredictionProvenance(
        match=match, half=1, scenario="clean", split_id="fold0", feature_sha256=HASH,
        components=[ComponentProvenance(
            name="fine", kind="fine_model", train_matches=list(trained), selection_matches=[],
            fit_sha256=HASH, model_sha256=HASH,
            ancestors=[ComponentProvenance(
                name="standardization", kind="normalization", train_matches=list(trained),
                selection_matches=[], fit_sha256=HASH, model_sha256=HASH)])],
        config_lineage=[ConfigLineage(
            name="historical_v6", sha256=HASH, training_matches=[],
            selection_matches=MATCHES, historical_reuse=True)]))


def validate(row):
    return validate_prediction_provenance(
        row, outer_assessment_matches=["B"], allowed_matches=MATCHES)


def test_safe_holdout_reports_historical_reuse_separately():
    audit = validate(prediction())
    assert audit["training_matches"] == list("CDEF")
    assert audit["selection_matches"] == []
    assert audit["historical_selection_matches"] == MATCHES
    assert "does not undo" in audit["historical_reuse_caveat"]


@pytest.mark.parametrize("leak", ["A", "B"])
def test_indirect_leakage_through_normalization_ancestor(leak):
    row = prediction()
    row["components"][0]["ancestors"][0]["train_matches"].append(leak)
    with pytest.raises(ValueError, match="leakage"):
        validate(row)


@pytest.mark.parametrize("key,value", [
    ("kind", "unknown_fit"), ("train_matches", []), ("fit_sha256", None),
    ("model_sha256", "corrupt"), ("ancestors", None),
])
def test_unknown_or_absent_fit_provenance_rejected(key, value):
    row = prediction()
    row["components"][0][key] = value
    with pytest.raises(ValueError):
        validate(row)


def test_missing_ancestor_key_and_unrecognized_metadata_rejected():
    row = prediction()
    del row["components"][0]["ancestors"]
    with pytest.raises(ValueError, match="missing or unrecognized"):
        validate(row)
    row = prediction()
    row["unrecognized_ancestry"] = []
    with pytest.raises(ValueError, match="missing or unrecognized"):
        validate(row)


@pytest.mark.parametrize("bad_id", ["A_half1", "../A", "A/clean", "A ", "", None])
def test_corrupt_match_ids_rejected(bad_id):
    row = prediction()
    row["match"] = bad_id
    with pytest.raises(ValueError):
        validate(row)


def test_unknown_match_id_rejected_even_if_not_forbidden():
    row = prediction()
    row["components"][0]["train_matches"].append("Z")
    with pytest.raises(ValueError, match="unknown match"):
        validate(row)


def test_false_complete_attestation_rejected():
    row = prediction()
    row["ancestry_complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        validate(row)


def test_expected_inventory_rejects_missing_coarse():
    with pytest.raises(ValueError, match="inventory"):
        validate_prediction_provenance(prediction(), outer_assessment_matches=["B"],
                                       expected_components=["fine", "coarse"])


def test_current_configuration_selection_is_leakage():
    row = prediction()
    row["config_lineage"][0]["historical_reuse"] = False
    with pytest.raises(ValueError, match="configuration lineage"):
        validate(row)


def test_historical_training_is_not_exempt():
    row = prediction()
    row["config_lineage"][0]["training_matches"] = ["B"]
    with pytest.raises(ValueError, match="configuration lineage"):
        validate(row)


def test_component_selection_is_checked_recursively():
    row = prediction()
    row["components"][0]["ancestors"][0]["selection_matches"] = ["B"]
    with pytest.raises(ValueError, match="leakage"):
        validate(row)


def test_cycles_fail_closed():
    row = prediction()
    row["components"][0]["ancestors"] = [row["components"][0]]
    with pytest.raises(ValueError, match="cyclic"):
        validate(row)


def test_record_coverage_and_duplicate_samples():
    rows = [prediction()]
    with pytest.raises(ValueError, match="cover"):
        validate_prediction_records(rows, outer_assessment_matches=["B"],
                                    expected_matches=["A", "C"])
    with pytest.raises(ValueError, match="duplicate"):
        validate_prediction_records(rows * 2, outer_assessment_matches=["B"])


@pytest.mark.parametrize("half,scenario", [(2, "clean"), (1, "short"), (2, "replay")])
def test_scenarios_and_halves_cannot_cross_splits(half, scenario):
    rows = [{"match": "A", "half": 1, "scenario": "clean", "split": "calibration"},
            {"match": "A", "half": half, "scenario": scenario, "split": "assessment"}]
    with pytest.raises(ValueError, match="same match"):
        validate_group_assignments(rows)
    rows[1]["split"] = "calibration"
    assert validate_group_assignments(rows) == {"A": "calibration"}


@pytest.mark.parametrize("search", [False, True])
def test_nested_fold_coverage_and_all_fit_exclusions(search):
    plan = build_nested_plan(MATCHES, search=search)
    assert plan["ordinary_oof_cache_is_sufficient"] is False
    assert sorted(m for fold in plan["folds"] for m in fold["assessment_matches"]) == MATCHES
    for fold in plan["folds"]:
        outer = set(fold["assessment_matches"])
        assert len(outer) == 2
        assert len(fold["training_matches"]) == 4
        assert len(fold["prediction_requests"]) == 5
        for request in fold["prediction_requests"]:
            fitted = set(request["train_matches"])
            assert not fitted & set(request["forbidden_matches"])
            assert not fitted & set(request["predict_matches"])
            assert not fitted & outer
            assert len(fitted) == (3 if request["role"] == "calibration" else 4)
        inner = fold["inner_selection"]
        assert len(inner) == (4 if search else 0)
        if search:
            assert sorted(m for item in inner for m in item["assessment_matches"]) == fold[
                "training_matches"]
        for item in inner:
            assert len(item["training_matches"]) == 3
            for request in item["prediction_requests"]:
                fitted = set(request["train_matches"])
                assert not fitted & set(request["forbidden_matches"])
                assert not fitted & set(item["assessment_matches"])
                assert not fitted & outer
                assert len(fitted) == (2 if request["role"] == "calibration" else 3)


def test_predeclared_folds_and_deterministic_default():
    folds = [["C", "A"], ["B", "D"], ["F", "E"]]
    plan = build_nested_plan(MATCHES, outer_assessment_folds=folds)
    assert plan["folds"][0]["assessment_matches"] == ["A", "C"]
    assert build_nested_plan(MATCHES) == build_nested_plan(MATCHES[::-1])
    with pytest.raises(ValueError, match="cover"):
        build_nested_plan(MATCHES, outer_assessment_folds=[["A", "B"], ["A", "C"], ["E", "F"]])
    with pytest.raises(ValueError):
        build_nested_plan(MATCHES + ["A"])


def test_holdout_keeps_same_base_for_calibration_and_assessment():
    plan = build_holdout_plan(list("CDEF"), ["A"], ["B"])
    assert plan["mode"] == "holdout_fixed"
    for request in plan["prediction_requests"]:
        assert request["train_matches"] == list("CDEF")
        assert request["forbidden_matches"] == ["A", "B"]
    with pytest.raises(ValueError, match="overlap"):
        build_holdout_plan(["A", "B"], ["B"], ["C"])


def test_canonical_hash_and_frozen_protocol_reuse(tmp_path):
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
    with pytest.raises(ValueError):
        canonical_sha256({"score": float("nan")})
    path = tmp_path / "protocol.json"
    protocol = build_nested_plan(MATCHES)
    assert freeze_protocol(path, protocol) == protocol
    initial = path.read_bytes()
    assert freeze_protocol(path, protocol) == protocol
    assert path.read_bytes() == initial
    with pytest.raises(ValueError, match="differs from requested"):
        freeze_protocol(path, build_nested_plan(MATCHES, search=True))
    assert path.read_bytes() == initial
    assert set(tmp_path.iterdir()) == {path, tmp_path / ".protocol.json.lock"}


def test_corrupt_frozen_protocol_and_duplicate_keys_rejected(tmp_path):
    path = tmp_path / "protocol.json"
    freeze_protocol(path, build_nested_plan(MATCHES))
    row = json.loads(path.read_text())
    row["payload"]["mode"] = "changed"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="checksum"):
        load_frozen_protocol(path)
    path.write_text('{"sha256":"a","sha256":"b","payload":{}}')
    with pytest.raises(ValueError, match="duplicate"):
        load_frozen_protocol(path)


def test_freeze_works_without_hard_link_support(tmp_path, monkeypatch):
    def unsupported_link(*args, **kwargs):
        raise OSError(45, "Operation not supported")
    monkeypatch.setattr(os, "link", unsupported_link)
    path = tmp_path / "protocol.json"
    protocol = build_nested_plan(MATCHES)
    assert freeze_protocol(path, protocol) == protocol
    lock = tmp_path / ".protocol.json.lock"
    inode = lock.stat().st_ino
    assert freeze_protocol(path, protocol) == protocol
    assert lock.stat().st_ino == inode


def test_concurrent_freeze_never_replaces_different_winners(tmp_path):
    path = tmp_path / "protocol.json"
    barrier = Barrier(8)
    def writer(index):
        barrier.wait(timeout=10)
        try:
            return freeze_protocol(path, {"winner": index})
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(writer, range(8)))
    accepted = [outcome for outcome in outcomes if outcome is not None]
    assert len(accepted) == 1
    assert load_frozen_protocol(path) == accepted[0]
    before = path.read_bytes()
    with pytest.raises(ValueError, match="differs from requested"):
        freeze_protocol(path, {"winner": -1})
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == {path, tmp_path / ".protocol.json.lock"}


def test_interrupted_publication_keeps_protocol_absent_and_releases_lock(tmp_path, monkeypatch):
    path = tmp_path / "protocol.json"
    def interrupted_replace(*args, **kwargs):
        raise OSError("simulated publication failure")
    with monkeypatch.context() as context:
        context.setattr(os, "replace", interrupted_replace)
        with pytest.raises(OSError, match="simulated"):
            freeze_protocol(path, {"config": 1})
    assert not path.exists()
    assert list(tmp_path.iterdir()) == [tmp_path / ".protocol.json.lock"]
    assert freeze_protocol(path, {"config": 2}) == {"config": 2}
