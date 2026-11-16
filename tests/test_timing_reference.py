"""Portable timing reference tests use seven tiny synthetic public matches."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation import possession_metrics as official

METHODS = ("identity", "global_offset", "family_offsets", "contextual")
WEIGHTS = {"clean": .5, "short": .2, "long": .15, "replay": .15}


def _module():
    from scripts import evaluate_timing_reference
    return evaluate_timing_reference


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _seal(path, payload):
    _write(path, {"sha256": _digest(payload), "payload": payload})


def _source_fixture(tmp_path):
    root = Path(__file__).resolve().parents[1]
    study, public = tmp_path / "study-v2", tmp_path / "public"
    matches = [f"DFL-{index}" for index in range(7)]
    arrays, records, splits, split_results = {}, [], [], []
    batches = ["batch_1", "batch_2"]
    truth = np.array([official.X, official.D, official.D, official.H, official.H,
                      official.A, official.A, official.D, official.D, official.X], np.uint8)
    identity = truth.copy()
    identity[3] = official.D
    for match in matches:
        for half in (1, 2):
            arrays[f"gt_{match}_p{half}"] = truth
        for batch in batches:
            for scenario in WEIGHTS:
                for half in (1, 2):
                    key = f"pred_{batch}_{match}_{scenario}_{half}"
                    arrays[key] = identity
                    records.append({"batch": batch, "match": match, "scenario": scenario,
                                    "period": half, "n_frames": len(truth),
                                    "scope": "outer" if match == matches[-1] else "inner",
                                    "ground_truth_key": f"gt_{match}_p{half}",
                                    "prediction_keys": {"coarse_v6": key}})
    public.mkdir()
    np.savez_compressed(public / "labels.npz", **arrays)
    manifest = {"schema": "public_dfl_decoded_reference_v1", "batches": batches,
                "scenario_weights": WEIGHTS, "scenario_order": list(WEIGHTS),
                "label_codes": {"H": 0, "A": 1, "D": 2, "X": 3}, "records": records,
                "labels_sha256": _sha(public / "labels.npz"),
                "label_array_hashes": {key: hashlib.sha256(value.tobytes()).hexdigest()
                                       for key, value in arrays.items()},
                "source_provenance": {"dataset_attribution": {
                    "license": "CC BY 4.0", "authors": "Synthetic fixture", "changes": "Labels only"}}}
    _write(public / "manifest.json", manifest)
    for index, match in enumerate(matches):
        guard = index == 6
        calibration = matches[:6] if guard else [matches[index ^ 1]]
        base = matches[:6] if guard else [item for item in matches[:6]
                                               if item not in [match, *calibration]]
        split = {"id": f"split_{index}", "kind": "oof_training_guard" if guard else "paired_holdout",
                 "assessment": [match], "calibration": calibration, "base_train": base,
                 "plan": None}
        splits.append(split)
    protocol = {"schema": "timing_paired_study_v1", "splits": splits,
                "scenario_weights": WEIGHTS, "target_labels_used": False,
                "historical_reuse_caveat": "Historically reused development evidence.",
                "sources": {name: _sha(root / name) for name in (
                    "possession/timing.py", "evaluation/possession_metrics.py")}}
    protocol_sha = _digest(protocol)
    _seal(study / "protocol.json", protocol)
    for split in splits:
        match = split["assessment"][0]
        directory = study / "splits" / split["id"]
        directory.mkdir(parents=True)
        np.savez_compressed(directory / "model.npz", metadata=np.array(json.dumps({
            "version": 2, "feature_names": ["typed_boundary_score"]})), coefficients=np.zeros(1))
        predictions, scores = {}, {}
        for method in METHODS:
            rows = []
            pred = truth if method == "contextual" else identity
            joined_gt = np.r_[truth, official.X, truth]
            measured = official.evaluate(joined_gt, np.r_[pred, official.X, pred], bootstrap_n=0)
            for batch in batches:
                for scenario in WEIGHTS:
                    rows.append({"batch": batch, "match": match, "scenario": scenario,
                                 **{key: measured[key] for key in (
                                     "spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells",
                                     "fragmentation", "n_graded")}})
                    for half in (1, 2):
                        predictions[f"{method}__{batch}__{match}__{scenario}__{half}"] = pred
            value = measured["spell_f1_mean"]
            scores[method] = {"objective": value, "missing_ball": value,
                              "means": dict.fromkeys(WEIGHTS, value), "records": rows}
        np.savez_compressed(directory / "predictions.npz", **predictions)
        audits = {}
        for role, samples in (("training_audit", split["calibration"]),
                              ("assessment_audit", split["assessment"])):
            audit = []
            for sample in samples:
                for number in range(len(batches) * len(WEIGHTS) * 2):
                    training = [item for item in split["base_train"] if item != sample]
                    audit.append({"match": sample, "training_matches": training,
                                  "selection_matches": [], "historical_selection_matches": matches,
                                  "provenance_sha256": _digest([sample, number]),
                                  "historical_reuse_caveat": protocol["historical_reuse_caveat"]})
            audits[role] = audit
        result = {"split": split, "protocol_sha256": protocol_sha, "scores": scores,
                  "model_sha256": _sha(directory / "model.npz"),
                  "predictions_sha256": _sha(directory / "predictions.npz"), **audits}
        split_results.append(result)
        _write(directory / "result.json", result)
    paired = {}
    for method in METHODS:
        score = split_results[0]["scores"][method]
        paired[method] = {key: score[key] for key in ("objective", "missing_ball", "means")}
        paired[method]["objective_delta"] = score["objective"] - split_results[0]["scores"]["identity"]["objective"]
    result = {"schema": "timing_paired_result_v1", "protocol_sha256": protocol_sha,
              "splits": split_results, "summary": {"paired_holdout": paired,
                  "historical_guard": split_results[-1]["scores"],
                  "promotion_authorized_by_this_study": False}}
    _seal(study / "result.json", result)
    return study, public


def test_export_and_recompute_separate_six_paired_matches_from_guard(tmp_path):
    module = _module()
    study, public = _source_fixture(tmp_path)
    destination = tmp_path / "portable"
    module.export_reference(study, destination, public)
    result = module.evaluate_reference(destination)
    assert result["independent_test"] is False
    assert result["paired_match_count"] == 6
    assert result["historical_guard_match_count"] == 1
    assert result["summary"]["paired_holdout"]["contextual"]["objective"] == 1
    assert result["summary"]["paired_holdout"]["contextual"]["objective_delta"] > 0
    assert len(result["delta_table"]) == 4
    manifest = json.loads((destination / "manifest.json").read_text())
    assert len(manifest["records"]) == 112
    with np.load(destination / "labels.npz", allow_pickle=False) as arrays:
        assert len([key for key in arrays if key.startswith("gt_")]) == 14
        assert len(arrays.files) == 14 + 4 * 112
    assert str(tmp_path) not in (destination / "manifest.json").read_text()
    assert manifest["source_provenance"]["dataset_attribution"]["license"] == "CC BY 4.0"


def test_source_artifact_hash_tamper_is_rejected_before_export(tmp_path):
    module = _module()
    study, public = _source_fixture(tmp_path)
    path = study / "splits/split_0/predictions.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash|checksum"):
        module.export_reference(study, tmp_path / "portable", public)


def test_recomputed_scores_detect_falsely_resealed_source_scores(tmp_path):
    module = _module()
    study, public = _source_fixture(tmp_path)
    wrapper = json.loads((study / "result.json").read_text())
    result = wrapper["payload"]
    result["splits"][0]["scores"]["contextual"]["objective"] = .1
    _write(study / "splits/split_0/result.json", result["splits"][0])
    _seal(study / "result.json", result)
    with pytest.raises(ValueError, match="score"):
        module.export_reference(study, tmp_path / "portable", public)


def test_guard_cannot_be_reclassified_as_paired_match(tmp_path):
    module = _module()
    study, public = _source_fixture(tmp_path)
    wrapper = json.loads((study / "protocol.json").read_text())
    protocol = wrapper["payload"]
    protocol["splits"][-1]["kind"] = "paired_holdout"
    _seal(study / "protocol.json", protocol)
    with pytest.raises(ValueError):
        module.export_reference(study, tmp_path / "portable", public)


def test_array_checksums_and_x_masks_are_verified_independently(tmp_path):
    module = _module()
    study, public = _source_fixture(tmp_path)
    destination = tmp_path / "portable"
    module.export_reference(study, destination, public)
    path = destination / "labels.npz"
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    key = next(key for key in arrays if key.startswith("pred_contextual"))
    arrays[key][0] = official.H
    np.savez_compressed(path, **arrays)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["labels_sha256"] = _sha(path)
    manifest["manifest_sha256"] = _digest({key: value for key, value in manifest.items()
                                            if key != "manifest_sha256"})
    _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="array|checksum"):
        module.evaluate_reference(destination)
    manifest["label_array_hashes"][key] = hashlib.sha256(arrays[key].tobytes()).hexdigest()
    manifest["manifest_sha256"] = _digest({key: value for key, value in manifest.items()
                                            if key != "manifest_sha256"})
    _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="mask"):
        module.evaluate_reference(destination)


def test_portable_protocol_snapshot_hash_is_rechecked(tmp_path):
    module = _module()
    study, public = _source_fixture(tmp_path)
    destination = tmp_path / "portable"
    module.export_reference(study, destination, public)
    path = destination / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_provenance"]["protocol"]["target_labels_used"] = True
    manifest["manifest_sha256"] = _digest({key: value for key, value in manifest.items()
                                            if key != "manifest_sha256"})
    _write(path, manifest)
    with pytest.raises(ValueError, match="protocol"):
        module.evaluate_reference(destination)


def test_portable_ancestry_summary_rejects_assessment_fit_ids(tmp_path):
    module = _module()
    study, public = _source_fixture(tmp_path)
    destination = tmp_path / "portable"
    module.export_reference(study, destination, public)
    path = destination / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["training_ancestry_id_audit"]["split_0"]["training_audit"]["training_matches"].append("DFL-0")
    manifest["manifest_sha256"] = _digest({key: value for key, value in manifest.items()
                                            if key != "manifest_sha256"})
    _write(path, manifest)
    with pytest.raises(ValueError, match="ancestry"):
        module.evaluate_reference(destination)
