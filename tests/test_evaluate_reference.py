import hashlib
import json

import numpy as np
import pytest

from evaluation import possession_metrics as official
from scripts.evaluate_reference import _score_rows, evaluate_reference


def write_reference(path, *, include_all_halves=True):
    truth = {
        ("M1", 1): np.asarray([official.H] * 40 + [official.D] * 10, dtype=np.uint8),
        ("M1", 2): np.asarray([official.A] * 30 + [official.D] * 20, dtype=np.uint8),
    }
    arrays = {"gt_M1_p1": truth[("M1", 1)], "gt_M1_p2": truth[("M1", 2)]}
    rows = []
    for period in (1, 2):
        if period == 2 and not include_all_halves:
            continue
        gt = truth[("M1", period)]
        pred_key = f"pred_v4_batch_1_M1_clean_p{period}"
        arrays[pred_key] = gt.copy()
        rows.append({"batch": "batch_1", "scope": "inner", "match": "M1",
                     "fold": 0, "scenario": "clean", "period": period,
                     "n_frames": len(gt), "ground_truth_key": f"gt_M1_p{period}",
                     "prediction_keys": {"v4": pred_key}})
    manifest = {"schema": "public_dfl_decoded_reference_v1", "label_codes": {
        "H": official.H, "A": official.A, "D": official.D, "X": official.X},
        "scenario_weights": {"clean": 1.0}, "variants": ["v4"],
        "batches": ["batch_1"], "records": rows}
    path.mkdir(parents=True)
    labels_path = path / "labels.npz"
    np.savez_compressed(labels_path, **arrays)
    manifest["labels_sha256"] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    if include_all_halves:
        outer_rows = [dict(row, scope="outer", match="M2",
                               ground_truth_key=row["ground_truth_key"].replace("M1", "M2"),
                               prediction_keys={"v4": row["prediction_keys"]["v4"].replace("M1", "M2")})
                          for row in rows]
        outer_arrays = dict(arrays)
        for old, new in (("gt_M1_p1", "gt_M2_p1"), ("gt_M1_p2", "gt_M2_p2"),
                         ("pred_v4_batch_1_M1_clean_p1", "pred_v4_batch_1_M2_clean_p1"),
                         ("pred_v4_batch_1_M1_clean_p2", "pred_v4_batch_1_M2_clean_p2")):
            outer_arrays[new] = arrays[old]
        rows.extend(outer_rows)
        arrays.update({key: value for key, value in outer_arrays.items() if key.startswith(("gt_M2", "pred_v4_batch_1_M2"))})
        np.savez_compressed(labels_path, **arrays)
        manifest["records"] = rows
        manifest["labels_sha256"] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
        manifest["expected_scores"] = {"v4": {"batch_1": {
            "inner": _score_rows(arrays, manifest, "v4", "batch_1", "inner"),
            "outer": _score_rows(arrays, manifest, "v4", "batch_1", "outer")}}}
    manifest["label_array_hashes"] = {
        key: hashlib.sha256(value.tobytes(order="C")).hexdigest()
        for key, value in arrays.items()}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest, arrays


def test_reference_recomputes_with_half_separator_and_declared_score(tmp_path):
    manifest, _ = write_reference(tmp_path / "reference")
    result = evaluate_reference(tmp_path / "reference")
    assert result["recomputed_from_saved_oof_predictions"] is True
    assert result["independent_test"] is False
    assert result["scores"]["v4"]["batch_1"]["inner"] == manifest["expected_scores"]["v4"]["batch_1"]["inner"]


def test_reference_rejects_incomplete_half_grid(tmp_path):
    write_reference(tmp_path / "reference", include_all_halves=False)
    with pytest.raises(ValueError, match="grid"):
        evaluate_reference(tmp_path / "reference")


def test_reference_rejects_duplicate_match_scenario_half_rows(tmp_path):
    reference = tmp_path / "reference"
    manifest, _ = write_reference(reference)
    manifest["records"].append(manifest["records"][0])
    (reference / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="grid"):
        evaluate_reference(reference)


def test_reference_rejects_negative_or_unknown_label_codes(tmp_path):
    reference = tmp_path / "reference"
    manifest, arrays = write_reference(reference)
    key = "pred_v4_batch_1_M1_clean_p1"
    arrays[key] = arrays[key].astype(np.int8)
    arrays[key][0] = -1
    labels_path = reference / "labels.npz"
    np.savez_compressed(labels_path, **arrays)
    manifest["labels_sha256"] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    manifest["label_array_hashes"][key] = hashlib.sha256(arrays[key].tobytes(order="C")).hexdigest()
    (reference / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unknown code"):
        evaluate_reference(reference)
