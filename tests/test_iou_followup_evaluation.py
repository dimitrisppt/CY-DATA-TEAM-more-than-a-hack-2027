import copy
import hashlib

import numpy as np
import pytest
from test_gpu_iou_evaluation import VARIANTS, _reference

from experiments.campaign_protocol import publish
from experiments.gpu_iou_evaluation import score as original_score
from experiments.iou_followup_evaluation import load_reference, score, validate_grid
from possession.storage import sha256_file


def reference():
    labels, manifest = _reference()
    manifest["variants"] = list(VARIANTS)
    manifest["schema"] = "followup_iou_labels_v1"
    return labels, manifest


def test_generalized_scoring_reproduces_original_whole_match_metrics():
    labels, manifest = reference()
    expected, actual = original_score(labels, manifest), score(labels, manifest)
    for group, data in actual["groups"].items():
        assert data["summary"] == expected["groups"][group]["summary"]
        assert data["variants"] == expected["groups"][group]["variants"]
        assert data["comparisons"] == expected["groups"][group]["comparisons"]


def test_single_match_guard_has_no_spurious_confidence_interval():
    labels, manifest = reference()
    manifest["assessment_matches"] = ["match_0"]
    manifest["records"] = [r for r in manifest["records"] if r["match"] == "match_0"]
    actual = score(labels, manifest)["groups"]["ensemble"]
    assert actual["summary"]["spell_balanced"]["objective"] == pytest.approx(1.)
    assert actual["prespecified_gate"]["spell_balanced"]["prespecified_checks_passed"]
    for comparisons in actual["comparisons"].values():
        for data in comparisons.values():
            assert data["n_matches"] == 1
            assert data["descriptive_match_bootstrap_95_interval"] is None


@pytest.mark.parametrize("damage", ["missing_row", "duplicate_row", "missing_group", "missing_variant"])
def test_incomplete_or_duplicate_reference_rejected(damage):
    _, manifest = reference()
    if damage == "missing_row":
        manifest["records"].pop()
    elif damage == "duplicate_row":
        manifest["records"][-1] = copy.deepcopy(manifest["records"][0])
    elif damage == "missing_group":
        manifest["records"][0]["group_prediction_keys"].pop("ensemble")
    else:
        manifest["records"][0]["group_prediction_keys"]["ensemble"].pop("control")
    with pytest.raises(ValueError, match="incomplete"):
        validate_grid(manifest)


def test_followup_reference_rejects_tampering(tmp_path):
    labels, manifest = reference()
    np.savez_compressed(tmp_path / "labels.npz", **labels)
    manifest["labels_sha256"] = sha256_file(tmp_path / "labels.npz")
    manifest["array_sha256"] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in labels.items()}
    publish(tmp_path / "manifest.json", manifest)
    load_reference(tmp_path)
    with (tmp_path / "labels.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash differs"):
        load_reference(tmp_path)


def test_factorial_interaction_uses_paired_whole_matches():
    labels, manifest = reference()
    for row in manifest["records"]:
        for keys in row["group_prediction_keys"].values():
            keys["sampling_only"] = keys["control"]
            keys["loss_only"] = keys["spell_balanced"]
    manifest["variants"] += ["sampling_only", "loss_only"]
    result = score(labels, manifest)
    for data in result["groups"].values():
        assert data["factorial_interaction"]["mean"] == pytest.approx(0.)
        assert data["factorial_interaction"]["descriptive_match_bootstrap_95_interval"] == pytest.approx([0., 0.])
