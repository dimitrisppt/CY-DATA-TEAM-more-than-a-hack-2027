"""Whole-match aggregation, CUDA inference, and incomplete-study rejection."""

import copy
import hashlib

import numpy as np
import pytest
import torch

from experiments.campaign_protocol import WEIGHTS, publish
from experiments.gpu_iou_evaluation import (
    GROUPS,
    VARIANTS,
    _blend,
    _predict,
    load_reference,
    score,
    verify_training,
    write_report,
)
from experiments.gpu_runtime import configure_device
from experiments.robustness import _digest
from possession.features import build_features
from possession.model import PossessionModel
from possession.predict import predict_features
from possession.schema import FrameTable
from possession.storage import sha256_file


def _reference():
    matches = [f"match_{i}" for i in range(6)]
    labels, rows = {}, []
    gt = np.repeat([0, 2, 1, 2, 0], [25, 5, 60, 4, 31]).astype(np.uint8)
    for batch in ("batch_1", "batch_2"):
        for match in matches:
            for scenario in WEIGHTS:
                for period in (1, 2):
                    key = f"{batch}_{match}_{scenario}_{period}"
                    labels[key + "gt"] = gt.copy()
                    groups = {}
                    for group in GROUPS:
                        groups[group] = {}
                        for variant in VARIANTS:
                            name = key + group + variant
                            labels[name] = gt.copy() if variant == "spell_balanced" else np.zeros_like(gt)
                            groups[group][variant] = name
                    rows.append({"batch": batch, "scope": "inner", "match": match,
                                 "period": period, "scenario": scenario,
                                 "ground_truth_key": key + "gt", "group_prediction_keys": groups,
                                 "n_frames": len(gt)})
    return labels, {"schema": "fresh_gpu_iou_labels_v1", "protocol_sha256": "test",
                    "assessment_matches": matches, "scenario_weights": WEIGHTS,
                    "scenario_order": list(WEIGHTS), "batches": ["batch_1", "batch_2"],
                    "groups": list(GROUPS), "records": rows}


def test_complete_match_scoring_and_report(tmp_path):
    labels, manifest = _reference()
    result = score(labels, manifest)
    assert result["independent_test"] is False
    for group in GROUPS:
        data = result["groups"][group]
        assert data["summary"]["spell_balanced"]["objective"] == pytest.approx(1.)
        assert data["comparisons"]["masked_teacher"]["control"]["objective_delta"] == 0.
        assert data["comparisons"]["masked_teacher"]["control"]["n_matches"] == 6
        for curve in data["variants"]["spell_balanced"]["curves"].values():
            assert curve["weighted_f1"] == pytest.approx(1.)
    gate = result["groups"]["ensemble"]["development_screen"]
    assert gate["spell_balanced"]["confirmation_eligible"]
    assert not gate["masked_teacher"]["confirmation_eligible"]
    write_report(result, tmp_path)
    assert (tmp_path / "report/iou-curves.pdf").stat().st_size > 100
    assert (tmp_path / "report/match-metrics.csv").read_text().count("\n") == 73


def test_incomplete_assessment_grid_rejected():
    labels, manifest = _reference()
    manifest["records"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        score(labels, manifest)


def test_reference_tampering_rejected(tmp_path):
    labels, manifest = _reference()
    np.savez_compressed(tmp_path / "labels.npz", **labels)
    manifest["labels_sha256"] = sha256_file(tmp_path / "labels.npz")
    manifest["array_sha256"] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in labels.items()}
    publish(tmp_path / "manifest.json", manifest)
    load_reference(tmp_path)
    with (tmp_path / "labels.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        load_reference(tmp_path)


def test_incomplete_training_rejected(tmp_path):
    plan = {"seeds": [8101, 8102], "selection": {"members": [{"id": "a"}, {"id": "b"}]},
            "variants": ["control", "spell_balanced", "masked_teacher"]}
    publish(tmp_path / "training.json", {"protocol_sha256": _digest(plan), "fits": []})
    with pytest.raises(ValueError, match="18 parents and 36"):
        verify_training(plan, tmp_path)


def test_blend_uses_logits_and_preserves_endpoints():
    first = {"neural_live": np.array([.1, .9]), "neural_home": np.array([.2, .8]),
             "neural_boundary": np.zeros((2, 3, 3))}
    second = {"neural_live": 1 - first["neural_live"], "neural_home": 1 - first["neural_home"],
              "neural_boundary": np.ones((2, 3, 3))}
    mixed = _blend(first, second, .5)
    for array in mixed.values():
        np.testing.assert_allclose(array, .5)
    for actual, expected in ((_blend(first, second, 1), first), (_blend(first, second, 0), second)):
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
    broken = copy.deepcopy(second)
    broken["neural_home"] = broken["neural_home"][:1]
    with pytest.raises(ValueError, match="shapes"):
        _blend(first, broken, .5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_cuda_inference_cached_without_cpu_fallback(tmp_path):
    device = configure_device()
    n = 32
    frames = FrameTable(np.arange(n), np.ones((n, 2, 2, 2), np.float32),
                        np.ones((n, 2, 2), bool), np.ones((n, 2), np.float32),
                        np.ones(n, bool), np.ones(n, bool))
    features = build_features(frames)
    model = PossessionModel(width=8, dilations=[1]).to(device)
    fit, source = {"identity": {"test": 1}, "checkpoint_sha256": "a"}, {"input_sha256": "b"}
    actual = _predict({"frame_id": frames.frame_id}, features, model, source, fit,
                      tmp_path / "prediction.npz", device)
    assert next(model.parameters()).is_cuda
    expected = predict_features(features, model, device=device)
    np.testing.assert_array_equal(actual["neural_live"], expected["live_probability"])
    cached = _predict({"frame_id": frames.frame_id}, features, model, source, fit,
                      tmp_path / "prediction.npz", device)
    for key in actual:
        np.testing.assert_array_equal(actual[key], cached[key])
