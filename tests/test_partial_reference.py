"""Portable scoring reproduces the stopped study and detects altered artifacts."""
import json
import shutil
from pathlib import Path

import pytest

from possession.storage import sha256_file
from scripts.evaluate_partial_reference import evaluate_reference

REFERENCE = Path(__file__).resolve().parents[1] / "benchmarks/partial-refinement-v1"


def _write(path, value):
    path.write_text(json.dumps(value))


def test_published_reference_reproduces_every_group_and_short_spell_count():
    result = evaluate_reference(REFERENCE)
    assert result["partial"] and not result["independent_test"]
    assert not result["target_inputs_included"] and not result["target_labels_used"]
    both = result["groups"]["completed_two_fine_seeds"]
    common = result["groups"]["matched_single_fine_seed"]
    assert both["summary"]["baseline"]["objective"] == 0.612520721554461
    assert both["summary"]["control"]["objective"] == 0.611244868343096
    assert both["summary"]["spell_balanced"]["objective"] == 0.6184427156237241
    assert common["summary"]["masked_teacher"]["objective"] == 0.5946678092280123
    assert common["summary"]["control"]["objective"] == 0.5948063564753762
    assert "masked_teacher" not in both["scores"]
    for variant, matched, spurious in (("baseline", 71, 141), ("spell_balanced", 83, 171)):
        rows = both["spell_diagnostics_batch_1"][variant]["clean"]
        bins = [row["duration_bins"][name] for row in rows
                for name in ("under_1s", "1_to_2s")]
        assert sum(row["gt"] for row in bins) == 238
        assert sum(row["matched_at_0_5"] for row in bins) == matched
        assert sum(row["counts"]["spurious_at_0_5"] for row in rows) == spurious


@pytest.mark.parametrize("filename", ["labels.npz", "result.json"])
def test_reference_rejects_file_hash_tampering(tmp_path, filename):
    reference = tmp_path / "reference"
    shutil.copytree(REFERENCE, reference)
    with (reference / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="hash differs"):
        evaluate_reference(reference)


def test_reference_rejects_changed_array_hash(tmp_path):
    reference = tmp_path / "reference"
    shutil.copytree(REFERENCE, reference)
    path = reference / "manifest.json"
    manifest = json.loads(path.read_text())
    key = next(iter(manifest["label_array_hashes"]))
    manifest["label_array_hashes"][key] = "0" * 64
    _write(path, manifest)
    with pytest.raises(ValueError, match="label-array hash differs"):
        evaluate_reference(reference)


def test_reference_rejects_score_mismatch_even_with_rebound_result_hash(tmp_path):
    reference = tmp_path / "reference"
    shutil.copytree(REFERENCE, reference)
    path = reference / "result.json"
    result = json.loads(path.read_text())
    result["groups"]["completed_two_fine_seeds"]["summary"]["baseline"]["objective"] += .01
    _write(path, result)
    manifest_path = reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["results_sha256"] = sha256_file(path)
    _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="differs from saved expected scores or counts"):
        evaluate_reference(reference)


@pytest.mark.parametrize("filename,field,value", [
    ("manifest.json", "target_inputs_included", True),
    ("result.json", "partial", False),
    ("result.json", "training_status", "running"),
    ("manifest.json", "scenario_weights", {"clean": 1., "short": 0., "long": 0., "replay": 0.}),
    ("manifest.json", "batches", ["batch_1"]),
])
def test_reference_rejects_changed_context_or_configuration(tmp_path, filename, field, value):
    reference = tmp_path / "reference"
    shutil.copytree(REFERENCE, reference)
    path = reference / filename
    record = json.loads(path.read_text())
    record[field] = value
    _write(path, record)
    if filename == "result.json":
        manifest_path = reference / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["results_sha256"] = sha256_file(path)
        _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="context differs|configuration differs"):
        evaluate_reference(reference)
