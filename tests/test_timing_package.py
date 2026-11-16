import json

import pytest

from scripts import evaluate_timing_reference, package_release


def test_optional_timing_reference_is_verified_and_allowlisted(tmp_path, monkeypatch):
    folder = tmp_path / "benchmarks" / "timing-calibration-v1"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({
        "schema": "public_dfl_timing_reference_v1", "target_inputs_included": False,
        "target_labels_used": False}))
    (folder / "labels.npz").write_bytes(b"fixture")
    (folder / "unlisted.npz").write_bytes(b"excluded")
    calls = []

    def verify(path):
        calls.append(path)
        return {"independent_test": False}

    monkeypatch.setattr(evaluate_timing_reference, "evaluate_reference", verify)
    payload = {}
    package_release._timing_benchmark_payload(tmp_path, payload)
    assert calls == [folder]
    assert set(payload) == {"benchmarks/timing-calibration-v1/manifest.json",
                            "benchmarks/timing-calibration-v1/labels.npz"}


def test_timing_reference_verification_failure_prevents_packaging(tmp_path, monkeypatch):
    (tmp_path / "benchmarks" / "timing-calibration-v1").mkdir(parents=True)

    def reject(_):
        raise ValueError("timing reference checksum differs")

    monkeypatch.setattr(evaluate_timing_reference, "evaluate_reference", reject)
    with pytest.raises(ValueError, match="checksum differs"):
        package_release._timing_benchmark_payload(tmp_path, {})
