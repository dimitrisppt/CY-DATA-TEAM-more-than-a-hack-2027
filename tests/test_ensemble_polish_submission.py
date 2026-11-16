import copy

import pytest

from experiments import ensemble_polish_submission as export
from experiments.campaign_protocol import publish
from experiments.robustness import _digest


def _study(directory, monkeypatch, *, passed=True):
    baseline = {"kind": "convex_pair", "members": [{"id": "m", "spec": {"width": 32}}],
                "blend_weight": .5, "decoder": {"setting": 1}}
    candidate = copy.deepcopy(baseline)
    candidate["decoder"] = {"setting": 2}
    protocol = {"baseline_selection": baseline, "source_sha256": {"fixture": "hash"},
                "runtime": {"fixture": True}}
    selected = {"selection": candidate, "protocol_sha256": _digest(protocol),
                "baseline_selection_sha256": _digest(baseline)}
    summary = {"status": "complete", "selection": candidate,
               "protocol_sha256": _digest(protocol), "confirmation": {
                   "eligible": passed, "gate": {"passed": passed},
                   "outer": {"gate_passed": True}}}
    publish(directory / "protocol.json", protocol)
    publish(directory / "selection.json", selected)
    publish(directory / "summary.json", summary)
    monkeypatch.setattr(export, "_source_hashes", lambda: {"fixture": "hash"})
    monkeypatch.setattr(export, "_execution_environment", lambda: {"fixture": True})
    bundle = {"recipe": {"selection": baseline}, "member_paths": {}}
    monkeypatch.setattr(export, "verify_bundle", lambda path: bundle)
    return bundle, candidate


def test_failed_confirmation_blocks_target_export_before_input_read(tmp_path, monkeypatch):
    _study(tmp_path / "study", monkeypatch, passed=False)
    monkeypatch.setattr(export, "_verified_inputs", lambda *args: pytest.fail("target opened"))
    output = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="promotion check"):
        export.infer_polished_submission(tmp_path / "study", tmp_path / "bundle",
                                         tmp_path / "data", output)
    assert not output.exists()


def test_passing_decoder_keeps_the_verified_networks(tmp_path, monkeypatch):
    bundle, candidate = _study(tmp_path / "study", monkeypatch)
    actual, selection, hashes = export.verify_promotion(tmp_path / "study", tmp_path / "bundle")
    assert actual == bundle
    assert selection == candidate
    assert set(hashes) == {"protocol.json", "selection.json", "summary.json"}


def test_different_model_bundle_cannot_receive_a_decoder_override(tmp_path, monkeypatch):
    bundle, _ = _study(tmp_path / "study", monkeypatch)
    bundle["recipe"]["selection"]["members"][0]["spec"]["width"] = 64
    with pytest.raises(ValueError, match="frozen neural models"):
        export.verify_promotion(tmp_path / "study", tmp_path / "bundle")


def test_changed_scoring_source_blocks_export(tmp_path, monkeypatch):
    _study(tmp_path / "study", monkeypatch)
    monkeypatch.setattr(export, "_source_hashes", lambda: {"fixture": "changed"})
    with pytest.raises(ValueError, match="execution source"):
        export.verify_promotion(tmp_path / "study", tmp_path / "bundle")
