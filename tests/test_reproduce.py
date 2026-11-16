import hashlib
import json
from pathlib import Path

import pytest

from scripts import reproduce

RAW_NAME = "BAR-ATH_tactical_atd.txt"
META_NAME = "BAR-ATH_tactical_atd.xml"


def _inputs(root: Path):
    tactical = root / "participant-export"
    tactical.mkdir(parents=True)
    raw = tactical / RAW_NAME
    metadata = tactical / META_NAME
    raw.write_bytes(b"automatic tracking fixture\n")
    metadata.write_bytes(b"<Metadata/>\n")
    return raw, metadata


def test_discover_inputs_requires_unique_known_files(tmp_path):
    raw, metadata = _inputs(tmp_path)
    assert reproduce.discover_inputs(tmp_path) == (raw.resolve(), metadata.resolve())
    (tmp_path / "duplicate").mkdir()
    (tmp_path / "duplicate" / RAW_NAME).write_bytes(b"other")
    with pytest.raises(ValueError, match="exactly one"):
        reproduce.discover_inputs(tmp_path)


def test_discover_inputs_supports_explicit_pair_and_requires_both(tmp_path):
    raw, metadata = _inputs(tmp_path)
    assert reproduce.discover_inputs(tmp_path / "missing", raw, metadata) == (
        raw.resolve(), metadata.resolve())
    with pytest.raises(ValueError, match="supplied together"):
        reproduce.discover_inputs(tmp_path, raw=raw)


def test_stage_inputs_copies_only_automatic_inputs_and_hashes_manifest(tmp_path):
    raw, metadata = _inputs(tmp_path)
    original_raw, original_metadata = raw.read_bytes(), metadata.read_bytes()
    staged = reproduce.stage_inputs(raw, metadata, tmp_path / "ignored-staging")
    manifest = json.loads((staged / "input-manifest.json").read_text())
    assert {row["path"] for row in manifest["files"]} == {
        f"tactical/{RAW_NAME}", f"tactical/{META_NAME}"}
    for row in manifest["files"]:
        path = staged / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert path.resolve() != (raw if path.name == RAW_NAME else metadata).resolve()
    assert raw.read_bytes() == original_raw
    assert metadata.read_bytes() == original_metadata
    assert reproduce.stage_inputs(raw, metadata, tmp_path / "ignored-staging") == staged


def test_stage_inputs_rejects_tampered_existing_staging(tmp_path):
    raw, metadata = _inputs(tmp_path)
    staged = reproduce.stage_inputs(raw, metadata, tmp_path / "ignored-staging")
    (staged / "tactical" / RAW_NAME).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="staged input differs"):
        reproduce.stage_inputs(raw, metadata, tmp_path / "ignored-staging")


def test_reproduce_uses_standard_inputs_and_verifies_csv_hash(tmp_path, monkeypatch):
    model, data = tmp_path / "model", tmp_path / "participant"
    model.mkdir()
    data.mkdir()
    output = tmp_path / "out.csv"
    content = b"verified csv fixture"
    digest = hashlib.sha256(content).hexdigest()
    seen = {}

    monkeypatch.setattr(reproduce, "_configure_torch", lambda: None)
    monkeypatch.setattr(reproduce, "_is_standard_input", lambda _: True)
    monkeypatch.setattr(reproduce, "_sha256", lambda path: hashlib.sha256(
        Path(path).read_bytes()).hexdigest())
    from experiments import campaign_submission, coarse_submission

    monkeypatch.setattr(coarse_submission, "verify_coarse_bundle",
                        lambda path: {"manifest": {"schema": "test"}})

    def inference(bundle_dir, data_dir, output_path):
        seen["data_dir"] = Path(data_dir)
        Path(output_path).write_bytes(content)
        return {"output": {"rows": 199319}, "halves": []}

    monkeypatch.setattr(coarse_submission, "infer_coarse_submission", inference)
    monkeypatch.setattr(campaign_submission, "_validate_output", lambda path: None)
    result = reproduce.reproduce(data, output, model, expected_sha256=digest)
    assert result["sha256"] == digest
    assert seen["data_dir"] == data.resolve()


def test_reproduce_stages_nonstandard_inputs_and_rejects_hash_drift(tmp_path, monkeypatch):
    raw, _metadata = _inputs(tmp_path / "participant")
    model, output = tmp_path / "model", tmp_path / "out.csv"
    model.mkdir()
    monkeypatch.setattr(reproduce, "_configure_torch", lambda: None)
    monkeypatch.setattr(reproduce, "_is_standard_input", lambda _: False)
    from experiments import campaign_submission, coarse_submission

    monkeypatch.setattr(coarse_submission, "verify_coarse_bundle", lambda _: {"manifest": {}})
    def inference(_, __, destination):
        Path(destination).write_bytes(b"wrong fixture")
        return {"output": {"rows": 199319}, "halves": []}

    monkeypatch.setattr(coarse_submission, "infer_coarse_submission", inference)
    monkeypatch.setattr(campaign_submission, "_validate_output", lambda _: None)
    with pytest.raises(ValueError, match="hash differs"):
        reproduce.reproduce(raw.parent.parent, output, model, expected_sha256="0" * 64,
                            staging_root=tmp_path / "artifacts" / "reproduce-input-v1")
    assert output.read_bytes() == b"wrong fixture"
    assert list((tmp_path / "artifacts" / "reproduce-input-v1").glob("*/input-manifest.json"))


def test_verify_only_does_not_discover_or_read_participant_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduce, "_configure_torch", lambda: None)
    from experiments import coarse_submission

    monkeypatch.setattr(coarse_submission, "verify_coarse_bundle", lambda _: {
        "manifest": {"schema": "coarse_context_submission_v1"}})
    monkeypatch.setattr(reproduce, "discover_inputs", lambda *args: pytest.fail("read inputs"))
    result = reproduce.reproduce(tmp_path, tmp_path / "out.csv", tmp_path / "model",
                                 verify_only=True)
    assert result == {"verified": True, "model_dir": str((tmp_path / "model").resolve()),
                      "schema": "coarse_context_submission_v1"}


def test_reproduce_rejects_outputs_inside_original_input_tree(tmp_path, monkeypatch):
    raw, metadata = _inputs(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(reproduce, "_configure_torch", lambda: None)
    monkeypatch.setattr(reproduce, "_is_standard_input", lambda _: False)
    from experiments import coarse_submission

    monkeypatch.setattr(coarse_submission, "verify_coarse_bundle", lambda _: {"manifest": {}})
    monkeypatch.setattr(coarse_submission, "infer_coarse_submission",
                        lambda *_: pytest.fail("inference should not run"))
    output = tmp_path / "participant-export" / "prediction.csv"
    with pytest.raises(ValueError, match="outside model and input directories"):
        reproduce.reproduce(tmp_path, output, model, raw=raw, metadata=metadata,
                            staging_root=tmp_path / "stage")
