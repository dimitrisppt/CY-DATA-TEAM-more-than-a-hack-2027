import hashlib
import importlib
import json
import zipfile

import pytest

from experiments import campaign_protocol
from possession.storage import sha256_file, write_json
from scripts import package_bagged_submission as package


def _fixture(root, monkeypatch):
    bundle = root / "bundle"
    submission = root / "submission"
    runtime = root / "runtime"
    docs = root / "docs"
    for directory in (bundle, submission, runtime, docs):
        directory.mkdir(parents=True)
    files = {}
    for relative, content in (("seed_20261116/manifest.json", b"manifest one"),
                              ("seed_20261116/member.pt", b"member one"),
                              ("seed_20261117/manifest.json", b"manifest two"),
                              ("seed_20261117/member.pt", b"member two"),
                              ("bagging-result.json", b"proof result"),
                              ("bagging-protocol.json", b"proof protocol")):
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files[relative] = sha256_file(path)
    (runtime / "experiments.py").write_text("print('frozen')\n")
    snapshot_files = {"experiments.py": sha256_file(runtime / "experiments.py")}
    write_json(runtime / "snapshot-manifest.json", {"source_root": "/original/workspace",
                                                      "files": snapshot_files})
    bundle_manifest = {"schema": "ensemble_bagging_submission_v1",
                       "seeds": [20261116, 20261117], "files": files,
                       "source_sha256": snapshot_files}
    campaign_protocol.publish(bundle / "manifest.json", bundle_manifest)
    bundle_sha = sha256_file(bundle / "manifest.json")
    monkeypatch.setattr(package, "verify_bagged_bundle", lambda path: {
        "manifest": bundle_manifest})

    csv = submission / "CY-DATA-TEAM.csv"
    csv.write_text("frame,label\n0,H\n")
    sidecar_request = {"bundle_sha256": bundle_sha}
    sidecar = {"request_sha256": package._digest_json(sidecar_request),
               "ground_truth_used": False, "format_validated": True,
               "request": sidecar_request, "output": {"sha256": sha256_file(csv)}}
    write_json(csv.with_suffix(".manifest.json"), sidecar)
    verification = {"status": "complete", "format_validated": True,
                    "target_ground_truth_used": False,
                    "independent_inference_byte_identical": True,
                    "model_seeds": [20261116, 20261117],
                    "csv": {"path": str(csv), "sha256": sha256_file(csv)},
                    "bundle": str(bundle)}
    write_json(submission / "verification.json", verification)
    for name in package.DOC_ALLOWLIST:
        (docs / name).write_text(f"# {name}\n")
    return bundle, submission, runtime, docs


def test_package_is_deterministic_and_all_entries_are_hash_pinned(tmp_path, monkeypatch):
    bundle, submission, runtime, docs = _fixture(tmp_path, monkeypatch)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    result_a = package.package_bagged_submission(bundle, submission, runtime, first, docs)
    result_b = package.package_bagged_submission(bundle, submission, runtime, second, docs)
    assert result_a["sha256"] == result_b["sha256"]
    assert result_a["files_count"] == 18
    with zipfile.ZipFile(first) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("package-manifest.json"))
        assert sorted(manifest["files"]) == sorted(
            name for name in archive.namelist() if name != "package-manifest.json")
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        readme = archive.read("README.md").decode()
        assert "TARGET_DATA_DIR" in readme
        assert "external machine" not in readme.lower()


def test_failed_verification_and_snapshot_tamper_are_rejected(tmp_path, monkeypatch):
    bundle, submission, runtime, docs = _fixture(tmp_path, monkeypatch)
    verify = package.verify_bagged_bundle
    monkeypatch.setattr(package, "verify_bagged_bundle",
                        lambda path: (_ for _ in ()).throw(ValueError("bad bagged gate")))
    with pytest.raises(ValueError, match="bad bagged gate"):
        package.package_bagged_submission(bundle, submission, runtime, tmp_path / "bad.zip", docs)
    monkeypatch.setattr(package, "verify_bagged_bundle", verify)
    (runtime / "experiments.py").write_text("changed\n")
    with pytest.raises(ValueError, match="snapshot hash differs"):
        package.package_bagged_submission(bundle, submission, runtime, tmp_path / "bad.zip", docs)
    assert not (tmp_path / "bad.zip").exists()


def test_verification_requires_exact_seeds_format_repeat_and_bundle_link(tmp_path, monkeypatch):
    bundle, submission, runtime, docs = _fixture(tmp_path, monkeypatch)
    verification_path = submission / "verification.json"
    verification = json.loads(verification_path.read_text())
    verification["model_seeds"] = [20261116]
    write_json(verification_path, verification)
    with pytest.raises(ValueError, match="verification status"):
        package.package_bagged_submission(bundle, submission, runtime, tmp_path / "bad.zip", docs)


def test_same_basename_does_not_substitute_for_bundle_manifest_hash(tmp_path, monkeypatch):
    bundle, submission, runtime, docs = _fixture(tmp_path, monkeypatch)
    sidecar_path = submission / "CY-DATA-TEAM.manifest.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["request"]["bundle_sha256"] = "wrong-bundle-hash"
    sidecar["request_sha256"] = package._digest_json(sidecar["request"])
    write_json(sidecar_path, sidecar)
    with pytest.raises(ValueError, match="does not link"):
        package.package_bagged_submission(bundle, submission, runtime, tmp_path / "bad.zip", docs)


def test_bundle_source_hash_must_match_frozen_snapshot(tmp_path, monkeypatch):
    bundle, submission, runtime, docs = _fixture(tmp_path, monkeypatch)
    manifest = {"seeds": [20261116, 20261117], "files": {},
                "source_sha256": {"experiments.py": "not-the-snapshot-hash"}}
    monkeypatch.setattr(package, "verify_bagged_bundle", lambda path: {"manifest": manifest})
    with pytest.raises(ValueError, match="snapshot differs"):
        package.package_bagged_submission(bundle, submission, runtime, tmp_path / "bad.zip", docs)


def test_different_existing_archive_is_never_replaced(tmp_path, monkeypatch):
    bundle, submission, runtime, docs = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "package.zip"
    output.write_bytes(b"existing archive")
    with pytest.raises(ValueError, match="different package archive"):
        package.package_bagged_submission(bundle, submission, runtime, output, docs)
    assert output.read_bytes() == b"existing archive"


def test_output_overlap_and_unallowlisted_docs_are_rejected(tmp_path, monkeypatch):
    bundle, submission, runtime, docs = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="separate"):
        package.package_bagged_submission(bundle, submission, runtime,
                                          submission / "nested.zip", docs)
    (docs / "README.md").write_text("unapproved")
    result = package.package_bagged_submission(bundle, submission, runtime,
                                               tmp_path / "okay.zip", docs)
    with zipfile.ZipFile(result["artifact_path"]) as archive:
        assert "docs/README.md" not in archive.namelist()


def test_import_does_not_change_process_thread_environment(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.setenv("OMP_DYNAMIC", "TRUE")
    importlib.reload(package)
    assert package.os.environ["OMP_NUM_THREADS"] == "7"
    assert package.os.environ["OMP_DYNAMIC"] == "TRUE"
