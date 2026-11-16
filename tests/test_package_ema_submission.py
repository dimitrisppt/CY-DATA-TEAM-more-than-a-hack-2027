import hashlib
import importlib
import json
import zipfile

import pytest

from possession.storage import sha256_file, write_json
from scripts import package_ema_submission as package


def _fixture(root, monkeypatch):
    bundle = root / "bundle"
    submission = root / "submission"
    runtime = root / "runtime"
    docs = root / "docs"
    for directory in (bundle, submission, runtime, docs):
        directory.mkdir(parents=True)

    files = {}
    for relative, content in (("members/20261116/mean_007/model.pt", b"ema model one"),
                              ("members/20261117/mean_007/model.pt", b"ema model two"),
                              ("proof/study/result.json", b"study result")):
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files[relative] = sha256_file(path)
    source = runtime / "experiments" / "ema_submission.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('frozen ema source')\n")
    snapshot_files = {"experiments/ema_submission.py": sha256_file(source)}
    write_json(runtime / "snapshot-manifest.json",
               {"source_root": "/original/workspace", "files": snapshot_files})
    manifest = {"schema": "ema_ensemble_submission_v1", "portable": True,
                "seeds": [20261116, 20261117], "files": files,
                "source_sha256": snapshot_files}
    write_json(bundle / "manifest.json", manifest)
    monkeypatch.setattr(package, "verify_ema_bundle", lambda path: {"manifest": manifest})

    csv = submission / "CY-DATA-TEAM.csv"
    csv.write_text("frame,label\n0,H\n")
    request = {"bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
               "selection": {"decoder": "frozen"}}
    sidecar = {"request": request, "request_sha256": package._base._digest_json(request),
               "ground_truth_used": False, "format_validated": True,
               "output": {"sha256": sha256_file(csv)}}
    write_json(csv.with_suffix(".manifest.json"), sidecar)
    verification = {
        "status": "complete", "format_validated": True,
        "independent_inference_byte_identical": True,
        "independent_process_and_bundle_copy": True,
        "target_ground_truth_used": False, "submitted_externally": False,
        "csv": {"path": str(csv), "sha256": sha256_file(csv)},
        "model_seeds": [20261116, 20261117], "steps_per_member": 7200,
        "ema_decay": .99, "ema_tail_updates": 200,
    }
    write_json(submission / "verification.json", verification)
    for name in package.DOC_ALLOWLIST:
        (docs / name).write_text(f"# {name}\n")
    (docs / "UNAPPROVED.md").write_text("exclude me\n")
    return bundle, submission, runtime, docs


def test_package_is_deterministic_and_hashes_every_payload_entry(tmp_path, monkeypatch):
    bundle, submission, runtime, docs = _fixture(tmp_path, monkeypatch)
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    result_a = package.package_ema_submission(bundle, submission, runtime, first, docs)
    result_b = package.package_ema_submission(bundle, submission, runtime, second, docs)
    assert result_a["sha256"] == result_b["sha256"]
    assert result_a["files_count"] == 14
    with zipfile.ZipFile(first) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("package-manifest.json"))
        assert sorted(manifest["files"]) == sorted(
            name for name in archive.namelist() if name != "package-manifest.json")
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        assert "docs/UNAPPROVED.md" not in archive.namelist()
        readme = archive.read("README.md").decode()
        assert "experiments.ema_submission" in readme
        assert "TARGET_DATA_DIR" in readme


def test_bundle_manifest_link_and_sidecar_digest_are_required(tmp_path, monkeypatch):
    bundle, submission, runtime, docs = _fixture(tmp_path, monkeypatch)
    sidecar_path = submission / "CY-DATA-TEAM.manifest.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["request"]["bundle_manifest_sha256"] = "different-manifest"
    sidecar["request_sha256"] = package._base._digest_json(sidecar["request"])
    write_json(sidecar_path, sidecar)
    with pytest.raises(ValueError, match="does not link"):
        package.package_ema_submission(bundle, submission, runtime, tmp_path / "bad.zip", docs)
    assert not (tmp_path / "bad.zip").exists()


def test_source_snapshot_must_match_ema_bundle_source_map(tmp_path, monkeypatch):
    bundle, submission, runtime, docs = _fixture(tmp_path, monkeypatch)
    manifest = {"schema": "ema_ensemble_submission_v1", "portable": True,
                "seeds": [20261116, 20261117], "files": {},
                "source_sha256": {"experiments/ema_submission.py": "wrong"}}
    monkeypatch.setattr(package, "verify_ema_bundle", lambda path: {"manifest": manifest})
    with pytest.raises(ValueError, match="snapshot differs"):
        package.package_ema_submission(bundle, submission, runtime, tmp_path / "bad.zip", docs)


def test_failed_verification_and_tampered_bundle_file_are_rejected(tmp_path, monkeypatch):
    bundle, submission, runtime, docs = _fixture(tmp_path, monkeypatch)
    verify = package.verify_ema_bundle
    monkeypatch.setattr(package, "verify_ema_bundle",
                        lambda path: (_ for _ in ()).throw(ValueError("EMA gate failed")))
    with pytest.raises(ValueError, match="EMA gate failed"):
        package.package_ema_submission(bundle, submission, runtime, tmp_path / "bad.zip", docs)
    monkeypatch.setattr(package, "verify_ema_bundle", verify)
    (bundle / "members/20261116/mean_007/model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bundle file changed"):
        package.package_ema_submission(bundle, submission, runtime, tmp_path / "bad.zip", docs)


def test_incomplete_verification_and_output_overlap_are_rejected(tmp_path, monkeypatch):
    bundle, submission, runtime, docs = _fixture(tmp_path, monkeypatch)
    verification_path = submission / "verification.json"
    verification = json.loads(verification_path.read_text())
    verification["independent_process_and_bundle_copy"] = False
    write_json(verification_path, verification)
    with pytest.raises(ValueError, match="verification is incomplete"):
        package.package_ema_submission(bundle, submission, runtime, tmp_path / "bad.zip", docs)
    with pytest.raises(ValueError, match="separate"):
        package.package_ema_submission(bundle, submission, runtime,
                                       submission / "nested.zip", docs)


def test_module_import_does_not_change_thread_environment(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.setenv("OMP_DYNAMIC", "TRUE")
    importlib.reload(package)
    assert package.os.environ["OMP_NUM_THREADS"] == "7"
    assert package.os.environ["OMP_DYNAMIC"] == "TRUE"
