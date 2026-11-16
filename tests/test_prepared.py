import json
import os
from pathlib import Path

import pytest

from possession.prepared import verify_prepared
from possession.storage import sha256_file


def prepared_fixture(tmp_path):
    first = tmp_path / "DFL-MAT-TEST_half1.npz"
    second = tmp_path / "DFL-MAT-TEST_half2.npz"
    first.write_bytes(b"valid-cache-first")
    second.write_bytes(b"valid-cache-second")
    entry = {
        "match_id": "DFL-MAT-TEST",
        "sources": {
            "positions": {"path": "/source/positions.xml", "sha256": "a" * 64},
            "match_info": {"path": "/source/match.xml", "sha256": "b" * 64},
        },
        "halves": [
            {"period": 1, "cache": first.name, "cache_sha256": sha256_file(first)},
            {"period": 2, "cache": second.name, "cache_sha256": sha256_file(second)},
        ],
    }
    manifest = {"version": 1, "matches": {"DFL-MAT-TEST": entry}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return [first, second], manifest


def test_verify_prepared_records_selected_cache_and_source_fingerprints(tmp_path):
    paths, _ = prepared_fixture(tmp_path)
    provenance = verify_prepared(tmp_path, paths)
    assert provenance["status"] == "verified_manifest"
    assert provenance["caches"][0]["cache"] == paths[0].name
    assert provenance["caches"][0]["sha256"] == sha256_file(paths[0])
    assert provenance["sources"]["DFL-MAT-TEST"]["positions"]["sha256"] == "a" * 64
    assert json.loads(json.dumps(provenance)) == provenance


def test_verify_prepared_accepts_paths_from_relative_glob(tmp_path):
    paths, _ = prepared_fixture(tmp_path)
    relative = [Path(os.path.relpath(path, Path.cwd())) for path in paths]
    assert verify_prepared(tmp_path, relative)["status"] == "verified_manifest"


def test_verify_prepared_rejects_tampered_cache(tmp_path):
    paths, _ = prepared_fixture(tmp_path)
    paths[0].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash|SHA|changed"):
        verify_prepared(tmp_path, paths)


def test_verify_prepared_rejects_missing_manifest_entry(tmp_path):
    paths, manifest = prepared_fixture(tmp_path)
    manifest["matches"] = {}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest|DFL-MAT-TEST"):
        verify_prepared(tmp_path, paths)


def test_verify_prepared_rejects_invalid_manifest_version(tmp_path):
    paths, manifest = prepared_fixture(tmp_path)
    manifest["version"] = 2
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="version|manifest"):
        verify_prepared(tmp_path, paths)


def test_verify_prepared_requires_both_selected_halves(tmp_path):
    paths, _ = prepared_fixture(tmp_path)
    with pytest.raises(ValueError, match="half|period"):
        verify_prepared(tmp_path, paths[:1])


def test_verify_prepared_rejects_duplicate_half_entry(tmp_path):
    paths, manifest = prepared_fixture(tmp_path)
    manifest["matches"]["DFL-MAT-TEST"]["halves"][1] = (
        manifest["matches"]["DFL-MAT-TEST"]["halves"][0].copy())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="half|period"):
        verify_prepared(tmp_path, paths)


def test_verify_prepared_marks_manifestless_synthetic_cache(tmp_path):
    cache = tmp_path / "a_half1.npz"
    cache.write_bytes(b"synthetic")
    provenance = verify_prepared(tmp_path, [cache])
    assert provenance["status"] == "unmanifested_local_cache"
    assert provenance["caches"][0]["sha256"] == sha256_file(cache)
