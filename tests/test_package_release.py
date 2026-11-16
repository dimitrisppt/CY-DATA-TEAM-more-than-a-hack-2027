import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts import package_release as package


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record(value: dict) -> dict:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode()
    return {"payload": value, "sha256": _digest(canonical)}


def _fixture(root: Path, monkeypatch):
    for name in (".gitignore", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
                 "pyproject.toml", "uv.lock",
                 "games.py", *package.OFFICIAL_DOCS):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"release fixture {name}\n")
    (root / ".python-version").write_text("3.12.11\n")
    for directory in package.SOURCE_DIRS:
        path = root / directory / "module.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {directory}\n")
    (root / "configs").mkdir()
    (root / "configs" / "model.json").write_text("{}\n")
    (root / "docs").mkdir()
    (root / "docs" / "CURATED.md").write_text("curated\n")

    pinned_file = root / "possession" / "module.py"
    model_file = root / "models" / "v6" / "coarse.bin"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"model checkpoint fixture")
    model = {"schema": "coarse_context_submission_v1", "portable": True,
             "files": {"coarse.bin": _digest(model_file.read_bytes())},
             "source_sha256": {"possession/module.py": package._sha_file(pinned_file)}}
    model_outer = _record(model)
    model_manifest = root / "models" / "v6" / "manifest.json"
    model_manifest.write_text(json.dumps(model_outer, sort_keys=True) + "\n")

    csv = root / "submission" / "CY-DATA-TEAM.csv"
    csv.parent.mkdir(parents=True)
    csv.write_bytes(b"csv fixture\n")
    csv_sha = _digest(csv.read_bytes())
    monkeypatch.setattr(package, "EXPECTED_CSV_SHA256", csv_sha)
    (csv.parent / "manifest.json").write_text(json.dumps({
        "version": "v6", "rows": 199319, "ground_truth_used": False,
        "model_directory": "models/v6", "model_manifest_sha256": package._sha_file(model_manifest),
        "sha256": csv_sha,
    }))

    benchmark = root / "benchmarks" / "v6" / "public-reference"
    benchmark.mkdir(parents=True)
    labels = benchmark / "labels.npz"
    labels.write_bytes(b"public decoded labels fixture")
    (benchmark / "manifest.json").write_text(json.dumps({
        "schema": "public_dfl_decoded_reference_v1", "target_ground_truth_used": False,
        "target_inputs_included": False, "raw_participant_assets_included": False,
        "labels_file": "labels.npz", "labels_sha256": package._sha_file(labels),
        "labels_bytes": labels.stat().st_size,
    }))
    return pinned_file, model_file, csv


@pytest.mark.parametrize("root_name", ["repo", ".hidden/repo"])
def test_build_release_is_deterministic_and_allowlisted(tmp_path, monkeypatch, root_name):
    root = tmp_path / root_name
    root.mkdir(parents=True)
    _fixture(root, monkeypatch)
    (root / "data").mkdir()
    (root / "data" / "private.txt").write_text("exclude")
    (root / "artifacts").mkdir()
    (root / "artifacts" / "cache.bin").write_bytes(b"exclude")
    (root / "models" / "v6" / "unlisted.bin").write_bytes(b"exclude")
    (root / "benchmarks" / "v6" / "public-reference" / "extra.npz").write_bytes(b"exclude")
    output = tmp_path / "release.zip"

    first = package.build_release(root, output)
    second = package.build_release(root, output)
    assert first["sha256"] == second["sha256"]
    assert first["reused"] is False and second["reused"] is True
    manifest = package.verify_archive(output)
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "package-manifest.json" not in manifest["files"]
        assert "possession/module.py" in names
        assert "scripts/module.py" in names
        assert "configs/model.json" in names
        assert "docs/CURATED.md" in names
        assert "models/v6/coarse.bin" in names
        assert "models/v6/manifest.json" in names
        assert "models/v6/unlisted.bin" not in names
        assert "benchmarks/v6/public-reference/labels.npz" in names
        assert "benchmarks/v6/public-reference/extra.npz" not in names
        assert "submission/CY-DATA-TEAM.csv" in names
        assert not any(name.startswith(("data/", "artifacts/", ".git/")) for name in names)
        encoded_manifest = archive.read("package-manifest.json").decode()
        assert str(root) not in encoded_manifest
    output.write_bytes(output.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="existing release archive differs"):
        package.build_release(root, output)


def test_release_rejects_changed_pinned_source_and_model_file(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    pinned, model_file, _ = _fixture(root, monkeypatch)
    output = tmp_path / "release.zip"
    pinned.write_text("changed source\n")
    with pytest.raises(ValueError, match="pinned source hash differs"):
        package.build_release(root, output)

    root = tmp_path / "repo-model-tampered"
    root.mkdir()
    _, model_file, _ = _fixture(root, monkeypatch)
    model_file.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="model file hash differs"):
        package.build_release(root, tmp_path / "tampered.zip")


def test_release_rejects_submission_or_benchmark_manifest_drift(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _fixture(root, monkeypatch)
    benchmark_manifest = root / "benchmarks" / "v6" / "public-reference" / "manifest.json"
    benchmark = json.loads(benchmark_manifest.read_text())
    benchmark["target_inputs_included"] = True
    benchmark_manifest.write_text(json.dumps(benchmark))
    with pytest.raises(ValueError, match="unexpected or includes target data"):
        package.build_release(root, tmp_path / "release.zip")


def test_default_dist_output_is_allowed_but_remains_excluded(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _fixture(root, monkeypatch)
    output = root / "dist" / "CY-DATA-TEAM-v6.zip"
    package.build_release(root, output)
    with zipfile.ZipFile(output) as archive:
        assert not any(name.startswith("dist/") for name in archive.namelist())


def test_verify_only_does_not_require_data_dir(tmp_path, monkeypatch):
    from experiments import coarse_submission
    from scripts import reproduce

    monkeypatch.setattr(reproduce, "_configure_torch", lambda: None)
    monkeypatch.setattr(coarse_submission, "verify_coarse_bundle", lambda _: {
        "manifest": {"schema": "coarse_context_submission_v1"}})
    result = reproduce.main(["--verify-only", "--model-dir", str(tmp_path / "models")])
    assert result == 0

