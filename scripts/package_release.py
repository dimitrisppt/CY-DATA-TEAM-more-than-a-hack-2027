"""Build a deterministic, allowlisted v6 source and artifact release archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ZIP_TIME = (2026, 1, 1, 0, 0, 0)
EXPECTED_CSV_SHA256 = "4c22b0171509ae1f93c57b72e9394a10623e19bcc1834290c2a9ec640edfea1b"
ROOT_FILES = (
    ".gitignore", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", ".python-version",
    "pyproject.toml", "uv.lock", "games.py", "CHALLENGE.md", "DATA.md",
    "EVALUATION.md", "FAQ.md", "RULES.md", "SUBMISSION.md", "OVERVIEW.md",
    "requirements.txt",
)
SOURCE_DIRS = ("possession", "experiments", "evaluation", "scripts", "tests")
OFFICIAL_DOCS = ("CHALLENGE.md", "DATA.md", "EVALUATION.md", "FAQ.md", "RULES.md",
                 "SUBMISSION.md", "OVERVIEW.md")
SCHEMA = "cy_data_team_v6_release_v1"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid release metadata: {path.name}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"release metadata must be an object: {path.name}")
    return value


def _safe_relative(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or str(path) != name):
        raise ValueError(f"unsafe release path: {name}")
    return name


def _add_file(root: Path, payload: dict[str, Path], relative: str) -> None:
    relative = _safe_relative(relative)
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required release file is missing or linked: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"release file escapes repository: {relative}")
    payload[relative] = path


def _source_payload(root: Path) -> dict[str, Path]:
    payload: dict[str, Path] = {}
    for name in ROOT_FILES:
        path = root / name
        if path.is_file():
            _add_file(root, payload, name)
    for required in (".gitignore", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
                     ".python-version",
                     "pyproject.toml", "uv.lock", "games.py", *OFFICIAL_DOCS):
        if required not in payload:
            raise ValueError(f"required release source or attribution file is missing: {required}")

    for directory in SOURCE_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if path.name.startswith("._") or any(
                    part.startswith(".") for part in path.relative_to(root).parts):
                continue
            _add_file(root, payload, relative)

    for directory, suffix in (("configs", ".json"), ("docs", ".md")):
        base = root / directory
        if base.is_dir():
            for path in sorted(base.rglob(f"*{suffix}")):
                if path.name.startswith("._") or any(
                        part.startswith(".") for part in path.relative_to(root).parts):
                    continue
                _add_file(root, payload, path.relative_to(root).as_posix())
    return payload


def _model_payload(root: Path, payload: dict[str, Path]) -> dict:
    model_root = root / "models" / "v6"
    outer = _json(model_root / "manifest.json")
    model = outer.get("payload")
    if (set(outer) != {"payload", "sha256"} or not isinstance(model, dict)
            or outer.get("sha256") != _sha_bytes(json.dumps(
                model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())):
        raise ValueError("v6 model manifest checksum differs")
    if model.get("schema") != "coarse_context_submission_v1" or model.get("portable") is not True:
        raise ValueError("v6 model manifest has an unexpected schema or is not portable")
    files = model.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("v6 model manifest has no file map")
    for relative, expected in sorted(files.items()):
        name = _safe_relative(relative)
        path = (model_root / name).resolve()
        if not path.is_relative_to(model_root.resolve()) or not path.is_file():
            raise ValueError(f"v6 model file is missing or unsafe: {name}")
        if _sha_file(path) != expected:
            raise ValueError(f"v6 model file hash differs: {name}")
        payload[f"models/v6/{name}"] = path
    for name, expected in model.get("source_sha256", {}).items():
        source = root / _safe_relative(name)
        if not source.is_file() or _sha_file(source) != expected:
            raise ValueError(f"pinned source hash differs: {name}")
        if name not in payload:
            raise ValueError(f"pinned source is excluded from release allowlist: {name}")
    payload["models/v6/manifest.json"] = model_root / "manifest.json"
    return model


def _submission_payload(root: Path, model_outer_sha: str,
                        payload: dict[str, Path]) -> dict:
    folder = root / "submission"
    record = _json(folder / "manifest.json")
    csv = folder / "CY-DATA-TEAM.csv"
    if (record.get("version") != "v6" or record.get("rows") != 199319
            or record.get("ground_truth_used") is not False
            or record.get("model_directory") != "models/v6"
            or record.get("model_manifest_sha256") != model_outer_sha
            or not csv.is_file() or _sha_file(csv) != EXPECTED_CSV_SHA256
            or record.get("sha256") != EXPECTED_CSV_SHA256):
        raise ValueError("v6 submission CSV or manifest differs from the verified output")
    payload["submission/CY-DATA-TEAM.csv"] = csv
    payload["submission/manifest.json"] = folder / "manifest.json"
    return record


def _benchmark_payload(root: Path, payload: dict[str, Path]) -> dict:
    base = root / "benchmarks" / "v6"
    nested = base / "public-reference"
    folder = nested if (nested / "manifest.json").is_file() else base
    record = _json(folder / "manifest.json")
    if (record.get("schema") != "public_dfl_decoded_reference_v1"
            or record.get("target_ground_truth_used") is not False
            or record.get("target_inputs_included") is not False
            or record.get("raw_participant_assets_included") is not False
            or record.get("labels_file") != "labels.npz"):
        raise ValueError("v6 public benchmark manifest is unexpected or includes target data")
    labels = folder / "labels.npz"
    if (not labels.is_file() or _sha_file(labels) != record.get("labels_sha256")
            or labels.stat().st_size != record.get("labels_bytes")):
        raise ValueError("v6 public benchmark label reference differs from its manifest")
    prefix = folder.relative_to(root).as_posix()
    payload[f"{prefix}/labels.npz"] = labels
    payload[f"{prefix}/manifest.json"] = folder / "manifest.json"
    return record


def _timing_benchmark_payload(root: Path, payload: dict[str, Path]) -> None:
    """Include the optional public timing comparison after recomputing its scores."""
    folder = root / "benchmarks" / "timing-calibration-v1"
    if not folder.exists():
        return
    from scripts.evaluate_timing_reference import evaluate_reference

    result = evaluate_reference(folder)
    record = _json(folder / "manifest.json")
    if (record.get("schema") != "public_dfl_timing_reference_v1"
            or record.get("target_inputs_included") is not False
            or record.get("target_labels_used") is not False
            or result.get("independent_test") is not False):
        raise ValueError("timing benchmark is unexpected or includes target data")
    for name in ("manifest.json", "labels.npz"):
        _add_file(root, payload, (folder / name).relative_to(root).as_posix())


def _zip_bytes(payload: dict[str, Path], package_manifest: dict) -> bytes:
    import io

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        entries = dict(payload)
        entries["package-manifest.json"] = json.dumps(
            package_manifest, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode() + b"\n"
        for name, path in sorted(entries.items()):
            info = zipfile.ZipInfo(_safe_relative(name), date_time=ZIP_TIME)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            data = path.read_bytes() if isinstance(path, Path) else path
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    return output.getvalue()


def verify_archive(archive_path: str | Path) -> dict:
    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "package-manifest.json" not in names:
            raise ValueError("release archive has duplicate entries or no package manifest")
        manifest = json.loads(archive.read("package-manifest.json"))
        files = manifest.get("files")
        if not isinstance(files, dict) or set(names) != {*files, "package-manifest.json"}:
            raise ValueError("release archive contains unrecorded or missing entries")
        for name in names:
            _safe_relative(name)
            if name != "package-manifest.json" and _sha_bytes(archive.read(name)) != files[name]:
                raise ValueError(f"release archive entry hash differs: {name}")
    return manifest


def build_release(repo_root: str | Path, output_zip: str | Path) -> dict:
    root = Path(repo_root).expanduser().resolve()
    output = Path(output_zip).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output == root or (output.is_relative_to(root)
                          and not output.is_relative_to(root / "dist")):
        raise ValueError("release archive inside the repository must be under dist/")

    payload = _source_payload(root)
    model = _model_payload(root, payload)
    submission = _submission_payload(root, _sha_file(root / "models" / "v6" / "manifest.json"),
                                     payload)
    benchmark = _benchmark_payload(root, payload)
    _timing_benchmark_payload(root, payload)
    hashes = {name: _sha_file(path) for name, path in sorted(payload.items())}
    benchmark_manifest_name = next(name for name in payload
                                   if name.startswith("benchmarks/v6/")
                                   and name.endswith("manifest.json"))
    manifest = {"schema": SCHEMA, "files": hashes,
                "model_manifest_sha256": _sha_file(root / "models" / "v6" / "manifest.json"),
                "submission_sha256": submission["sha256"],
                "benchmark_manifest_sha256": _sha_file(root / benchmark_manifest_name),
                "benchmark_schema": benchmark["schema"],
                "model_schema": model["schema"]}
    archive_bytes = _zip_bytes(payload, manifest)

    if output.exists():
        if not output.is_file() or output.read_bytes() != archive_bytes:
            raise ValueError("existing release archive differs, refusing to replace it")
        verify_archive(output)
        return {"archive": str(output), "sha256": _sha_bytes(archive_bytes),
                "files": len(hashes), "reused": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp",
                                                    dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(archive_bytes)
        os.replace(temporary_name, output)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    verify_archive(output)
    return {"archive": str(output), "sha256": _sha_bytes(archive_bytes),
            "files": len(hashes), "reused": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("dist/CY-DATA-TEAM-v6.zip"))
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build_release(args.repo_root, args.output), indent=2, sort_keys=True))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Release packaging failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
