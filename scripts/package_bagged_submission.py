"""Build a deterministic local reproduction archive for a verified bagged submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

MODEL_SEEDS = [20261116, 20261117]
DOC_ALLOWLIST = ("ENSEMBLE_BAGGING.md", "CY-DATA-TEAM-ensemble-report.md",
                 "ENSEMBLE_DELIVERY.md", "ENSEMBLE_POLISH.md")
ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _safe_relative(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or str(path) != name):
        raise ValueError(f"unsafe package path: {name}")
    return name


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _validate_submission(bundle_dir: Path, submission_dir: Path) -> dict:
    verification_path = submission_dir / "verification.json"
    if not verification_path.is_file():
        raise ValueError("submission verification.json is required")
    verification = json.loads(verification_path.read_text())
    if (verification.get("status") != "complete"
            or verification.get("format_validated") is not True
            or verification.get("target_ground_truth_used") is not False
            or verification.get("model_seeds") != MODEL_SEEDS):
        raise ValueError("submission verification status, format, labels, or model seeds differ")
    if verification.get("independent_inference_byte_identical") is not True:
        raise ValueError("submission verification lacks byte-identical repeated inference")
    csv_record = verification.get("csv")
    if not isinstance(csv_record, dict) or not isinstance(csv_record.get("path"), str):
        raise TypeError("verification must identify its CSV output")
    csv_name = Path(csv_record["path"]).name
    if not csv_name.lower().endswith(".csv"):
        raise ValueError("verification CSV path must name a CSV")
    csv_path = submission_dir / csv_name
    sidecar_path = csv_path.with_suffix(".manifest.json")
    if not csv_path.is_file() or not sidecar_path.is_file():
        raise ValueError("submission CSV and inference sidecar are required")
    csv_sha = _digest_file(csv_path)
    if csv_record.get("sha256") != csv_sha:
        raise ValueError("submission CSV hash differs from verification")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar_sha = _digest_file(sidecar_path)
    request = sidecar.get("request")
    if not isinstance(request, dict) or sidecar.get("request_sha256") != _digest_json(request):
        raise ValueError("inference sidecar request hash differs")
    if (sidecar.get("output", {}).get("sha256") != csv_sha
            or sidecar.get("format_validated") is not True
            or sidecar.get("ground_truth_used") is not False):
        raise ValueError("CSV sidecar does not verify the output")
    bundle_sha = _digest_file(bundle_dir / "manifest.json")
    if request.get("bundle_sha256") != bundle_sha:
        raise ValueError("submission proof does not link to the verified bagged bundle")
    return {"csv": csv_path, "sidecar": sidecar_path,
            "verification": verification_path, "csv_sha256": csv_sha,
            "sidecar_sha256": sidecar_sha, "bundle_manifest_sha256": bundle_sha}


def _snapshot_files(runtime_dir: Path) -> tuple[dict, dict[str, Path]]:
    manifest_path = runtime_dir / "snapshot-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("runtime snapshot-manifest.json is required")
    snapshot = json.loads(manifest_path.read_text())
    files = snapshot.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("runtime snapshot manifest has no file hashes")
    paths = {}
    for raw_name, expected_sha in files.items():
        name = _safe_relative(raw_name)
        relative = PurePosixPath(name)
        forbidden_parts = {"data", "dataset", "datasets", "target", "targets", "cache",
                           "caches", "checkpoints", "predictions", "fits", "models", "processed"}
        if any(part.lower() in forbidden_parts for part in relative.parts):
            raise ValueError(f"runtime snapshot includes a data or training directory: {name}")
        if Path(name).suffix.lower() not in {".py", ".toml", ".lock", ".json"}:
            raise ValueError(f"runtime snapshot includes a non-source file: {name}")
        path = runtime_dir / name
        resolved = path.resolve()
        if runtime_dir.resolve() not in resolved.parents or not resolved.is_file():
            raise ValueError(f"snapshot file is missing or escapes runtime: {name}")
        if path.suffix.lower() in {".npz", ".pt", ".pth", ".ckpt", ".onnx"}:
            raise ValueError(f"runtime snapshot includes a training or data artifact: {name}")
        if _digest_file(path) != expected_sha:
            raise ValueError(f"frozen runtime snapshot hash differs: {name}")
        paths[f"source/{name}"] = path
    paths["source/snapshot-manifest.json"] = manifest_path
    return snapshot, paths


def _readme() -> bytes:
    return b"""# Bagged submission reproduction

This archive contains the verified submission, portable four-model bundle, and frozen source snapshot. It contains no participant tracking data or training caches.

Use Python 3.12.11. From `source/`, sync the frozen dependencies with `uv sync --frozen --python 3.12.11 --extra learn --extra tune --extra dev`. Set `TARGET_DATA_DIR` to a directory containing the participant input manifest and raw ATD XML files. The command below writes a fresh submission beside this archive.

```sh
cd source
uv sync --frozen --python 3.12.11 --extra learn --extra tune --extra dev
export TARGET_DATA_DIR=/path/to/participant-data
env -u OMP_DYNAMIC OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 uv run --no-sync python - <<'PY'
import os
import torch
from experiments.ensemble_bagging_submission import infer_bagged_submission
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
infer_bagged_submission('../model', os.environ['TARGET_DATA_DIR'], '../reproduced.csv')
PY
```

The inference routine verifies bundle hashes and the target input manifest. It uses the two recorded model seeds, averages seed predictions within each architecture, then applies the frozen architecture blend and decoder.
"""


def verify_bagged_bundle(bundle_dir):
    """Load the verifier only after the caller has configured its runtime."""
    from experiments.ensemble_bagging_submission import verify_bagged_bundle as verify

    return verify(bundle_dir)


def package_bagged_submission(bundle_dir, submission_dir, runtime_dir, output_zip, docs_dir) -> dict:
    """Package verified inference evidence, bundle, frozen source, and allowlisted docs."""
    bundle_dir, submission_dir, runtime_dir, output_zip, docs_dir = (
        Path(path).resolve() for path in (bundle_dir, submission_dir, runtime_dir, output_zip, docs_dir))
    inputs = (bundle_dir, submission_dir, runtime_dir, docs_dir)
    if any(output_zip == path or output_zip.is_relative_to(path) or path.is_relative_to(output_zip)
           for path in inputs):
        raise ValueError("output archive must be separate from all inputs")
    verified_bundle = verify_bagged_bundle(bundle_dir)
    if verified_bundle["manifest"].get("seeds") != MODEL_SEEDS:
        raise ValueError("verified model seeds differ from required pair")
    proof = _validate_submission(bundle_dir, submission_dir)
    _, source_paths = _snapshot_files(runtime_dir)
    snapshot_files = json.loads((runtime_dir / "snapshot-manifest.json").read_text())["files"]
    for source_name, expected_sha in verified_bundle["manifest"].get("source_sha256", {}).items():
        if snapshot_files.get(source_name) != expected_sha:
            raise ValueError(f"frozen source snapshot differs from bagged bundle: {source_name}")

    payload: dict[str, Path | bytes] = {}
    for relative, digest in verified_bundle["manifest"]["files"].items():
        name = _safe_relative(relative)
        source = bundle_dir / name
        if not source.is_file() or _digest_file(source) != digest:
            raise ValueError(f"verified bundle file changed before packaging: {name}")
        payload[f"model/{name}"] = source
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("bagged bundle manifest is missing")
    payload["model/manifest.json"] = manifest_path
    payload.update(source_paths)
    payload["submission/" + proof["csv"].name] = proof["csv"]
    payload["submission/" + proof["sidecar"].name] = proof["sidecar"]
    payload["submission/verification.json"] = proof["verification"]
    for name in DOC_ALLOWLIST:
        path = docs_dir / name
        if not path.is_file():
            raise ValueError(f"allowlisted delivery document is missing: {name}")
        payload[f"docs/{name}"] = path
    payload["README.md"] = _readme()

    hashes = {name: (hashlib.sha256(value).hexdigest() if isinstance(value, bytes)
                     else _digest_file(value)) for name, value in sorted(payload.items())}
    package_manifest = {"schema": "bagged_submission_reproduction_v1", "files": hashes,
                        "model_seeds": MODEL_SEEDS,
                        "bundle_manifest_sha256": _digest_file(manifest_path),
                        "csv_sha256": proof["csv_sha256"],
                        "inference_sidecar_sha256": proof["sidecar_sha256"],
                        "runtime_snapshot_sha256": _digest_file(runtime_dir / "snapshot-manifest.json")}
    payload["package-manifest.json"] = json.dumps(
        package_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output_zip.name}.", suffix=".tmp",
                                     dir=output_zip.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as archive:
            for name, value in sorted(payload.items()):
                info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                data = value if isinstance(value, bytes) else value.read_bytes()
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        with zipfile.ZipFile(temp_path, "r") as archive:
            if archive.testzip() is not None:
                raise ValueError("generated package ZIP failed CRC validation")
            if sorted(archive.namelist()) != sorted(payload):
                raise ValueError("generated package ZIP entries differ")
            for name, expected_sha in hashes.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected_sha:
                    raise ValueError(f"generated package entry hash differs: {name}")
            if json.loads(archive.read("package-manifest.json")) != package_manifest:
                raise ValueError("generated package manifest differs")
        archive_sha = _digest_file(temp_path)
        if output_zip.exists():
            if not output_zip.is_file() or _digest_file(output_zip) != archive_sha:
                raise ValueError("refusing to replace a different package archive")
            temp_path.unlink()
        else:
            os.replace(temp_path, output_zip)
        return {"artifact_path": str(output_zip), "sha256": archive_sha,
                "files_count": len(payload), "size_bytes": output_zip.stat().st_size}
    finally:
        temp_path.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for thread_var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
                       "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[thread_var] = "4"
    os.environ.pop("OMP_DYNAMIC", None)
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    result = package_bagged_submission(args.bundle_dir, args.submission_dir,
                                       args.runtime_dir, args.output_zip, args.docs_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
