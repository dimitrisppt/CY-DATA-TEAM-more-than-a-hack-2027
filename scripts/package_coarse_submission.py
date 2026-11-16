"""Build a deterministic reproduction archive for a verified coarse submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

from scripts import package_bagged_submission as _base

MODEL_SEEDS = [20261116, 20261117]
DOC_ALLOWLIST = ("COARSE_DELIVERY.md", "COARSE_CONTEXT.md", "EMA_STABILITY.md")
ZIP_TIME = _base.ZIP_TIME


def _digest_file(path: Path) -> str:
    return _base._digest_file(path)


def _validate_submission(bundle_dir: Path, submission_dir: Path) -> dict:
    verification_path = submission_dir / "verification.json"
    if not verification_path.is_file():
        raise ValueError("submission verification.json is required")
    verification = json.loads(verification_path.read_text())
    if (verification.get("status") != "complete"
            or verification.get("format_validated") is not True
            or verification.get("target_ground_truth_used") is not False
            or verification.get("submitted_externally") is not False
            or verification.get("independent_inference_byte_identical") is not True
            or verification.get("independent_process_and_bundle_copy") is not True
            or verification.get("model_seeds") != MODEL_SEEDS
            or verification.get("steps_per_member") != 7200
            or verification.get("ema_decay") != .99
            or verification.get("ema_tail_updates") != 200
            or verification.get("coarse_models") != 2
            or verification.get("coarse_hz") != 1
            or verification.get("model_count") != 6):
        raise ValueError("coarse verification is incomplete or has different model evidence")
    csv_record = verification.get("csv")
    if not isinstance(csv_record, dict) or not isinstance(csv_record.get("path"), str):
        raise TypeError("coarse verification must identify its CSV output")
    csv_name = Path(csv_record["path"]).name
    if not csv_name.lower().endswith(".csv"):
        raise ValueError("verification CSV path must name a CSV")
    csv_path = submission_dir / csv_name
    sidecar_path = csv_path.with_suffix(".manifest.json")
    if not csv_path.is_file() or not sidecar_path.is_file():
        raise ValueError("coarse submission CSV and inference sidecar are required")
    if any(not path.resolve().is_relative_to(submission_dir.resolve())
           for path in (csv_path, sidecar_path, verification_path)):
        raise ValueError("submission evidence path escapes submission directory")
    csv_sha = _digest_file(csv_path)
    if csv_record.get("sha256") != csv_sha:
        raise ValueError("submission CSV hash differs from verification")
    sidecar = json.loads(sidecar_path.read_text())
    request = sidecar.get("request")
    if (not isinstance(request, dict)
            or sidecar.get("request_sha256") != _base._digest_json(request)):
        raise ValueError("coarse sidecar request hash differs")
    if (sidecar.get("output", {}).get("sha256") != csv_sha
            or sidecar.get("format_validated") is not True
            or sidecar.get("ground_truth_used") is not False):
        raise ValueError("coarse sidecar does not verify the output")
    bundle_sha = _digest_file(bundle_dir / "manifest.json")
    if request.get("bundle_manifest_sha256") != bundle_sha:
        raise ValueError("coarse inference proof does not link to this bundle manifest")
    return {"csv": csv_path, "sidecar": sidecar_path,
            "verification": verification_path, "csv_sha256": csv_sha,
            "sidecar_sha256": _digest_file(sidecar_path),
            "bundle_manifest_sha256": bundle_sha}


def _readme() -> bytes:
    return b"""# Coarse-context submission reproduction

This archive contains the verified coarse-context submission, portable model bundle, and frozen source snapshot. It contains no participant tracking data or training caches.

Use Python 3.12.11. From the source directory, sync the frozen dependencies:

    cd source
    uv sync --frozen --python 3.12.11 --extra learn --extra tune --extra dev

Set TARGET_DATA_DIR to a directory containing the participant input manifest and raw ATD XML files. Then run:

    export TARGET_DATA_DIR=/path/to/participant-data
    env -u OMP_DYNAMIC OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 uv run --no-sync python -m experiments.coarse_submission --bundle ../model --target "$TARGET_DATA_DIR" --output ../reproduced.csv

Inference verifies the contained coarse proof, EMA base bundle, source models, and hashes before reading participant inputs. The fixed one-hertz player-only specialist supplies gated live and owner context on missing-ball frames. The decoder and fine boundary scores remain frozen.
"""


def verify_coarse_bundle(bundle_dir):
    """Call the strict coarse bundle verifier in the caller's pinned runtime."""
    from experiments.coarse_submission import verify_coarse_bundle as verify

    return verify(bundle_dir)


def package_coarse_submission(bundle_dir, submission_dir, runtime_dir, output_zip,
                              docs_dir) -> dict:
    """Package verified coarse evidence, portable weights, frozen source, and docs."""
    bundle_dir, submission_dir, runtime_dir, output_zip, docs_dir = (
        Path(path).resolve()
        for path in (bundle_dir, submission_dir, runtime_dir, output_zip, docs_dir))
    inputs = (bundle_dir, submission_dir, runtime_dir, docs_dir)
    if any(output_zip == path or output_zip.is_relative_to(path) or path.is_relative_to(output_zip)
           for path in inputs):
        raise ValueError("output archive must be separate from all inputs")

    verified = verify_coarse_bundle(bundle_dir)
    manifest = verified["manifest"]
    if (manifest.get("schema") != "coarse_context_submission_v1"
            or manifest.get("base_seeds") != MODEL_SEEDS
            or manifest.get("portable") is not True
            or not isinstance(manifest.get("source_sha256"), dict)
            or not manifest["source_sha256"]):
        raise ValueError("verified coarse bundle schema, seeds, or source hashes differ")
    proof = _validate_submission(bundle_dir, submission_dir)
    _snapshot, source_paths = _base._snapshot_files(runtime_dir)
    snapshot_files = json.loads((runtime_dir / "snapshot-manifest.json").read_text())["files"]
    for source_name, expected_sha in manifest.get("source_sha256", {}).items():
        if snapshot_files.get(source_name) != expected_sha:
            raise ValueError(f"frozen source snapshot differs from coarse bundle: {source_name}")

    payload: dict[str, Path | bytes] = {}
    bundle_files = manifest.get("files")
    if not isinstance(bundle_files, dict) or not bundle_files:
        raise ValueError("coarse bundle manifest has no contained file hashes")
    for relative, digest in bundle_files.items():
        name = _base._safe_relative(relative)
        source = bundle_dir / name
        if not source.is_file() or _digest_file(source) != digest:
            raise ValueError(f"coarse bundle file changed before packaging: {name}")
        payload[f"model/{name}"] = source
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file() or _digest_file(manifest_path) != proof["bundle_manifest_sha256"]:
        raise ValueError("coarse bundle manifest changed before packaging")
    payload["model/manifest.json"] = manifest_path
    payload.update(source_paths)
    payload[f"submission/{proof['csv'].name}"] = proof["csv"]
    payload[f"submission/{proof['sidecar'].name}"] = proof["sidecar"]
    payload["submission/verification.json"] = proof["verification"]
    for name in DOC_ALLOWLIST:
        path = docs_dir / name
        if (not path.is_file()
                or not path.resolve().is_relative_to(docs_dir.resolve())):
            raise ValueError(f"allowlisted coarse delivery document is missing: {name}")
        payload[f"docs/{name}"] = path
    payload["README.md"] = _readme()

    hashes = {name: (hashlib.sha256(value).hexdigest() if isinstance(value, bytes)
                     else _digest_file(value)) for name, value in sorted(payload.items())}
    package_manifest = {
        "schema": "coarse_submission_reproduction_v1", "files": hashes,
        "model_seeds": MODEL_SEEDS, "steps_per_member": 7200,
        "coarse_models": 2, "coarse_hz": 1, "model_count": 6,
        "bundle_manifest_sha256": proof["bundle_manifest_sha256"],
        "csv_sha256": proof["csv_sha256"],
        "inference_sidecar_sha256": proof["sidecar_sha256"],
        "runtime_snapshot_sha256": _digest_file(runtime_dir / "snapshot-manifest.json"),
    }
    payload["package-manifest.json"] = json.dumps(
        package_manifest, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode() + b"\n"

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
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED,
                                 compresslevel=9)
        with zipfile.ZipFile(temp_path, "r") as archive:
            if archive.testzip() is not None:
                raise ValueError("generated coarse package ZIP failed CRC validation")
            if sorted(archive.namelist()) != sorted(payload):
                raise ValueError("generated coarse package ZIP entries differ")
            for name, expected_sha in hashes.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected_sha:
                    raise ValueError(f"generated coarse package entry hash differs: {name}")
            if json.loads(archive.read("package-manifest.json")) != package_manifest:
                raise ValueError("generated coarse package manifest differs")
        archive_sha = _digest_file(temp_path)
        if output_zip.exists():
            if not output_zip.is_file() or _digest_file(output_zip) != archive_sha:
                raise ValueError("refusing to replace a different coarse package archive")
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
    result = package_coarse_submission(args.bundle_dir, args.submission_dir,
                                       args.runtime_dir, args.output_zip, args.docs_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
