"""Reproduce the verified v6 tactical-tracking submission locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

EXPECTED_CSV_SHA256 = "4c22b0171509ae1f93c57b72e9394a10623e19bcc1834290c2a9ec640edfea1b"


def _set_thread_environment() -> None:
    for name in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "4"
    os.environ.pop("OMP_DYNAMIC", None)


_set_thread_environment()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _known_names() -> tuple[str, str]:
    from games import asset_paths

    assets = asset_paths()
    return Path(assets["atd_raw"]).name, Path(assets["atd_meta"]).name


def discover_inputs(data_dir: Path, raw: Path | None = None,
                    metadata: Path | None = None) -> tuple[Path, Path]:
    """Resolve the exact tactical ATD pair by explicit paths or unique known filenames."""
    root = Path(data_dir).expanduser().resolve()
    raw_name, metadata_name = _known_names()
    if (raw is None) != (metadata is None):
        raise ValueError("--raw and --metadata must be supplied together")
    if raw is not None:
        pair = Path(raw).expanduser().resolve(), Path(metadata).expanduser().resolve()
        if not all(path.is_file() for path in pair):
            raise FileNotFoundError("explicit raw and metadata files must both exist")
        return pair

    if not root.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {root}")
    found = []
    for name in (raw_name, metadata_name):
        matches = sorted(path.resolve() for path in root.rglob(name) if path.is_file())
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {name} under data directory, found {len(matches)}")
        found.append(matches[0])
    return found[0], found[1]


def stage_inputs(raw: Path, metadata: Path, staging_root: Path) -> Path:
    """Copy the two automatic inputs into a normalized, hash-manifested local directory."""
    raw, metadata = Path(raw).resolve(), Path(metadata).resolve()
    raw_name, metadata_name = _known_names()
    if not raw.is_file() or not metadata.is_file():
        raise FileNotFoundError("raw tracking and metadata inputs must exist")
    fingerprints = {raw_name: _sha256(raw), metadata_name: _sha256(metadata)}
    key = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()[:20]
    target = Path(staging_root).expanduser().resolve() / key
    tactical = target / "tactical"
    tactical.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, name in ((raw, raw_name), (metadata, metadata_name)):
        destination = tactical / name
        expected = fingerprints[name]
        if destination.exists():
            if not destination.is_file() or _sha256(destination) != expected:
                raise ValueError(f"staged input differs from source hash: {name}")
        else:
            shutil.copyfile(source, destination)
            if _sha256(destination) != expected:
                destination.unlink(missing_ok=True)
                raise ValueError(f"staged input hash differs: {name}")
        rows.append({"path": f"tactical/{name}", "sha256": expected,
                     "bytes": destination.stat().st_size})
    manifest_path = target / "input-manifest.json"
    manifest = {"files": rows}
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != encoded:
        raise ValueError("existing staged input manifest differs")
    if not manifest_path.exists():
        manifest_path.write_text(encoded)
    return target


def _is_standard_input(data_dir: Path) -> bool:
    """Return whether standard layout and manifest already pass the submission verifier."""
    try:
        from experiments.campaign_submission import _verified_inputs

        _verified_inputs(data_dir)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def _configure_torch() -> None:
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)


def reproduce(data_dir: Path | None, output: Path, model_dir: Path, *, raw: Path | None = None,
              metadata: Path | None = None, verify_only: bool = False,
              expected_sha256: str = EXPECTED_CSV_SHA256,
              staging_root: Path | None = None) -> dict:
    _configure_torch()
    from experiments import campaign_submission
    from experiments.coarse_submission import infer_coarse_submission, verify_coarse_bundle

    repo = Path(__file__).resolve().parents[1]
    output, model_dir = (Path(value).expanduser().resolve() for value in (output, model_dir))
    bundle = verify_coarse_bundle(model_dir)
    if verify_only:
        return {"verified": True, "model_dir": str(model_dir),
                "schema": bundle["manifest"].get("schema")}

    if data_dir is None:
        data_dir = Path("data/target/2027")
    data_dir = Path(data_dir).expanduser().resolve()

    if raw is None and metadata is None and _is_standard_input(data_dir):
        normalized = data_dir
        source_paths = ()
    else:
        resolved_raw, resolved_metadata = discover_inputs(data_dir, raw, metadata)
        source_paths = (resolved_raw, resolved_metadata)
        stage_root = staging_root or repo / "artifacts" / "reproduce-input-v1"
        normalized = stage_inputs(resolved_raw, resolved_metadata, stage_root)

    sidecar = output.with_suffix(".manifest.json")
    protected_roots = [model_dir, normalized, data_dir]
    protected_roots.extend(path.parent for path in source_paths)
    if any(path.is_relative_to(base) for path in (output, sidecar)
           for base in protected_roots) or output in source_paths or sidecar in source_paths:
        raise ValueError("output must be outside model and input directories")
    report = infer_coarse_submission(model_dir, normalized, output)
    campaign_submission._validate_output(output)
    actual = _sha256(output)
    if actual != expected_sha256:
        raise ValueError(f"reproduced CSV hash differs from verified v6 output: {actual}")
    return {"verified": True, "output": str(output), "sha256": actual,
            "rows": report["output"]["rows"], "format_validated": True,
            "halves": report["halves"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/target/2027"),
                        help="participant input root containing the tactical ATD files")
    parser.add_argument("--output", type=Path, default=Path("outputs/CY-DATA-TEAM.csv"))
    parser.add_argument("--model-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models" / "v6")
    parser.add_argument("--raw", type=Path, help="explicit raw tactical ATD text file")
    parser.add_argument("--metadata", type=Path, help="explicit tactical ATD XML metadata file")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify the portable model and runtime without reading inputs")
    args = parser.parse_args(argv)
    try:
        result = reproduce(args.data_dir, args.output, args.model_dir, raw=args.raw,
                           metadata=args.metadata, verify_only=args.verify_only)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f"v6 reproduction failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
