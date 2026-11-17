"""Freeze source files and launch the eight-hour Linux GPU campaign detached."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from possession.storage import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = repo / "artifacts/gpu-tuning-8h-v1"
    runtime = repo / "artifacts/runtime/gpu-tuning-8h-v1"
    if args.resume:
        if not (output / "launch.json").exists():
            raise RuntimeError("no existing campaign to resume")
        with (output / ".supervisor.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = json.loads((runtime / "snapshot-manifest.json").read_text())
        for name, digest in snapshot["files"].items():
            if sha256_file(runtime / name) != digest:
                raise RuntimeError(f"frozen source differs: {name}")
    else:
        if output.exists() or runtime.exists():
            raise RuntimeError("campaign exists; inspect it or use --resume without resetting its clock")
        if shutil.disk_usage(repo).free < 2_000_000_000:
            raise RuntimeError("at least 2 GB free required before launching")
        preflight = repo / "artifacts/gpu-tuning-integration-v1/verification.json"
        if not preflight.is_file():
            raise RuntimeError("run the end-to-end integration check before launch")
        output.mkdir(parents=True)
        runtime.mkdir(parents=True)
        sources = [repo / name for name in ("games.py", "pyproject.toml", "uv.lock")]
        for name in ("possession", "experiments", "evaluation", "scripts"):
            sources.extend((repo / name).rglob("*.py"))
        sources.extend((repo / "configs").glob("*.json"))
        hashes = {}
        for source in sorted(set(sources)):
            relative = source.relative_to(repo)
            destination = runtime / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            hashes[str(relative)] = sha256_file(destination)
            if hashes[str(relative)] != sha256_file(source):
                raise RuntimeError("source changed while snapshotting")
            destination.chmod(0o444)
        write_json(runtime / "snapshot-manifest.json", {"source_root": str(repo), "files": hashes})
    command = [sys.executable, "-u", "-m", "scripts.watch_gpu_tuning",
               "--repo", str(repo), "--output", str(output), "--hours", "8"]
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(runtime)
    with (output / "supervisor.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=runtime, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"status": "submitted", "supervisor_pid": process.pid, "command": command,
              "runtime": str(runtime), "output": str(output), "hours": 8,
              "snapshot_sha256": sha256_file(runtime / "snapshot-manifest.json"),
              "resume": args.resume}
    if not args.resume:
        record["integration_verification_sha256"] = sha256_file(preflight)
    write_json(output / "launch.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
