"""Freeze sources and submit one bounded local campaign to the macOS job manager."""

from __future__ import annotations

import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from possession.storage import sha256_file, write_json


def main():
    repo = Path(__file__).resolve().parents[1]
    local = repo / "artifacts/overnight-joint-v1"
    runtime = repo / "artifacts/runtime/overnight-joint-v1"
    external = Path("/Volumes/T7/Data/more-than-a-hack-2027/artifacts/overnight-joint-v1")
    label = "org.cy-data-team.overnight-joint-v1"
    if sys.platform != "darwin" or not Path("/Volumes/T7").is_mount():
        raise RuntimeError("this launcher requires macOS and the mounted T7 drive")
    if (local / "launch.json").exists() or runtime.exists() or external.exists():
        raise RuntimeError("campaign artifacts already exist; inspect/resume the recorded job instead")
    if shutil.disk_usage("/Volumes/T7").free < 40_000_000_000:
        raise RuntimeError("campaign requires at least 40 GB free on T7")
    local.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True)
    external.mkdir(parents=True)
    sources = [repo / name for name in ("games.py", "pyproject.toml", "uv.lock")]
    for directory in ("possession", "experiments", "evaluation", "scripts"):
        sources.extend((repo / directory).rglob("*.py"))
    sources.extend((repo / "configs").glob("*.json"))
    hashes = {}
    for source in sorted(set(sources)):
        relative = source.relative_to(repo)
        destination = runtime / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = sha256_file(destination)
        if hashes[str(relative)] != sha256_file(source):
            raise RuntimeError(f"source changed while snapshotting: {source}")
        destination.chmod(0o444)
    baseline = runtime / "baseline-recipe.json"
    shutil.copy2(repo / "artifacts/focused-temporal-v1/baseline-recipe.json", baseline)
    hashes[baseline.name] = sha256_file(baseline)
    baseline.chmod(0o444)
    entry = runtime / "run_campaign.py"
    entry.write_text(
        "from pathlib import Path\nimport os, runpy, sys\n"
        "root = Path(__file__).resolve().parent\nos.chdir(root)\n"
        "sys.path.insert(0, str(root))\n"
        "for name in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', "
        "'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):\n    os.environ[name] = '4'\n"
        "os.environ['PYTHONDONTWRITEBYTECODE'] = '1'\n"
        "runpy.run_module('experiments.overnight_tuning', run_name='__main__')\n")
    hashes[entry.name] = sha256_file(entry)
    entry.chmod(0o444)
    write_json(runtime / "snapshot-manifest.json", {"source_root": str(repo), "files": hashes})
    python_command = [str(repo / ".venv/bin/python"), "-u", str(entry),
                      "--data-dir", str(repo / "data/processed/dfl"),
                      "--target-dir", str(repo / "data/target/2027"),
                      "--baseline", str(baseline), "--output-dir", str(external),
                      "--database", str(local / "optuna.sqlite3"), "--hours", "12",
                      "--per-family", "12", "--decoder-trials", "12", "32", "64"]
    stdout, stderr = local / "stdout.log", local / "stderr.log"
    job_path = local / "job.plist"
    with job_path.open("wb") as handle:
        plistlib.dump({"Label": label, "ProgramArguments": ["/usr/bin/caffeinate", "-i", *python_command],
                      "RunAtLoad": True, "KeepAlive": False,
                      "WorkingDirectory": str(runtime), "StandardOutPath": str(stdout),
                      "StandardErrorPath": str(stderr)}, handle)
    command = ["/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(job_path)]
    record = {"label": label, "launchd_domain": f"gui/{os.getuid()}", "runtime": str(runtime),
              "output_dir": str(external), "python_command": python_command,
              "submit_command": command, "stdout": str(stdout), "stderr": str(stderr),
              "job_plist": str(job_path), "job_plist_sha256": sha256_file(job_path),
              "source_manifest_sha256": sha256_file(runtime / "snapshot-manifest.json"),
              "approved_hours": 12, "planned_initial_configurations": 36,
              "status_path": str(external / "status.json"), "status": "prepared"}
    write_json(local / "launch.json", record)
    (local / "run").symlink_to(external, target_is_directory=True)
    subprocess.run(command, check=True, capture_output=True, text=True)
    record["status"] = "submitted"
    write_json(local / "launch.json", record)
    print(json.dumps({key: record[key] for key in
                      ("label", "status", "status_path", "stdout", "stderr")}, indent=2))


if __name__ == "__main__":
    main()
