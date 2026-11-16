"""Freeze and run verified final EMA replay, CSV inference, and local/T7 copies."""

from __future__ import annotations

import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from possession.storage import sha256_file, write_json

RUNNER = '''from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

root = Path(__file__).resolve().parent
os.chdir(root)
sys.path.insert(0, str(root))
for name in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"
os.environ.pop("OMP_DYNAMIC", None)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch
from experiments.ema_submission import export_ema_bundle, verify_ema_bundle, infer_ema_submission
from possession.storage import sha256_file, write_json

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
task = json.loads((root / "delivery-task.json").read_text())
status_path = Path(task["local_output"]) / "delivery-status.json"
started = time.time()

def status(stage, **extra):
    write_json(status_path, {"stage": stage, "started_unix": started,
                            "updated_unix": time.time(), "pid": os.getpid(), **extra})

def copy_checked(source, destination):
    digest = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) != digest:
        raise ValueError(f"existing delivery file differs: {destination}")
    shutil.copy2(source, destination)
    if sha256_file(destination) != digest:
        raise ValueError(f"delivery copy differs: {destination}")

try:
    snapshot = json.loads((root / "snapshot-manifest.json").read_text())
    for name, digest in snapshot["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"frozen runtime changed: {name}")
    status("replaying_final_tails")
    exported = export_ema_bundle(task["study"], task["first_bundle"],
                                task["second_bundle"], task["prepared"], task["bundle"])
    status("copying_verified_bundle")
    local_bundle = Path(task["local_bundle"])
    for name in [*exported["manifest"]["files"], "manifest.json"]:
        copy_checked(Path(task["bundle"]) / name, local_bundle / name)
    verify_ema_bundle(local_bundle)
    status("inferring")
    output = Path(task["local_output"]) / "CY-DATA-TEAM.csv"
    first = infer_ema_submission(task["bundle"], task["target"], output)
    status("independent_process_inference")
    repeat = Path(task["external_output"]) / "repeated-inference.csv"
    subprocess.run([sys.executable, "-m", "experiments.ema_submission",
                    "--bundle", str(local_bundle), "--target", task["target"],
                    "--output", str(repeat)], cwd=root, check=True)
    second = json.loads(repeat.with_suffix(".manifest.json").read_text())
    if first["output"]["sha256"] != second["output"]["sha256"]:
        raise ValueError("independent inference processes produced different CSV hashes")
    verification = {"status": "complete", "format_validated": True,
                    "independent_inference_byte_identical": True,
                    "independent_process_and_bundle_copy": True,
                    "target_ground_truth_used": False, "submitted_externally": False,
                    "csv": first["output"], "bundle": str(local_bundle),
                    "model_seeds": [20261116, 20261117], "steps_per_member": 7200,
                    "ema_decay": .99, "ema_tail_updates": 200,
                    "source_manifest_sha256": sha256_file(root / "snapshot-manifest.json"),
                    "elapsed_seconds": time.time() - started}
    write_json(Path(task["local_output"]) / "verification.json", verification)
    for source in (output, output.with_suffix(".manifest.json"),
                   Path(task["local_output"]) / "verification.json"):
        copy_checked(source, Path(task["external_output"]) / source.name)
    status("complete", verification=verification)
    print(json.dumps(verification, indent=2), flush=True)
except BaseException as error:
    status("failed", error=str(error), error_type=type(error).__name__)
    traceback.print_exc()
    raise
'''


def main():
    repo = Path(__file__).resolve().parents[1]
    external = Path("/Volumes/T7/Data/more-than-a-hack-2027/artifacts")
    runtime = repo / "artifacts/runtime/ema-delivery-v1"
    local = repo / "artifacts/submission-candidate-v5"
    label = "org.cy-data-team.ema-delivery-v1"
    task = {"study": str(external / "ema-stability-v1"),
            "first_bundle": str(external / "final-ensemble-v2"),
            "second_bundle": str(external / "final-ensemble-seed20261117"),
            "prepared": str(repo / "data/processed/dfl"),
            "target": str(repo / "data/target/2027"),
            "bundle": str(external / "final-ema-v1"),
            "local_bundle": str(repo / "artifacts/final-ema-v1"),
            "local_output": str(local),
            "external_output": str(external / "submission-candidate-v5")}
    if sys.platform != "darwin" or not Path("/Volumes/T7").is_mount():
        raise RuntimeError("EMA delivery requires macOS and the mounted T7 drive")
    if any(path.exists() for path in (runtime, local, Path(task["bundle"]),
                                     Path(task["local_bundle"]), Path(task["external_output"]))):
        raise RuntimeError("delivery paths exist, inspect or resume the recorded job")
    if shutil.disk_usage(external).free < 1_000_000_000:
        raise RuntimeError("EMA delivery needs 1 GB free on T7")
    runtime.mkdir(parents=True)
    local.mkdir(parents=True)
    sources = [repo / name for name in ("games.py", "pyproject.toml", "uv.lock")]
    for directory in ("possession", "experiments", "evaluation", "scripts"):
        sources.extend((repo / directory).rglob("*.py"))
    hashes = {}
    for source in sorted(set(sources)):
        if source.name.startswith("._") or source.name == "ema_calibration_audit.py":
            continue
        relative = source.relative_to(repo)
        destination = runtime / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = sha256_file(destination)
        if hashes[str(relative)] != sha256_file(source):
            raise RuntimeError(f"source changed while snapshotting: {source}")
        destination.chmod(0o444)
    entry = runtime / "run_delivery.py"
    entry.write_text(RUNNER)
    write_json(runtime / "delivery-task.json", task)
    for path in (entry, runtime / "delivery-task.json"):
        hashes[path.name] = sha256_file(path)
        path.chmod(0o444)
    write_json(runtime / "snapshot-manifest.json", {"source_root": str(repo), "files": hashes})
    command = ["/usr/bin/caffeinate", "-i", str(repo / ".venv/bin/python"), "-u", str(entry)]
    job = local / "job.plist"
    with job.open("wb") as handle:
        plistlib.dump({"Label": label, "ProgramArguments": command, "RunAtLoad": True,
                      "KeepAlive": False, "WorkingDirectory": str(runtime),
                      "StandardOutPath": str(local / "stdout.log"),
                      "StandardErrorPath": str(local / "stderr.log")}, handle)
    domain = f"gui/{os.getuid()}"
    launch = {"label": label, "domain": domain, "runtime": str(runtime),
              "job_plist": str(job), "command": command, "task": task,
              "status": "prepared", "snapshot_sha256": sha256_file(runtime / "snapshot-manifest.json")}
    write_json(local / "launch.json", launch)
    subprocess.run(["/bin/launchctl", "bootstrap", domain, str(job)],
                   capture_output=True, text=True, check=True)
    launch["status"] = "submitted"
    write_json(local / "launch.json", launch)
    print(json.dumps({"status": "submitted", "label": label, "local_output": str(local)}))


if __name__ == "__main__":
    main()
