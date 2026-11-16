"""Run the frozen all-seven ensemble refit and verified CSV export as a local Mac job."""

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
from experiments.campaign_refit import run_refit, verify_bundle
from experiments.campaign_submission import infer_campaign_submission
from possession.storage import sha256_file, write_json

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
task = json.loads((root / "delivery-task.json").read_text())
status_path = Path(task["local_output"]) / "delivery-status.json"
started = time.time()

def status(stage, **extra):
    write_json(status_path, {"stage": stage, "started_unix": started,
                            "updated_unix": time.time(), **extra})

try:
    snapshot = json.loads((root / "snapshot-manifest.json").read_text())
    for name, digest in snapshot["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"frozen runtime changed: {name}")
    status("refitting")
    result = run_refit(task["campaign"], task["prepared"], task["bundle"], seed=20261116)
    status("inferring")
    output = Path(task["local_output"]) / "CY-DATA-TEAM.csv"
    first = infer_campaign_submission(task["bundle"], task["target"], output)
    repeat = Path(task["external_output"]) / "repeated-inference.csv"
    second = infer_campaign_submission(task["bundle"], task["target"], repeat)
    if first["output"]["sha256"] != second["output"]["sha256"]:
        raise ValueError("separate inference runs produced different CSV hashes")
    verified = verify_bundle(task["bundle"])
    local_bundle = Path(task["local_bundle"])
    for name in [*verified["manifest"]["files"], "manifest.json"]:
        source = Path(task["bundle"]) / name
        destination = local_bundle / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise ValueError(f"existing local bundle differs: {name}")
        shutil.copy2(source, destination)
    verify_bundle(local_bundle)
    verification = {"status": "complete", "format_validated": True,
                    "independent_inference_byte_identical": True,
                    "target_ground_truth_used": False, "submitted_externally": False,
                    "csv": first["output"], "bundle": str(local_bundle),
                    "training_seed": 20261116, "steps_per_member": 7200,
                    "source_manifest_sha256": sha256_file(root / "snapshot-manifest.json"),
                    "elapsed_seconds": time.time() - started}
    write_json(Path(task["local_output"]) / "verification.json", verification)
    external = Path(task["external_output"])
    external.mkdir(parents=True, exist_ok=True)
    for source in (output, output.with_suffix(".manifest.json"),
                   Path(task["local_output"]) / "verification.json"):
        destination = external / source.name
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise ValueError(f"existing external delivery differs: {source.name}")
        shutil.copy2(source, destination)
        if sha256_file(destination) != sha256_file(source):
            raise ValueError(f"external delivery copy differs: {source.name}")
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
    runtime = repo / "artifacts/runtime/final-ensemble-v2"
    local = repo / "artifacts/submission-candidate-v2"
    label = "org.cy-data-team.final-ensemble-v2"
    task = {"campaign": str(external / "overnight-joint-v1"),
            "prepared": str(repo / "data/processed/dfl"),
            "target": str(repo / "data/target/2027"),
            "bundle": str(external / "final-ensemble-v2"),
            "local_bundle": str(repo / "artifacts/final-ensemble-v2"),
            "local_output": str(local),
            "external_output": str(external / "submission-candidate-v2")}
    if sys.platform != "darwin" or not Path("/Volumes/T7").is_mount():
        raise RuntimeError("this delivery requires macOS and the mounted T7 drive")
    if any(path.exists() for path in (runtime, local, Path(task["bundle"]),
                                     Path(task["local_bundle"]), Path(task["external_output"]))):
        raise RuntimeError("delivery paths exist, inspect or resume the recorded job")
    if shutil.disk_usage(external).free < 1_000_000_000:
        raise RuntimeError("delivery needs 1 GB free on T7")
    runtime.mkdir(parents=True)
    local.mkdir(parents=True)
    sources = [repo / name for name in ("games.py", "pyproject.toml", "uv.lock")]
    for directory in ("possession", "experiments", "evaluation", "scripts"):
        sources.extend((repo / directory).rglob("*.py"))
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
