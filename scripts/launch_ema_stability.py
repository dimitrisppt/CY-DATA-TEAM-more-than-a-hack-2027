"""Freeze and launch the fixed tail-EMA comparison as a bounded macOS job."""

from __future__ import annotations

import json
import os
import plistlib
import shutil
import subprocess
from pathlib import Path

from possession.storage import sha256_file, write_json

RUNNER = '''from pathlib import Path
import json
import os
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
from experiments.campaign_protocol import BudgetExceeded
from experiments.ema_stability import run_ema_stability
from possession.storage import sha256_file, write_json

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
task = json.loads((root / "task.json").read_text())
status_path = Path(task["local"]) / "status.json"
started = time.time()

def status(stage, **extra):
    write_json(status_path, {"stage": stage, "started_unix": started,
                            "updated_unix": time.time(),
                            "target_ground_truth_used": False,
                            "submission_replaced": False, **extra})

try:
    snapshot = json.loads((root / "snapshot-manifest.json").read_text())
    for name, digest in snapshot["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"frozen source differs: {name}")
    status("running_fixed_tail_ema_comparison")
    result = run_ema_stability(task["campaign"], task["polish"], task["bagging"],
                               task["data"], task["target"], task["study"],
                               max_seconds=task["max_seconds"])
    for name in ("protocol.json", "result.json"):
        source = Path(task["study"]) / name
        if source.is_file():
            value = json.loads(source.read_text())
            write_json(Path(task["local"]) / name, value)
            if sha256_file(source) != sha256_file(Path(task["local"]) / name):
                raise ValueError(f"local record copy differs: {name}")
    status("complete_eligible" if result["eligible"] else "complete_no_promotion",
           result_path=str(Path(task["study"]) / "result.json"),
           elapsed_seconds=time.time() - started, eligible=result["eligible"])
    print(status_path.read_text(), flush=True)
except BudgetExceeded as error:
    status("budget_reached", error=str(error), elapsed_seconds=time.time() - started)
    print(status_path.read_text(), flush=True)
except BaseException as error:
    status("failed", error=str(error), error_type=type(error).__name__,
           elapsed_seconds=time.time() - started)
    traceback.print_exc()
    raise
'''


def main():
    repo = Path(__file__).resolve().parents[1]
    external = Path("/Volumes/T7/Data/more-than-a-hack-2027/artifacts")
    name = "ema-stability-v1"
    runtime = repo / "artifacts/runtime" / name
    local = repo / "artifacts" / name
    label = f"org.cy-data-team.{name}"
    task = {
        "local": str(local), "campaign": str(external / "overnight-joint-v1"),
        "polish": str(external / "ensemble-polish-v1"),
        "bagging": str(external / "ensemble-bagging-v1"),
        "data": str(repo / "data/processed/dfl"), "target": str(repo / "data/target/2027"),
        "study": str(external / name), "max_seconds": 5400,
    }
    if any(path.exists() for path in (runtime, local, Path(task["study"]))):
        raise RuntimeError("EMA job paths exist, inspect the recorded job before resuming")
    if not Path("/Volumes/T7").is_mount() or shutil.disk_usage(external).free < 10_000_000_000:
        raise RuntimeError("EMA comparison needs mounted T7 with at least 10 GB free")
    for name in ("experiments/ema_stability.py", "experiments/ema_training.py"):
        if not (repo / name).is_file():
            raise FileNotFoundError(name)
    runtime.mkdir(parents=True)
    local.mkdir(parents=True)
    sources = [repo / name for name in ("games.py", "pyproject.toml", "uv.lock")]
    for directory in ("possession", "experiments", "evaluation", "scripts"):
        sources.extend((repo / directory).rglob("*.py"))
    hashes = {}
    for source in sorted(set(sources)):
        if source.name.startswith("._"):
            continue
        relative = source.relative_to(repo)
        destination = runtime / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = sha256_file(destination)
        if hashes[str(relative)] != sha256_file(source):
            raise RuntimeError(f"source changed while snapshotting: {source}")
        destination.chmod(0o444)
    entry = runtime / "run_ema.py"
    entry.write_text(RUNNER)
    write_json(runtime / "task.json", task)
    for path in (entry, runtime / "task.json"):
        hashes[path.name] = sha256_file(path)
        path.chmod(0o444)
    write_json(runtime / "snapshot-manifest.json", {"source_root": str(repo), "files": hashes})
    job = local / "job.plist"
    command = ["/usr/bin/caffeinate", "-i", str(repo / ".venv/bin/python"), "-u", str(entry)]
    with job.open("wb") as handle:
        plistlib.dump({"Label": label, "ProgramArguments": command, "RunAtLoad": True,
                      "KeepAlive": False, "WorkingDirectory": str(runtime),
                      "StandardOutPath": str(local / "stdout.log"),
                      "StandardErrorPath": str(local / "stderr.log")}, handle)
    domain = f"gui/{os.getuid()}"
    launch = {"label": label, "domain": domain, "runtime": str(runtime),
              "job_plist": str(job), "command": command, "task": task, "status": "prepared",
              "snapshot_sha256": sha256_file(runtime / "snapshot-manifest.json")}
    write_json(local / "launch.json", launch)
    subprocess.run(["/bin/launchctl", "bootstrap", domain, str(job)],
                   capture_output=True, text=True, check=True)
    launch["status"] = "submitted"
    write_json(local / "launch.json", launch)
    print(json.dumps({"status": "submitted", "label": label, "local": str(local)}))


if __name__ == "__main__":
    main()
