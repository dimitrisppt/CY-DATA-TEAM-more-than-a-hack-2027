"""Freeze and run the bounded ensemble refinement independently of the chat app."""

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
from experiments.ensemble_polish import run_polish
from experiments.ensemble_polish_submission import infer_polished_submission
from possession.storage import sha256_file, write_json

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
task = json.loads((root / "task.json").read_text())
status_path = Path(task["local"]) / "delivery-status.json"
started = time.time()
def status(stage, **extra):
    write_json(status_path, {"stage": stage, "started_unix": started,
                            "updated_unix": time.time(), **extra})
try:
    snapshot = json.loads((root / "snapshot-manifest.json").read_text())
    for name, digest in snapshot["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"frozen source differs: {name}")
    status("experiment_running")
    result = run_polish(task["campaign"], task["data"], task["target"], task["study"],
                        trials=64, seed=20261116, hours=1.5)
    if result["status"] != "complete":
        status(result["status"], baseline_replaced=False)
    elif not result["confirmation"]["eligible"]:
        status("complete_no_promotion", gate=result["confirmation"]["gate"],
               baseline_replaced=False)
    else:
        status("waiting_for_final_models")
        deadline = time.time() + 3600
        while not (Path(task["bundle"]) / "manifest.json").is_file():
            if time.time() > deadline:
                raise TimeoutError("confirmed decoder awaits final model bundle")
            time.sleep(15)
        status("exporting_confirmed_improvement")
        output = Path(task["submission"]) / "CY-DATA-TEAM.csv"
        first = infer_polished_submission(task["study"], task["bundle"], task["target"], output)
        repeat = Path(task["external_submission"]) / "repeated-inference.csv"
        second = infer_polished_submission(task["study"], task["bundle"], task["target"], repeat)
        if first["output"]["sha256"] != second["output"]["sha256"]:
            raise ValueError("repeated inference CSV differs")
        verification = {"status": "complete", "format_validated": True,
                        "independent_inference_byte_identical": True,
                        "csv": first["output"], "study": task["study"],
                        "target_ground_truth_used": False, "submitted_externally": False,
                        "baseline_replaced": False}
        write_json(output.parent / "verification.json", verification)
        for source in (output, output.with_suffix(".manifest.json"), output.parent / "verification.json"):
            destination = Path(task["external_submission"]) / source.name
            if destination.exists() and sha256_file(destination) != sha256_file(source):
                raise ValueError(f"existing destination differs: {destination}")
            shutil.copy2(source, destination)
            if sha256_file(destination) != sha256_file(source):
                raise ValueError("external copy hash differs")
        status("complete_promoted_candidate", verification=verification)
    print(status_path.read_text(), flush=True)
except BaseException as error:
    status("failed", error=str(error), error_type=type(error).__name__)
    traceback.print_exc()
    raise
'''


def main():
    repo = Path(__file__).resolve().parents[1]
    external = Path("/Volumes/T7/Data/more-than-a-hack-2027/artifacts")
    runtime = repo / "artifacts/runtime/ensemble-polish-v1"
    local = repo / "artifacts/ensemble-polish-v1"
    label = "org.cy-data-team.ensemble-polish-v1"
    task = {"local": str(local), "campaign": str(external / "overnight-joint-v1"),
            "data": str(repo / "data/processed/dfl"), "target": str(repo / "data/target/2027"),
            "study": str(external / "ensemble-polish-v1"),
            "bundle": str(external / "final-ensemble-v2"),
            "submission": str(repo / "artifacts/submission-candidate-v3"),
            "external_submission": str(external / "submission-candidate-v3")}
    if any(path.exists() for path in (runtime, local, Path(task["study"]),
                                     Path(task["submission"]), Path(task["external_submission"]))):
        raise RuntimeError("polish paths exist, inspect the recorded job before resuming")
    if not Path("/Volumes/T7").is_mount() or shutil.disk_usage(external).free < 15_000_000_000:
        raise RuntimeError("polish needs the mounted T7 with at least 15 GB free")
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
    entry = runtime / "run_polish.py"
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
