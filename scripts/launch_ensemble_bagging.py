"""Confirm fixed seed averaging, then conditionally fit and export a four-network candidate."""

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
from experiments.campaign_protocol import Budget, BudgetExceeded, read_record
from experiments.ensemble_bagging import run_bagging_confirmation
from experiments.ensemble_bagging_submission import (
    export_bagged_bundle, verify_bagged_bundle, infer_bagged_submission)
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
    budget = Budget(task["study"], 2.)
    status("confirming_seed_averaging")
    result = run_bagging_confirmation(task["campaign"], task["polish"], task["data"],
                                      task["target"], task["study"],
                                      check=lambda: budget.check(.4))
    if not result["eligible"]:
        status("complete_no_promotion", summary=result["summary"], baseline_replaced=False)
    else:
        budget.check(.9)
        if not (Path(task["first_bundle"]) / "manifest.json").is_file():
            status("waiting_for_first_refit", summary=result["summary"])
        while not (Path(task["first_bundle"]) / "manifest.json").is_file():
            budget.check(.9)
            time.sleep(15)
        status("refitting_additional_seed", summary=result["summary"], seed=20261117)
        subprocess.run([sys.executable, "-m", "experiments.campaign_refit",
                        "--campaign-dir", task["campaign"], "--data-dir", task["data"],
                        "--output-dir", task["second_bundle"], "--seed", "20261117"],
                       check=True, timeout=max(1., budget.seconds_left(.9)))
        status("packaging_verified_models")
        export_bagged_bundle(task["study"], task["first_bundle"], task["second_bundle"], task["bundle"])
        manifest = read_record(Path(task["bundle"]) / "manifest.json")
        for name in [*manifest["files"], "manifest.json"]:
            source = Path(task["bundle"]) / name
            destination = Path(task["local_bundle"]) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256_file(destination) != sha256_file(source):
                raise ValueError(f"existing local bundle differs: {name}")
            shutil.copy2(source, destination)
        verify_bagged_bundle(task["local_bundle"])
        budget.check()
        status("exporting_confirmed_seed_average")
        output = Path(task["submission"]) / "CY-DATA-TEAM.csv"
        first = infer_bagged_submission(task["bundle"], task["target"], output)
        repeat = Path(task["external_submission"]) / "repeated-inference.csv"
        second = infer_bagged_submission(task["bundle"], task["target"], repeat)
        if first["output"]["sha256"] != second["output"]["sha256"]:
            raise ValueError("separate inference CSV hashes differ")
        verification = {"status": "complete", "format_validated": True,
                        "independent_inference_byte_identical": True,
                        "csv": first["output"], "study": task["study"],
                        "bundle": task["local_bundle"], "validation_summary": result["summary"],
                        "model_seeds": [20261116, 20261117],
                        "target_ground_truth_used": False, "submitted_externally": False,
                        "baseline_replaced": False}
        write_json(output.parent / "verification.json", verification)
        for source in (output, output.with_suffix(".manifest.json"), output.parent / "verification.json"):
            destination = Path(task["external_submission"]) / source.name
            if destination.exists() and sha256_file(destination) != sha256_file(source):
                raise ValueError(f"existing external output differs: {source.name}")
            shutil.copy2(source, destination)
            if sha256_file(destination) != sha256_file(source):
                raise ValueError(f"external copy differs: {source.name}")
        status("complete_promoted_candidate", verification=verification)
    print(status_path.read_text(), flush=True)
except (BudgetExceeded, subprocess.TimeoutExpired) as error:
    status("budget_reached", error=str(error), baseline_replaced=False)
    print(status_path.read_text(), flush=True)
except BaseException as error:
    status("failed", error=str(error), error_type=type(error).__name__)
    traceback.print_exc()
    raise
'''


def main():
    repo = Path(__file__).resolve().parents[1]
    external = Path("/Volumes/T7/Data/more-than-a-hack-2027/artifacts")
    runtime = repo / "artifacts/runtime/ensemble-bagging-v1"
    local = repo / "artifacts/ensemble-bagging-v1"
    label = "org.cy-data-team.ensemble-bagging-v1"
    task = {"local": str(local), "campaign": str(external / "overnight-joint-v1"),
            "polish": str(external / "ensemble-polish-v1"),
            "data": str(repo / "data/processed/dfl"), "target": str(repo / "data/target/2027"),
            "study": str(external / "ensemble-bagging-v1"),
            "first_bundle": str(external / "final-ensemble-v2"),
            "second_bundle": str(external / "final-ensemble-seed20261117"),
            "bundle": str(external / "final-bagged-v1"),
            "local_bundle": str(repo / "artifacts/final-bagged-v1"),
            "submission": str(repo / "artifacts/submission-candidate-v4"),
            "external_submission": str(external / "submission-candidate-v4")}
    new_paths = [runtime, local] + [Path(task[key]) for key in
                                   ("study", "second_bundle", "bundle", "local_bundle",
                                    "submission", "external_submission")]
    if any(path.exists() for path in new_paths):
        raise RuntimeError("bagging paths exist, inspect the recorded job before resuming")
    if not Path("/Volumes/T7").is_mount() or shutil.disk_usage(external).free < 20_000_000_000:
        raise RuntimeError("bagging needs the mounted T7 with at least 20 GB free")
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
    entry = runtime / "run_bagging.py"
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
