"""Reproduce an extracted submission while blocking access to original artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

RUNNER = '''import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

task = json.loads(Path(sys.argv[1]).read_text())
root = Path(task["extract"]).resolve()
source = root / "source"
os.chdir(source)
sys.path.insert(0, str(source))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for name in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"
os.environ.pop("OMP_DYNAMIC", None)
blocked_roots = [Path(value).resolve() for value in task["blocked_roots"]]
venv = Path(task["venv"]).resolve()
target = Path(task["target"]).resolve()
allowed_inputs = {(target / name).resolve() for name in
                  ("input-manifest.json", "tactical/BAR-ATH_tactical_atd.txt",
                   "tactical/BAR-ATH_tactical_atd.xml")}
input_reads = set()

def guard(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if path in allowed_inputs:
        input_reads.add(path.name)
        return
    if path.is_relative_to(root) or path.is_relative_to(venv):
        return
    if any(path == base or path.is_relative_to(base) for base in blocked_roots):
        raise PermissionError(f"portable inference attempted original-source/data access: {path}")

sys.addaudithook(guard)
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
module = importlib.import_module(task["module"])
if not Path(module.__file__).resolve().is_relative_to(source):
    raise ValueError("inference module was not loaded from the extracted archive")
function = getattr(module, task["function"])
started = time.perf_counter()
result = function(root / "model", target, root / "reproduced.csv")
actual = hashlib.sha256((root / "reproduced.csv").read_bytes()).hexdigest()
if actual != task["csv_sha256"] or result["output"]["sha256"] != actual:
    raise ValueError("extracted inference differs from the delivered CSV")
proof = {"status": "complete", "archive_sha256": task["archive_sha256"],
         "archive_files_verified": task["files_count"], "csv_sha256": actual,
         "csv_byte_identical": True, "relocated_source_and_models": True,
         "original_training_and_source_access_blocked": True,
         "target_inputs_read": sorted(input_reads), "target_ground_truth_used": False,
         "inference_seconds": time.perf_counter() - started}
(root / "portability-result.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\\n")
print(json.dumps(proof, indent=2), flush=True)
'''


def verify_archive(archive_path: Path, target: Path, output_dir: Path, kind: str) -> dict:
    repo = Path(__file__).resolve().parents[1]
    archive_path, target, output_dir = (p.resolve() for p in (archive_path, target, output_dir))
    extract = output_dir / "verified-extract"
    if extract.exists():
        raise ValueError("extraction directory exists, inspect the prior verification first")
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("package-manifest.json"))
        hashes = manifest["files"]
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {*hashes, "package-manifest.json"}:
            raise ValueError("archive contains duplicate or unrecorded entries")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
                raise ValueError(f"unsafe archive entry: {name}")
            if name in hashes and hashlib.sha256(archive.read(name)).hexdigest() != hashes[name]:
                raise ValueError(f"archive entry hash differs: {name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        archive.extractall(extract)
    task = {"extract": str(extract), "target": str(target),
            "venv": str(repo / ".venv"),
            "blocked_roots": [str(repo), "/Volumes/T7/Data/more-than-a-hack-2027"],
            "module": f"experiments.{kind}_submission", "function": f"infer_{kind}_submission",
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "csv_sha256": manifest["csv_sha256"], "files_count": len(names)}
    task_path = extract / "portability-task.json"
    task_path.write_text(json.dumps(task, indent=2) + "\n")
    runner = extract / "verify_portability.py"
    runner.write_text(RUNNER)
    subprocess.run([sys.executable, str(runner), str(task_path)], cwd=extract, check=True)
    proof = json.loads((extract / "portability-result.json").read_text())
    (output_dir / "package-verification.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("ema", "coarse"), required=True)
    args = parser.parse_args()
    verify_archive(args.archive, args.target, args.output_dir, args.kind)


if __name__ == "__main__":
    main()
