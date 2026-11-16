"""Code and runtime fingerprints accompanying local experiment artifacts."""

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

from possession.storage import sha256_file


def runtime_provenance():
    root = Path(__file__).resolve().parent.parent
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "torch", "optuna", "scikit-learn", "numba"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    revision = None
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                  capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    paths = list((root / "possession").glob("*.py")) + list((root / "experiments").glob("*.py"))
    paths += [root / "games.py", root / "evaluation" / "possession_metrics.py"]
    return {"versions": versions, "git_revision": revision,
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            "source_sha256": {str(p.relative_to(root)): sha256_file(p)
                              for p in sorted(paths) if p.is_file()}}
