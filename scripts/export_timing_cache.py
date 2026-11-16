"""Export authenticated public DFL features for temporal calibration experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.timing_cache import export_timing_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = export_timing_cache(args.output_dir, artifacts_root=args.artifacts_root)
    print(json.dumps({"records": result["record_count"],
                      "exact_v6_parity": result["exact_v6_parity"],
                      "manifest": str(args.output_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
