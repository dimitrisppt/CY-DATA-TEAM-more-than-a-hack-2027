"""Print saved GPU-study progress without importing Torch or reading tensors."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/gpu-iou-v1"))
    args = parser.parse_args()
    root = args.output_dir.resolve()
    status = json.loads((root / "status.json").read_text())
    saved = {}
    for path in (root / "fits").rglob("state_*.pt"):
        if not path.with_suffix(".pt.sha256.json").exists():
            continue
        role = path.relative_to(root / "fits").parts[0]
        step = int(path.stem.split("_")[-1])
        updates = step if role in ("parent", "coarse") else max(0, step - 7200)
        saved[path.parent] = max(saved.get(path.parent, 0), updates)
    total = 18 * 7200 + 36 * 1200
    historical = root / "historical-masks"
    folds = [json.loads(p.read_text()) for p in sorted(root.glob("status-fold*.json"))]
    for fold in folds:
        if fold["stage"] == "training_complete":
            fold["status"] = "complete"
    output = {"stage": status["stage"], "saved_updates": sum(saved.values()),
              "total_updates": total, "training_percent": round(100 * sum(saved.values()) / total, 1),
              "completed_fits": len(list((root / "fits").rglob("fit.json"))), "total_fits": 54,
              "folds": folds,
              "output_dir": str(root), "report_ready": (root / "result.json").is_file(),
              "historical_mask_report_ready": (historical / "result.json").is_file(),
              "historical_mask_stage": (json.loads((historical / "status.json").read_text())["stage"]
                                        if (historical / "status.json").exists() else "not_started"),
              "prediction_files": {
                  "additional_masks": len(list((root / "inference").rglob("*.npz"))),
                  "historical_masks": len(list((historical / "inference").rglob("*.npz"))),
                  "expected_per_assessment": 1536},
              "all_assessments_complete": ((root / "result.json").is_file()
                                           and (historical / "result.json").is_file()),
              "note": "Training and prediction counts exclude subsequent decoding and scoring."}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
