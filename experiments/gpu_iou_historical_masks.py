"""Evaluate fresh GPU fits on the exact historical refinement outage presets.

The initial fresh GPU protocol froze the earlier campaign search/confirmation
presets. Preserve that evaluation, and assess the same completed fits on the
historical refinement masks in a separately frozen, outcome-independent pass.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.campaign_protocol import publish, read_record
from experiments.ensemble_bagging import MASK_BATCHES, _scenarios
from experiments.gpu_iou import ROOT, status, verify_hash
from experiments.gpu_iou_evaluation import (
    infer,
    load_reference,
    score,
    verify_training,
    write_report,
)
from experiments.gpu_runtime import configure_device
from experiments.robustness import _digest
from possession.storage import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT / "artifacts/gpu-iou-v1")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    source = args.study_dir.resolve()
    output = source / "historical-masks"
    plan = read_record(source / "protocol.json")
    if plan is None:
        raise FileNotFoundError("freeze the GPU training protocol first")
    original_evaluation = read_record(source / "evaluation-protocol.json")
    if original_evaluation is None:
        raise FileNotFoundError("the first assessment protocol must be frozen")
    for name, digest in original_evaluation["source_sha256"].items():
        verify_hash(ROOT / name, digest)
    evaluation = {**plan, "batches": {b["name"]: _scenarios(b) for b in MASK_BATCHES},
                  "training_protocol_sha256": _digest(plan),
                  "assessment_mask_set": "historical_refinement_bagging_presets",
                  "assessment_source_sha256": {
                      "experiments/gpu_iou_historical_masks.py": sha256_file(__file__),
                      "experiments/ensemble_bagging.py": sha256_file(ROOT / "experiments/ensemble_bagging.py")}}
    publish(output / "protocol.json", evaluation)
    if args.score_only:
        labels, manifest = load_reference(output / "reference")
        result = score(labels, manifest)
        saved = read_record(output / "result.json")
        if saved != result:
            raise ValueError("historical-mask label-only reproduction differs")
        print(json.dumps({"historical_mask_label_reproduction": True}))
        return
    if args.wait:
        status(output, "waiting_for_full_training_and_first_assessment")
        while not (source / "result.json").is_file():
            state = json.loads((source / "status.json").read_text())
            if state["stage"] in ("failed", "interrupted"):
                raise RuntimeError(f"upstream study {state['stage']}: {state.get('error')}")
            time.sleep(30)
    configure_device(plan["resources"]["device"])
    status(output, "verifying_training")
    fits = verify_training(plan, source)
    if (output / "reference/manifest.json").exists():
        labels, manifest = load_reference(output / "reference")
        if manifest["protocol_sha256"] != _digest(evaluation):
            raise ValueError("historical-mask reference protocol differs")
    else:
        labels, manifest = infer(evaluation, output, fits)
    status(output, "scoring")
    result = score(labels, manifest)
    write_report(result, output)
    publish(output / "result.json", result)
    status(output, "complete", status="complete", report=str(output / "report/report.md"))


if __name__ == "__main__":
    main()
