"""Cap concurrent inference at two workers while the WMX training pool retains RAM."""
import time
from pathlib import Path

from experiments import iou_followup_evaluation as evaluation
from experiments.campaign_protocol import publish, read_record
from experiments.gpu_iou import verify_hash
from experiments.gpu_runtime import configure_device
from experiments.iou_followup_training import OUTPUT, freeze
from experiments.robustness import _digest
from possession.storage import sha256_file
from scripts.run_iou_followup_evaluation import publish_completed_arm

_EXECUTOR = evaluation.ProcessPoolExecutor


def bounded_executor(*args, **kwargs):
    kwargs["max_workers"] = min(2, kwargs["max_workers"])
    return _EXECUTOR(*args, **kwargs)


def main():
    plan = freeze()
    frozen = read_record(OUTPUT / "evaluation-protocol.json")
    root = Path(__file__).resolve().parents[1]
    for name, digest in frozen["source_sha256"].items():
        verify_hash(root / name, digest)
    publish(OUTPUT / "evaluation-scheduling-bounded.json", {
        "script_sha256": sha256_file(Path(__file__)), "protocol_sha256": _digest(plan),
        "maximum_inference_workers": 2,
        "reason": "The completed training workers retain approximately 2.5 GB each until the WMX job ends. "
                  "Cap inference concurrency to keep the combined processes within 14 GiB RAM.",
        "science_parameters_changed": False, "new_results_seen_before_change": False})
    evaluation.ProcessPoolExecutor = bounded_executor
    while not all(publish_completed_arm(plan, arm) for arm in ("confirmation", "ablation")):
        time.sleep(20)
    configure_device("cuda")
    for arm in ("confirmation", "ablation", "guard"):
        while not publish_completed_arm(plan, arm):
            time.sleep(20)
        evaluation.evaluate_arm(plan["arms"][arm], OUTPUT / arm)


if __name__ == "__main__":
    main()
