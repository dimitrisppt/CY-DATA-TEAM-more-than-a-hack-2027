"""Overlap completed-arm assessment with the final, separately frozen WMX fits."""
import time
from pathlib import Path

from experiments.campaign_protocol import publish, read_record
from experiments.gpu_iou import verify_hash
from experiments.gpu_runtime import configure_device
from experiments.iou_followup_evaluation import evaluate_arm
from experiments.iou_followup_training import OUTPUT, freeze
from experiments.robustness import _digest
from possession.storage import sha256_file


def publish_completed_arm(plan, arm):
    config = plan["arms"][arm]
    directory = OUTPUT / arm
    paths = [directory / f"training-fold{i}.json" for i in range(len(config["folds"]))]
    if not all(p.exists() for p in paths):
        return False
    rows = [read_record(p) for p in paths]
    if any(r["protocol_sha256"] != _digest(config) for r in rows):
        raise ValueError("completed-fold protocol differs")
    fits = [f for row in rows for f in row["fits"]]
    if len(fits) != plan["expected_new_fits"][arm]:
        raise ValueError("incomplete completed-arm training grid")
    fits.sort(key=lambda r: (r["identity"]["fold"], r["identity"]["base_seed"],
                             r["identity"]["member"], r["identity"]["role"]))
    publish(directory / "training.json", {"protocol_sha256": _digest(config), "fits": fits})
    return True


def main():
    plan = freeze()
    evaluation = read_record(OUTPUT / "evaluation-protocol.json")
    root = Path(__file__).resolve().parents[1]
    for name, digest in evaluation["source_sha256"].items():
        verify_hash(root / name, digest)
    publish(OUTPUT / "evaluation-scheduling.json", {
        "script_sha256": sha256_file(Path(__file__)), "protocol_sha256": _digest(plan),
        "policy": "Wait for all confirmation and ablation fits before assessment, then overlap "
                  "three inference workers with the single WMX training worker. Each arm must have its "
                  "complete authenticated fit grid before inference. Candidate and analysis stay fixed."})
    # Avoid six concurrent heavy workers on this four-core host.
    while not all(publish_completed_arm(plan, arm) for arm in ("confirmation", "ablation")):
        time.sleep(20)
    configure_device("cuda")
    for arm in ("confirmation", "ablation", "guard"):
        while not publish_completed_arm(plan, arm):
            time.sleep(20)
        evaluate_arm(plan["arms"][arm], OUTPUT / arm)


if __name__ == "__main__":
    main()
