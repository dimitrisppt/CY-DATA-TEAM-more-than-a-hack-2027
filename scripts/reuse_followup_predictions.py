"""Reuse predictions between frozen mask batches only when full input hashes agree."""
import json
import time
from pathlib import Path

from experiments.gpu_iou import verify_hash
from experiments.iou_followup_training import OUTPUT, freeze
from possession.storage import sha256_file, write_json
from scripts.reuse_iou_predictions import reuse_once


def index_inputs(config, output):
    index = {}
    for batch in config["batches"]:
        path = Path(config["input_root"]) / batch / "input-manifest.json"
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value.get("status") != "complete":
            return None
        verify_hash(path, json.loads(path.with_suffix(".sha256.json").read_text())["sha256"])
        for row in value["records"]:
            key = row["match"], row["period"], row["input_sha256"]
            index.setdefault(key, []).append((output, batch, row["scenario"]))
    return index


def main():
    plan = freeze()
    seen = {arm: set() for arm in plan["arms"]}
    receipts, indices = [], {}
    while True:
        for arm, config in plan["arms"].items():
            if arm not in indices:
                index = index_inputs(config, OUTPUT / arm)
                if index is None:
                    continue
                indices[arm] = index
            receipts += reuse_once(OUTPUT / arm, indices[arm], seen[arm])
        write_json(OUTPUT / "prediction-reuse.json", {
            "script_sha256": sha256_file(Path(__file__)), "helper_sha256": sha256_file(
                Path(__file__).with_name("reuse_iou_predictions.py")),
            "linked_predictions": len(receipts), "records": receipts})
        if all((OUTPUT / arm / "result.json").exists() for arm in plan["arms"]):
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
