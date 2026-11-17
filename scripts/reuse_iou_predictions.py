"""Reuse immutable predictions only for byte-identical assessment inputs."""

import argparse
import json
import os
import time
from pathlib import Path

from experiments.campaign_protocol import read_record
from possession.storage import sha256_file, write_json


def input_index(study):
    index = {}
    for root in (study, study / "historical-masks"):
        for manifest in sorted((root / "assessment").glob("*/input-manifest.json")):
            expected = json.loads(manifest.with_suffix(".sha256.json").read_text())["sha256"]
            if sha256_file(manifest) != expected:
                raise ValueError("input manifest hash differs")
            value = json.loads(manifest.read_text())
            if value.get("status") != "complete":
                raise ValueError("finish assessment input preparation before cache reuse")
            for row in value["records"]:
                key = row["match"], row["period"], row["input_sha256"]
                index.setdefault(key, []).append((root, manifest.parent.name, row["scenario"]))
    return index


def reuse_once(study, index, seen):
    receipts = []
    for meta in sorted((study / "inference").rglob("*.json")):
        if meta in seen:
            continue
        relative = meta.relative_to(study / "inference")
        if len(relative.parts) != 7:
            continue
        _batch, variant, seed, member, match, half, _ = relative.parts
        value = read_record(meta)
        source = meta.with_suffix(".npz")
        if value is None or sha256_file(source) != value["sha256"]:
            raise ValueError("source prediction hash differs")
        key = match, int(half.removeprefix("half")), value["identity"]["input_sha256"]
        for root, target_batch, scenario in index.get(key, []):
            destination = (root / "inference" / target_batch / variant / seed / member
                           / match / half / f"{scenario}.npz")
            sidecar = destination.with_suffix(".json")
            # Never replace a prediction or interfere with an active writer.
            if destination.exists() or sidecar.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except FileExistsError:
                continue
            try:
                os.link(meta, sidecar)
            except FileExistsError:
                # A writer won the race. Its normal identity/hash checks apply.
                continue
            receipts.append({"source": str(source), "destination": str(destination),
                             "prediction_sha256": value["sha256"], "input_sha256": key[2]})
        seen.add(meta)
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("artifacts/gpu-iou-v1"))
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    study = args.study_dir.resolve()
    index, seen, receipts = input_index(study), set(), []
    while True:
        added = reuse_once(study, index, seen)
        receipts.extend(added)
        if added:
            write_json(study / "prediction-reuse.json", {
                "schema": "identical_input_prediction_reuse_v1", "records": receipts,
                "code_sha256": sha256_file(__file__), "linked_predictions": len(receipts)})
            print(json.dumps({"new_links": len(added), "total_links": len(receipts)}), flush=True)
        if not args.wait or (study / "result.json").exists():
            break
        current = json.loads((study / "status.json").read_text())
        if current["stage"] in ("failed", "interrupted"):
            raise RuntimeError(f"source study {current['stage']}")
        time.sleep(5)


if __name__ == "__main__":
    main()
