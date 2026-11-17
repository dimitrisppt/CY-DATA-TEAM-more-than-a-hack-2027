"""Recompute the selected v7 recipe's public OOF scores, without training or target data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.campaign_protocol import read_record
from experiments.gpu_tuning_evaluation import average_scores, score_labels
from possession.storage import sha256_file
from scripts.v7_bundle import ROOT, contained


def evaluate(directory):
    directory = Path(directory).resolve()
    manifest = read_record(directory / "manifest.json")
    if manifest["schema"] != "v7_public_validation_reference_v1":
        raise ValueError("unsupported public reference")
    for name, digest in manifest["files"].items():
        path = contained(directory, name)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"reference file missing or changed: {name}")
    collected = {}
    for row in manifest["records"]:
        path = contained(directory, row["directory"])
        result = read_record(path / "result.json")
        if sha256_file(path / "labels.npz") != result["labels_sha256"]:
            raise ValueError("label archive differs")
        with np.load(path / "labels.npz", allow_pickle=False) as archive:
            labels = {name: archive[name] for name in archive.files}
        hashes = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in labels.items()}
        if hashes != result["array_sha256"]:
            raise ValueError("reference label arrays differ")
        scores = score_labels(labels, result["matches"])
        if scores != result["scores"]:
            raise ValueError("recomputed official scores differ")
        collected.setdefault((row["scope"], row["variant"]), []).append(scores)
    output = {}
    for scope in ("confirmation", "guard"):
        output[scope] = {}
        for variant, label in (("incumbent", "incumbent"), ("ensemble", "candidate")):
            rows = collected[scope, variant]
            if len(rows) != 2:
                raise ValueError("expected both mask batches")
            score = average_scores(rows)
            if score != manifest["expected"][scope][label]:
                raise ValueError("aggregate development scores differ")
            output[scope][label] = score["objective"]
        output[scope]["delta"] = output[scope]["candidate"] - output[scope]["incumbent"]
    return {"verified": True, "records": len(manifest["records"]), "scores": output,
            "independent_test": False, "competition_score": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "benchmarks/v7")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.reference), indent=2))


if __name__ == "__main__":
    main()
