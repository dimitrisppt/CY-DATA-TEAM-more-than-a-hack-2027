"""Export a confirmed decoder improvement without changing the frozen model bundle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.campaign_protocol import read_record
from experiments.campaign_refit import verify_bundle
from experiments.campaign_submission import (
    N_FRAMES,
    TARGET_SPANS,
    _validate_output,
    _verified_inputs,
    infer_spans,
)
from experiments.robustness import _digest, _execution_environment, _source_hashes
from games import GAME, GAMES
from possession.cli import write_submission
from possession.io_atd import read_atd
from possession.predict import load_model_checkpoint
from possession.storage import sha256_file, write_json


def verify_promotion(study_dir, bundle_dir):
    study_dir, bundle_dir = Path(study_dir).resolve(), Path(bundle_dir).resolve()
    protocol = read_record(study_dir / "protocol.json")
    selected = read_record(study_dir / "selection.json")
    summary = read_record(study_dir / "summary.json")
    if not protocol or not selected or not summary:
        raise ValueError("polish study is incomplete")
    if (summary.get("status") != "complete"
            or summary.get("protocol_sha256") != _digest(protocol)
            or selected.get("protocol_sha256") != _digest(protocol)
            or selected.get("selection") != summary.get("selection")
            or selected.get("baseline_selection_sha256") != _digest(protocol["baseline_selection"])):
        raise ValueError("polish selection ancestry differs")
    confirmation = summary["confirmation"]
    if (confirmation.get("eligible") is not True
            or confirmation.get("gate", {}).get("passed") is not True
            or confirmation.get("outer", {}).get("gate_passed") is not True):
        raise ValueError("decoder improvement did not pass every promotion check")
    current = _source_hashes()
    if any(current.get(name) != digest for name, digest in protocol["source_sha256"].items()):
        raise ValueError("polish execution source differs")
    if _execution_environment() != protocol["runtime"]:
        raise ValueError("polish execution runtime differs")
    bundle = verify_bundle(bundle_dir)
    baseline, candidate = protocol["baseline_selection"], selected["selection"]
    if (bundle["recipe"]["selection"] != baseline
            or candidate["members"] != baseline["members"]
            or candidate["kind"] != baseline["kind"]):
        raise ValueError("polish selection changed the frozen neural models")
    return bundle, candidate, {name: sha256_file(study_dir / name) for name in
                               ("protocol.json", "selection.json", "summary.json")}


def infer_polished_submission(study_dir, bundle_dir, data_dir, output_path):
    started = time.perf_counter()
    study_dir, bundle_dir, data_dir, output = (Path(path).resolve() for path in
                                              (study_dir, bundle_dir, data_dir, output_path))
    sidecar = output.with_suffix(".manifest.json")
    if output == sidecar or any(path.is_relative_to(base) for path in (output, sidecar)
                               for base in (study_dir, bundle_dir, data_dir)):
        raise ValueError("output must be outside the study, bundle and inputs")
    bundle, selection, study_hashes = verify_promotion(study_dir, bundle_dir)
    inputs = _verified_inputs(data_dir)
    request = {"study_sha256": study_hashes, "bundle_sha256": sha256_file(bundle_dir / "manifest.json"),
               "selection": selection, "inputs": inputs,
               "runtime": _execution_environment(), "source_sha256": _source_hashes()}
    fingerprint = _digest(request)
    if output.exists() or sidecar.exists():
        if not (output.is_file() and sidecar.is_file()):
            raise ValueError("incomplete existing output, use a new path")
        previous = json.loads(sidecar.read_text())
        if (previous.get("request_sha256") != fingerprint
                or previous.get("output", {}).get("sha256") != sha256_file(output)):
            raise ValueError("existing output differs from this verified recipe")
        _validate_output(output)
        return previous
    models = {name: load_model_checkpoint(path, device="cpu")[0]
              for name, path in bundle["member_paths"].items()}
    frames = read_atd(inputs["raw"]["path"], inputs["metadata"]["path"], n_frames=N_FRAMES,
                      pitch=GAMES[GAME]["pitch"], fps=GAMES[GAME]["fps"], graded_spans=TARGET_SPANS)
    codes, halves = infer_spans(frames, models, selection, TARGET_SPANS)
    write_submission(output, codes, n_expected=N_FRAMES)
    codes = _validate_output(output)
    report = {"schema": "ensemble_polish_submission_v1", "ground_truth_used": False,
              "request": request, "request_sha256": fingerprint,
              "format_validated": True, "inference_overrides": {},
              "output": {"path": str(output), "rows": N_FRAMES,
                         "sha256": sha256_file(output),
                         "label_counts": {label: int((codes == index).sum())
                                          for index, label in enumerate("HADX")}},
              "halves": halves, "elapsed_seconds": time.perf_counter() - started,
              "score_note": "The confirmation score is public development evidence, not target accuracy."}
    write_json(sidecar, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    report = infer_polished_submission(args.study_dir, args.bundle, args.data_dir, args.output)
    print(json.dumps(report["output"], indent=2))


if __name__ == "__main__":
    main()
