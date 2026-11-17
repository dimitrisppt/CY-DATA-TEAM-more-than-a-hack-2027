"""Regenerate v7 from its eight bundled models and the tactical tracking pair."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.campaign_submission import N_FRAMES, _validate_output, _verified_inputs
from experiments.gpu_runtime import configure_device
from experiments.robustness import _execution_environment
from games import GAME, GAMES
from possession.cli import TARGET_SPANS, write_submission
from possession.io_atd import read_atd
from possession.storage import sha256_file, write_json
from scripts.reproduce import _is_standard_input, discover_inputs, stage_inputs
from scripts.v7_bundle import ROOT, infer_spans, load_models, verify_bundle


def reproduce(model_dir, data_dir, output, *, device="cuda:0", raw=None, metadata=None,
              expected_sha256=None):
    started = time.monotonic()
    device = configure_device(device, threads=4 if device == "cpu" else 1)
    manifest, recipe = verify_bundle(model_dir)
    output, data_dir, model_dir = (Path(p).resolve() for p in (output, data_dir, model_dir))
    sidecar = output.with_suffix(".manifest.json")
    if output == sidecar or output.exists() or sidecar.exists():
        raise ValueError("choose a new CSV output path; existing files are preserved")
    if any(p.is_relative_to(root) for p in (output, sidecar) for root in (data_dir, model_dir)):
        raise ValueError("output must be outside model and input directories")
    if raw is None and metadata is None and _is_standard_input(data_dir):
        normalized = data_dir
    else:
        raw, metadata = discover_inputs(data_dir, raw, metadata)
        normalized = stage_inputs(raw, metadata, ROOT / "artifacts/v7-inputs")
    inputs = _verified_inputs(normalized)
    models = load_models(model_dir, manifest, device)
    frames = read_atd(inputs["raw"]["path"], inputs["metadata"]["path"], n_frames=N_FRAMES,
                      pitch=GAMES[GAME]["pitch"], fps=GAMES[GAME]["fps"], graded_spans=TARGET_SPANS)
    codes, halves = infer_spans(frames, models, recipe, TARGET_SPANS, device)
    write_submission(output, codes, n_expected=N_FRAMES)
    _validate_output(output)
    digest = sha256_file(output)
    report = {"schema": "v7_inference_v1", "version": "v7", "ground_truth_used": False,
              "model_manifest_sha256": sha256_file(model_dir / "manifest.json"),
              "recipe_sha256": sha256_file(model_dir / "recipe.json"),
              "pipeline": recipe["pipeline"], "inputs": inputs,
              "device": device, "environment": _execution_environment(),
              "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
              "output": {"sha256": digest, "rows": N_FRAMES,
                         "label_counts": dict(zip("HADX", np.bincount(codes, minlength=4).tolist()))},
              "halves": halves, "format_validated": True,
              "expected_sha256": expected_sha256,
              "matches_expected": digest == expected_sha256 if expected_sha256 else None,
              "elapsed_seconds": time.monotonic() - started,
              "score_note": "Format validation does not measure target accuracy."}
    write_json(sidecar, report)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"CSV differs from the release hash: {digest}; see {sidecar}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/v7")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/target/2027")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/CY-DATA-TEAM-v7.csv")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest, _ = verify_bundle(args.model_dir)
        print(json.dumps({"verified": True, "version": "v7", "models": len(manifest["models"])}))
        return
    expected = json.loads((ROOT / "submission/manifest.json").read_text())
    expected_sha = expected["sha256"] if expected["version"] == "v7" else None
    report = reproduce(args.model_dir, args.data_dir, args.output, device=args.device,
                       raw=args.raw, metadata=args.metadata, expected_sha256=expected_sha)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
