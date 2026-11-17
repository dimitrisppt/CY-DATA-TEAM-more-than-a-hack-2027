"""Final integrity, complete-grid and independent label-score reproduction receipt."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.campaign_protocol import publish, read_record
from experiments.gpu_iou import verify_hash
from experiments.gpu_iou_evaluation import load_reference as load_original
from experiments.gpu_runtime import configure_device
from experiments.iou_followup_evaluation import (
    assessment_inputs,
    load_reference,
    score,
    verify_training,
)
from experiments.iou_followup_training import MODES, OUTPUT, SOURCE, freeze
from experiments.robustness import _digest
from possession.storage import sha256_file


def main():
    plan = freeze()
    configure_device("cuda")
    evaluation = read_record(OUTPUT / "evaluation-protocol.json")
    root = Path(__file__).resolve().parents[1]
    for name, digest in evaluation["source_sha256"].items():
        verify_hash(root / name, digest)
    upstream_evaluation = read_record(SOURCE / "evaluation-protocol.json")
    upstream_masks = read_record(SOURCE / "historical-masks/protocol.json")
    for mapping in (upstream_evaluation["source_sha256"], upstream_masks["assessment_source_sha256"]):
        for name, digest in mapping.items():
            verify_hash(root / name, digest)
    verify_hash(root / "models/v6/manifest.json", read_record(SOURCE / "protocol.json")["release_manifest_sha256"])
    verified_files, arms, input_roots = {}, {}, {}
    for arm, config in plan["arms"].items():
        output = OUTPUT / arm
        fits = verify_training(config, output)
        labels, manifest = load_reference(output / "reference")
        if config["input_root"] not in input_roots:
            input_roots[config["input_root"]] = {
                (r["batch"], r["match"], r["period"], r["scenario"]): r["input_sha256"]
                for r in assessment_inputs(config)}
        if any(input_roots[config["input_root"]][r["batch"], r["match"], r["period"], r["scenario"]]
               != r["input_sha256"] for r in manifest["records"]):
            raise ValueError("reference input hashes differ from prepared assessment inputs")
        reproduced = score(labels, manifest)
        reproduced["arm"] = arm
        if reproduced != read_record(output / "result.json"):
            raise ValueError("independently recomputed label scores differ")
        variants = MODES if arm == "ablation" else ("gpu_baseline", *config["variants"])
        expected = set()
        for row in manifest["records"]:
            for variant in variants:
                role = "parent" if variant == "gpu_baseline" else variant
                for seed in config["seeds"]:
                    for member in config["selection"]["members"]:
                        fit = fits[row["fold"], seed, member["id"], role]
                        name = "ema.pt" if role == "parent" else "model.pt"
                        path = (output / "inference" / row["batch"] / variant / f"seed_{seed}"
                                / member["id"] / row["match"] / f"half{row['period']}" / f"{row['scenario']}.npz")
                        expected.add(path)
                        meta = read_record(path.with_suffix(".json"))
                        identity = {"checkpoint_sha256": fit["files"][name], "fit": fit["identity"],
                                    "input_sha256": row["input_sha256"], "device": config["resources"]["device"],
                                    "center_size": 2048}
                        if meta["identity"] != identity:
                            raise ValueError("prediction checkpoint/input identity differs")
                        stat = path.stat()
                        key = stat.st_dev, stat.st_ino
                        if key not in verified_files:
                            verify_hash(path, meta["sha256"])
                            with np.load(path, allow_pickle=False) as archive:
                                if set(archive.files) != {"neural_live", "neural_home", "neural_boundary"}:
                                    raise ValueError("prediction fields differ")
                                if any(len(archive[k]) != row["n_frames"] or not np.isfinite(archive[k]).all()
                                       for k in archive.files):
                                    raise ValueError("prediction shape or values invalid")
                            verified_files[key] = meta["sha256"]
                        elif verified_files[key] != meta["sha256"]:
                            raise ValueError("linked prediction hashes disagree")
        if set((output / "inference").rglob("*.npz")) != expected:
            raise ValueError("missing or unexpected new prediction files")
        arms[arm] = {"new_fits": len(read_record(output / "training.json")["fits"]),
                     "new_predictions": len(expected), "decoded_arrays": len(labels),
                     "label_scores_reproduced_exactly": True,
                     "result_sha256": sha256_file(output / "result.json"),
                     "reference_manifest_sha256": sha256_file(output / "reference/manifest.json")}
    original, _ = load_original(SOURCE / "historical-masks/reference")
    ablation, manifest = load_reference(OUTPUT / "ablation/reference")
    checked = set()
    for row in manifest["records"]:
        keys = [row["ground_truth_key"]]
        keys += [variants[v] for variants in row["group_prediction_keys"].values()
                 for v in ("gpu_baseline", "control", "spell_balanced")]
        for key in keys:
            if not np.array_equal(original[key], ablation[key]):
                raise ValueError("reused conditions do not exactly reproduce original decoded labels")
            checked.add(key)
    progress = __import__("scripts.iou_followup_status", fromlist=["progress"]).progress()
    if progress["complete_fits"] != 80 or progress["completed_updates"] != 240000:
        raise ValueError("final fixed training budget incomplete")
    preflight = read_record(OUTPUT / "preflight-verification.json")
    resource = read_record(OUTPUT / "resource-verification.json")
    if resource:
        verify_hash(resource["resumed_parent_state"], resource["resumed_parent_state_sha256"])
        before = torch.load(resource["resumed_parent_state"], map_location="cpu", weights_only=True)
        after = torch.load(Path(resource["resumed_parent_state"]).with_name("state_007200.pt"),
                           map_location="cpu", weights_only=True)
        if not after["state"]["training_transcript"].startswith(before["state"]["training_transcript"]):
            raise ValueError("resumed guard trajectory lost its original prefix")
    receipt = {"protocol_sha256": _digest(plan), "arms": arms,
               "new_fits": 80, "optimizer_updates": 240000,
               "tests_passed": preflight["test_count"] + (resource["additional_tests_passed"] if resource else 0),
               "guard_resume_prefix_verified": bool(resource),
               "new_prediction_files_verified": sum(a["new_predictions"] for a in arms.values()),
               "unique_prediction_files_hashed": len(verified_files),
               "input_files_verified": sum(len(r) for r in input_roots.values()),
               "original_condition_arrays_reproduced_exactly": len(checked),
               "all_source_and_upstream_hashes_preserved": True,
               "verifier_sha256": sha256_file(Path(__file__)), "automatic_promotion": False}
    publish(OUTPUT / "completion-verification.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
