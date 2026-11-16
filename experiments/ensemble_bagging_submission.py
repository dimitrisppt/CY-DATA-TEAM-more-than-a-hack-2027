"""Portable four-checkpoint inference for a two-seed, two-architecture ensemble."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from experiments import campaign_protocol, campaign_refit, campaign_submission
from experiments.campaign_decode import decode_arrays, reference_basis
from experiments.campaign_evaluation import _gap_seconds
from experiments.robustness import _execution_environment, _source_hashes
from games import GAME, GAMES
from possession.features import build_features
from possession.io_atd import read_atd
from possession.predict import load_model_checkpoint, predict_features
from possession.storage import sha256_file, write_json

SEEDS = (20261116, 20261117)
MEMBERS = campaign_refit.MEMBER_IDS
N_FRAMES = campaign_submission.N_FRAMES
CENTER_SIZE = campaign_submission.CENTER_SIZE
TARGET_SPANS = campaign_submission.TARGET_SPANS


def _digest(value):
    return campaign_refit._digest(value)


def _inside(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def _checks_pass(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value) and all(_checks_pass(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_checks_pass(item) for item in value)
    return False


def _gate_pass(value) -> bool:
    return (isinstance(value, dict) and value.get("passed") is True
            and _checks_pass(value.get("checks")))


def _selection_valid(selection: dict, expected: dict | None = None) -> None:
    if not isinstance(selection, dict) or selection.get("kind") != "convex_pair":
        raise ValueError("bagging selection must be a convex pair")
    if [member.get("id") for member in selection.get("members", [])] != list(MEMBERS):
        raise ValueError("bagging selection members or order differ")
    weight = selection.get("blend_weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("bagging selection blend weight is invalid")
    from experiments.campaign_decode import _validate_config

    _validate_config(selection.get("decoder"))
    if expected is not None:
        for key in ("kind", "members"):
            if selection.get(key) != expected.get(key):
                raise ValueError(f"candidate member specs or order change frozen {key}")
        for left, right in zip(selection["members"], expected["members"], strict=True):
            if (left.get("id") != right.get("id") or left.get("spec") != right.get("spec")
                    or left.get("steps") != right.get("steps")):
                raise ValueError("candidate member specs, order, or fit steps differ")


def _study_records(study_dir: Path) -> tuple[dict, dict, dict]:
    protocol = campaign_protocol.read_record(study_dir / "protocol.json")
    result = campaign_protocol.read_record(study_dir / "result.json")
    if protocol is None or result is None:
        raise ValueError("bagging study protocol and result are required")
    protocol_sha = _digest(protocol)
    if result.get("protocol_sha256") != protocol_sha:
        raise ValueError("bagging result protocol hash differs")
    if (protocol.get("baseline_selection") != result.get("baseline_selection")
            or protocol.get("candidate_selection") != result.get("candidate_selection")):
        raise ValueError("bagging protocol selections differ from result")
    study_sources = protocol.get("source_sha256")
    current_sources = _source_hashes()
    if (not isinstance(study_sources, dict) or any(
            current_sources.get(path) != digest for path, digest in study_sources.items())):
        raise ValueError("bagging study source hashes differ from current code")
    if protocol.get("runtime") != _execution_environment():
        raise ValueError("bagging study runtime differs from current environment")
    _validate_gate(result, protocol)
    return protocol, result, {"protocol_sha256": protocol_sha}


def _validate_gate(result: dict, protocol: dict | None = None) -> None:
    if result.get("eligible") is not True:
        raise ValueError("bagging confirmation did not pass eligibility")
    batches = result.get("batches")
    if not isinstance(batches, list) or len(batches) != 2:
        raise ValueError("bagging result must contain two confirmation batches")
    frozen_batches = protocol.get("mask_batches") if isinstance(protocol, dict) else None
    if frozen_batches is not None and len(frozen_batches) != 2:
        raise ValueError("bagging protocol must freeze two mask batches")
    for index, batch in enumerate(batches):
        if not isinstance(batch, dict):
            raise TypeError("bagging batch evidence must be a mapping")
        seeds = batch.get("mask_seeds")
        seed_values = list(seeds.values()) if isinstance(seeds, dict) else seeds
        if (not isinstance(seed_values, list) or len(seed_values) < 2
                or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seed_values)
                or len(set(seed_values)) != len(seed_values)
                or not isinstance(batch.get("baseline"), list) or len(batch["baseline"]) != 2
                or not isinstance(batch.get("candidate"), dict)):
            raise ValueError("bagging batch evidence is incomplete")
        if frozen_batches is not None:
            frozen = frozen_batches[index]
            scenarios = frozen.get("scenarios", {})
            replay_seed = scenarios.get("replay", {}).get("seed")
            expected_seeds = {"short": frozen["short_seed"], "long": frozen["long_seed"],
                              "replay": replay_seed}
            if not isinstance(replay_seed, int) or isinstance(replay_seed, bool):
                raise ValueError("bagging protocol must freeze a replay mask seed")
            if seeds != expected_seeds:
                raise ValueError("bagging result mask seeds differ from protocol")
        if not _gate_pass(batch.get("checks")):
            raise ValueError("bagging batch checks failed")
        outer = batch.get("outer")
        if (not isinstance(outer, dict) or not isinstance(outer.get("baseline"), list)
                or len(outer["baseline"]) != 2 or not isinstance(outer.get("candidate"), dict)):
            raise ValueError("bagging outer evidence is incomplete")
        if not _gate_pass(outer.get("checks")):
            raise ValueError("bagging outer checks failed")


def _copy_verified(source: Path, target: Path, expected_sha: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or sha256_file(target) != expected_sha:
            raise ValueError(f"refusing to replace different bundle file: {target}")
        return
    shutil.copyfile(source, target)
    if sha256_file(target) != expected_sha:
        target.unlink(missing_ok=True)
        raise ValueError(f"copied bundle file hash differs: {target}")


def export_bagged_bundle(study_dir, first_bundle, second_bundle, output_dir) -> dict:
    """Verify the passed two-seed gate and publish a portable four-model bundle."""
    study_dir, first_bundle, second_bundle, output_dir = (
        Path(path).resolve() for path in (study_dir, first_bundle, second_bundle, output_dir))
    if any(_inside(output_dir, path) for path in (study_dir, first_bundle, second_bundle)):
        raise ValueError("bagged output must be separate from study and source bundles")
    protocol, result, protocol_ref = _study_records(study_dir)

    # Gate validation deliberately precedes source verification and every output write.
    first = campaign_refit.verify_bundle(first_bundle)
    second = campaign_refit.verify_bundle(second_bundle)
    bundles = {first["manifest"]["seed"]: first, second["manifest"]["seed"]: second}
    if tuple(sorted(bundles)) != tuple(sorted(SEEDS)):
        raise ValueError("source bundles must use the two frozen refit seeds")
    first_manifest, second_manifest = bundles[SEEDS[0]]["manifest"], bundles[SEEDS[1]]["manifest"]
    if (first_manifest["selection"] != second_manifest["selection"]
            or first_manifest["steps"] != 7200 or second_manifest["steps"] != 7200
            or first_manifest["selection_matches"] != second_manifest["selection_matches"]
            or first_manifest["fit_matches"] != second_manifest["fit_matches"]):
        raise ValueError("seed bundles do not share the original selection and fit cohort")
    baseline_selection = result.get("baseline_selection")
    candidate_selection = result.get("candidate_selection")
    _selection_valid(baseline_selection, first_manifest["selection"])
    _selection_valid(candidate_selection, first_manifest["selection"])
    if baseline_selection != first_manifest["selection"]:
        raise ValueError("bagging baseline selection differs from source bundles")
    if protocol.get("baseline_selection") != baseline_selection:
        raise ValueError("bagging protocol baseline selection differs from result")
    if protocol.get("candidate_selection") != candidate_selection:
        raise ValueError("bagging protocol candidate selection differs from result")
    first_sources, second_sources = first_manifest["source_sha256"], second_manifest["source_sha256"]
    required_sources = {"experiments/campaign_refit.py", "experiments/campaign_training.py",
                        "experiments/campaign_decode.py", "experiments/campaign_evaluation.py",
                        "experiments/campaign_protocol.py", "experiments/campaign_submission.py",
                        "possession/predict.py", "possession/features.py", "possession/model.py"}
    overlap = set(first_sources) & set(second_sources)
    if (not required_sources <= set(first_sources) or not required_sources <= set(second_sources)
            or any(first_sources[name] != second_sources[name] for name in overlap)
            or any(first_sources[name] != second_sources[name] for name in required_sources)
            or first_manifest["runtime"] != second_manifest["runtime"]):
        raise ValueError("seed bundles have different or incomplete source/runtime ancestry")
    output_dir.mkdir(parents=True, exist_ok=True)
    child_hashes = {}
    for seed in SEEDS:
        verified = bundles[seed]
        source_dir = Path(verified["bundle_dir"])
        relative_root = Path(f"seed_{seed}")
        required = {"manifest.json": sha256_file(source_dir / "manifest.json"),
                    **verified["manifest"]["files"]}
        for relative, digest in required.items():
            _copy_verified(source_dir / relative, output_dir / relative_root / relative, digest)
            child_hashes[f"{relative_root.as_posix()}/{relative}"] = digest
    for source, target in ((study_dir / "protocol.json", output_dir / "bagging-protocol.json"),
                           (study_dir / "result.json", output_dir / "bagging-result.json")):
        digest = sha256_file(source)
        _copy_verified(source, target, digest)
        child_hashes[target.name] = digest
    manifest = {
        "schema": "ensemble_bagging_submission_v1", "seeds": list(SEEDS),
        "seed_weights": {str(seed): .5 for seed in SEEDS},
        "member_ids": list(MEMBERS), "candidate_selection": candidate_selection,
        "baseline_selection": baseline_selection, "protocol_sha256": protocol_ref["protocol_sha256"],
        "study_protocol_sha256": sha256_file(study_dir / "protocol.json"),
        "study_result_sha256": sha256_file(study_dir / "result.json"),
        "files": child_hashes, "source_sha256": _source_hashes(),
        "runtime": _execution_environment(),
        "refit_source_sha256": {str(SEEDS[0]): first_sources, str(SEEDS[1]): second_sources},
        "study_source_sha256": protocol.get("source_sha256", {}),
        "study_runtime": protocol.get("runtime", {}),
        "selection_matches": first_manifest["selection_matches"],
        "fit_matches": first_manifest["fit_matches"], "steps": 7200,
        "architecture_seed_blend_weight": .5,
        "candidate_architecture_blend_weight": candidate_selection["blend_weight"],
    }
    campaign_protocol.publish(output_dir / "manifest.json", manifest)
    verify_bagged_bundle(output_dir)
    return {"bundle_dir": str(output_dir), "manifest": manifest}


def verify_bagged_bundle(bundle_dir) -> dict:
    bundle_dir = Path(bundle_dir).resolve()
    manifest = campaign_protocol.read_record(bundle_dir / "manifest.json")
    if manifest is None or manifest.get("schema") != "ensemble_bagging_submission_v1":
        raise ValueError("invalid bagged submission manifest")
    if (manifest.get("seeds") != list(SEEDS) or manifest.get("member_ids") != list(MEMBERS)
            or manifest.get("steps") != 7200 or manifest.get("seed_weights") != {str(s): .5 for s in SEEDS}):
        raise ValueError("bagged submission seed or member configuration differs")
    for relative, digest in manifest.get("files", {}).items():
        path = (bundle_dir / relative).resolve()
        if bundle_dir not in path.parents or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"bagged bundle file differs: {relative}")
    protocol = campaign_protocol.read_record(bundle_dir / "bagging-protocol.json")
    result = campaign_protocol.read_record(bundle_dir / "bagging-result.json")
    if (protocol is None or result is None or _digest(protocol) != manifest.get("protocol_sha256")
            or result.get("protocol_sha256") != manifest.get("protocol_sha256")
            or sha256_file(bundle_dir / "bagging-protocol.json") != manifest.get("study_protocol_sha256")
            or sha256_file(bundle_dir / "bagging-result.json") != manifest.get("study_result_sha256")):
        raise ValueError("bagged study evidence differs from manifest")
    if result.get("eligible") is not True:
        raise ValueError("bagging confirmation is not eligible")
    _validate_gate(result, protocol)
    if (result.get("baseline_selection") != manifest.get("baseline_selection")
            or result.get("candidate_selection") != manifest.get("candidate_selection")
            or protocol.get("baseline_selection") != manifest.get("baseline_selection")
            or protocol.get("candidate_selection") != manifest.get("candidate_selection")):
        raise ValueError("bagged proof selections differ from manifest")
    _selection_valid(manifest["candidate_selection"], manifest["baseline_selection"])
    children = {}
    for seed in SEEDS:
        child = campaign_refit.verify_bundle(bundle_dir / f"seed_{seed}")
        if (child["manifest"]["seed"] != seed
                or child["manifest"]["selection"] != manifest["baseline_selection"]
                or child["manifest"]["source_sha256"] != manifest["refit_source_sha256"].get(str(seed))
                or child["manifest"]["runtime"] != manifest["runtime"]):
            raise ValueError(f"bagged seed child ancestry differs: {seed}")
        children[seed] = child
    source_hashes = _source_hashes()
    if any(source_hashes.get(name) != digest for name, digest in manifest["source_sha256"].items()):
        raise ValueError("bagged bundle source code differs")
    if _execution_environment() != manifest["runtime"]:
        raise ValueError("bagged bundle runtime differs")
    if (manifest.get("study_source_sha256", {}) != protocol.get("source_sha256", {})
            or manifest.get("study_runtime", {}) != protocol.get("runtime", {})):
        raise ValueError("bagging study source/runtime differs from manifest")
    if (any(source_hashes.get(name) != digest
            for name, digest in protocol.get("source_sha256", {}).items())
            or protocol.get("runtime") != _execution_environment()):
        raise ValueError("bagging study source/runtime differs from current execution")
    return {"manifest": manifest, "protocol": protocol, "result": result, "children": children,
            "bundle_dir": str(bundle_dir)}


def _predict_half(frames, models: dict[int, dict], selection: dict) -> np.ndarray:
    if selection.get("kind") != "convex_pair" or len(selection.get("members", [])) != 2:
        raise ValueError("bagged inference requires a frozen two-member pair")
    features = build_features(frames)
    predictions = {seed: {} for seed in SEEDS}
    for member in selection["members"]:
        member_id = member["id"]
        for seed in SEEDS:
            predictions[seed][member_id] = predict_features(
                features, models[seed][member_id], device="cpu", center_size=CENTER_SIZE,
                halo=models[seed][member_id].radius)
    scores = combine_four_predictions(predictions, selection)
    arrays = {"neural_live": scores["live_probability"],
              "neural_home": scores["home_probability"], "neural_boundary": scores["boundary"],
              "ball_valid": frames.ball_valid,
              "nearest_ball_seconds": _gap_seconds(frames.ball_valid, frames.fps),
              **reference_basis(frames, features)}
    return decode_arrays(arrays, selection["decoder"])


def combine_four_predictions(predictions: dict, selection: dict) -> dict:
    """Blend seed logits within each architecture, then blend architectures."""
    if selection.get("kind") != "convex_pair" or len(selection.get("members", [])) != 2:
        raise ValueError("bagged inference requires a frozen two-member pair")
    architecture_scores = {}
    for member in selection["members"]:
        member_id = member["id"]
        architecture_scores[member_id] = campaign_submission.blend_neural(
            predictions[SEEDS[0]][member_id], predictions[SEEDS[1]][member_id], .5)
    return campaign_submission.blend_neural(
        architecture_scores[selection["members"][0]["id"]],
        architecture_scores[selection["members"][1]["id"]], selection["blend_weight"])


def infer_bagged_submission(bundle_dir, data_dir, output_path) -> dict:
    started = time.perf_counter()
    bundle_dir, data_dir, output = (Path(path).resolve()
                                   for path in (bundle_dir, data_dir, output_path))
    sidecar = output.with_suffix(".manifest.json")
    if output == sidecar:
        raise ValueError("output must be a CSV path distinct from its manifest")
    if any(path.is_relative_to(base) for path in (output, sidecar) for base in (bundle_dir, data_dir)):
        raise ValueError("output must be outside bundle and input directories")
    bundle = verify_bagged_bundle(bundle_dir)
    inputs = campaign_submission._verified_inputs(data_dir)
    signature = {"bundle_sha256": sha256_file(bundle_dir / "manifest.json"), "inputs": inputs,
                 "selection": bundle["manifest"]["candidate_selection"],
                 "rows": N_FRAMES, "spans": [list(span) for span in TARGET_SPANS],
                 "source_sha256": _source_hashes(), "runtime": _execution_environment(),
                 "center_size": CENTER_SIZE, "device": "cpu", "seed_weights": {str(s): .5 for s in SEEDS}}
    fingerprint = _digest(signature)
    if output.exists() or sidecar.exists():
        if not (output.is_file() and sidecar.is_file()):
            raise ValueError("existing output is incomplete, use a new output path")
        previous = json.loads(sidecar.read_text())
        if previous.get("request_sha256") != fingerprint or previous.get("output", {}).get("sha256") != sha256_file(output):
            raise ValueError("existing output belongs to a different or changed inference run")
        campaign_submission._validate_output(output)
        return previous

    models = {}
    for seed, child in bundle["children"].items():
        models[seed] = {member: load_model_checkpoint(path, device="cpu")[0]
                        for member, path in child["member_paths"].items()}
    frames = read_atd(inputs["raw"]["path"], inputs["metadata"]["path"], n_frames=N_FRAMES,
                      pitch=GAMES[GAME]["pitch"], fps=GAMES[GAME]["fps"],
                      graded_spans=TARGET_SPANS)
    codes = np.full(len(frames.frame_id), 3, dtype=np.int8)
    halves = []
    previous_stop = 0
    for period, (start, stop) in enumerate(TARGET_SPANS, 1):
        if not previous_stop <= start < stop <= len(frames.frame_id):
            raise ValueError("target half spans are invalid")
        previous_stop = stop
        half = frames.subset(start, stop)
        half.period = period
        path = _predict_half(half, models, bundle["manifest"]["candidate_selection"])
        if path.shape != (stop - start,) or not np.isin(path, [0, 1, 2]).all():
            raise ValueError("decoder returned invalid half labels")
        codes[start:stop] = path
        halves.append({"period": period, "span": [start, stop],
                       "switches": int(np.count_nonzero(np.diff(path))),
                       "ball_observed_fraction": float(half.ball_valid.mean())})
    from possession.cli import write_submission

    write_submission(output, codes, n_expected=N_FRAMES)
    campaign_submission._validate_output(output)
    report = {"schema": "ensemble_bagging_submission_v1", "ground_truth_used": False,
              "request_sha256": fingerprint, "request": signature,
              "output": {"path": str(output), "sha256": sha256_file(output), "rows": N_FRAMES,
                         "label_counts": dict(zip("HADX", np.bincount(codes, minlength=4).tolist()))},
              "halves": halves, "elapsed_seconds": time.perf_counter() - started,
              "format_validated": True}
    write_json(sidecar, report)
    return report
