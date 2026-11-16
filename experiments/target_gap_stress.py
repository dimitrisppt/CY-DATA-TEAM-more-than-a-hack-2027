"""Post-selection replay of the real target ball-gap pattern on DFL validation halves.

This is a development stress diagnostic. It never fits, tunes, selects, or
promotes a model, and never reads target possession labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluation import possession_metrics as official
from experiments import frozen_transfer
from experiments.augmentation_evaluation import _compact
from experiments.augmentation_tuning import _load_record, _save_record
from experiments.evaluate import _join_with_x, _target_codes
from experiments.focused_tuning import _verify_fit
from experiments.frozen_transfer import _records, _runtime_ancestry
from experiments.oof_selection import decode_oof_half
from experiments.prediction_store import load_prediction_bundle, save_prediction_bundle
from games import GAME, GAMES, half_video_blocks
from possession.diagnostics import merge_spell_diagnostics, spell_error_diagnostics
from possession.features import build_features
from possession.outages import gap_statistics
from possession.predict import load_model_checkpoint, predict_features
from possession.prepared import verify_prepared
from possession.reference import reference_scores
from possession.storage import load_half, sha256_file, write_json

SCENARIO_SEEDS = (5101, 5102)
PROTOCOL = "target_gap_replay_postselection_v1_development"
FRAME_BINS = ("0", "(0,10]", "(10,30]", "(30,60]", "(60,120]", ">120")


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _target_patterns(target_dir: Path) -> tuple[dict[int, np.ndarray], dict]:
    """Read only the tactical ball field inside published graded video spans."""
    manifest_path = target_dir / "input-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    name = "tactical/BAR-ATH_tactical_atd.txt"
    rows = [row for row in manifest.get("files", []) if row.get("path") == name]
    if len(rows) != 1:
        raise ValueError("target input manifest lacks a unique tactical ATD")
    raw = target_dir / name
    if raw.stat().st_size != rows[0]["bytes"] or sha256_file(raw) != rows[0]["sha256"]:
        raise ValueError("target tactical ATD hash or size differs from input manifest")
    blocks = half_video_blocks(GAME)
    if len(blocks) != 2 or [half for half, _, _ in blocks] != [1, 2]:
        raise ValueError("published target graded spans differ")
    spans = {half: (start, stop + 1) for half, start, stop in blocks}
    values = {half: [] for half in spans}
    total_frames = 0
    with raw.open(encoding="utf-8") as stream:
        for frame, line in enumerate(stream):
            total_frames += 1
            if int(line.split(":", 1)[0]) != frame + 1:
                raise ValueError("target tactical ATD frame order differs")
            for half, (start, stop) in spans.items():
                if start <= frame < stop:
                    ball = line.rsplit(":", 1)[-1].strip().lower()
                    if ball in ("*", "nan"):
                        missing = True
                    else:
                        parts = ball.split(",")
                        if len(parts) < 2:
                            raise ValueError("target ball field is malformed")
                        missing = all(value in ("nan", "*") for value in parts[:2])
                        if not missing and (any(value in ("nan", "*") for value in parts[:2])
                                            or not np.isfinite([float(value) for value in parts[:2]]).all()):
                            raise ValueError("target ball observation is malformed")
                    values[half].append(missing)
                    break
    if total_frames != GAMES[GAME]["n_video_frames"]:
        raise ValueError("target tactical ATD frame count differs")
    patterns = {half: np.asarray(values[half], dtype=bool) for half in spans}
    if any(len(patterns[half]) != stop - start for half, (start, stop) in spans.items()):
        raise ValueError("target tactical graded-span length differs")
    evidence = {"manifest_sha256": sha256_file(manifest_path),
                "raw_sha256": rows[0]["sha256"],
                "patterns": {str(half): {"span": list(spans[half]),
                                         "frames": len(pattern),
                                         "missing_frames": int(pattern.sum()),
                                         "missing_sha256": hashlib.sha256(pattern.tobytes()).hexdigest()}
                             for half, pattern in patterns.items()}}
    return patterns, evidence


def _offset(match: str, period: int, seed: int, length: int) -> int:
    return int.from_bytes(hashlib.sha256(
        f"{PROTOCOL}:{match}:{period}:{seed}".encode()).digest()[:8], "big") % length


def _replay_mask(pattern: np.ndarray, n: int, match: str, period: int, seed: int) -> tuple[np.ndarray, dict]:
    """Cyclically replay one target half; split any missing run at a cycle seam."""
    source = np.asarray(pattern)
    if source.ndim != 1 or source.dtype != np.bool_ or not len(source) or n < 0:
        raise ValueError("target pattern must be a nonempty boolean vector and n nonnegative")
    if seed not in SCENARIO_SEEDS or period not in (1, 2) or not match:
        raise ValueError("invalid replay identity")
    offset = _offset(match, period, seed, len(source))
    indices = (np.arange(n, dtype=np.int64) + offset) % len(source)
    mask = source[indices].copy()
    seams = (indices == 0) & (np.arange(n) > 0)
    clipped = seams & source[-1] & source[0]
    mask[clipped] = False

    def source_context(start: int, direction: int) -> int:
        count = 0
        index = start % len(source)
        while count < len(source) and source[index]:
            count += 1
            index = (index + direction) % len(source)
        return count

    start_clip = source_context(offset - 1, -1) if n and source[offset] else 0
    end_clip = source_context(indices[-1] + 1, 1) if n and source[indices[-1]] else 0
    evidence = {"source_offset": offset, "source_length": len(source),
                "cycle_seams": int(seams.sum()),
                "cycle_seam_clipped_missing_frames": int(clipped.sum()),
                "half_start_clipped_source_context_frames": start_clip,
                "half_end_clipped_source_context_frames": end_clip,
                "replayed_missing_frames": int(mask.sum()),
                "replayed_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()}
    return mask, evidence


def _frame_errors(gt: np.ndarray, pred: np.ndarray, ball_valid: np.ndarray, fps: float) -> dict:
    """Frame error counts by duration of each resulting graded ball gap."""
    graded = gt != official.X
    missing = graded & ~ball_valid
    length = np.zeros(len(gt), dtype=np.int32)
    edges = np.diff(np.r_[False, missing, False].astype(np.int8))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        length[start:stop] = stop - start
    seconds = length / fps
    masks = {"0": graded & ball_valid,
             "(0,10]": missing & (seconds <= 10),
             "(10,30]": missing & (seconds > 10) & (seconds <= 30),
             "(30,60]": missing & (seconds > 30) & (seconds <= 60),
             "(60,120]": missing & (seconds > 60) & (seconds <= 120),
             ">120": missing & (seconds > 120)}
    result = {name: {"n_frames": int(mask.sum()),
                     "n_errors": int(np.count_nonzero((gt != pred) & mask))}
              for name, mask in masks.items()}
    for bucket in result.values():
        bucket["error_rate"] = (bucket["n_errors"] / bucket["n_frames"]
                                if bucket["n_frames"] else None)
    assert sum(bucket["n_frames"] for bucket in result.values()) == int(graded.sum())
    return result


def _checkpoint_entries(focused_dir: Path, plan: dict, confirmation: dict) -> dict:
    seed = plan["payload"]["resources"]["confirmation_seed"]
    folds = plan["payload"]["fold"]["inner"]
    result = {}
    for role in ("baseline", "selected"):
        entries = confirmation[role]["entries"]
        if set(entries) != {str(seed)} or len(entries[str(seed)]) != len(folds):
            raise ValueError("confirmation seed or fold count differs")
        result[role] = []
        for index, (entry, fold) in enumerate(zip(entries[str(seed)], folds, strict=True)):
            fit = entry["fit"]
            job = fit["job"]
            if (job["seed"] != seed or job["fold_index"] != index
                    or job["train_matches"] != fold["train"]
                    or job["validation_matches"] != fold["validation"]
                    or job["protocol_sha256"] != plan["sha256"]):
                raise ValueError("confirmation checkpoint fold ancestry differs")
            own = focused_dir / "confirmation" / role / f"seed_{seed}" / f"fold_{index}"
            baseline = focused_dir / "confirmation" / "baseline" / f"seed_{seed}" / f"fold_{index}"
            if role == "selected" and own.exists() and not (own / "model.pt").is_file():
                raise ValueError("incomplete selected confirmation checkpoint directory")
            directory = own if (own / "model.pt").is_file() else baseline
            if role == "selected" and directory == baseline:
                base_entry = confirmation["baseline"]["entries"][str(seed)][index]
                if entry != base_entry:
                    raise ValueError("selected checkpoint reuse differs from baseline fit")
            checkpoint = directory / "model.pt"
            if not checkpoint.is_file() or sha256_file(checkpoint) != fit["checkpoint_sha256"]:
                raise ValueError("confirmation checkpoint hash differs")
            if _load_record(directory / "fit.json") != fit:
                raise ValueError("confirmation fit record differs from published entry")
            _verify_fit(directory, plan, job, fit)
            result[role].append({"fold_index": index, "checkpoint": str(checkpoint),
                                 "checkpoint_sha256": fit["checkpoint_sha256"],
                                 "fit_sha256": sha256_file(directory / "fit.json"),
                                 "validation_matches": fold["validation"]})
    return result


def _manifest(focused_dir: Path, data_dir: Path, target_dir: Path) -> tuple[dict, dict]:
    plan, summary, selection, confirmation, baseline, selected = _records(focused_dir)
    source, environment = _runtime_ancestry(plan)
    if (summary["status"] != "complete" or selection["locked_before_confirmation"] is not True
            or summary["confirmation_scores"] != {role: confirmation[role]["scores"]
                                                     for role in ("baseline", "selected")}):
        raise ValueError("focused confirmation is not complete and frozen")
    patterns, target = _target_patterns(target_dir)
    matches = plan["payload"]["fold"]["outer_train"]
    paths = [data_dir / f"{match}_half{period}.npz" for match in matches for period in (1, 2)]
    prepared_provenance = verify_prepared(data_dir, paths)
    prepared = {path.name: sha256_file(path) for path in paths}
    if any(plan["payload"]["prepared_sha256"].get(name) != digest
           for name, digest in prepared.items()):
        raise ValueError("prepared development data differ from focused protocol")
    checkpoints = _checkpoint_entries(focused_dir, plan, confirmation)
    for role, folds in checkpoints.items():
        validation = [match for fold in folds for match in fold["validation_matches"]]
        if len(folds) != 3 or sorted(validation) != sorted(matches):
            raise ValueError(f"{role} confirmation folds do not cover six matches exactly once")
    artifact_names = ("protocol", "summary", "selection", "confirmation",
                      "baseline-recipe", "selected-recipe")
    artifacts = {f"{name}.json": sha256_file(focused_dir / f"{name}.json")
                 for name in artifact_names}
    result = {"protocol": PROTOCOL, "selection_frozen": True,
              "scope": "seed13 six-match development stress diagnostic; no fitting or promotion",
              "focused_protocol_sha256": plan["sha256"],
              "confirmation_seed": plan["payload"]["resources"]["confirmation_seed"],
              "development_matches": matches, "source_sha256": source,
              "execution_environment": environment,
              "generation": {"scenario_seeds": list(SCENARIO_SEEDS),
                             "offset": "first 8 bytes SHA256(protocol:match:period:seed) modulo source span length",
                             "mapping": "target period pattern cyclically offset, cropped/tiled to DFL half",
                             "cycle_rule": "if both source boundary frames are missing, set first frame after each internal cycle seam observed; report count",
                             "union_existing_missing": True, "force_missing_fraction": False,
                             "fps": 25.0},
              "target": target, "target_root": str(target_dir),
              "prepared_root": str(data_dir), "prepared_manifest_sha256": sha256_file(data_dir / "manifest.json"),
              "prepared_sha256": prepared, "prepared_provenance": prepared_provenance,
              "focused_root": str(focused_dir), "focused_artifact_sha256": artifacts,
              "recipes": {"baseline": baseline, "selected": selected},
              "checkpoints": checkpoints}
    return result, patterns


def _prediction(output_dir: Path, manifest: dict, role: str, fold: dict, match: str,
                period: int, seed: int, pattern: np.ndarray, model) -> tuple[dict, dict]:
    scenario = f"targetgap_seed{seed}"
    path = (output_dir / "predictions" / fold["checkpoint_sha256"] / match
            / scenario / f"half{period}.npz")
    record_path = path.with_name(f"{path.stem}.replay.json")
    existing = _load_record(record_path)
    expected = {"manifest_sha256": _digest(manifest), "checkpoint_sha256": fold["checkpoint_sha256"],
                "match": match, "period": period, "scenario": scenario}
    if existing is not None:
        if any(existing.get(key) != value for key, value in expected.items()):
            raise ValueError("cached target-gap prediction provenance differs")
        arrays = load_prediction_bundle(existing["bundle"])
        replayed, replay = _replay_mask(pattern, len(arrays["gt"]), match, period, seed)
        if any(existing["mask"].get(key) != value for key, value in replay.items()):
            raise ValueError("cached target-gap replay mask differs")
        with np.load(Path(manifest["prepared_root"]) / f"{match}_half{period}.npz",
                     allow_pickle=False) as stored:
            original_valid = stored["ball_valid"]
        if not np.array_equal(arrays["ball_valid"], original_valid & ~replayed):
            raise ValueError("cached target-gap ball validity differs from replay")
        return arrays, existing["mask"]
    if path.exists() or record_path.exists():
        raise ValueError("incomplete target-gap prediction cache requires inspection")
    frames, targets = load_half(Path(manifest["prepared_root"]) / f"{match}_half{period}.npz")
    if frames.match_id != match or frames.period != period or frames.fps != 25.0:
        raise ValueError("prepared half identity or FPS differs")
    original_gt = _target_codes(frames, targets)
    mask, replay = _replay_mask(pattern, len(frames.frame_id), match, period, seed)
    altered = frames.subset(0, len(frames.frame_id))
    additional = mask & altered.ball_valid
    altered.ball_valid[mask] = False
    altered.ball_xy[~altered.ball_valid] = 0
    if not np.array_equal(altered.graded, frames.graded) or not np.array_equal(original_gt, _target_codes(frames, targets)):
        raise AssertionError("target-gap replay changed labels or graded status")
    features = build_features(altered)
    neural = predict_features(features, model, device="cpu",
                              center_size=manifest["recipes"][role]["training_args"]["center_size"],
                              halo=model.radius)
    reference = reference_scores(altered)
    bundle = save_prediction_bundle(
        path, neural=neural, reference=reference, gt=original_gt,
        ball_valid=altered.ball_valid,
        nearest_ball_seconds=10.0 * np.minimum(
            features.global_tokens[:, 6], features.global_tokens[:, 7]).astype(np.float64),
        frame_id=altered.frame_id, match=match, period=period, scenario=scenario)
    evidence = {**replay, "additional_missing_frames": int(additional.sum()),
                "existing_missing_frames": int((~frames.ball_valid).sum()),
                "resulting_missing_mask_sha256": hashlib.sha256(
                    (~altered.ball_valid).tobytes()).hexdigest(),
                "resulting_gap_statistics": gap_statistics(
                    (~altered.ball_valid) & altered.graded, frames.fps)}
    _save_record(record_path, {**expected, "bundle": bundle, "mask": evidence})
    return load_prediction_bundle(bundle), evidence


def _score_match(output_dir: Path, manifest: dict, role: str, fold: dict,
                 match: str, seed: int, patterns: dict[int, np.ndarray], model) -> dict:
    gt_parts, pred_parts, diagnostics, frame_rows, masks = [], [], [], [], {}
    recipe = manifest["recipes"][role]
    for period in (1, 2):
        arrays, mask = _prediction(output_dir, manifest, role, fold, match, period,
                                   seed, patterns[period], model)
        pred = decode_oof_half(arrays, recipe)
        gt_parts.append(arrays["gt"])
        pred_parts.append(pred)
        diagnostics.append(spell_error_diagnostics(arrays["gt"], pred, arrays["ball_valid"], fps=25.0))
        frame_rows.append(_frame_errors(arrays["gt"], pred, arrays["ball_valid"], 25.0))
        masks[str(period)] = mask
    scores = _compact(official.evaluate(_join_with_x(gt_parts), _join_with_x(pred_parts),
                                        bootstrap_n=0))
    frame = {name: {"n_frames": sum(row[name]["n_frames"] for row in frame_rows),
                    "n_errors": sum(row[name]["n_errors"] for row in frame_rows)}
             for name in FRAME_BINS}
    for bucket in frame.values():
        bucket["error_rate"] = (bucket["n_errors"] / bucket["n_frames"]
                                if bucket["n_frames"] else None)
    return {"official": scores, "spell_diagnostics": merge_spell_diagnostics(diagnostics),
            "frame_errors_by_gap_duration": frame, "masks": masks}


def _set_frozen_threads(focused_dir: Path) -> None:
    plan = _records(focused_dir)[0]
    threads = plan["payload"]["environment"]["torch_num_threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 4:
        raise ValueError("focused runtime must use at most four Torch threads for replay")
    torch.set_num_threads(threads)


def run(focused_dir, data_dir, target_dir, output_dir, *, selection_frozen=False) -> dict:
    """Publish immutable provenance, then score only confirmed validation fits."""
    if selection_frozen is not True:
        raise ValueError("explicit selection_frozen=True required")
    focused_dir, data_dir, target_dir, output_dir = (
        Path(path).resolve() for path in (focused_dir, data_dir, target_dir, output_dir))
    _set_frozen_threads(focused_dir)
    with frozen_transfer._lock(output_dir):
        manifest, patterns = _manifest(focused_dir, data_dir, target_dir)
        manifest_path = output_dir / "frozen-manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("target-gap replay provenance changed")
        else:
            write_json(manifest_path, manifest)
        results = {}
        for role in ("baseline", "selected"):
            results[role] = {}
            for fold in manifest["checkpoints"][role]:
                if _manifest(focused_dir, data_dir, target_dir)[0] != manifest:
                    raise ValueError("target-gap replay provenance changed before inference")
                model, _ = load_model_checkpoint(fold["checkpoint"])
                for match in fold["validation_matches"]:
                    for seed in SCENARIO_SEEDS:
                        path = output_dir / "results" / role / f"fold{fold['fold_index']}" / match / f"seed{seed}.json"
                        cached = _load_record(path)
                        expected = {"manifest_sha256": _digest(manifest), "role": role,
                                    "fold_index": fold["fold_index"], "match": match,
                                    "seed": seed, "checkpoint_sha256": fold["checkpoint_sha256"]}
                        if cached is None:
                            scored = _score_match(output_dir, manifest, role, fold,
                                                  match, seed, patterns, model)
                            cached = {**expected, **scored}
                            _save_record(path, cached)
                        elif any(cached.get(key) != value for key, value in expected.items()):
                            raise ValueError("cached target-gap score provenance differs")
                        else:
                            for period in (1, 2):
                                _, mask = _prediction(output_dir, manifest, role, fold, match,
                                                      period, seed, patterns[period], model)
                                if cached["masks"].get(str(period)) != mask:
                                    raise ValueError("cached target-gap score mask differs")
                        results[role].setdefault(match, {})[str(seed)] = cached
        if _manifest(focused_dir, data_dir, target_dir)[0] != manifest:
            raise ValueError("target-gap replay provenance changed after inference")
        means = {}
        for role, by_match in results.items():
            rows = [record["official"] for seeds in by_match.values() for record in seeds.values()]
            means[role] = {"spell_f1_mean": float(np.mean([row["spell_f1_mean"] for row in rows])),
                           "macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
                           "transition_f1_at_12": float(np.mean([
                               row["transitions"]["12"]["f1"] for row in rows]))}
        summary = {"protocol": PROTOCOL, "selection_frozen": True,
                   "post_selection_development_only": True, "promotion": False,
                   "manifest_sha256": sha256_file(manifest_path),
                   "mean_official_by_role": means, "by_role": results}
        existing = _load_record(output_dir / "summary.json")
        if existing is not None and existing != summary:
            raise ValueError("completed target-gap summary differs")
        if existing is None:
            _save_record(output_dir / "summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focused", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/dfl"))
    parser.add_argument("--target-dir", type=Path, default=Path("data/target/2027"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-frozen", action="store_true", required=True)
    args = parser.parse_args(argv)
    result = run(args.focused, args.data_dir, args.target_dir, args.output,
                 selection_frozen=args.selection_frozen)
    print(json.dumps({"summary": str(args.output / "summary.json"),
                      "roles": list(result["by_role"])}, indent=2))


if __name__ == "__main__":
    main()
