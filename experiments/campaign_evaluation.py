"""Cached public-match OOF inputs, neural predictions, and official scoring."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from experiments.augmentation_evaluation import _compact
from experiments.campaign_decode import decode_arrays, reference_basis
from experiments.evaluate import _join_with_x, _target_codes
from possession.features import build_features
from possession.outages import generate_outage_mask
from possession.predict import predict_features
from possession.prepared import verify_prepared
from possession.schema import FeatureBatch
from possession.storage import load_half, sha256_file, write_json

INPUT_SCHEMA = "campaign_input_v1"
PREDICTION_SCHEMA = "campaign_prediction_v1"


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray], *, compressed: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer = np.savez_compressed if compressed else np.savez
            writer(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _safe_id(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _validate_scenarios(scenarios):
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("scenarios must be a nonempty mapping")
    result = {}
    for name, raw in scenarios.items():
        _safe_id(name, "scenario")
        if not isinstance(raw, dict) or raw.get("kind") not in (
                "clean", "synthetic", "target_replay"):
            raise ValueError(f"invalid scenario definition: {name}")
        kind = raw["kind"]
        expected = {"kind"} if kind == "clean" else (
            {"kind", "seed", "missing_fraction", "min_seconds", "max_seconds"}
            if kind == "synthetic" else {"kind", "seed"})
        if set(raw) != expected:
            raise ValueError(f"scenario fields differ for {name}")
        if kind == "synthetic":
            numbers = (raw["missing_fraction"], raw["min_seconds"], raw["max_seconds"])
            if (not isinstance(raw["seed"], int) or isinstance(raw["seed"], bool)
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not np.isfinite(value) for value in numbers)
                    or not 0 <= raw["missing_fraction"] <= 1
                    or not 0 < raw["min_seconds"] <= raw["max_seconds"]):
                raise ValueError(f"invalid synthetic scenario values: {name}")
        elif kind == "target_replay" and (
                isinstance(raw["seed"], bool) or not isinstance(raw["seed"], int)
                or raw["seed"] not in (5101, 5102)):
            raise ValueError(f"invalid target replay seed: {name}")
        result[name] = _canonical(raw)
    return dict(sorted(result.items()))


def _gap_seconds(ball_valid, fps):
    """Uncapped bidirectional distance to observed ball; finite for no-observation halves."""
    valid = np.asarray(ball_valid, dtype=bool)
    n = len(valid)
    seen = np.flatnonzero(valid)
    if not len(seen):
        return np.full(n, n / float(fps), dtype=np.float32)
    index = np.arange(n)
    previous = np.maximum.accumulate(np.where(valid, index, -1))
    following = np.minimum.accumulate(np.where(valid, index, n)[::-1])[::-1]
    before = np.where(previous >= 0, index - previous, n)
    after = np.where(following < n, following - index, n)
    return (np.minimum(before, after) / float(fps)).astype(np.float32)


def _input_arrays(frames, targets, scenario, match, period, target_patterns):
    gt = _target_codes(frames, targets)
    altered = frames.subset(0, len(frames.frame_id))
    kind = scenario["kind"]
    if kind == "synthetic":
        seed = int.from_bytes(hashlib.sha256(
            f"campaign-gap:{scenario['seed']}:{match}:{period}".encode()).digest()[:8], "big")
        additional = generate_outage_mask(
            altered.ball_valid, altered.fps, np.random.default_rng(seed),
            missing_fraction=scenario["missing_fraction"],
            min_seconds=scenario["min_seconds"], max_seconds=scenario["max_seconds"])
        altered.ball_valid &= ~additional
    elif kind == "target_replay":
        from experiments.target_gap_stress import _replay_mask

        mask, _ = _replay_mask(target_patterns[period], len(altered.frame_id), match,
                               period, scenario["seed"])
        altered.ball_valid &= ~mask
    altered.ball_xy[~altered.ball_valid] = 0
    features = build_features(altered)
    basis = reference_basis(altered, features)
    return {
        "frame_id": altered.frame_id.copy(), "gt": gt,
        "ball_valid": altered.ball_valid.copy(),
        "nearest_ball_seconds": _gap_seconds(altered.ball_valid, altered.fps),
        "player_tokens": features.player_tokens,
        "player_mask": features.player_mask,
        "global_tokens": features.global_tokens,
        "time_valid": features.time_valid,
        **basis,
    }


def _input_path(output_dir, match, period, scenario):
    return output_dir / "inputs" / match / f"half{period}" / f"{scenario}.npz"


def _publish_input_manifest(path, manifest):
    write_json(path, manifest)
    write_json(path.with_suffix(".sha256.json"), {"sha256": sha256_file(path)})


def prepare_inputs(data_dir, target_dir, output_dir, matches, scenarios, cache=None,
                   progress_callback=None) -> list[dict]:
    """Build or verify public DFL feature/reference caches; target access is input-only."""
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    target_dir = Path(target_dir).resolve() if target_dir is not None else None
    match_values = [_safe_id(value, "match") for value in matches]
    if len(match_values) != len(set(match_values)):
        raise ValueError("duplicate match ID")
    matches = sorted(match_values)
    if not matches:
        raise ValueError("matches must be nonempty")
    scenarios = _validate_scenarios(scenarios)
    periods = {}
    for match in matches:
        paths = sorted(data_dir.glob(f"{match}_half*.npz"))
        if len(paths) != 2:
            raise ValueError(f"expected exactly two prepared halves for {match}")
        by_period = {}
        for path in paths:
            found = re.fullmatch(re.escape(match) + r"_half([12])\.npz", path.name)
            if not found:
                raise ValueError(f"unexpected prepared half path: {path}")
            by_period[int(found.group(1))] = path
        if set(by_period) != {1, 2}:
            raise ValueError(f"prepared half period set differs for {match}")
        periods[match] = by_period
    sources = [periods[m][p] for m in matches for p in (1, 2)]
    provenance = verify_prepared(data_dir, sources)
    source_hashes = {path.name: sha256_file(path) for path in sources}
    target_patterns, target_evidence = None, None
    if any(config["kind"] == "target_replay" for config in scenarios.values()):
        if target_dir is None:
            raise ValueError("target_dir is required for target_replay scenarios")
        from experiments.target_gap_stress import _target_patterns

        target_patterns, target_evidence = _target_patterns(target_dir)
    signature = {"schema": INPUT_SCHEMA, "matches": matches, "scenarios": scenarios,
                 "prepared_sha256": source_hashes, "prepared_provenance": provenance,
                 "target_input_evidence": target_evidence}
    manifest_path = output_dir / "input-manifest.json"
    expected = _digest(signature)
    if manifest_path.exists():
        sidecar = manifest_path.with_suffix(".sha256.json")
        if not sidecar.is_file() or json.loads(sidecar.read_text()).get("sha256") != sha256_file(manifest_path):
            raise ValueError("input manifest hash differs")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("signature_sha256") != expected or manifest.get("signature") != signature:
            raise ValueError("input cache source or scenario identity differs")
    else:
        stray = list((output_dir / "inputs").rglob("*.npz")) if (output_dir / "inputs").exists() else []
        if stray:
            raise ValueError("input NPZ files exist without manifest; inspect incomplete cache")
        manifest = {"signature": signature, "signature_sha256": expected, "records": []}
        _publish_input_manifest(manifest_path, manifest)
    indexed = {(r["match"], r["period"], r["scenario"]): r for r in manifest["records"]}
    if len(indexed) != len(manifest["records"]):
        raise ValueError("duplicate record in input manifest")
    records = []
    for match in matches:
        for period in (1, 2):
            if cache is not None:
                frames, targets, _, _ = cache.get(periods[match][period])
            else:
                frames, targets = load_half(periods[match][period])
            if frames.match_id != match or frames.period != period:
                raise ValueError("prepared half identity differs from filename")
            for name, scenario in scenarios.items():
                key = (match, period, name)
                existing = indexed.get(key)
                path = _input_path(output_dir, match, period, name)
                if existing is not None:
                    if (existing.get("input_identity") != expected
                            or existing.get("input_sha256") != sha256_file(path)):
                        raise ValueError("cached input array hash differs")
                    records.append({**existing, "input_path": str(path)})
                    continue
                if progress_callback is not None:
                    progress_callback()
                arrays = _input_arrays(frames, targets, scenario, match, period, target_patterns)
                _atomic_npz(path, arrays, compressed=False)
                record = {"match": match, "period": period, "scenario": name,
                          "input_path": str(path), "input_sha256": sha256_file(path),
                          "input_identity": expected}
                manifest["records"].append({key: value for key, value in record.items()
                                            if key != "input_path"})
                indexed[key] = manifest["records"][-1]
                _publish_input_manifest(manifest_path, manifest)
                records.append(record)
    expected_keys = {(m, p, scenario) for m in matches for p in (1, 2) for scenario in scenarios}
    if set(indexed) != expected_keys:
        raise ValueError("input manifest grid differs")
    manifest["status"] = "complete"
    _publish_input_manifest(manifest_path, manifest)
    return records


def _prediction_sidecar(path):
    return path.with_suffix(".json")


def predict_records(model, input_records, output_dir, identity: dict,
                    center_size: int = 2048, progress_callback=None) -> list[dict]:
    """Predict all cached halves and atomically reuse only matching model/input bundles."""
    output_dir = Path(output_dir).resolve()
    identity = _canonical(identity)
    if not identity or not isinstance(center_size, int) or center_size < 1:
        raise ValueError("prediction identity must be nonempty and center_size positive")
    records, seen = [], set()
    for source in input_records:
        if progress_callback is not None:
            progress_callback()
        key = (source["match"], source["period"], source["scenario"])
        if key in seen:
            raise ValueError("duplicate input record")
        seen.add(key)
        input_path = Path(source["input_path"])
        if sha256_file(input_path) != source["input_sha256"]:
            raise ValueError("input array hash differs before inference")
        model_key = _digest(identity)[:16]
        path = output_dir / "predictions" / model_key / source["match"] / f"half{source['period']}" / f"{source['scenario']}.npz"
        sidecar = _prediction_sidecar(path)
        expected = {"schema": PREDICTION_SCHEMA, "input_sha256": source["input_sha256"],
                    "model_identity": identity, "center_size": center_size,
                    "match": source["match"], "period": source["period"],
                    "scenario": source["scenario"]}
        if path.is_file() and sidecar.is_file():
            metadata = json.loads(sidecar.read_text())
            if metadata.get("identity") != expected or metadata.get("prediction_sha256") != sha256_file(path):
                raise ValueError("prediction bundle identity or hash differs")
        else:
            arrays = load_arrays(source)
            features = FeatureBatch(arrays["player_tokens"], arrays["player_mask"],
                                    arrays["global_tokens"], arrays["time_valid"])
            prediction = predict_features(features, model, device="cpu", center_size=center_size,
                                          halo=model.radius)
            scores = {"neural_live": np.asarray(prediction["live_probability"], dtype=np.float64),
                      "neural_home": np.asarray(prediction["home_probability"], dtype=np.float64),
                      "neural_boundary": np.asarray(prediction["boundary"], dtype=np.float64)}
            _atomic_npz(path, scores, compressed=True)
            write_json(sidecar, {"identity": expected, "prediction_sha256": sha256_file(path),
                                 "n_frames": len(scores["neural_live"])})
        record = {**source, "prediction_path": str(path),
                  "prediction_sha256": sha256_file(path), "model_identity": identity,
                  "center_size": center_size}
        records.append(record)
    return records


def load_arrays(record: dict) -> dict[str, np.ndarray]:
    """Load and verify one prepared input plus its optional neural prediction bundle."""
    path = Path(record["input_path"])
    if sha256_file(path) != record["input_sha256"]:
        raise ValueError("input array hash differs")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    n = len(arrays["frame_id"])
    frame_id = arrays["frame_id"]
    if (frame_id.dtype.kind not in "iu" or n == 0
            or (n > 1 and not np.all(np.diff(frame_id) == 1))):
        raise ValueError("cached frame IDs must be a nonempty contiguous integer timeline")
    if (arrays["gt"].dtype.kind not in "iu" or arrays["gt"].shape != (n,)
            or arrays["ball_valid"].shape != (n,) or arrays["ball_valid"].dtype.kind != "b"
            or arrays["nearest_ball_seconds"].shape != (n,)
            or arrays["player_tokens"].shape[:1] != (n,)
            or arrays["player_mask"].shape != arrays["player_tokens"].shape[:-1]
            or arrays["global_tokens"].shape != (n, 9)
            or arrays["time_valid"].shape != (n,)):
        raise ValueError("cached model inputs, labels or timeline shapes differ")
    if not np.isclose(float(arrays["ref_fps"]), official.FPS):
        raise ValueError("cached reference FPS differs from the official scoring rate")
    if np.any(arrays["gt"] < 0) or np.any(arrays["gt"] > official.X):
        raise ValueError("cached public labels contain invalid codes")
    if "prediction_path" in record:
        prediction_path = Path(record["prediction_path"])
        if sha256_file(prediction_path) != record["prediction_sha256"]:
            raise ValueError("prediction array hash differs")
        sidecar = json.loads(_prediction_sidecar(prediction_path).read_text())
        expected = {"schema": PREDICTION_SCHEMA, "input_sha256": record["input_sha256"],
                    "model_identity": record["model_identity"],
                    "center_size": record["center_size"], "match": record["match"],
                    "period": record["period"], "scenario": record["scenario"]}
        if sidecar.get("identity") != expected or sidecar.get("prediction_sha256") != record["prediction_sha256"]:
            raise ValueError("prediction sidecar provenance differs")
        with np.load(prediction_path, allow_pickle=False) as prediction:
            for name in prediction.files:
                arrays[name] = prediction[name]
        if (arrays["neural_live"].shape != (n,) or arrays["neural_home"].shape != (n,)
                or arrays["neural_boundary"].shape != (n, 3, 3)):
            raise ValueError("cached neural prediction shapes differ")
    return arrays


def load_records(records: list[dict]) -> list[dict]:
    """Explicitly load records in memory for repeated decoder trials; no global cache."""
    loaded = []
    for record in records:
        arrays = load_arrays(record)
        for feature in ("player_tokens", "player_mask", "global_tokens", "time_valid"):
            arrays.pop(feature, None)
        loaded.append({**record, "arrays": arrays})
    return loaded


def score_records(records: list[dict], decoder: dict, scenario_weights: dict[str, float]) -> dict:
    """Official whole-match scoring, equal-match means, weighted across scenarios."""
    if not records or not isinstance(scenario_weights, dict) or not scenario_weights:
        raise ValueError("records and scenario_weights must be nonempty")
    index = {}
    for record in records:
        key = (record["match"], record["period"], record["scenario"])
        if key in index:
            raise ValueError("duplicate match-period-scenario record")
        index[key] = record
    matches = sorted({key[0] for key in index})
    scenarios = sorted({key[2] for key in index})
    if set(scenario_weights) != set(scenarios):
        raise ValueError("scenario weight names differ from record grid")
    weights = {name: float(value) for name, value in scenario_weights.items()}
    if (any(not np.isfinite(value) or value < 0 for value in weights.values())
            or not np.isclose(sum(weights.values()), 1.0)):
        raise ValueError("scenario weights must be finite, nonnegative and sum to one")
    expected = {(match, period, scenario) for match in matches for period in (1, 2)
                for scenario in scenarios}
    if set(index) != expected:
        raise ValueError("records must cover each match x scenario x period exactly once")
    by_match, score_rows = {}, {scenario: [] for scenario in scenarios}
    labels_by_half = {}
    for match in matches:
        by_match[match] = {}
        for scenario in scenarios:
            gt_parts, pred_parts = [], []
            for period in (1, 2):
                record = index[(match, period, scenario)]
                arrays = record.get("arrays") or load_arrays(record)
                timeline_key = (match, period)
                timeline = (arrays["frame_id"], arrays["gt"])
                if timeline_key in labels_by_half and any(
                        not np.array_equal(a, b) for a, b in zip(labels_by_half[timeline_key], timeline)):
                    raise ValueError("paired scenario frame IDs or public labels differ")
                labels_by_half[timeline_key] = tuple(value.copy() for value in timeline)
                gt_parts.append(arrays["gt"])
                pred_parts.append(decode_arrays(arrays, decoder))
            result = _compact(official.evaluate(_join_with_x(gt_parts), _join_with_x(pred_parts),
                                                bootstrap_n=0))
            by_match[match][scenario] = result
            score_rows[scenario].append(result)
    means = {}
    for scenario, rows in score_rows.items():
        transition_scores = [row["transitions"]["12"] for row in rows]
        means[scenario] = {
            "spell_f1_mean": float(np.mean([row["spell_f1_mean"] for row in rows])),
            "macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
            "transition_f1_at_12": float(np.mean([entry["f1"] for entry in transition_scores])),
        }
    return {"objective": float(sum(weights[name] * means[name]["spell_f1_mean"]
                                  for name in scenarios)),
            "scenario_weights": weights, "means": means, "by_match": by_match}
