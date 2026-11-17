"""Memory-bounded OOF probabilities and exact official possession scoring."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from experiments import campaign_evaluation as inputs
from experiments.campaign_decode import decode_arrays
from experiments.campaign_protocol import WEIGHTS, publish, read_record
from experiments.coarse_context import predict_coarse_native
from experiments.evaluate import _join_with_x
from experiments.gpu_iou_evaluation import _blend
from experiments.gpu_tuning_training import check_file
from possession.predict import load_model_checkpoint, predict_features
from possession.schema import FeatureBatch
from possession.storage import load_half, sha256_file

FEATURES = ("player_tokens", "player_mask", "global_tokens", "time_valid")


def key(row):
    return (row["match"], row["period"], row["scenario"])


def checked_probabilities(path, expected_model, expected_input):
    metadata = read_record(path.with_suffix(".json"))
    if (not metadata or metadata["identity"]["checkpoint_sha256"] != expected_model
            or metadata["identity"]["input_sha256"] != expected_input):
        raise ValueError("incumbent probability provenance differs")
    check_file(path, metadata["sha256"])
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


class Bank:
    """Reuse the completed combined ensemble and its existing input caches."""
    def __init__(self, repo, output, scope, batch, tick=lambda **kw: None):
        self.repo, self.output = Path(repo), Path(output)
        self.scope, self.batch = scope, batch
        arm = "guard" if scope == "guard" else "confirmation"
        previous = self.repo / "artifacts/gpu-iou-followup-v1"
        self.plan = read_record(previous / "protocol.json")["arms"][arm]
        self.matches = sorted({m for f in self.plan["folds"] for m in f["validation"]})
        self.folds = self.plan["folds"]
        source_dir = previous / arm
        fit_rows = read_record(source_dir / "training.json")["fits"]
        fits = {tuple(r["identity"][k] for k in ("fold", "base_seed", "member", "role")): r
                for r in fit_rows}
        input_dir = Path(self.plan["input_root"]) / batch
        manifest_path = input_dir / "input-manifest.json"
        check_file(manifest_path, json.loads(manifest_path.with_suffix(".sha256.json").read_text())["sha256"])
        manifest = json.loads(manifest_path.read_text())
        if (manifest["status"] != "complete" or manifest["signature"]["matches"] != self.matches
                or manifest["signature"]["scenarios"] != self.plan["batches"][batch]):
            raise ValueError("input cache scope or scenarios differ")
        self.sources = [{**r, "input_path": str(inputs._input_path(
            input_dir, r["match"], r["period"], r["scenario"]))} for r in manifest["records"]]
        expected = {(m, p, s) for m in self.matches for p in (1, 2) for s in WEIGHTS}
        if len(self.sources) != len(expected) or {key(r) for r in self.sources} != expected:
            raise ValueError("incomplete assessment grid")
        self.records = {}
        coarse_cache = {}
        for source in self.sources:
            tick(input_match=source["match"], input_scenario=source["scenario"])
            match, period, scenario = key(source)
            fold_index = next(i for i, f in enumerate(self.folds) if match in f["validation"])
            arrays = inputs.load_arrays(source)
            for name in FEATURES:
                arrays.pop(name)
            fine = []
            for member in self.plan["selection"]["members"]:
                predictions = []
                for seed in self.plan["seeds"]:
                    fit = fits[fold_index, seed, member["id"], "spell_balanced"]
                    if match in fit["identity"]["train_matches"]:
                        raise ValueError("incumbent assessment leakage")
                    path = (source_dir / "inference" / batch / "spell_balanced" / f"seed_{seed}"
                            / member["id"] / match / f"half{period}" / f"{scenario}.npz")
                    predictions.append(checked_probabilities(
                        path, fit["files"]["model.pt"], source["input_sha256"]))
                fine.append(_blend(*predictions, .5))
            arrays.update(_blend(*fine, self.plan["selection"]["blend_weight"]))
            if (match, period) not in coarse_cache:
                coarse_fits = [fits[fold_index, seed, "coarse_player_only", "coarse"]
                               for seed in self.plan["seeds"]]
                coarse_cache[match, period] = self._coarse(match, period, coarse_fits)
            arrays.update(coarse_cache[match, period])
            self.records[key(source)] = {**source, "arrays": arrays}
        gc.collect()

    def _coarse(self, match, period, fits):
        path = self.output / "coarse-cache" / self.scope / match / f"half{period}.npz"
        identity = {"models": [r["files"]["model.pt"] for r in fits],
                    "source": sha256_file(Path(self.plan["data_dir"]) / f"{match}_half{period}.npz")}
        metadata = read_record(path.with_suffix(".json"))
        if metadata:
            if metadata["identity"] != identity:
                raise ValueError("coarse cache identity differs")
            check_file(path, metadata["sha256"])
            with np.load(path, allow_pickle=False) as data:
                return {name: data[name] for name in data.files}
        frames, _ = load_half(Path(self.plan["data_dir"]) / f"{match}_half{period}.npz")
        predictions = []
        for fit in fits:
            path_model = Path(fit["directory"]) / "model.pt"
            check_file(path_model, fit["files"]["model.pt"])
            model, payload = load_model_checkpoint(path_model, device="cuda:0")
            if match in payload["training_config"]["train_matches"]:
                raise ValueError("coarse model assessment leakage")
            result = predict_coarse_native(frames, model)
            predictions.append({"neural_live": result["live_probability"],
                                "neural_home": result["home_probability"],
                                "neural_boundary": np.zeros((len(frames.frame_id), 3, 3))})
            del model
        mixed = _blend(*predictions, .5)
        arrays = {"coarse_live": mixed["neural_live"], "coarse_home": mixed["neural_home"]}
        inputs._atomic_npz(path, arrays, compressed=True)
        publish(path.with_suffix(".json"), {"identity": identity, "sha256": sha256_file(path)})
        return arrays

    def predict(self, fits, tick=lambda **kw: None):
        output = {}
        for fitted in fits:
            check_file(fitted["model_path"], fitted["model_sha256"])
            model, payload = load_model_checkpoint(fitted["model_path"], device="cuda:0")
            valid = fitted["identity"]["validation_matches"]
            train = payload["training_config"]["train_matches"]
            if set(train) != set(fitted["identity"]["train_matches"]) or set(train) & set(valid):
                raise ValueError("candidate checkpoint split differs")
            for source in self.sources:
                if source["match"] not in valid:
                    continue
                tick(inference_match=source["match"], inference_scenario=source["scenario"])
                arrays = inputs.load_arrays(source)
                features = FeatureBatch(*(arrays[name] for name in FEATURES))
                predicted = predict_features(features, model, device="cuda:0", center_size=2048)
                values = {"neural_live": predicted["live_probability"],
                          "neural_home": predicted["home_probability"],
                          "neural_boundary": predicted["boundary"]}
                if key(source) in output or any(not np.isfinite(v).all() for v in values.values()):
                    raise ValueError("duplicate or nonfinite inference")
                output[key(source)] = values
            del model
        if set(output) != set(self.records):
            raise ValueError("incomplete candidate predictions")
        return output


def coarse_blend(arrays, maximum, seconds):
    if not 0 <= maximum <= 1 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid coarse gate")
    alpha = maximum * np.clip(np.asarray(arrays["nearest_ball_seconds"], dtype=np.float64) / seconds, 0., 1.)
    alpha[arrays["ball_valid"]] = 0.
    output = dict(arrays)
    for name, coarse in (("neural_live", "coarse_live"), ("neural_home", "coarse_home")):
        fine = np.asarray(arrays[name], dtype=np.float64)
        result = fine.copy()
        active = alpha > 0
        p, q = (np.clip(np.asarray(arrays[k])[active], 1e-7, 1 - 1e-7) for k in (name, coarse))
        logits = (1 - alpha[active]) * (np.log(p) - np.log1p(-p)) + alpha[active] * (np.log(q) - np.log1p(-q))
        result[active] = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        output[name] = result
    return output


def spell_result(gt, pred):
    runs = official.valid_runs(gt)
    return official.spell_scores(gt, official.fill_abstentions(pred, runs), runs,
                                 ious=[i / 10 for i in range(1, 10)])


def score_labels(labels, matches):
    by_match, curves = {}, {str(i / 10): {} for i in range(1, 10)}
    diagnostics = {"spurious": 0, "matched": 0, "true": 0, "short_matched": 0, "short_true": 0}
    for match in matches:
        by_match[match] = {}
        for scenario in WEIGHTS:
            gt = _join_with_x([labels[f"gt_{match}_{p}"] for p in (1, 2)])
            pred = _join_with_x([labels[f"pred_{match}_{p}_{scenario}"] for p in (1, 2)])
            metric = spell_result(gt, pred)
            by_match[match][scenario] = float(np.mean([metric["per_iou"][t]["f1"] for t in official.SPELL_IOUS]))
            for threshold, values in metric["per_iou"].items():
                curves[str(threshold)].setdefault(match, {})[scenario] = values["f1"]
            if scenario == "clean":
                values = metric["per_iou"][.5]
                diagnostics["spurious"] += values["n_spurious"]
                diagnostics["matched"] += values["n_matched"]
                diagnostics["true"] += metric["n_gt"]
                runs = official.valid_runs(gt)
                truth = official.possession_spells(gt, runs)
                prediction = official.possession_spells(official.fill_abstentions(pred, runs), runs)
                flags = official.matched_gt_flags(official.spell_overlaps(truth, prediction), len(truth), .5)
                short = np.array([b - a < 50 for a, b, _ in truth], dtype=bool)
                diagnostics["short_true"] += int(short.sum())
                diagnostics["short_matched"] += int(flags[short].sum())
    means = {s: float(np.mean([row[s] for row in by_match.values()])) for s in WEIGHTS}
    return {"objective": sum(WEIGHTS[s] * means[s] for s in WEIGHTS), "means": means,
            "by_match": by_match, "diagnostics": diagnostics,
            "iou_curve": {t: float(np.mean([sum(WEIGHTS[s] * row[s] for s in WEIGHTS)
                                             for row in rows.values()])) for t, rows in curves.items()}}


def evaluate(bank, predictions, pipeline, save=None):
    if predictions is not None and set(predictions) != set(bank.records):
        raise ValueError("candidate prediction grid differs")
    if predictions is None and pipeline["candidate_weight"] != 0:
        raise ValueError("nonzero candidate weight requires predictions")
    labels = {}
    for k, row in bank.records.items():
        arrays = row["arrays"]
        mixed = (arrays if pipeline["candidate_weight"] == 0 else
                 {**arrays, **_blend(predictions[k], arrays, pipeline["candidate_weight"])})
        blended = coarse_blend(mixed, pipeline["coarse_weight"], pipeline["coarse_seconds"])
        match, period, scenario = k
        truth_key = f"gt_{match}_{period}"
        if truth_key in labels and not np.array_equal(labels[truth_key], arrays["gt"]):
            raise ValueError("ground truth differs across scenarios")
        labels[truth_key] = arrays["gt"].astype(np.uint8)
        labels[f"pred_{match}_{period}_{scenario}"] = decode_arrays(blended, pipeline["decoder"]).astype(np.uint8)
    result = score_labels(labels, bank.matches)
    if save is not None:
        directory = Path(save)
        inputs._atomic_npz(directory / "labels.npz", labels, compressed=True)
        publish(directory / "result.json", {
            "scores": result, "pipeline": pipeline, "scope": bank.scope, "batch": bank.batch,
            "matches": bank.matches, "labels_sha256": sha256_file(directory / "labels.npz"),
            "array_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in labels.items()},
            "inputs": [{k: r[k] for k in ("match", "period", "scenario", "input_sha256")} for r in bank.sources]})
    return result


def average_scores(rows):
    matches = sorted(rows[0]["by_match"])
    if any(sorted(row["by_match"]) != matches for row in rows):
        raise ValueError("score match sets differ")
    by_match = {m: {s: float(np.mean([r["by_match"][m][s] for r in rows])) for s in WEIGHTS} for m in matches}
    means = {s: float(np.mean([v[s] for v in by_match.values()])) for s in WEIGHTS}
    return {"objective": sum(WEIGHTS[s] * means[s] for s in WEIGHTS), "means": means, "by_match": by_match}


def comparison(candidate, baseline, guard=False):
    matches = sorted(baseline["by_match"])
    if sorted(candidate["by_match"]) != matches:
        raise ValueError("comparison match sets differ")
    delta = candidate["objective"] - baseline["objective"]
    scenarios = {s: candidate["means"][s] - baseline["means"][s] for s in WEIGHTS}
    by_match = {m: sum(WEIGHTS[s] * (candidate["by_match"][m][s] - baseline["by_match"][m][s])
                       for s in WEIGHTS) for m in matches}
    checks = {"objective": delta >= -.01 if guard else delta > .002,
              "scenarios_preserved": min(scenarios.values()) >= -.01}
    if not guard:
        checks.update(breadth=sum(v > 0 for v in by_match.values()) >= 4,
                      missing_improved=sum(WEIGHTS[s] * scenarios[s] for s in WEIGHTS if s != "clean") > 0)
    result = {"delta": delta, "scenario_deltas": scenarios, "by_match": by_match,
              "checks": checks, "passed": all(checks.values()), "independent_test": False}
    if len(matches) > 1:
        rng = np.random.default_rng(20261117)
        values = np.array(list(by_match.values()))
        draws = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
        result["descriptive_95_interval"] = np.quantile(draws, [.025, .975]).tolist()
    return result
