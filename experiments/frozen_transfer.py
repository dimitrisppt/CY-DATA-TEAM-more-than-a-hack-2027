"""One-shot, frozen six-match refits and two reserved SkillCorner proxy checks.

This command is intentionally unavailable until the development focused study has
published its selection, confirmation, and both recipes. Proxy labels are used
only after a provenance manifest and both seed refits have been published.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from evaluation import possession_metrics as official
from experiments import skillcorner_proxy
from experiments.augmentation_tuning import _load_record, _save_record
from experiments.oof_selection import decode_oof_half
from experiments.robustness import _execution_environment, _source_hashes
from possession.features import build_features
from possession.outages import OutageConfig
from possession.predict import load_model_checkpoint, predict_features
from possession.prepared import verify_prepared
from possession.reference import reference_scores
from possession.schema import TrainingTargets
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache, TrainingSession

SEEDS = (20, 21)
MAX_CACHE_BYTES = 2_000_000_000


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _records(focused_dir: Path) -> tuple[dict, dict, dict, dict, dict, dict]:
    names = ("protocol", "summary", "selection", "confirmation",
             "baseline-recipe", "selected-recipe")
    values = []
    for name in names:
        value = _load_record(focused_dir / f"{name}.json")
        if value is None:
            raise ValueError(f"frozen focused-study artifact missing: {name}.json")
        values.append(value)
    plan, summary, selection, confirmation, baseline, selected = values
    control = _load_record(focused_dir / "trials" / "trial_000" / "trial.json")
    if control is None:
        raise ValueError("focused control trial record is missing")
    if (plan.get("sha256") != _digest(plan.get("payload"))
            or summary.get("status") != "complete"
            or summary.get("selection") != selection
            or not selection.get("locked_before_confirmation")
            or selection.get("protocol_sha256") != plan["sha256"]
            or confirmation.get("protocol_sha256") != plan["sha256"]
            or summary.get("protocol_sha256") != plan["sha256"]
            or summary.get("selected_rung") != plan["payload"].get("selected_rung")):
        raise ValueError("focused selection or confirmation ancestry differs")
    if not {"baseline", "selected", "comparison", "no_op"} <= confirmation.keys():
        raise ValueError("focused confirmation is incomplete")
    if confirmation["no_op"] is not (selection.get("selected") == "baseline"):
        raise ValueError("focused confirmation no-op status differs from selection")
    if summary.get("confirmation_comparison") != confirmation["comparison"]:
        raise ValueError("focused confirmation comparison differs from summary")
    if (summary.get("confirmation_scores") != {
            name: confirmation[name]["scores"] for name in ("baseline", "selected")}
            or summary.get("model_trials", [{}])[0].get("canonical") != control.get("canonical")
            or summary.get("baseline_search_scores") != control.get("scores")):
        raise ValueError("focused control or confirmation scores differ from summary")
    if (selected.get("decoder") != selection["frozen"].get("decoder")
            or selected.get("calibration") != selection["frozen"].get("calibration")
            or selected.get("ensemble") != selection["frozen"].get("ensemble")):
        raise ValueError("selected recipe differs from locked selection")
    for recipe in (baseline, selected):
        if (recipe.get("steps") != summary["selected_rung"]
                or recipe.get("development_only") is not True
                or not isinstance(recipe.get("model_kwargs"), dict)
                or not isinstance(recipe.get("training_args"), dict)
                or recipe.get("decoder", {}).get("boundary_scale") != 0):
            raise ValueError("recipe is not the focused frozen rung/decoder")
    base_model = plan["payload"]["fixed"]["selected_model_recipe"]

    def expected_recipe(spec, frozen):
        return {"model_kwargs": {**base_model["model_kwargs"], "pooling": spec["pooling"]},
                "training_args": {**base_model["training_args"],
                                  "learning_rate": spec["learning_rate"], "augment": True},
                "augmentation_config": spec["augmentation_config"],
                "outage_config": {"mode": "legacy"},
                "decoder": frozen["decoder"], "calibration": frozen["calibration"],
                "ensemble": frozen["ensemble"], "steps": summary["selected_rung"],
                "development_only": True}

    control_spec = {"pooling": "mean", "learning_rate": control["canonical"]["learning_rate"],
                    "augmentation_config": control["canonical"]["augmentation_config"]}
    if (baseline != expected_recipe(control_spec, control["decoder"]["frozen"])
            or selected != expected_recipe(selection["model_spec"], selection["frozen"])):
        raise ValueError("frozen recipes differ from focused control or selected model")
    matches = plan["payload"]["fold"]["outer_train"]
    if (len(matches) != 6 or len(set(matches)) != 6
            or plan["payload"]["fold"]["outer_test"] != ["DFL-MAT-J03WMX"]
            or set(matches) & set(skillcorner_proxy.MATCHES)):
        raise ValueError("six development matches or reserved split differ")
    return tuple(values)


def _fit_key(recipe: dict, seed: int, matches: list[str]) -> str:
    """Decoder differences do not cause another fit."""
    return _digest({"seed": seed, "matches": matches, "steps": recipe["steps"],
                    "model_kwargs": recipe["model_kwargs"],
                    "training_args": recipe["training_args"],
                    "augmentation_config": recipe["augmentation_config"],
                    "outage_config": recipe["outage_config"]})


def _hashed_files(paths: list[Path], root: Path) -> dict[str, str]:
    if any(not path.is_file() for path in paths):
        raise ValueError("provenance source file is missing")
    return {str(path.relative_to(root)): sha256_file(path) for path in paths}


def _runtime_ancestry(plan: dict) -> tuple[dict[str, str], dict]:
    """Compare every focused-study source and runtime setting before touching a fit.

    The source fingerprint covers all possession, experiments, and evaluation
    Python modules, plus project lockfiles. Additional experiment modules may
    be added after the focused study, but its recorded code must remain byte
    identical, including transitive model dependencies such as interaction.py.
    """
    expected_source = plan["payload"]["source_sha256"]
    if not isinstance(expected_source, dict) or not expected_source:
        raise ValueError("focused source fingerprint is missing")
    current_source = _source_hashes()
    changed = sorted(name for name, digest in expected_source.items()
                     if current_source.get(name) != digest)
    added = sorted(set(current_source) - set(expected_source))
    forbidden = [name for name in added
                 if not (name.startswith("experiments/") and name.endswith(".py"))]
    if changed or forbidden or "experiments/frozen_transfer.py" not in current_source:
        raise ValueError(f"focused source fingerprint differs: changed={changed}, "
                         f"nonexperiment_added={forbidden}")
    expected_environment = plan["payload"]["environment"]
    current_environment = _execution_environment()
    if current_environment != expected_environment:
        differences = sorted(name for name in set(current_environment) | set(expected_environment)
                             if current_environment.get(name) != expected_environment.get(name))
        raise ValueError(f"focused execution environment differs: {differences}; "
                         f"expected={expected_environment}, actual={current_environment}")
    return current_source, current_environment


def _manifest(focused_dir: Path, data_dir: Path, proxy_dir: Path,
              cache_bytes: int) -> dict:
    plan, summary, _selection, confirmation, baseline, selected = _records(focused_dir)
    if cache_bytes < 1 or cache_bytes > MAX_CACHE_BYTES:
        raise ValueError("cache must be positive and at most 2 GB")
    source, environment = _runtime_ancestry(plan)
    matches = plan["payload"]["fold"]["outer_train"]
    prepared = [data_dir / f"{match}_half{period}.npz"
                for match in matches for period in (1, 2)]
    provenance = verify_prepared(data_dir, prepared)
    actual_prepared = {p.name: sha256_file(p) for p in prepared}
    expected_prepared = plan["payload"]["prepared_sha256"]
    if any(actual_prepared[name] != expected_prepared.get(name)
           for name in actual_prepared):
        raise ValueError("six development prepared caches differ from focused protocol")
    skillcorner_proxy.verify_sources(proxy_dir)
    provider = json.loads((proxy_dir / "manifest.json").read_text())
    if (provider.get("commit") != skillcorner_proxy.COMMIT
            or {row.get("filename"): row.get("sha256") for row in provider.get("files", [])}
            != skillcorner_proxy.SOURCE_SHA256):
        raise ValueError("SkillCorner provider manifest differs from pinned sources")
    proxy_paths = [proxy_dir / "manifest.json", proxy_dir / "coverage-audit.json"]
    for match in skillcorner_proxy.MATCHES:
        proxy_paths += [proxy_dir / row["filename"] for row in skillcorner_proxy.source_files()
                        if row["match_id"] == match]
        for period in (1, 2):
            proxy_paths += [proxy_dir / f"{match}_half{period}_{suffix}.npz"
                            for suffix in ("input25", "proxy10_labels", "source_map")]
    artifacts = _hashed_files([focused_dir / f"{name}.json" for name in
                               ("protocol", "summary", "selection", "confirmation",
                                "baseline-recipe", "selected-recipe")]
                              + [focused_dir / "trials" / "trial_000" / "trial.json"], focused_dir)
    gate = {"complete_and_hash_verified": True,
            "selected_comparison_eligible": bool(confirmation["comparison"].get("eligible")),
            "no_op": confirmation["no_op"],
            "status": ("no_op" if confirmation["no_op"] else
                       "passed" if confirmation["comparison"].get("eligible") else "failed")}
    return {"protocol": "frozen_transfer_skillcorner_proxy_v1",
            "selection_frozen": True, "confirmation_gate": gate,
            "development_matches": matches, "reserved_proxy_matches": list(skillcorner_proxy.MATCHES),
            "seeds": list(SEEDS), "selected_rung": summary["selected_rung"],
            "resources": {"device": "cpu", "workers": 1, "cache_bytes": cache_bytes},
            "roots": {"focused": str(focused_dir), "prepared": str(data_dir),
                      "proxy": str(proxy_dir)},
            "artifact_sha256": artifacts, "source_sha256": source,
            "focused_source_sha256": plan["payload"]["source_sha256"],
            "execution_environment": environment,
            "prepared_sha256": actual_prepared,
            "prepared_manifest_sha256": sha256_file(data_dir / "manifest.json"),
            "prepared_provenance": provenance,
            "proxy_sha256": _hashed_files(proxy_paths, proxy_dir),
            "recipes": {"baseline": baseline, "selected": selected},
            "fit_keys": {str(seed): {label: _fit_key(recipe, seed, matches)
                                     for label, recipe in (("baseline", baseline),
                                                           ("selected", selected))}
                         for seed in SEEDS},
            "proxy_label_limit": "Provider H/A agreement subset only; dead-ball truth unavailable; conditional metrics are not official challenge scores"}


@contextmanager
def _lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".lock").open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _assert_manifest(path: Path, expected: dict) -> None:
    if not path.is_file() or json.loads(path.read_text()) != expected:
        raise ValueError("frozen transfer provenance changed during run")


def _fit(data_dir: Path, output_dir: Path, manifest: dict, recipe: dict,
         seed: int, key: str, cache: HalfCache) -> tuple[Path, dict]:
    directory = output_dir / "fits" / str(seed) / key
    checkpoint, progress_path = directory / "model.pt", directory / "progress.json"
    existing = _load_record(progress_path)
    if existing is not None:
        if (not checkpoint.is_file() or existing.get("checkpoint_sha256") != sha256_file(checkpoint)
                or existing.get("fit_key") != key
                or existing.get("manifest_sha256") != _digest(manifest)):
            raise ValueError("completed refit checkpoint ancestry differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = payload.get("training_config", {})
        args = recipe["training_args"]
        expected_model = {"player_dim": 8, "global_dim": 9, "width": 64,
                          "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                          **recipe["model_kwargs"]}
        if (config.get("train_matches") != manifest["development_matches"]
                or config.get("validation_matches") != []
                or config.get("seed") != seed or config.get("steps") != recipe["steps"]
                or config.get("transfer_fit_key") != key
                or config.get("transfer_manifest_sha256") != _digest(manifest)
                or config.get("augmentation_config") != recipe["augmentation_config"]
                or config.get("outage_config") != OutageConfig(
                    **recipe["outage_config"]).to_dict()
                or any(config.get(name) != args[name] for name in (
                    "learning_rate", "weight_decay", "boundary_weight",
                    "boundary_pos_weight", "center_size", "augment"))
                or payload.get("model_kwargs") != expected_model
                or payload.get("prepared_sha256") != manifest["prepared_sha256"]):
            raise ValueError("completed refit training provenance differs")
        return checkpoint, existing
    if checkpoint.exists() or progress_path.exists():
        raise ValueError("incomplete refit artifacts require manual inspection")
    args = recipe["training_args"]
    session = TrainingSession(
        data_dir, train_matches=manifest["development_matches"], seed=seed,
        device="cpu", model_kwargs=recipe["model_kwargs"],
        learning_rate=args["learning_rate"], weight_decay=args["weight_decay"],
        boundary_weight=args["boundary_weight"],
        boundary_pos_weight=args["boundary_pos_weight"],
        augment=args["augment"], center_size=args["center_size"], cache=cache,
        outage_config=recipe["outage_config"],
        augmentation_config=recipe["augmentation_config"])
    progress = session.advance_to(recipe["steps"])
    directory.mkdir(parents=True, exist_ok=True)
    session.save_checkpoint(checkpoint, extra_training_config={
        "transfer_fit_key": key, "transfer_manifest_sha256": _digest(manifest),
        "transfer_role": "six_development_match_refit_no_proxy_labels"})
    record = {"fit_key": key, "seed": seed, "steps": recipe["steps"],
              "checkpoint_sha256": sha256_file(checkpoint),
              "manifest_sha256": _digest(manifest), "progress": progress,
              "train_matches": manifest["development_matches"]}
    _save_record(progress_path, record)
    return checkpoint, record


def _score_match(proxy_dir: Path, match: str, model, recipe: dict) -> dict:
    """Decode label-free 25 Hz inputs, then score mapped native 10 Hz halves."""
    halves = skillcorner_proxy.prepare_match(proxy_dir, match)
    if len(halves) != 2 or [half.input_25hz.period for half in halves] != [1, 2]:
        raise ValueError("reserved proxy has missing or reordered halves")
    reports = []
    native_parts = []
    model_parts = []
    map_parts = []
    model_offset = 0
    for half in halves:
        frames = half.input_25hz
        features = build_features(frames)
        reference = reference_scores(frames)
        neural = predict_features(features, model, device="cpu",
                                  center_size=recipe["training_args"]["center_size"],
                                  halo=model.radius)
        arrays = {"neural_live": neural["live_probability"],
                  "neural_home": neural["home_probability"],
                  "reference_live": reference["live_probability"],
                  "reference_home": reference["home_probability"],
                  "ball_valid": frames.ball_valid,
                  "nearest_ball_seconds": 10.0 * np.minimum(
                      features.global_tokens[:, 6], features.global_tokens[:, 7]).astype(np.float64)}
        prediction = decode_oof_half(arrays, recipe)
        metrics = skillcorner_proxy.score_native_proxy(
            half.proxy_10hz, prediction, half.source_to_model, selection_frozen=True)
        native_parts.append(half.proxy_10hz)
        model_parts.append(prediction)
        map_parts.append(half.source_to_model + model_offset)
        model_offset += len(prediction) + 1
        reports.append({"period": frames.period, "provider_label_audit": half.audit,
                        "native10_conditional": metrics})
    # One X separator keeps the two half timelines disjoint for official spell
    # and transition matching, while all durations remain native 10 Hz.
    fields = ("live", "live_valid", "owner", "owner_valid", "state", "state_valid")
    dummy = TrainingTargets(np.array([0], np.int8), np.array([False]),
                            np.array([0], np.int8), np.array([False]),
                            np.array([0], np.int8), np.array([False]))
    labels = TrainingTargets(*(np.concatenate((getattr(native_parts[0], field),
                                               getattr(dummy, field),
                                               getattr(native_parts[1], field)))
                               for field in fields))
    prediction = np.concatenate((model_parts[0], np.array([official.X], np.int8), model_parts[1]))
    mapping = np.concatenate((map_parts[0], np.array([len(model_parts[0])], np.int64),
                              map_parts[1]))
    native_metrics = skillcorner_proxy.score_native_proxy(
        labels, prediction, mapping, selection_frozen=True)
    return {"match": match, "halves": reports,
            "native10_conditional": native_metrics,
            "provider_label_coverage": {
                "graded_HA_frames": native_metrics["graded_HA_frames"],
                "total_native_frames_excluding_half_separator":
                    sum(h["native10_conditional"]["total_native_frames"] for h in reports),
                "coverage_excluding_half_separator":
                    sum(h["native10_conditional"]["graded_HA_frames"] for h in reports)
                    / sum(h["native10_conditional"]["total_native_frames"] for h in reports)}}


def run(focused_dir, data_dir, proxy_dir, output_dir, *, selection_frozen=False,
        cache_bytes=MAX_CACHE_BYTES) -> dict:
    if selection_frozen is not True:
        raise ValueError("explicit selection_frozen=True required before proxy evaluation")
    focused_dir, data_dir, proxy_dir, output_dir = (
        Path(path).resolve() for path in (focused_dir, data_dir, proxy_dir, output_dir))
    torch.set_num_threads(4)
    with _lock(output_dir):
        manifest = _manifest(focused_dir, data_dir, proxy_dir, cache_bytes)
        manifest_path = output_dir / "frozen-manifest.json"
        if manifest_path.exists():
            _assert_manifest(manifest_path, manifest)
        else:
            write_json(manifest_path, manifest)  # Published before first fit or proxy score.
        cache = HalfCache(data_dir, max_bytes=cache_bytes)
        fits = {}
        for seed in SEEDS:
            fits[str(seed)] = {}
            for label in ("baseline", "selected"):
                _assert_manifest(manifest_path, _manifest(focused_dir, data_dir, proxy_dir, cache_bytes))
                recipe = manifest["recipes"][label]
                key = manifest["fit_keys"][str(seed)][label]
                print(f"[frozen-transfer] verify/refit seed={seed} role={label} "
                      f"steps={recipe['steps']}", flush=True)
                checkpoint, progress = _fit(data_dir, output_dir, manifest, recipe, seed, key, cache)
                fits[str(seed)][label] = {"checkpoint": str(checkpoint), "progress": progress}
            left = fits[str(seed)]["baseline"]["progress"]["progress"]
            right = fits[str(seed)]["selected"]["progress"]["progress"]
            for name in ("training_window_sha256", "training_schedule_blocks", "trained_halves"):
                if left[name] != right[name]:
                    raise ValueError(f"paired refit windows differ: {name}")
        by_seed = {}
        for seed in SEEDS:
            by_seed[str(seed)] = {}
            for label in ("baseline", "selected"):
                recipe = manifest["recipes"][label]
                checkpoint = Path(fits[str(seed)][label]["checkpoint"])
                model, _ = load_model_checkpoint(checkpoint, device="cpu")
                by_match = {}
                for match in skillcorner_proxy.MATCHES:
                    _assert_manifest(manifest_path, _manifest(focused_dir, data_dir, proxy_dir, cache_bytes))
                    print(f"[frozen-transfer] verify/score seed={seed} role={label} "
                          f"proxy={match}", flush=True)
                    path = output_dir / "proxy" / str(seed) / label / f"{match}.json"
                    cached = _load_record(path)
                    if cached is None:
                        cached = {"seed": seed, "role": label, "match": match,
                                  "fit_key": manifest["fit_keys"][str(seed)][label],
                                  "checkpoint_sha256": sha256_file(checkpoint),
                                  "manifest_sha256": _digest(manifest),
                                  **_score_match(proxy_dir, match, model, recipe)}
                        _save_record(path, cached)
                    elif (cached.get("manifest_sha256") != _digest(manifest)
                          or cached.get("checkpoint_sha256") != sha256_file(checkpoint)
                          or cached.get("fit_key") != manifest["fit_keys"][str(seed)][label]):
                        raise ValueError("cached proxy result provenance differs")
                    by_match[match] = cached
                by_seed[str(seed)][label] = by_match
        _assert_manifest(manifest_path, _manifest(focused_dir, data_dir, proxy_dir, cache_bytes))
        result = {"protocol": manifest["protocol"], "selection_frozen": True,
                  "confirmation_gate": manifest["confirmation_gate"],
                  "manifest_sha256": sha256_file(manifest_path),
                  "selected_rung": manifest["selected_rung"],
                  "seeds": list(SEEDS), "fit_records": fits,
                  "by_seed": by_seed, "proxy_label_limit": manifest["proxy_label_limit"],
                  "official_challenge_score": None}
        existing = _load_record(output_dir / "summary.json")
        if existing is not None and existing != result:
            raise ValueError("completed transfer summary differs")
        if existing is None:
            _save_record(output_dir / "summary.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focused", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/dfl"))
    parser.add_argument("--proxy-dir", type=Path, default=Path("data/external/skillcorner-proxy"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-frozen", action="store_true", required=True)
    parser.add_argument("--cache-gb", type=float, default=2.0)
    args = parser.parse_args(argv)
    result = run(args.focused, args.data_dir, args.proxy_dir, args.output,
                 selection_frozen=args.selection_frozen,
                 cache_bytes=int(args.cache_gb * 1_000_000_000))
    print(json.dumps({"summary": str(args.output / "summary.json"),
                      "seeds": result["seeds"], "matches": list(skillcorner_proxy.MATCHES)}, indent=2))


if __name__ == "__main__":
    main()
