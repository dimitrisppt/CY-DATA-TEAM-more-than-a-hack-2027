"""Resumable public-only train-three producer for fixed nested timing studies.

No training starts on import or planning. Twelve fresh recipes are required.
Assessment predictions reuse the verified train-four v6 cache. Every newly
produced calibration prediction declares its actual three training matches.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments import campaign_evaluation, campaign_training
from experiments.campaign_protocol import publish, read_record
from experiments.coarse_context import blend_coarse_context, predict_coarse_native
from experiments.coarse_stability import _average_logits, _bagged_loaded
from experiments.ema_stability import _verify_input_cache
from experiments.ema_training import replay_tail_ema
from experiments.robustness import _source_hashes
from experiments.timing_cache import _check_hash, _input_arrays, _required
from experiments.timing_validation import (
    HISTORICAL_REUSE_CAVEAT,
    BasePredictionProvenance,
    ComponentProvenance,
    ConfigLineage,
    build_nested_plan,
    canonical_sha256,
    freeze_protocol,
    validate_prediction_provenance,
)
from experiments.tune import _seed_for_fold
from possession.prepared import verify_prepared
from possession.storage import load_half, sha256_file, write_json
from possession.timing import TimingConfig
from possession.training_session import HalfCache

DEFAULT_ARTIFACTS = Path("/Volumes/T7/Data/more-than-a-hack-2027/artifacts")
REPO = Path(__file__).resolve().parents[1]
HISTORICAL_FIT_SECONDS = {
    "mean_007": 902.9,
    "ball_attention_002": 403.8,
    "coarse_player_only": 64.1,
}
HISTORICAL_TAIL_SECONDS = {"mean_007": 32.3, "ball_attention_002": 13.4}
SEEDS = (8101, 8102)
SCENARIOS = ("clean", "short", "long", "replay")


class NestedBudgetExceeded(RuntimeError):
    """A resumable run reached its invocation budget or first-fit projection gate."""


def make_nested_plan(cache_manifest, config=None):
    """Build the immutable twelve-recipe schedule from existing fold metadata."""
    if isinstance(cache_manifest, (str, Path)):
        cache_manifest = _required(Path(cache_manifest))
    if cache_manifest.get("schema") != "public_dfl_timing_cache_v1" or not cache_manifest.get(
        "exact_v6_parity"
    ):
        raise ValueError("nested study requires an authenticated exact-v6 public cache")
    folds = cache_manifest["inner_folds"]
    if len(folds) != 3:
        raise ValueError("exactly three existing assessment folds are required")
    matches = sorted({m for fold in folds for m in fold["validation"]})
    universe = sorted({row["match"] for row in cache_manifest["records"]})
    if len(universe) != 7:
        raise ValueError("the seven historical public matches must be explicit")
    plan = build_nested_plan(
        matches,
        search=False,
        outer_assessment_folds=[fold["validation"] for fold in folds],
        historical_selection_matches=universe,
    )
    recipes = []
    for index, fold in enumerate(folds):
        if sorted(fold["train"]) != plan["folds"][index]["training_matches"]:
            raise ValueError("existing assessment fit training set differs from nested plan")
        for match in fold["train"]:
            train = sorted(set(fold["train"]) - {match})
            forbidden = sorted(set(fold["validation"]) | {match})
            if len(train) != 3 or set(train) & set(forbidden):
                raise ValueError("nested calibration fit leaks a held-out match")
            recipes.append(
                {
                    "id": f"nested_{index}_{match}",
                    "fold": index,
                    "train_matches": train,
                    "calibration_match": match,
                    "assessment_matches": fold["validation"],
                    "forbidden_matches": forbidden,
                }
            )
    return {
        "schema": "timing_nested_producer_plan_v1",
        "plan": plan,
        "recipes": recipes,
        "config": asdict(config or TimingConfig()),
        "seeds": list(SEEDS),
        "batches": ["batch_1", "batch_2"],
        "scenarios": list(SCENARIOS),
        "steps": 7200,
        "tail": {"start_steps": 7000, "end_steps": 7200, "decay": 0.99},
        "search": False,
        "historical_reuse_caveat": HISTORICAL_REUSE_CAVEAT,
    }


def projected_training_seconds(first_fit_seconds):
    """Scale the measured first mean fit to all declared historical fit costs."""
    if not np.isfinite(first_fit_seconds) or first_fit_seconds <= 0:
        raise ValueError("first-fit seconds must be positive and finite")
    historical = (
        12
        * len(SEEDS)
        * (sum(HISTORICAL_FIT_SECONDS.values()) + sum(HISTORICAL_TAIL_SECONDS.values()))
    )
    return historical * float(first_fit_seconds) / HISTORICAL_FIT_SECONDS["mean_007"]


class _Budget:
    def __init__(self, output, max_seconds):
        if not np.isfinite(max_seconds) or not 0 < max_seconds <= 43200:
            raise ValueError("max_seconds must be positive and at most the 12 hour budget")
        self.output, self.max_seconds = output, float(max_seconds)
        self.started = time.monotonic()

    def status(self, phase, **detail):
        row = {
            "phase": phase,
            "invocation_elapsed_seconds": time.monotonic() - self.started,
            "max_seconds": self.max_seconds,
            **detail,
        }
        write_json(self.output / "status.json", row)
        print(json.dumps(row), flush=True)

    def check(self):
        if time.monotonic() - self.started >= self.max_seconds:
            self.status("budget_exhausted_resumable")
            raise NestedBudgetExceeded(
                "nested producer invocation budget exhausted, states preserved"
            )


def _identity(recipe, name, spec, seed, protocol_sha):
    return {
        "protocol_sha256": protocol_sha,
        "spec_sha256": canonical_sha256(spec),
        "train_matches": recipe["train_matches"],
        "seed": _seed_for_fold(seed, f"timing-nested-{recipe['id']}", recipe["fold"]),
        "name": name,
    }


def _committed_states(directory):
    return sorted(
        (p for p in directory.glob("state_*.pt") if p.with_suffix(".pt.sha256.json").is_file()),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )


def _prune_states(directory, *, fine):
    states = _committed_states(directory)
    keep = set(states[-2:])
    if fine:
        keep.update(p for p in states if p.stem in ("state_007000", "state_007200"))
    for obsolete in states:
        if obsolete not in keep:
            obsolete.unlink()
            obsolete.with_suffix(".pt.sha256.json").unlink(missing_ok=True)


def _verify_fit(directory, identity, spec, recipe):
    import torch

    fit = read_record(directory / "fit.json")
    if fit is None:
        return None
    if fit.get("identity") != identity or fit.get("steps") != 7200:
        raise ValueError("nested fit identity differs")
    _check_hash(directory / "model.pt", fit.get("checkpoint_sha256"))
    payload = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    config = payload.get("training_config", {})
    if (
        config.get("train_matches") != recipe["train_matches"]
        or config.get("validation_matches") != [recipe["calibration_match"]]
        or config.get("seed") != identity["seed"]
        or config.get("steps") != 7200
        or config.get("campaign_spec") != spec
        or config.get("timing_nested_recipe") != recipe["id"]
    ):
        raise ValueError("nested checkpoint training ancestry differs")
    if set(config["train_matches"]) & set(recipe["forbidden_matches"]):
        raise ValueError("nested checkpoint leaks excluded matches")
    return fit


def _train_raw(directory, data_dir, spec, identity, recipe, cache, budget, *, fine):
    completed = _verify_fit(directory, identity, spec, recipe)
    if completed is not None:
        return completed
    directory.mkdir(parents=True, exist_ok=True)
    session = campaign_training.build_session(
        data_dir, spec, recipe["train_matches"], identity["seed"], cache
    )
    states = _committed_states(directory)
    if states:
        campaign_training.load_training_state(states[-1], session, identity)
    if session.steps > 7200:
        raise ValueError("nested state exceeds frozen step budget")
    timing_path = directory / "training-time.json"
    previous_seconds = 0.0
    if timing_path.exists():
        timing = json.loads(timing_path.read_text())
        if timing.get("identity") != identity or timing.get("steps", 0) > session.steps:
            raise ValueError("nested training time record differs from resumed state")
        previous_seconds = float(timing["seconds"])
    started = time.monotonic()
    while session.steps < 7200:
        budget.check()
        session.advance_to(min(7200, session.steps + 200))
        campaign_training.save_training_state(
            directory / f"state_{session.steps:06d}.pt", session, identity
        )
        _prune_states(directory, fine=fine)
        write_json(
            timing_path,
            {
                "identity": identity,
                "steps": session.steps,
                "seconds": previous_seconds + time.monotonic() - started,
            },
        )
        budget.status(
            "training_base",
            recipe=recipe["id"],
            member=identity["name"],
            seed=identity["seed"],
            step=session.steps,
        )
    session.save_checkpoint(
        directory / "model.pt",
        validation_matches=[recipe["calibration_match"]],
        extra_training_config={
            "timing_nested_recipe": recipe["id"],
            "timing_nested_protocol_sha256": identity["protocol_sha256"],
            "timing_nested_forbidden_matches": recipe["forbidden_matches"],
        },
    )
    fit = publish(
        directory / "fit.json",
        {
            "schema": "timing_nested_base_fit_v1",
            "identity": identity,
            "steps": 7200,
            "checkpoint_sha256": sha256_file(directory / "model.pt"),
            "training_seconds": previous_seconds + time.monotonic() - started,
            "trained_halves": session._progress()["trained_halves"],
            "validation_matches": [recipe["calibration_match"]],
        },
    )
    del session
    gc.collect()
    return fit


def _fit_recipe(
    recipe,
    output,
    data_dir,
    coarse_data,
    selection,
    coarse_spec,
    protocol_sha,
    budget,
    projection_check,
):
    fine_cache, coarse_cache = (
        HalfCache(data_dir, max_bytes=1_000_000_000),
        HalfCache(coarse_data, max_bytes=100_000_000),
    )
    fine, coarse, components = {}, {}, []
    for seed in SEEDS:
        for member in selection["members"]:
            name, spec = member["id"], member["spec"]
            directory = output / "fits" / recipe["id"] / f"seed_{seed}" / name
            identity = _identity(recipe, name, spec, seed, protocol_sha)
            raw = _train_raw(
                directory / "raw", data_dir, spec, identity, recipe, fine_cache, budget, fine=True
            )
            projection_check(raw, name)
            budget.check()
            ema_fit = replay_tail_ema(
                data_dir=data_dir,
                spec=spec,
                identity=identity,
                state_path=directory / "raw/state_007000.pt",
                baseline_checkpoint=directory / "raw/model.pt",
                baseline_sha256=raw["checkpoint_sha256"],
                output_dir=directory / "ema",
                validation_matches=[recipe["calibration_match"]],
                decay=0.99,
                start_steps=7000,
                end_steps=7200,
                cache=fine_cache,
                progress_callback=budget.check,
            )
            raw_component = ComponentProvenance(
                f"{name}_raw_seed_{seed}",
                "fine_model",
                recipe["train_matches"],
                [],
                sha256_file(directory / "raw/fit.json"),
                raw["checkpoint_sha256"],
            )
            component = ComponentProvenance(
                f"{name}_seed_{seed}",
                "fine_model",
                recipe["train_matches"],
                [],
                sha256_file(directory / "ema/fit.json"),
                ema_fit["checkpoint_sha256"],
                [raw_component],
            )
            fine[(seed, name)] = {
                "fit": ema_fit,
                "fit_path": directory / "ema/fit.json",
                "model_path": directory / "ema/model.pt",
                "component": component,
            }
            components.append(component)
        name = "coarse_player_only"
        directory = output / "fits" / recipe["id"] / f"seed_{seed}" / name
        identity = _identity(recipe, name, coarse_spec, seed, protocol_sha)
        fit = _train_raw(
            directory, coarse_data, coarse_spec, identity, recipe, coarse_cache, budget, fine=False
        )
        component = ComponentProvenance(
            f"{name}_seed_{seed}",
            "coarse_model",
            recipe["train_matches"],
            [],
            sha256_file(directory / "fit.json"),
            fit["checkpoint_sha256"],
        )
        coarse[seed] = {
            "fit": fit,
            "fit_path": directory / "fit.json",
            "model_path": directory / "model.pt",
            "component": component,
        }
        components.append(component)
    return fine, coarse, components


def _verify_completed_recipe(
    completed, recipe, output, selection, coarse_spec, protocol_sha, sources, universe
):
    """Reauthenticate model ancestry before reusing a completed compact recipe."""
    if completed.get("protocol_sha256") != protocol_sha or completed.get("recipe") != recipe:
        raise ValueError("completed nested recipe identity differs")
    expected_components = {}
    for seed in SEEDS:
        for member in selection["members"]:
            name, spec = member["id"], member["spec"]
            directory = output / "fits" / recipe["id"] / f"seed_{seed}" / name
            identity = _identity(recipe, name, spec, seed, protocol_sha)
            raw = _verify_fit(directory / "raw", identity, spec, recipe)
            if raw is None:
                raise ValueError("completed recipe has missing raw fit ancestry")
            ema = _required(directory / "ema/fit.json")
            if ema.get("identity") != identity or ema.get("steps") != 7200:
                raise ValueError("completed recipe EMA fit ancestry differs")
            tail = ema.get("tail_ema", {})
            if any(
                tail.get(k) != v
                for k, v in {
                    "decay": 0.99,
                    "start_steps": 7000,
                    "end_steps": 7200,
                    "updates": 200,
                }.items()
            ) or not all(
                ema.get("raw_parity", {}).get(k)
                for k in (
                    "raw_weights_match_endpoint_state",
                    "raw_weights_match_baseline_checkpoint",
                    "raw_training_progress_matches_endpoint",
                )
            ):
                raise ValueError("completed recipe EMA tail recipe or raw parity differs")
            _check_hash(directory / "ema/model.pt", ema["checkpoint_sha256"])
            if ema["inputs"]["baseline_checkpoint_sha256"] != raw["checkpoint_sha256"]:
                raise ValueError("completed recipe EMA raw ancestor differs")
            for step, key in ((7000, "start_state_sha256"), (7200, "end_state_sha256")):
                _check_hash(directory / "raw" / f"state_{step:06d}.pt", ema["inputs"][key])
            expected_components[f"{name}_seed_{seed}"] = (
                sha256_file(directory / "ema/fit.json"),
                ema["checkpoint_sha256"],
                [(sha256_file(directory / "raw/fit.json"), raw["checkpoint_sha256"])],
            )
        name = "coarse_player_only"
        directory = output / "fits" / recipe["id"] / f"seed_{seed}" / name
        identity = _identity(recipe, name, coarse_spec, seed, protocol_sha)
        coarse = _verify_fit(directory, identity, coarse_spec, recipe)
        if coarse is None:
            raise ValueError("completed recipe has missing coarse ancestry")
        expected_components[f"{name}_seed_{seed}"] = (
            sha256_file(directory / "fit.json"),
            coarse["checkpoint_sha256"],
            [],
        )
    fixed_sha = canonical_sha256(
        {
            "reference": "default",
            "feature_scales": "physical",
            "blend_weight": selection["blend_weight"],
        }
    )
    expected_components["fixed_features_reference_blend"] = (fixed_sha, fixed_sha, [])
    rows = completed.get("records", [])
    expected_grid = {
        (batch, half, scenario)
        for batch in ("batch_1", "batch_2")
        for half in (1, 2)
        for scenario in SCENARIOS
    }
    if len(rows) != 16 or {(r["batch"], r["half"], r["scenario"]) for r in rows} != expected_grid:
        raise ValueError("completed nested recipe grid differs")
    for row in rows:
        expected_path = (
            output
            / "calibration_cache"
            / recipe["id"]
            / row["batch"]
            / f"half{row['half']}"
            / f"{row['scenario']}.npz"
        )
        if (
            Path(row["path"]).resolve() != expected_path.resolve()
            or row.get("recipe_id") != recipe["id"]
            or row.get("scope") != "nested"
            or row.get("fold") != recipe["fold"]
            or row.get("match") != recipe["calibration_match"]
        ):
            raise ValueError("completed recipe row identity differs")
        _check_hash(expected_path, row["sha256"])
        source = next(
            r
            for r in sources[row["batch"]]
            if r["match"] == row["match"]
            and r["period"] == row["half"]
            and r["scenario"] == row["scenario"]
        )
        if row["source_input_sha256"] != source["input_sha256"]:
            raise ValueError("completed recipe source input differs")
        provenance = row["provenance"]
        if (
            provenance["feature_sha256"] != row["sha256"]
            or provenance["split_id"] != f"nested_fold_{recipe['fold']}"
            or (provenance["match"], provenance["half"], provenance["scenario"])
            != (row["match"], row["half"], row["scenario"])
        ):
            raise ValueError("completed recipe compact evidence is detached from ancestry")
        validate_prediction_provenance(
            provenance,
            outer_assessment_matches=recipe["assessment_matches"],
            allowed_matches=universe,
            expected_components=sorted(expected_components),
        )
        for component in provenance["components"]:
            expected = expected_components[component["name"]]
            actual = (
                component["fit_sha256"],
                component["model_sha256"],
                [(a["fit_sha256"], a["model_sha256"]) for a in component["ancestors"]],
            )
            if actual != expected:
                raise ValueError("completed recipe declared ancestor hashes differ from saved fits")
            learned = component["name"] != "fixed_features_reference_blend"
            expected_train = recipe["train_matches"] if learned else []
            if component["train_matches"] != expected_train or component["selection_matches"]:
                raise ValueError(
                    "completed recipe actual training matches differ from declared ancestry"
                )
            for ancestor in component["ancestors"]:
                if (
                    ancestor["train_matches"] != recipe["train_matches"]
                    or ancestor["selection_matches"]
                ):
                    raise ValueError("completed recipe raw ancestor training matches differ")
    return rows


def _coarse_predictions(recipe, models, data_dir, output, protocol_sha, budget):
    from possession.predict import load_model_checkpoint

    values = {}
    match = recipe["calibration_match"]
    for seed, model in models.items():
        net = None
        for half in (1, 2):
            budget.check()
            path = (
                output
                / "coarse_predictions"
                / recipe["id"]
                / f"seed_{seed}"
                / f"{match}_half{half}.npz"
            )
            sidecar = path.with_suffix(".json")
            identity = {
                "protocol_sha256": protocol_sha,
                "recipe": recipe["id"],
                "seed": seed,
                "match": match,
                "half": half,
                "model_sha256": model["fit"]["checkpoint_sha256"],
                "fit_sha256": sha256_file(model["fit_path"]),
                "source_sha256": sha256_file(data_dir / f"{match}_half{half}.npz"),
            }
            saved = read_record(sidecar)
            if saved is not None:
                if saved.get("identity") != identity:
                    raise ValueError("nested coarse prediction identity differs")
                _check_hash(path, saved["sha256"])
                with np.load(path, allow_pickle=False) as stored:
                    prediction = {name: stored[name] for name in stored.files}
            else:
                if net is None:
                    net, _ = load_model_checkpoint(model["model_path"], device="cpu")
                frames, _ = load_half(data_dir / f"{match}_half{half}.npz")
                prediction = predict_coarse_native(frames, net, center_size=64)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **prediction)
                publish(sidecar, {"identity": identity, "sha256": sha256_file(path)})
            values[(seed, half)] = prediction
        del net
    return {
        half: {
            name: _average_logits(values[(SEEDS[0], half)][name], values[(SEEDS[1], half)][name])
            for name in ("live_probability", "home_probability")
        }
        for half in (1, 2)
    }


def _new_prediction_arrays(record):
    path = Path(record["prediction_path"])
    _check_hash(path, record["prediction_sha256"])
    meta = json.loads(path.with_suffix(".json").read_text())
    expected = {
        "schema": campaign_evaluation.PREDICTION_SCHEMA,
        "input_sha256": record["input_sha256"],
        "model_identity": record["model_identity"],
        "center_size": record["center_size"],
        "match": record["match"],
        "period": record["period"],
        "scenario": record["scenario"],
    }
    if (
        meta.get("identity") != expected
        or meta.get("prediction_sha256") != record["prediction_sha256"]
    ):
        raise ValueError("nested fine prediction sidecar differs")
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in ("neural_live", "neural_home", "neural_boundary")}


def _predict_recipe(
    recipe,
    fine,
    coarse,
    components,
    input_records,
    data_dir,
    selection,
    output,
    protocol_sha,
    universe,
    budget,
):
    from possession.predict import load_model_checkpoint

    averaged = _coarse_predictions(recipe, coarse, data_dir, output, protocol_sha, budget)
    fixed_sha = canonical_sha256(
        {
            "reference": "default",
            "feature_scales": "physical",
            "blend_weight": selection["blend_weight"],
        }
    )
    components = components + [
        ComponentProvenance(
            "fixed_features_reference_blend", "deterministic", [], [], fixed_sha, fixed_sha
        )
    ]
    historical = ConfigLineage("v6_recipe", canonical_sha256(selection), [], universe, True)
    rows = []
    for batch, sources in input_records.items():
        selected = [r for r in sources if r["match"] == recipe["calibration_match"]]
        expected = {
            (recipe["calibration_match"], half, scenario)
            for half in (1, 2)
            for scenario in SCENARIOS
        }
        if {(r["match"], r["period"], r["scenario"]) for r in selected} != expected or len(
            selected
        ) != 8:
            raise ValueError("nested public source grid differs")
        indexed = {}
        for key, model in fine.items():
            budget.check()
            identity = {
                "protocol_sha256": protocol_sha,
                "recipe": recipe["id"],
                "train_matches": recipe["train_matches"],
                "validation_matches": [recipe["calibration_match"]],
                "fit_sha256": sha256_file(model["fit_path"]),
                "model_sha256": model["fit"]["checkpoint_sha256"],
            }
            net, _ = load_model_checkpoint(model["model_path"], device="cpu")
            predictions = campaign_evaluation.predict_records(
                net,
                selected,
                output / "fine_predictions" / recipe["id"] / batch / f"seed_{key[0]}" / key[1],
                identity,
                center_size=2048,
                progress_callback=budget.check,
            )
            publish(
                output
                / "prediction_manifests"
                / recipe["id"]
                / batch
                / f"seed_{key[0]}_{key[1]}.json",
                {"identity": identity, "records": predictions},
            )
            indexed[key] = {(r["period"], r["scenario"]): r for r in predictions}
            del net
        for source in selected:
            budget.check()
            half, scenario = source["period"], source["scenario"]
            arrays = _input_arrays(source)
            bundles, raw_predictions = {}, []
            for seed in SEEDS:
                members = []
                for member in selection["members"]:
                    record = indexed[(seed, member["id"])][(half, scenario)]
                    prediction = _new_prediction_arrays(record)
                    raw_predictions.append(prediction)
                    members.append(
                        [
                            {
                                **source,
                                "arrays": {**arrays, **prediction},
                                "model_identity": record["model_identity"],
                            }
                        ]
                    )
                bundles[seed] = members
            fine_arrays = _bagged_loaded(bundles, selection)[0]["arrays"]
            coarse_arrays = averaged[half]
            mixed = blend_coarse_context(
                fine_arrays, coarse_arrays["live_probability"], coarse_arrays["home_probability"]
            )
            from experiments.campaign_decode import decode_arrays

            labels = np.asarray(decode_arrays(mixed, selection["decoder"]), dtype=np.uint8)
            compact = {
                **mixed,
                "v6_labels": labels,
                "fine_live": fine_arrays["neural_live"],
                "fine_home": fine_arrays["neural_home"],
                "coarse_live": coarse_arrays["live_probability"],
                "coarse_home": coarse_arrays["home_probability"],
                "member_live_std": np.std([p["neural_live"] for p in raw_predictions], axis=0),
                "member_home_std": np.std([p["neural_home"] for p in raw_predictions], axis=0),
            }
            path = (
                output
                / "calibration_cache"
                / recipe["id"]
                / batch
                / f"half{half}"
                / f"{scenario}.npz"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                with np.load(path, allow_pickle=False) as old:
                    if set(old.files) != set(compact) or any(
                        not np.array_equal(old[n], compact[n], equal_nan=True) for n in compact
                    ):
                        raise ValueError("existing nested compact evidence differs")
            else:
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **compact)
                temporary.replace(path)
            checksum = sha256_file(path)
            provenance = asdict(
                BasePredictionProvenance(
                    recipe["calibration_match"],
                    half,
                    scenario,
                    f"nested_fold_{recipe['fold']}",
                    checksum,
                    components,
                    [historical],
                )
            )
            validate_prediction_provenance(
                provenance,
                outer_assessment_matches=recipe["assessment_matches"],
                allowed_matches=universe,
            )
            rows.append(
                {
                    "match": recipe["calibration_match"],
                    "half": half,
                    "period": half,
                    "batch": batch,
                    "scenario": scenario,
                    "fold": recipe["fold"],
                    "scope": "nested",
                    "recipe_id": recipe["id"],
                    "path": str(path),
                    "sha256": checksum,
                    "n_frames": len(labels),
                    "provenance": provenance,
                    "source_input_sha256": source["input_sha256"],
                }
            )
    return rows


def _verify_assessment_rows(rows, fold, index, universe):
    expected_names = sorted(
        [
            f"{name}_seed_{seed}"
            for seed in SEEDS
            for name in ("mean_007", "ball_attention_002", "coarse_player_only")
        ]
        + ["fixed_features_reference_blend"]
    )
    expected_grid = {
        (batch, match, half, scenario)
        for batch in ("batch_1", "batch_2")
        for match in fold["validation"]
        for half in (1, 2)
        for scenario in SCENARIOS
    }
    if (
        len(rows) != 32
        or {(r["batch"], r["match"], r["half"], r["scenario"]) for r in rows} != expected_grid
    ):
        raise ValueError("nested assessment cache grid differs")
    for row in rows:
        provenance = row["provenance"]
        if (
            row["scope"] != "inner"
            or row["fold"] != index
            or provenance["feature_sha256"] != row["sha256"]
            or provenance["split_id"] != f"inner_fold_{index}"
            or (provenance["match"], provenance["half"], provenance["scenario"])
            != (row["match"], row["half"], row["scenario"])
        ):
            raise ValueError("nested assessment metadata is detached from its array or ancestry")
        _check_hash(Path(row["path"]), row["sha256"])
        audit = validate_prediction_provenance(
            provenance,
            outer_assessment_matches=fold["validation"],
            allowed_matches=universe,
            expected_components=expected_names,
        )
        if audit["training_matches"] != sorted(fold["train"]) or audit[
            "historical_selection_matches"
        ] != sorted(universe):
            raise ValueError("nested assessment training or historical recipe ancestry differs")
    return rows


def produce_nested_cache(
    output_dir,
    cache_manifest=None,
    *,
    artifacts_root=DEFAULT_ARTIFACTS,
    config=None,
    max_seconds=43200,
    projection_gate=True,
    inference_reserve_seconds=900,
    stop_after_first_recipe=False,
    prepare_only=False,
):
    """Fit and cache the twelve fresh recipes, resuming authenticated states.

    A first completed mean fit scales historical total training cost. The default
    gate stops when that estimate plus the declared inference reserve exceeds
    this invocation's budget. No timing HPO is performed. Returned split records
    can be consumed by the generic fixed-config timing assessment runner.
    """
    import torch

    if (
        not np.isfinite(inference_reserve_seconds)
        or inference_reserve_seconds < 0
        or type(projection_gate) is not bool
    ):
        raise ValueError("invalid inference reserve or projection gate")
    output, root = Path(output_dir).resolve(), Path(artifacts_root).resolve()
    cache_path = Path(
        cache_manifest or root / "timing-calibration-v1/cache/manifest.json"
    ).resolve()
    if (
        output == root
        or root.is_relative_to(output)
        or any(
            output.is_relative_to(root / name)
            for name in (
                "overnight-joint-v1",
                "ensemble-bagging-v1",
                "ema-stability-v1",
                "coarse-stability-v1",
            )
        )
    ):
        raise ValueError("nested output must be separate from frozen artifacts")
    output.mkdir(parents=True, exist_ok=True)
    budget = _Budget(output, max_seconds)
    cache = _required(cache_path)
    plan = make_nested_plan(cache, config)
    protocols = {
        name: _required(root / name / "protocol.json")
        for name in (
            "overnight-joint-v1",
            "ensemble-bagging-v1",
            "ema-stability-v1",
            "coarse-stability-v1",
        )
    }
    campaign, bag, ema, coarse = (protocols[n] for n in protocols)
    selection = coarse["candidate_selection"]
    if (
        selection != ema["candidate_selection"]
        or selection["decoder"]["reference_mode"] != "default"
        or coarse["ema_protocol_sha256"] != canonical_sha256(ema)
        or coarse["campaign_protocol_sha256"] != canonical_sha256(campaign)
        or coarse["bagging_protocol_sha256"] != canonical_sha256(bag)
        or coarse["baseline_kind"] != "ema"
    ):
        raise ValueError("nested source recipe ancestry differs")
    if any(
        _source_hashes().get(name) != digest for name, digest in coarse["source_sha256"].items()
    ):
        raise ValueError("pinned base implementation source differs")
    for name, protocol in protocols.items():
        if name != "overnight-joint-v1" and cache["source_protocol_sha256"].get(
            name
        ) != canonical_sha256(protocol):
            raise ValueError("nested source recipe differs from verified assessment cache")
    if (
        cache["decoder"] != selection["decoder"]
        or cache["inner_folds"] != campaign["fold"]["inner"]
    ):
        raise ValueError("nested assessment cache uses a different recipe or folds")
    data_dir = (REPO / "data/processed/dfl").resolve()
    if Path(bag["data_dir"]).resolve() != data_dir:
        raise ValueError("nested producer accepts only the declared public prepared DFL directory")
    if (
        verify_prepared(data_dir, sorted(data_dir.glob("DFL-MAT-*_half*.npz")))
        != campaign["prepared"]
    ):
        raise ValueError("prepared public DFL source provenance differs")
    coarse_data = root / "coarse-stability-v1/derived_data"
    derivation = _required(coarse_data / "coarse-derivation.json")
    if canonical_sha256(derivation) != coarse["coarse_derivation_sha256"]:
        raise ValueError("coarse public derivation differs")
    verify_prepared(coarse_data, sorted(coarse_data.glob("DFL-MAT-*_half*.npz")))
    source_files = (
        "experiments/timing_nested.py",
        "experiments/timing_cache.py",
        "experiments/timing_validation.py",
        "possession/timing.py",
    )
    protocol = {
        **plan,
        "selection": selection,
        "coarse_spec": coarse["model_spec"],
        "assessment_cache_manifest_sha256": sha256_file(cache_path),
        "source_protocol_sha256": {name: canonical_sha256(p) for name, p in protocols.items()},
        "producer_source_sha256": {name: sha256_file(REPO / name) for name in source_files},
        "methods": ["identity", "global_offset", "family_offsets", "contextual"],
        "projection_gate": bool(projection_gate),
        "inference_reserve_seconds": inference_reserve_seconds,
        "target_inputs_accessed": False,
        "target_ground_truth_used": False,
    }
    freeze_protocol(output / "protocol.json", protocol)
    protocol_sha = canonical_sha256(protocol)
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    universe = plan["plan"]["historical_selection_matches"]
    sources = {}
    for batch in plan["batches"]:
        raw = _required(root / "ensemble-bagging-v1" / batch / "result.json")
        if raw.get("protocol_sha256") != canonical_sha256(bag):
            raise ValueError("public input batch protocol differs")
        sources[batch], _ = _verify_input_cache(
            root / "ensemble-bagging-v1" / batch / "inner",
            raw["inner_inputs_sha256"],
            set(plan["plan"]["matches"]),
        )
    if prepare_only:
        budget.status("preflight_complete_no_training", recipes=len(plan["recipes"]))
        return {
            "schema": "timing_nested_preflight_v1",
            "protocol_sha256": protocol_sha,
            "plan": plan,
            "complete": False,
            "training_started": False,
        }
    gate_path = output / "first-fit-projection.json"
    gate_done = gate_path.exists()
    if gate_done:
        gate = _required(gate_path)
        if gate.get("protocol_sha256") != protocol_sha:
            raise ValueError("first-fit projection belongs to another protocol")
        if (
            projection_gate
            and gate["projected_training_seconds"] + inference_reserve_seconds > max_seconds
        ):
            raise NestedBudgetExceeded("saved first-fit projection exceeds invocation budget")

    def projection_check(fit, name):
        nonlocal gate_done
        if gate_done or name != "mean_007":
            return
        estimate = projected_training_seconds(fit["training_seconds"])
        gate = {
            "protocol_sha256": protocol_sha,
            "first_fit_seconds": fit["training_seconds"],
            "projected_training_seconds": estimate,
            "inference_reserve_seconds": inference_reserve_seconds,
            "historical_fit_seconds": HISTORICAL_FIT_SECONDS,
            "historical_tail_seconds": HISTORICAL_TAIL_SECONDS,
            "interpretation": "Projection only, not a measured complete nested runtime.",
        }
        publish(gate_path, gate)
        gate_done = True
        budget.status("first_fit_projection", **gate)
        if projection_gate and estimate + inference_reserve_seconds > max_seconds:
            raise NestedBudgetExceeded(
                "first-fit projection exceeds budget, completed fit preserved"
            )

    all_rows = []
    for recipe in plan["recipes"]:
        budget.check()
        recipe_path = output / "recipes" / f"{recipe['id']}.json"
        completed = read_record(recipe_path)
        if completed is not None:
            verified_rows = _verify_completed_recipe(
                completed,
                recipe,
                output,
                selection,
                coarse["model_spec"],
                protocol_sha,
                sources,
                universe,
            )
            all_rows.extend(verified_rows)
            if stop_after_first_recipe:
                return {
                    "schema": "timing_nested_partial_v1",
                    "protocol_sha256": protocol_sha,
                    "records": all_rows,
                    "complete": False,
                }
            continue
        budget.status("recipe_start", recipe=recipe["id"], train_matches=recipe["train_matches"])
        fine, coarse_models, components = _fit_recipe(
            recipe,
            output,
            data_dir,
            coarse_data,
            selection,
            coarse["model_spec"],
            protocol_sha,
            budget,
            projection_check,
        )
        rows = _predict_recipe(
            recipe,
            fine,
            coarse_models,
            components,
            sources,
            data_dir,
            selection,
            output,
            protocol_sha,
            universe,
            budget,
        )
        publish(recipe_path, {"protocol_sha256": protocol_sha, "recipe": recipe, "records": rows})
        all_rows.extend(rows)
        budget.status("recipe_complete", recipe=recipe["id"], records=len(rows))
        if stop_after_first_recipe:
            return {
                "schema": "timing_nested_partial_v1",
                "protocol_sha256": protocol_sha,
                "records": all_rows,
                "complete": False,
            }
    splits = []
    for index, fold in enumerate(cache["inner_folds"]):
        training = [r for r in all_rows if r["fold"] == index]
        assessment = [r for r in cache["records"] if r["scope"] == "inner" and r["fold"] == index]
        _verify_assessment_rows(assessment, fold, index, universe)
        splits.append(
            {
                "id": f"nested_{index}",
                "kind": "strict_nested",
                "calibration": fold["train"],
                "assessment": fold["validation"],
                "training_records": training,
                "assessment_records": assessment,
            }
        )
    manifest = {
        "schema": "timing_nested_cache_v1",
        "protocol_sha256": protocol_sha,
        "records": all_rows,
        "record_count": len(all_rows),
        "splits": splits,
        "complete": True,
        "config": plan["config"],
        "historical_reuse_caveat": HISTORICAL_REUSE_CAVEAT,
    }
    publish(output / "manifest.json", manifest)
    budget.status("nested_cache_complete", records=len(all_rows), splits=len(splits))
    return manifest


def summarize_nested(results):
    """Aggregate official assessment records, giving each of six matches equal mass."""
    from experiments.timing_study import METHODS, WEIGHTS

    if len(results) != 3 or any(r["split"]["kind"] != "strict_nested" for r in results):
        raise ValueError("nested summary requires exactly three completed strict splits")
    assessment = [m for r in results for m in r["split"]["assessment"]]
    if len(assessment) != 6 or len(set(assessment)) != 6:
        raise ValueError("nested assessment matches must cover six unique groups")
    methods = {}
    for method in METHODS:
        records = [row for result in results for row in result["scores"][method]["records"]]
        expected_grid = {
            (batch, match, scenario)
            for batch in ("batch_1", "batch_2")
            for match in assessment
            for scenario in WEIGHTS
        }
        if (
            len(records) != len(expected_grid)
            or {(r["batch"], r["match"], r["scenario"]) for r in records} != expected_grid
        ):
            raise ValueError("nested official assessment score grid differs")
        by_match = {}
        for match in sorted(assessment):
            scenarios = {
                name: float(
                    np.mean(
                        [
                            r["spell_f1_mean"]
                            for r in records
                            if r["match"] == match and r["scenario"] == name
                        ]
                    )
                )
                for name in WEIGHTS
            }
            if not all(np.isfinite(value) for value in scenarios.values()):
                raise ValueError("nested assessment scenario is absent or nonfinite")
            by_match[match] = {
                "objective": float(sum(WEIGHTS[name] * scenarios[name] for name in WEIGHTS)),
                "means": scenarios,
            }
        means = {
            name: float(np.mean([r["means"][name] for r in by_match.values()])) for name in WEIGHTS
        }
        methods[method] = {
            "objective": float(np.mean([r["objective"] for r in by_match.values()])),
            "missing_ball": float(
                sum(WEIGHTS[name] * means[name] for name in ("short", "long", "replay")) / 0.5
            ),
            "means": means,
            "by_match": by_match,
        }
    for method in METHODS:
        delta = {
            match: methods[method]["by_match"][match]["objective"]
            - methods["identity"]["by_match"][match]["objective"]
            for match in sorted(assessment)
        }
        values = np.array(list(delta.values()))
        draws = (
            np.random.default_rng(20261116).choice(values, size=(10000, len(values))).mean(axis=1)
        )
        methods[method].update(
            {
                "by_match_objective_delta": delta,
                "objective_delta": float(values.mean()),
                "matches_improved": int(np.sum(values > 1e-12)),
                "descriptive_match_bootstrap_95pct": np.quantile(draws, [0.025, 0.975]).tolist(),
            }
        )
    contextual, identity = methods["contextual"], methods["identity"]
    return {
        "methods": methods,
        "gates": {
            "objective_gain_gt_0_002": contextual["objective_delta"] > 0.002,
            "four_of_six_matches_improved": contextual["matches_improved"] >= 4,
            "missing_ball_improved": contextual["missing_ball"] > identity["missing_ball"],
            "no_scenario_loss_over_0_01": all(
                contextual["means"][name] - identity["means"][name] >= -0.01 for name in WEIGHTS
            ),
        },
        "independent_test_claim": False,
        "promotion_authorized_by_this_study": False,
        "historical_reuse_caveat": HISTORICAL_REUSE_CAVEAT,
    }


def run_nested_study(
    output_dir,
    cache_manifest=None,
    *,
    artifacts_root=DEFAULT_ARTIFACTS,
    config=None,
    max_seconds=43200,
    inference_reserve_seconds=900,
    stop_after_first_recipe=False,
):
    """Produce all nested evidence and compare four fixed methods before deadline."""
    from experiments.timing_study import (
        EXPECTED_COMPONENTS,
        OFFSET_GRID,
        OFFSET_PENALTY,
        _run_split,
    )

    output, root = Path(output_dir).resolve(), Path(artifacts_root).resolve()
    cache_path = Path(
        cache_manifest or root / "timing-calibration-v1/cache/manifest.json"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    budget = _Budget(output, max_seconds)
    config = config or TimingConfig()
    cache = _required(cache_path)
    plan = make_nested_plan(cache, config)
    protocol = {
        "schema": "timing_nested_fixed_study_v1",
        "config": asdict(config),
        "plan": plan,
        "cache_manifest_sha256": sha256_file(cache_path),
        "offset_grid": list(OFFSET_GRID),
        "offset_penalty": OFFSET_PENALTY,
        "family_search_passes": 2,
        "timing_search": False,
        "source_sha256": {
            name: sha256_file(REPO / name)
            for name in (
                "experiments/timing_nested.py",
                "experiments/timing_study.py",
                "experiments/timing_validation.py",
                "possession/timing.py",
                "evaluation/possession_metrics.py",
            )
        },
        "historical_reuse_caveat": HISTORICAL_REUSE_CAVEAT,
    }
    freeze_protocol(output / "study-protocol.json", protocol)
    protocol_sha = canonical_sha256(protocol)
    budget.check()
    produced = produce_nested_cache(
        output,
        cache_path,
        artifacts_root=root,
        config=config,
        max_seconds=max_seconds - (time.monotonic() - budget.started),
        inference_reserve_seconds=inference_reserve_seconds,
        stop_after_first_recipe=stop_after_first_recipe,
    )
    if not produced["complete"]:
        return produced
    components = {"nested": EXPECTED_COMPONENTS["inner"], "inner": EXPECTED_COMPONENTS["inner"]}
    universe = plan["plan"]["historical_selection_matches"]
    results = []
    study_output = output / "assessment"

    def status(phase, **kwargs):
        budget.check()
        budget.status(phase, **kwargs)

    for split in produced["splits"]:
        budget.check()
        result_path = study_output / "splits" / split["id"] / "result.json"
        receipt_path = result_path.parent / "verified-result.json"
        receipt = read_record(receipt_path)
        expected_split = {
            k: v for k, v in split.items() if k not in ("training_records", "assessment_records")
        }
        if receipt is not None:
            if (
                receipt.get("protocol_sha256") != protocol_sha
                or receipt.get("split") != expected_split
            ):
                raise ValueError("completed nested assessment receipt differs")
            _check_hash(result_path, receipt["result_sha256"])
            result = json.loads(result_path.read_text())
            if (
                result.get("protocol_sha256") != protocol_sha
                or result.get("split") != expected_split
            ):
                raise ValueError("completed nested assessment protocol or split differs")
            for filename, checksum in (
                ("model.npz", result["model_sha256"]),
                ("predictions.npz", result["predictions_sha256"]),
            ):
                _check_hash(result_path.parent / filename, checksum)
        else:
            result = _run_split(
                split, [], study_output, config, protocol_sha, universe, components, status
            )
            if result.get("split") != expected_split:
                raise ValueError("nested assessment returned a different split")
            publish(
                receipt_path,
                {
                    "protocol_sha256": protocol_sha,
                    "split": expected_split,
                    "result_sha256": sha256_file(result_path),
                },
            )
        results.append(result)
    final = {
        "schema": "timing_nested_fixed_result_v1",
        "protocol_sha256": protocol_sha,
        "producer_protocol_sha256": produced["protocol_sha256"],
        "splits": results,
        "summary": summarize_nested(results),
        "complete": True,
        "invocation_elapsed_seconds": time.monotonic() - budget.started,
    }
    # Invocation runtime is mutable operational metadata, not part of frozen score content.
    existing = read_record(output / "result.json")
    if existing is not None:
        comparable = {k: v for k, v in existing.items() if k != "invocation_elapsed_seconds"}
        if comparable != {k: v for k, v in final.items() if k != "invocation_elapsed_seconds"}:
            raise ValueError("completed nested result differs from verified split artifacts")
        return existing
    publish(output / "result.json", final)
    budget.status(
        "nested_study_complete",
        objective={name: value["objective"] for name, value in final["summary"]["methods"].items()},
    )
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--inference-reserve-seconds", type=float, default=900)
    parser.add_argument("--stop-after-first-recipe", action="store_true")
    parser.add_argument("--produce-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    runner = produce_nested_cache if args.produce_only or args.plan_only else run_nested_study
    try:
        kwargs = {"prepare_only": True} if args.plan_only else {}
        runner(
            args.output_dir,
            args.cache_manifest,
            artifacts_root=args.artifacts_root,
            max_seconds=args.hours * 3600,
            inference_reserve_seconds=args.inference_reserve_seconds,
            stop_after_first_recipe=args.stop_after_first_recipe,
            **kwargs,
        )
    except NestedBudgetExceeded as exc:
        print(json.dumps({"phase": "stopped_resumable", "reason": str(exc)}), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
