"""Bounded, label-free compute pilot for the overnight campaign."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from experiments.campaign_decode import baseline_decoder, sample_decoder
from experiments.campaign_evaluation import (
    load_records,
    predict_records,
    prepare_inputs,
    score_records,
)
from experiments.campaign_protocol import WEIGHTS, scenarios
from experiments.campaign_space import TOTAL_STEPS, baseline_spec
from experiments.campaign_training import build_session
from experiments.cross_validation import nested_splits
from possession.predict import predict_features
from possession.schema import FeatureBatch
from possession.storage import write_json
from possession.training_session import HalfCache


def _recipe(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if "selected_model_recipe" in payload:
        return payload
    return payload


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS returns bytes; Linux returns KiB.
    return value if platform.system() == "Darwin" else value * 1024


def _project_staged_budget(models: dict, *, per_family: int = 12) -> dict:
    """Project the protocol's incremental rungs from measured 7,200-step rates."""
    families = ("mean", "ball_attention", "interaction")
    full = {name: float(models[name]["estimated_seconds_7200_steps"]) for name in families}

    def stage(configs_per_family: int, folds: int, extra_steps: int) -> float:
        return sum(full[name] * configs_per_family * folds * extra_steps / TOTAL_STEPS
                   for name in families)

    screen = stage(per_family, 3, 1200)
    promote = stage(3, 3, 2400)  # best three from each pooling family
    refine = stage(1, 3, 3600)  # best one from each family
    mean_full = full["mean"]
    min_full, max_full = min(full.values()), max(full.values())
    confirmation = {
        "baseline_seconds": 6 * mean_full * 3600 / TOTAL_STEPS,
        "candidate_seconds_range": [6 * min_full, 6 * max_full],
    }
    outer = {
        "baseline_seconds": 2 * mean_full * 3600 / TOTAL_STEPS,
        "candidate_seconds_range": [2 * min_full, 2 * max_full],
    }
    low = screen + promote + refine + confirmation["baseline_seconds"] + \
        confirmation["candidate_seconds_range"][0] + outer["baseline_seconds"] + \
        outer["candidate_seconds_range"][0]
    high = screen + promote + refine + confirmation["baseline_seconds"] + \
        confirmation["candidate_seconds_range"][1] + outer["baseline_seconds"] + \
        outer["candidate_seconds_range"][1]
    # OOF inference for the three search stages: 16 scenario/half predictions
    # per validation fold model; one per-family clean half timing is the proxy.
    inference = sum(float(models[name]["one_clean_half_inference_seconds"])
                    * (per_family * 3 + 3 * 3 + 1 * 3) * 16 for name in families)
    return {
        "protocol_rungs": [1200, 3600, 7200],
        "screen": {"configurations": per_family * 3, "fold_fits": per_family * 3 * 3,
                   "steps_per_fit": 1200, "fit_seconds": screen},
        "promote": {"configurations": 9, "fold_fits": 27,
                    "additional_steps_per_fit": 2400, "fit_seconds": promote},
        "refine": {"configurations": 3, "fold_fits": 9,
                   "additional_steps_per_fit": 3600, "fit_seconds": refine},
        "confirmation": {"fit_count": 12, "baseline_3600_fit_count": 6,
                         "candidate_7200_fit_count": 6,
                         "baseline_fit_seconds": confirmation["baseline_seconds"],
                         "candidate_fit_seconds_range": confirmation["candidate_seconds_range"]},
        "outer": {"fit_count": 4, "baseline_3600_fit_count": 2,
                  "candidate_7200_fit_count": 2,
                  "baseline_fit_seconds": outer["baseline_seconds"],
                  "candidate_fit_seconds_range": outer["candidate_seconds_range"]},
        "search_fit_hours": (screen + promote + refine) / 3600,
        "fit_only_total_hours_range": [low / 3600, high / 3600],
        "search_validation_inference_hours": inference / 3600,
        "screen_cutoff_hours_of_12": 12 * .45,
        "remaining_before_screen_cutoff_hours": 12 * .45 - screen / 3600,
        "fits_only_estimate": True,
        "limitations": ("Linear extrapolation from clean-policy 200-step baseline fits. "
                        "Candidate corruption overhead, decoder search, repeated data prep, "
                        "confirmation/outer inference, and contention are not fully included."),
    }


def _decoder_samples(seed: int = 0) -> tuple[dict, dict]:
    import optuna

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=seed))
    full = sample_decoder(study.ask())
    reference_only = sample_decoder(study.ask(), reference_only=True)
    return full, reference_only


def run(output_dir: Path, data_dir: Path, target_dir: Path, baseline_path: Path) -> dict:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _recipe(baseline_path)
    paths = sorted(data_dir.glob("DFL-MAT-*_half*.npz"))
    matches = sorted({path.name.rsplit("_half", 1)[0] for path in paths})
    outer = next(item for item in nested_splits(matches, inner_folds=3, seed=0)
                 if item["outer_test"] == ["DFL-MAT-J03WMX"])
    inner = outer["inner"][0]
    match = inner["validation"][0]
    input_dir = output_dir / "input-cache"
    input_cache_reused = (input_dir / "input-manifest.json").is_file()
    run_id = time.time_ns()
    cache = HalfCache(data_dir, max_bytes=1_000_000_000)
    timings: dict[str, float] = {}
    start = time.perf_counter()
    input_records = prepare_inputs(data_dir, target_dir, input_dir, [match], scenarios(), cache=cache)
    timings["prepare_inputs_seconds"] = time.perf_counter() - start
    clean_records = [record for record in input_records if record["scenario"] == "clean"]
    if len(input_records) != 8 or len(clean_records) != 2:
        raise RuntimeError("pilot input grid must contain eight scenario/half records")

    base_spec = baseline_spec(baseline)
    decoder_baseline = baseline_decoder({key: baseline[key]
                                         for key in ("ensemble", "calibration", "decoder")})
    sampled_full, sampled_reference = _decoder_samples(0)
    families = {
        "mean": {**base_spec["model_kwargs"], "pooling": "mean", "width": 32},
        "ball_attention": {**base_spec["model_kwargs"], "pooling": "ball_attention", "width": 64},
        "interaction": {**base_spec["model_kwargs"], "pooling": "interaction", "width": 64},
    }
    result: dict = {
        "status": "running", "purpose": "200-step compute pilot; no selection or performance claim",
        "data_dir": str(data_dir), "target_dir": str(target_dir),
        "baseline_path": str(baseline_path), "outer_match_excluded": "DFL-MAT-J03WMX",
        "inner_fold_index": 0, "train_matches": inner["train"],
        "validation_match_used": match, "input_records": 8,
        "pilot_run_id": run_id, "input_cache_reused": input_cache_reused,
        "scenarios": scenarios(), "seed": 7100, "fit_steps": 200,
        "target_possession_labels_opened": False,
        "thread_settings": {"torch_threads": torch.get_num_threads(), "torch_interop_threads": 1,
                            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS")},
        "models": {}, "timings": timings,
    }
    write_json(output_dir / "pilot-progress.json", result)
    for name, model_kwargs in families.items():
        spec = json.loads(json.dumps(base_spec))
        spec["model_kwargs"] = model_kwargs
        spec["schedule"] = {"kind": "constant", "warmup_steps": 0,
                            "min_lr_ratio": 1.0, "total_steps": TOTAL_STEPS}
        setup_start = time.perf_counter()
        session = build_session(data_dir, spec, inner["train"], seed=7100, cache=cache)
        setup_seconds = time.perf_counter() - setup_start
        fit_start = time.perf_counter()
        session.advance_to(200)
        fit_seconds = time.perf_counter() - fit_start
        parameter_count = sum(parameter.numel() for parameter in session.model.parameters())
        clean_input = next(record for record in clean_records if record["period"] == 1)
        with np.load(clean_input["input_path"], allow_pickle=False) as source:
            features = FeatureBatch(source["player_tokens"], source["player_mask"],
                                    source["global_tokens"], source["time_valid"])
        infer_start = time.perf_counter()
        predict_features(features, session.model, device="cpu", center_size=2048,
                         halo=session.model.radius)
        one_clean_half_seconds = time.perf_counter() - infer_start
        prediction_records = clean_records
        baseline_all_scenario_seconds = None
        decoder_timings = {}
        if name == "mean":
            predict_start = time.perf_counter()
            prediction_records = predict_records(
                session.model, input_records, output_dir / "predictions",
                identity={"pilot": "mean-200", "seed": 7100, "run_id": run_id},
                center_size=2048)
            baseline_all_scenario_seconds = time.perf_counter() - predict_start
            loaded_start = time.perf_counter()
            loaded = load_records(prediction_records)
            timings["load_records_seconds"] = time.perf_counter() - loaded_start
            for label, decoder in (("focused_baseline", decoder_baseline),
                                   ("random_seed0_full", sampled_full),
                                   ("random_seed0_reference_only", sampled_reference)):
                score_start = time.perf_counter()
                score_records(loaded, decoder, WEIGHTS)
                decoder_timings[label] = time.perf_counter() - score_start
        result["models"][name] = {
            "model_kwargs": model_kwargs, "parameter_count": int(parameter_count),
            "training_args": spec["training_args"], "schedule": spec["schedule"],
            "fit_steps": session.steps, "session_setup_seconds": setup_seconds,
            "fit_seconds_200_steps": fit_seconds,
            "estimated_seconds_7200_steps": fit_seconds * TOTAL_STEPS / 200,
            "one_clean_half_inference_seconds": one_clean_half_seconds,
            "all_eight_inference_seconds": baseline_all_scenario_seconds,
            "three_decoder_official_score_seconds": decoder_timings,
            "official_scoring_performed": bool(decoder_timings),
            "scoring_used_for_selection": False,
        }
        if name == "mean":
            result["sampled_decoders"] = {
                "full": sampled_full, "reference_only": sampled_reference,
                "seed": 0,
            }
        result["timings"] = timings
        result["peak_rss_bytes"] = _peak_rss_bytes()
        result["status"] = "running"
        write_json(output_dir / "pilot-progress.json", result)
        del session

    result["rough_screen_estimate"] = _project_staged_budget(result["models"])
    result["peak_rss_bytes"] = _peak_rss_bytes()
    result["timings"] = timings
    result["status"] = "complete"
    write_json(output_dir / "pilot-summary.json", result)
    return result


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/Volumes/T7/Data/more-than-a-hack-2027/artifacts/overnight-pilot-v1"))
    parser.add_argument("--data-dir", type=Path, default=repo / "data/processed/dfl")
    parser.add_argument("--target-dir", type=Path, default=repo / "data/target/2027")
    parser.add_argument("--baseline", type=Path,
                        default=repo / "artifacts/focused-temporal-v1/baseline-recipe.json")
    args = parser.parse_args()
    result = run(args.output_dir, args.data_dir, args.target_dir, args.baseline)
    print(json.dumps({"status": result["status"], "validation_match_used": result["validation_match_used"],
                      "models": {name: {"fit_seconds_200_steps": model["fit_seconds_200_steps"],
                                        "one_clean_half_inference_seconds": model["one_clean_half_inference_seconds"],
                                        "parameter_count": model["parameter_count"]}
                                 for name, model in result["models"].items()},
                      "staged_budget_estimate": result["rough_screen_estimate"]}, indent=2))


if __name__ == "__main__":
    main()
