"""Bounded, half-local evaluation of a trained augmentation candidate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from experiments.evaluate import _join_with_x, _target_codes
from experiments.prediction_store import save_prediction_bundle
from experiments.robustness import SCENARIOS, _decode, _mask
from experiments.select import _prepared_halves
from possession.diagnostics import merge_spell_diagnostics, spell_error_diagnostics
from possession.features import build_features
from possession.predict import predict_features
from possession.reference import reference_scores

_SCORE_FIELDS = (
    "spell_f1_mean", "macro_f1", "accuracy", "alive_dead_f1",
    "n_frames", "n_graded", "n_gt_spells", "n_pred_spells", "fragmentation",
)
_TRANSITION_FIELDS = (
    "precision", "recall", "f1", "n_gt", "n_pred", "n_matched", "n_missed",
    "n_spurious", "n_ambiguous", "mean_latency", "median_latency",
    "mean_abs_latency", "p90_abs_latency",
)


def _compact(metrics: dict) -> dict:
    """Retain actionable official scores without per-boundary match lists."""
    return {
        **{name: metrics[name] for name in _SCORE_FIELDS},
        "transitions": {
            str(tolerance): {name: scores[name] for name in _TRANSITION_FIELDS}
            for tolerance, scores in metrics["transitions"].items()
        },
    }


def evaluate_augmented_model(
    data_dir,
    match,
    model,
    frozen,
    cache,
    *,
    scenarios=("clean", "block41_seed3101", "block41_seed3102"),
    inference_center_size=2048,
    include_diagnostics=False,
    prediction_dir=None,
) -> dict:
    """Score one model on original labels with clean or deterministically masked inputs.

    The caller supplies the frozen decoder and a bounded HalfCache. Features and
    reference scores are built afresh for one altered half at a time; inference
    uses the model's real context radius and never applies training augmentation.
    Optional stored predictions belong with the caller's fit result, which binds
    the checkpoint and protocol; their own record verifies arrays and half identity.
    """
    if not isinstance(match, str) or not match:
        raise ValueError("match must be a nonempty string")
    scenarios = tuple(scenarios)
    if not scenarios or len(scenarios) != len(set(scenarios)):
        raise ValueError("scenarios must be nonempty and unique")
    unknown = set(scenarios) - SCENARIOS.keys()
    if unknown:
        raise ValueError(f"unknown evaluation scenarios: {sorted(unknown)}")
    if (not isinstance(inference_center_size, int) or isinstance(inference_center_size, bool)
            or inference_center_size < 1):
        raise ValueError("inference_center_size must be a positive integer")
    if prediction_dir is not None:
        if frozen.get("decoder", {}).get("boundary_scale") != 0:
            raise ValueError("stored predictions support boundary_scale == 0 only")
        prediction_dir = Path(prediction_dir).resolve()

    paths = _prepared_halves(Path(data_dir), match)
    results = {}
    for scenario in scenarios:
        gt_parts = []
        pred_parts = {diagnostic: [] for diagnostic in ("blend", "neural_only")}
        diagnostic_parts = {diagnostic: [] for diagnostic in pred_parts} if include_diagnostics else None
        masks = {}
        prediction_records = {} if prediction_dir is not None else None
        for period, path in enumerate(paths, 1):
            frames, targets, _, _ = cache.get(path)
            if frames.match_id != match or frames.period != period or frames.fps != official.FPS:
                raise ValueError("prepared half identity or FPS differs")

            gt = _target_codes(frames, targets)
            altered, evidence = _mask(frames, match, period, scenario)
            features = build_features(altered)
            reference = reference_scores(altered)
            neural = predict_features(
                features, model, device="cpu", center_size=inference_center_size,
                halo=model.radius,
            )
            if len(neural["live_probability"]) != len(gt):
                raise ValueError("neural prediction length differs from source timeline")
            if prediction_records is not None:
                prediction_records[str(period)] = save_prediction_bundle(
                    prediction_dir / scenario / f"half{period}.npz", neural=neural,
                    reference=reference, gt=gt, ball_valid=altered.ball_valid,
                    nearest_ball_seconds=10.0 * np.minimum(
                        features.global_tokens[:, 6], features.global_tokens[:, 7]).astype(np.float64),
                    frame_id=altered.frame_id, match=match, period=period, scenario=scenario)
            for diagnostic, predictions in pred_parts.items():
                pred = _decode(neural, reference, frozen,
                               neural_only=diagnostic == "neural_only")
                if len(pred) != len(gt):
                    raise ValueError("decoded prediction length differs from source timeline")
                predictions.append(pred)
                if diagnostic_parts is not None:
                    diagnostic_parts[diagnostic].append(spell_error_diagnostics(
                        gt, pred, altered.ball_valid, fps=frames.fps))
            gt_parts.append(gt)
            masks[str(period)] = evidence

        joined_gt = _join_with_x(gt_parts)
        scores = {
            diagnostic: _compact(official.evaluate(
                joined_gt, _join_with_x(parts), bootstrap_n=0))
            for diagnostic, parts in pred_parts.items()
        }
        results[scenario] = {"scores": scores, "masks": masks}
        if diagnostic_parts is not None:
            results[scenario]["diagnostics"] = {
                diagnostic: merge_spell_diagnostics(parts)
                for diagnostic, parts in diagnostic_parts.items()}
        if prediction_records is not None:
            results[scenario]["predictions"] = prediction_records
    return results
