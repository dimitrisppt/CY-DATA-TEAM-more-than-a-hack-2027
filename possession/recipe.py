"""Resolve a saved selection recipe for evaluation and deployment."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from possession.calibrate import CalibrationConfig


@dataclass(frozen=True)
class InferenceConfig:
    switch_cost: float = 2.0
    boundary_scale: float = 0.0
    neural_weight: float = 1.0
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)


def load_frozen_recipe(path, checkpoint=None):
    """A reference-only winner can be deployed without a neural checkpoint."""
    path = Path(path)
    recipe = json.loads(path.read_text())
    required = {"decoder", "calibration", "ensemble", "selection_fingerprint",
                "selected_model_recipe"}
    if not isinstance(recipe, dict) or not required <= recipe.keys():
        raise ValueError("not a frozen model-selection recipe")
    weight = recipe["ensemble"].get("neural_weight")
    if weight is None or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("frozen recipe neural_weight must lie in [0,1]")
    if weight > 0 and checkpoint is None:
        checkpoint = path.parent / "selected.pt"
    return recipe, checkpoint


def resolve_config(training=None, *, switch_cost=None, boundary_scale=None,
                   neural_weight=None, calibration=None, has_model=True):
    """Explicit overrides take precedence; absent flags keep saved tuned values."""
    training = {} if training is None else training
    decoder = training.get("decoder", {})
    ensemble = training.get("ensemble", {})
    cost = decoder.get("switch_cost", 2.0) if switch_cost is None else switch_cost
    scale = decoder.get("boundary_scale", 0.0) if boundary_scale is None else boundary_scale
    weight = ensemble.get("neural_weight", 1.0) if neural_weight is None else neural_weight
    if not has_model:
        if neural_weight is not None and neural_weight != 0:
            raise ValueError("a nonzero neural_weight requires a checkpoint")
        weight = 0.0
    if not np.isfinite(cost) or not np.isfinite(scale) or scale < 0:
        raise ValueError("switch_cost must be finite and boundary_scale finite and nonnegative")
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("neural_weight must lie in [0,1]")
    cfg = (CalibrationConfig(**training.get("calibration", {}))
           if calibration is None else calibration)
    return InferenceConfig(float(cost), float(scale), float(weight), cfg)
