"""Search-space recipes stay valid, canonical and conditionally sampled."""

import json
from pathlib import Path

import pytest

from experiments.campaign_space import POLICIES, baseline_spec, sample_spec
from possession.augmentation import AugmentationConfig
from possession.outages import OutageConfig


class Trial:
    def __init__(self, policy="none", outage="legacy", schedule="constant"):
        self.policy, self.outage, self.schedule = policy, outage, schedule
        self.calls = []

    def suggest_categorical(self, name, values):
        self.calls.append(name)
        chosen = {"augmentation_policy": self.policy, "outage_mode": self.outage,
                  "schedule_kind": self.schedule}.get(name, values[0])
        return chosen if chosen in values else values[0]

    def suggest_float(self, name, low, high, log=False):
        self.calls.append(name)
        return low


@pytest.mark.parametrize("policy", POLICIES)
def test_sampled_augmentation_policies_are_valid_and_only_sample_active_magnitudes(policy):
    trial = Trial(policy=policy)
    spec = sample_spec(trial, "mean")
    AugmentationConfig.from_dict(spec["augmentation_config"])
    active = set(spec["augmentation_config"]["operations"])
    magnitude_by_op = {
        "coordinate_noise": {"noise_player_std", "noise_ball_std", "noise_correlation"},
        "player_dropout": {"player_dropout_fraction"},
        "spatial_occlusion": {"patch_fraction"}, "translation": {"translation_max_meters"},
        "rotation": {"rotation_max_degrees"}, "time_warp": {"time_warp_max_rate", "time_warp_knots"},
        "time_shift": {"temporal_shift_max_seconds"}, "frame_dropout": {"frame_dropout_fraction"},
        "frame_hold": {"hold_max_frames"}, "temporal_blur": {"blur_radius"},
    }
    for operation, names in magnitude_by_op.items():
        assert (names & set(trial.calls)) == (names if operation in active else set())
    assert ("dropout_max_seconds" in trial.calls) == bool(
        active & {"player_dropout", "spatial_occlusion"})
    assert len(active) == len(set(active))
    assert json.loads(json.dumps(spec, sort_keys=True)) == spec


@pytest.mark.parametrize("pooling", ("mean", "ball_attention", "interaction"))
def test_sampled_model_and_correlated_outage_configs_are_valid(pooling):
    spec = sample_spec(Trial(policy="combined", outage="correlated", schedule="cosine"), pooling)
    assert spec["model_kwargs"]["pooling"] == pooling
    assert len(spec["model_kwargs"]["dilations"]) in (7, 8, 9)
    assert spec["schedule"]["kind"] == "cosine"
    assert OutageConfig(**spec["outage_config"]).mode == "correlated"


def test_legacy_outage_and_constant_schedule_have_no_inactive_sampled_knobs():
    trial = Trial(policy="none", outage="legacy", schedule="constant")
    spec = sample_spec(trial, "mean")
    assert "outage_clean_probability" not in trial.calls
    assert "warmup_steps" not in trial.calls and "min_lr_ratio" not in trial.calls
    assert spec["schedule"] == {"kind": "constant", "warmup_steps": 0,
                                "min_lr_ratio": 1.0, "total_steps": 7200}
    assert spec["training_args"]["boundary_weight"] == 0
    assert spec["training_args"]["boundary_pos_weight"] == 1
    assert "boundary_pos_weight" not in trial.calls


@pytest.mark.parametrize("policy", ("translation", "noise", "spatial_occlusion", "time_warp"))
def test_single_operation_policy_caps_transform_count(policy):
    trial = Trial(policy=policy)
    spec = sample_spec(trial, "mean")
    assert spec["augmentation_config"]["max_transforms"] == 1
    assert "augmentation_max_transforms" not in trial.calls


def test_spatial_occlusion_exposes_its_temporal_duration():
    trial = Trial(policy="spatial_occlusion")
    sample_spec(trial, "mean")
    assert "dropout_max_seconds" in trial.calls


def test_baseline_spec_keeps_frozen_training_values_and_constant_lr():
    baseline = {"selected_model_recipe": {
        "model_kwargs": {"width": 32, "pooling": "mean", "dilations": [1, 2]},
        "training_args": {"steps": 3600, "learning_rate": .001, "weight_decay": .0001,
                          "boundary_weight": .25, "boundary_pos_weight": 1.,
                          "center_size": 512, "augment": True}},
        "outage_config": {"mode": "legacy"}, "augmentation_config": {"mode": "none"}}
    spec = baseline_spec(baseline)
    assert {key: spec["model_kwargs"][key] for key in
            baseline["selected_model_recipe"]["model_kwargs"]} == baseline["selected_model_recipe"]["model_kwargs"]
    assert spec["training_args"]["learning_rate"] == .001
    assert spec["schedule"] == {"kind": "constant", "warmup_steps": 0,
                                "min_lr_ratio": 1.0, "total_steps": 3600}
    OutageConfig(**spec["outage_config"])
    AugmentationConfig.from_dict(spec["augmentation_config"])


def test_actual_focused_baseline_artifact_normalizes_to_campaign_spec():
    path = Path(__file__).resolve().parents[1] / "configs/focused-baseline.json"
    baseline = json.loads(path.read_text())
    spec = baseline_spec(baseline)
    assert spec["model_kwargs"] == baseline["model_kwargs"]
    assert spec["training_args"] == baseline["training_args"]
    assert spec["outage_config"] == OutageConfig(**baseline["outage_config"]).to_dict()
    assert spec["augmentation_config"] == AugmentationConfig.from_dict(
        baseline["augmentation_config"]).to_dict()
    assert spec["schedule"]["total_steps"] == baseline["steps"]


def test_rejects_unknown_pooling():
    with pytest.raises(ValueError, match="unsupported pooling"):
        sample_spec(Trial(), "mps")
