"""Numerical/RNG equivalence and real CUDA checkpoint continuation checks."""

import copy

import numpy as np
import pytest
import torch

from experiments import campaign_training as campaign
from experiments import gpu_runtime as gpu
from experiments.ema_training import _tree_equal
from experiments.refinement_training import (
    configure_refinement,
    load_refinement_state,
    save_refinement_state,
)
from possession.augmentation import AugmentationConfig, _coordinate_noise
from possession.features import build_features
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half


def _half(n=113, players=3):
    rng = np.random.default_rng(13)
    xy = np.cumsum(rng.normal(0, .05, (n, 2, players, 2)), axis=0).astype(np.float32) + 30
    ball = np.cumsum(rng.normal(0, .1, (n, 2)), axis=0).astype(np.float32) + 30
    return FrameTable(np.arange(n), xy, rng.random((n, 2, players)) > .15,
                      ball, rng.random(n) > .4, rng.random(n) > .1, match_id="train")


@pytest.mark.parametrize("players", [0, 1, 3, 11])
@pytest.mark.parametrize("visibility", ["mixed", "none", "all"])
def test_features_exact_with_masked_observations(players, visibility):
    frames = _half(players=players)
    if visibility != "mixed":
        frames.ball_valid[:] = visibility == "all"
    expected, actual = build_features(frames), gpu.build_features(frames)
    for key in vars(expected):
        np.testing.assert_array_equal(getattr(expected, key), getattr(actual, key))


@pytest.mark.parametrize("rho", [0., .5, .95, 1.])
@pytest.mark.parametrize("strength", [0., .1, 1.])
def test_noise_preserves_values_and_generator_state(rho, strength):
    frames = _half()
    original, accelerated = copy.deepcopy(frames), copy.deepcopy(frames)
    first, second = np.random.default_rng(739), np.random.default_rng(739)
    config = AugmentationConfig(noise_correlation=rho)
    assert _coordinate_noise(original, first, config, strength) == gpu.coordinate_noise(
        accelerated, second, config, strength)
    np.testing.assert_array_equal(original.players_xy, accelerated.players_xy)
    np.testing.assert_array_equal(original.ball_xy, accelerated.ball_xy)
    assert first.bit_generator.state == second.bit_generator.state


def _data_spec(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    frames = _half()
    state = np.resize(np.repeat([0, 2, 1], [9, 3, 8]), len(frames.frame_id)).astype(np.int8)
    live = (state != 2).astype(np.int8)
    targets = TrainingTargets(live, frames.graded, (state == 0).astype(np.int8),
                              live.astype(bool) & frames.graded, state, frames.graded)
    save_half(data / "train_half1.npz", frames, targets)
    return data, {
        "model_kwargs": {"width": 8, "dilations": [1], "pooling": "mean"},
        "training_args": {"learning_rate": .001, "weight_decay": .0001,
                          "boundary_weight": .25, "boundary_pos_weight": 2.,
                          "center_size": 32, "augment": True},
        "augmentation_config": AugmentationConfig(
            mode="spatial", operations=("coordinate_noise",), warmup_steps=0).to_dict(),
        "outage_config": {"mode": "legacy"},
        "schedule": {"kind": "constant", "total_steps": 8,
                     "warmup_steps": 0, "min_lr_ratio": 1.},
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_training_preprocessing_parity_and_exact_resume(tmp_path):
    device = gpu.configure_device()
    data, spec = _data_spec(tmp_path)
    original = gpu.build_session(data, spec, ["train"], 91, device=device)
    original.advance_to(6)
    with gpu.accelerated_preprocessing():
        staged = gpu.build_session(data, spec, ["train"], 91, device=device)
        assert next(staged.model.parameters()).is_cuda
        staged.advance_to(3)
        campaign.save_training_state(tmp_path / "state.pt", staged, {"test": 1})
        resumed = gpu.build_session(data, spec, ["train"], 91, device=device)
        campaign.load_training_state(tmp_path / "state.pt", resumed, {"test": 1})
        resumed.advance_to(6)
    assert _tree_equal(original.model.state_dict(), resumed.model.state_dict())
    assert _tree_equal(original.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert original._progress() == resumed._progress()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("variant", ["control", "spell_balanced", "masked_teacher"])
def test_cuda_refinement_resume_preserves_optimizer_rng_and_ema(tmp_path, variant):
    device = gpu.configure_device()
    data, spec = _data_spec(tmp_path)
    with gpu.accelerated_preprocessing():
        parent = gpu.build_session(data, spec, ["train"], 13, device=device)
        parent.advance_to(2)
        campaign.save_training_state(tmp_path / "parent.pt", parent, {"parent": 1})
        teacher = copy.deepcopy(parent.model).eval() if variant == "masked_teacher" else None

        def receiver():
            session = gpu.build_session(data, spec, ["train"], 13, device=device)
            campaign.load_training_state(tmp_path / "parent.pt", session, {"parent": 1})
            return configure_refinement(session, variant=variant, teacher=teacher,
                                         start_steps=2, total_extra_steps=4)

        whole, staged = receiver(), receiver()
        whole.advance_to(6)
        staged.advance_to(4)
        save_refinement_state(tmp_path / "child.pt", staged, {"child": variant})
        resumed = receiver()
        load_refinement_state(tmp_path / "child.pt", resumed, {"child": variant})
        resumed.advance_to(6)
    assert _tree_equal(whole.model.state_dict(), resumed.model.state_dict())
    assert _tree_equal(whole.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert _tree_equal(whole._tail_ema, resumed._tail_ema)
    assert whole._progress() == resumed._progress()
