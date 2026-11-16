"""Campaign wrappers preserve TrainingSession behavior and exact restarts."""

import numpy as np
import pytest
import torch

from experiments.campaign_training import build_session, load_training_state, save_training_state
from possession.augmentation import AugmentationConfig
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half
from possession.training_session import TrainingSession


def _half(match, n=80):
    t = np.arange(n)
    players = np.zeros((n, 2, 2, 2), np.float32)
    players[:, 0, :, 0] = 20 + t[:, None] * .05
    players[:, 1, :, 0] = 70 - t[:, None] * .05
    players[..., 1] = 30
    state = np.resize(np.repeat([0, 2, 1], [9, 3, 8]), n).astype(np.int8)
    live = (state != 2).astype(np.int8)
    frames = FrameTable(t, players, np.ones((n, 2, 2), bool),
                        np.column_stack((25 + t * .1, np.full(n, 31))),
                        np.ones(n, bool), np.ones(n, bool), match_id=match)
    targets = TrainingTargets(live, np.ones(n, bool), (state == 0).astype(np.int8),
                              live.astype(bool), state, np.ones(n, bool))
    return frames, targets


def _data(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    for match in ("a", "b", "c"):
        save_half(data / f"{match}_half1.npz", *_half(match))
    return data


def _spec(total=8):
    return {
        "model_kwargs": {"player_dim": 8, "global_dim": 9, "width": 8,
                         "pooling": "mean", "dilations": [1]},
        "training_args": {"learning_rate": .003, "weight_decay": .0001,
                          "boundary_weight": .25, "boundary_pos_weight": 2.,
                          "center_size": 16, "augment": True},
        "outage_config": {"mode": "legacy"},
        "augmentation_config": AugmentationConfig(
            mode="combined", operations=("time_shift", "coordinate_noise"),
            clean_probability=.2, strength=.5, warmup_steps=0).to_dict(),
        "schedule": {"kind": "cosine", "warmup_steps": 2,
                     "min_lr_ratio": .1, "total_steps": total},
    }


def test_constant_campaign_wrapper_matches_ordinary_training_session(tmp_path):
    data = _data(tmp_path)
    spec = _spec()
    spec["schedule"] = {"kind": "constant", "warmup_steps": 0,
                        "min_lr_ratio": 1., "total_steps": 8}
    actual = build_session(data, spec, ["a", "b"], 16)
    expected = TrainingSession(
        data, train_matches=["a", "b"], seed=16, model_kwargs=spec["model_kwargs"],
        learning_rate=.003, weight_decay=.0001, boundary_weight=.25,
        boundary_pos_weight=2., center_size=16, augment=True,
        outage_config=spec["outage_config"], augmentation_config=spec["augmentation_config"])
    assert actual.advance_to(5)["training_window_sha256"] == expected.advance_to(5)["training_window_sha256"]
    for key, value in expected.model.state_dict().items():
        torch.testing.assert_close(value, actual.model.state_dict()[key], rtol=0, atol=0)


def test_prediction_checkpoint_records_campaign_schedule(tmp_path):
    data = _data(tmp_path)
    spec = _spec()
    session = build_session(data, spec, ["a", "b"], 16)
    path = session.save_checkpoint(tmp_path / "model.pt", validation_matches=["c"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["training_config"]["campaign_spec"] == spec
    assert checkpoint["training_config"]["campaign_spec"]["schedule"] == spec["schedule"]


def test_resumed_cosine_trajectory_matches_uninterrupted_weights_optimizer_and_windows(tmp_path):
    data = _data(tmp_path)
    spec, identity = _spec(), {"campaign": "test", "source_sha256": "abc"}
    whole = build_session(data, spec, ["a", "b"], 16)
    expected = whole.advance_to(8)
    staged = build_session(data, spec, ["a", "b"], 16)
    staged.advance_to(4)
    path = save_training_state(tmp_path / "states" / "state.pt", staged, identity)
    resumed = build_session(data, spec, ["a", "b"], 16)
    assert load_training_state(path, resumed, identity)["steps"] == 4
    actual = resumed.advance_to(8)
    assert actual == expected
    for key, value in whole.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    expected_optimizer, actual_optimizer = whole.optimizer.state_dict(), resumed.optimizer.state_dict()
    assert expected_optimizer["param_groups"] == actual_optimizer["param_groups"]
    assert expected_optimizer["state"].keys() == actual_optimizer["state"].keys()
    for index, values in expected_optimizer["state"].items():
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, actual_optimizer["state"][index][key],
                                           rtol=0, atol=0)
            else:
                assert value == actual_optimizer["state"][index][key]


def test_state_rejects_changed_identity_before_session_mutation(tmp_path):
    data = _data(tmp_path)
    spec, identity = _spec(), {"source_sha256": "first"}
    source = build_session(data, spec, ["a", "b"], 16)
    source.advance_to(2)
    path = save_training_state(tmp_path / "state.pt", source, identity)
    resumed = build_session(data, spec, ["a", "b"], 16)
    before = {key: value.clone() for key, value in resumed.model.state_dict().items()}
    with pytest.raises(ValueError, match="identity/config/data"):
        load_training_state(path, resumed, {"source_sha256": "changed"})
    assert resumed.steps == 0
    for key, value in before.items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)


def test_state_rejects_tampering(tmp_path):
    data = _data(tmp_path)
    session = build_session(data, _spec(), ["a", "b"], 16)
    session.advance_to(1)
    path = save_training_state(tmp_path / "state.pt", session, {"x": 1})
    path.write_bytes(path.read_bytes()[:-5] + b"abcde")
    other = build_session(data, _spec(), ["a", "b"], 16)
    with pytest.raises(ValueError, match="SHA256"):
        load_training_state(path, other, {"x": 1})
