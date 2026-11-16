"""Bounded training uses prepared halves without crossing split or window edges."""

import json

import numpy as np
import pytest
import torch

from possession.features import build_features
from possession.model import PossessionModel
from possession.outages import OutageConfig
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half
from possession.train import _loss_targets_for_window, _selected_features, train_model
from possession.training_session import TrainingSession


def sample_half(match_id: str, frames: int = 10) -> tuple[FrameTable, TrainingTargets]:
    time = np.arange(frames)
    players = np.zeros((frames, 2, 2, 2), dtype=np.float32)
    players[:, 0, 0] = [20, 30]
    players[:, 0, 1] = [35, 30]
    players[:, 1, 0] = [70, 30]
    players[:, 1, 1] = [85, 30]
    states = np.resize(np.array([2, 0, 0, 1, 1, 2], dtype=np.int8), frames)
    live = (states != 2).astype(np.int8)
    owner = (states == 0).astype(np.int8)
    frame_table = FrameTable(
        frame_id=100 + time,
        players_xy=players,
        player_valid=np.ones((frames, 2, 2), dtype=bool),
        ball_xy=np.column_stack((25 + time, np.full(frames, 30))),
        ball_valid=np.ones(frames, dtype=bool),
        graded=np.ones(frames, dtype=bool),
        match_id=match_id,
        period=1,
    )
    targets = TrainingTargets(
        live=live,
        live_valid=np.ones(frames, dtype=bool),
        owner=owner,
        owner_valid=live.astype(bool),
        state=states,
        state_valid=np.ones(frames, dtype=bool),
    )
    return frame_table, targets


def test_window_masks_supervise_center_and_its_incoming_boundary_only():
    frames, targets = sample_half("a", 8)
    features = build_features(frames)
    loss_targets = _loss_targets_for_window(targets, features.time_valid, frames.frame_id,
                                            left=1, right=7, center_start=3, center_stop=5)
    assert loss_targets["live_valid"].tolist() == [[0, 0, 1, 1, 0, 0]]
    assert loss_targets["owner_valid"].tolist() == [[0, 0, 1, 1, 0, 0]]
    assert loss_targets["state_valid"].tolist() == [[0, 1, 1, 1, 0, 0]]
    # State at t=2 is context; its only supervised transition is into t=3.
    assert loss_targets["frame_id"].tolist() == [[101, 102, 103, 104, 105, 106]]


def test_train_rejects_overlapping_and_missing_match_splits(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    frames, targets = sample_half("a")
    save_half(data / "a_half1.npz", frames, targets)
    with pytest.raises(ValueError, match="disjoint"):
        train_model(data, tmp_path / "out", train_matches=["a"], validation_matches=["a"],
                    steps=1, width=8, dilations=(1,), center_size=4, halo=1)
    with pytest.raises(ValueError, match="missing"):
        train_model(data, tmp_path / "out", train_matches=["absent"], steps=1,
                    width=8, dilations=(1,), center_size=4, halo=1)


def test_short_match_ids_resolve_before_split_validation(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    frames, targets = sample_half("DFL-MAT-J03WMX")
    save_half(data / "DFL-MAT-J03WMX_half1.npz", frames, targets)
    with pytest.raises(ValueError, match="disjoint"):
        train_model(data, tmp_path / "out", train_matches=["J03WMX"],
                    validation_matches=["DFL-MAT-J03WMX"], steps=1,
                    width=8, dilations=(1,), center_size=4, halo=1)
    report = train_model(data, tmp_path / "out", train_matches=["J03WMX"], steps=1,
                         width=8, dilations=(1,), center_size=4, halo=1, augment=False)
    assert report["train_matches"] == ["DFL-MAT-J03WMX"]


def test_tiny_fit_writes_reloadable_checkpoint_and_holdout_report(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    for match_id in ("a", "b"):
        frames, targets = sample_half(match_id)
        save_half(data / f"{match_id}_half1.npz", frames, targets)
    output = tmp_path / "fit"
    report = train_model(data, output, train_matches=["a"], validation_matches=["b"],
                         steps=4, seed=9, width=8, dilations=(1,), center_size=4, halo=1,
                         learning_rate=0.02, augment=False)
    assert report == json.loads((output / "training-report.json").read_text())
    assert report["train_matches"] == ["a"]
    assert report["validation_matches"] == ["b"]
    assert report["steps"] == 4
    assert report["train_counts"]["live"] > 0
    assert report["train_counts"]["boundary"] > 0
    assert report["train_excluded"]["live"] == 0
    assert np.isfinite(report["validation_loss"])
    assert len(report["checkpoint_sha256"]) == 64
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["model_kwargs"] == {
        "player_dim": 8, "global_dim": 9, "width": 8, "dilations": [1],
    }
    assert checkpoint["training_config"]["train_matches"] == ["a"]
    model = PossessionModel(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    features = build_features(sample_half("a")[0])
    with torch.inference_mode():
        scores = model(torch.from_numpy(features.player_tokens)[None],
                       torch.from_numpy(features.player_mask)[None],
                       torch.from_numpy(features.global_tokens)[None])
    assert all(torch.isfinite(value).all() for value in scores.values())
    assert set(checkpoint["prepared_sha256"]) == {"a_half1.npz", "b_half1.npz"}


def test_ball_outage_is_contiguous_and_recomputed_without_mutating_source():
    frames, _ = sample_half("a", 10)
    original = build_features(frames)
    player, _, context, outage = _selected_features(
        frames, original, left=0, right=10, center_start=2, center_stop=8,
        rng=np.random.default_rng(2), augment=True,
    )
    assert outage
    missing = np.flatnonzero(context[:, 2] == 0)
    assert len(missing) >= 4
    np.testing.assert_array_equal(missing, np.arange(missing[0], missing[-1] + 1))
    assert not player[missing, ..., 7].any()
    assert frames.ball_valid.all()
    assert original.global_tokens[:, 2].all()


def test_training_is_seeded_and_uses_each_declared_half(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    for match_id in ("a", "b"):
        frames, targets = sample_half(match_id)
        save_half(data / f"{match_id}_half1.npz", frames, targets)
    arguments = {"train_matches": ["a", "b"], "steps": 3, "seed": 17, "width": 8,
                 "dilations": (1,), "center_size": 4, "halo": 1, "augment": True}
    first = train_model(data, tmp_path / "first", **arguments)
    second = train_model(data, tmp_path / "second", **arguments)
    assert set(first["trained_halves"]) == {"a_half1.npz", "b_half1.npz"}
    assert first["augmentation_counts"] == second["augmentation_counts"]
    first_weights = torch.load(tmp_path / "first" / "last.pt", weights_only=True)["state_dict"]
    second_weights = torch.load(tmp_path / "second" / "last.pt", weights_only=True)["state_dict"]
    for name in first_weights:
        torch.testing.assert_close(first_weights[name], second_weights[name], rtol=0, atol=0)


def test_training_revisits_halves_in_bounded_shuffled_rounds(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    for match_id in ("a", "b"):
        frames, targets = sample_half(match_id)
        save_half(data / f"{match_id}_half1.npz", frames, targets)
    report = train_model(data, tmp_path / "out", train_matches=["a", "b"], steps=12,
                         seed=5, width=8, dilations=(1,), center_size=4, halo=1,
                         augment=False)
    blocks = report["training_schedule_blocks"]
    assert sum(block["steps"] for block in blocks) == 12
    assert max(block["steps"] for block in blocks) <= 4
    assert blocks[0]["half"] != blocks[1]["half"]
    assert len(blocks) == 3


@pytest.mark.parametrize(
    ("outage_config", "pooling"),
    [
        (OutageConfig(mode="legacy"), "mean"),
        (OutageConfig(mode="correlated", clean_probability=0,
                      full_probability=0.4, fraction_min=0.5, fraction_max=0.5),
         "ball_attention"),
    ],
)
def test_train_model_matches_staged_training_session(tmp_path, outage_config, pooling):
    data = tmp_path / "prepared"
    data.mkdir()
    for match_id in ("a", "b", "c"):
        frames, targets = sample_half(match_id, 12)
        save_half(data / f"{match_id}_half1.npz", frames, targets)
    kwargs = {
        "train_matches": ["a", "b"], "seed": 31, "device": "cpu",
        "center_size": 4, "halo": 1, "learning_rate": 0.02,
        "weight_decay": 0.003, "boundary_weight": 0.7,
        "boundary_pos_weight": 3.0, "augment": True,
        "outage_config": outage_config,
    }
    model_kwargs = {"width": 8, "dilations": (1,)}
    if pooling != "mean":
        model_kwargs["pooling"] = pooling
    staged = TrainingSession(
        data, model_kwargs=model_kwargs, schedule_block_size=3, **kwargs,
    )
    staged.advance_to(2)
    progress = staged.advance_to(6)
    report = train_model(
        data, tmp_path / "fit", validation_matches=["c"], steps=6,
        width=8, dilations=(1,), pooling=pooling, **kwargs,
    )
    checkpoint = torch.load(tmp_path / "fit" / "last.pt", weights_only=True)
    for name, tensor in staged.model.state_dict().items():
        torch.testing.assert_close(tensor, checkpoint["state_dict"][name], rtol=0, atol=0)
    assert report["training_window_sha256"] == progress["training_window_sha256"]
    assert report["training_schedule_blocks"] == progress["training_schedule_blocks"]
    assert report["train_counts"] == progress["train_counts"]
    assert report["train_excluded"] == {
        head: progress["train_possible"][head] - progress["train_counts"][head]
        for head in progress["train_counts"]
    }
    assert report["train_losses"] == progress["train_losses"]
    assert report["boundary_positive_count"] == progress["boundary_positive_count"]
    assert report["center_ball_missing_frames"] == progress["center_ball_missing_frames"]
    assert report["sampled_training_loss_trace"][-1] == progress["last_step_losses"]
    assert checkpoint["training_config"]["weight_decay"] == 0.003
    assert checkpoint["training_config"]["boundary_pos_weight"] == 3.0
    assert checkpoint["model_kwargs"].get("pooling", "mean") == pooling


def test_validation_does_not_change_fit_weights_or_inputs(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    for match_id in ("a", "b"):
        frames, targets = sample_half(match_id, 12)
        save_half(data / f"{match_id}_half1.npz", frames, targets)
    kwargs = {"train_matches": ["a"], "steps": 5, "seed": 37,
              "width": 8, "dilations": (1,), "center_size": 4,
              "halo": 1, "augment": True}
    with_validation = train_model(data, tmp_path / "with", validation_matches=["b"], **kwargs)
    without_validation = train_model(data, tmp_path / "without", **kwargs)
    assert with_validation["validation_loss"] is not None
    assert without_validation["validation_loss"] is None
    for key in ("training_window_sha256", "train_counts", "train_losses",
                "augmentation_counts"):
        assert with_validation[key] == without_validation[key]
    with_weights = torch.load(tmp_path / "with" / "last.pt", weights_only=True)["state_dict"]
    without_weights = torch.load(tmp_path / "without" / "last.pt", weights_only=True)["state_dict"]
    for name in with_weights:
        torch.testing.assert_close(with_weights[name], without_weights[name], rtol=0, atol=0)
