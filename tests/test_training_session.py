"""Incremental training keeps one reproducible trajectory and bounded cache."""

import hashlib

import numpy as np
import pytest
import torch

from possession import training_session
from possession.outages import OutageConfig
from possession.predict import load_model_checkpoint
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half
from possession.training_session import HalfCache, TrainingSession


def _half(match_id, period=1, frames=12):
    frame_id = np.arange(100, 100 + frames)
    players = np.zeros((frames, 2, 2, 2), dtype=np.float32)
    players[:, 0, :, 0] = 20
    players[:, 1, :, 0] = 80
    players[:, :, :, 1] = 30
    states = np.resize(np.array([2, 0, 0, 1, 1, 2], dtype=np.int8), frames)
    live = (states != 2).astype(np.int8)
    return (
        FrameTable(frame_id, players, np.ones((frames, 2, 2), dtype=bool),
                   np.column_stack((25 + np.arange(frames), np.full(frames, 30))),
                   np.ones(frames, dtype=bool), np.ones(frames, dtype=bool),
                   match_id=match_id, period=period),
        TrainingTargets(live, np.ones(frames, dtype=bool),
                        (states == 0).astype(np.int8), live.astype(bool),
                        states, np.ones(frames, dtype=bool)),
    )


def _prepared(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    for match_id in ("a", "b", "c"):
        frames, targets = _half(match_id)
        save_half(data / f"{match_id}_half1.npz", frames, targets)
    return data


def _session(data, cache=None, outage_config=None):
    return TrainingSession(data, train_matches=["a", "b"], seed=23,
                           model_kwargs={"width": 8, "dilations": (1,)},
                           center_size=4, halo=1, augment=True, cache=cache,
                           outage_config=outage_config)


def test_staged_advance_is_identical_to_uninterrupted_training(tmp_path):
    data = _prepared(tmp_path)
    staged = _session(data)
    first = staged.advance_to(5)
    assert staged.steps == 5
    assert first["steps"] == 5
    staged.advance_to(10)
    direct = _session(data)
    direct_progress = direct.advance_to(10)
    assert staged.steps == direct.steps == 10
    assert staged.advance_to(10) == direct_progress
    assert len(direct_progress["training_window_sha256"]) == 64
    assert staged.config["train_matches"] == ["a", "b"]
    for name, tensor in staged.model.state_dict().items():
        torch.testing.assert_close(tensor, direct.model.state_dict()[name], rtol=0, atol=0)
    assert staged.advance_to(10)["steps"] == 10
    with pytest.raises(ValueError, match="backward"):
        staged.advance_to(9)


def test_correlated_staged_advance_and_config_checkpoint(tmp_path):
    data = _prepared(tmp_path)
    config = OutageConfig(mode="correlated", clean_probability=0,
                          full_probability=0, fraction_min=0.5, fraction_max=0.5,
                          min_gap_seconds=0.04, max_gap_seconds=0.08)
    staged = _session(data, outage_config=config.to_dict())
    staged.advance_to(4)
    progress = staged.advance_to(9)
    direct = _session(data, outage_config=config)
    assert progress == direct.advance_to(9)
    assert progress["center_ball_frames"] == 36
    assert 0 < progress["center_ball_missing_frames"] <= 36
    assert progress["center_ball_missing_fraction"] == (
        progress["center_ball_missing_frames"] / 36
    )
    for name, tensor in staged.model.state_dict().items():
        torch.testing.assert_close(tensor, direct.model.state_dict()[name], rtol=0, atol=0)
    path = tmp_path / "correlated.pt"
    staged.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    assert payload["training_config"]["outage_config"] == config.to_dict()


def test_policy_does_not_change_window_schedule(tmp_path, monkeypatch):
    data = _prepared(tmp_path)
    original = training_session._window
    windows = []

    def recorded(frames, targets, features, start, stop, *args, **kwargs):
        windows.append((frames.match_id, start, stop))
        return original(frames, targets, features, start, stop, *args, **kwargs)

    monkeypatch.setattr(training_session, "_window", recorded)
    legacy = _session(data, outage_config=OutageConfig(mode="legacy"))
    legacy_progress = legacy.advance_to(10)
    first = windows.copy()
    expected = hashlib.sha256()
    for match_id, start, stop in first:
        expected.update(f"{match_id}_half1.npz:{start}:{stop - start}\n".encode())
    assert legacy_progress["training_window_sha256"] == expected.hexdigest()
    windows.clear()
    correlated = _session(data, outage_config=OutageConfig(mode="correlated"))
    correlated_progress = correlated.advance_to(10)
    assert first == windows
    assert correlated_progress["training_window_sha256"] == legacy_progress["training_window_sha256"]


def test_checkpoint_resolves_and_proves_split_without_training_labels_as_inputs(tmp_path):
    data = _prepared(tmp_path)
    session = TrainingSession(data, train_matches=["a"], seed=3,
                              model_kwargs={"width": 8, "dilations": (1,)},
                              center_size=4, halo=1, augment=False)
    session.advance_to(2)
    path = tmp_path / "fit" / "last.pt"
    session.save_checkpoint(path, validation_matches=["b"],
                            extra_training_config={"trial_number": 7})
    model, payload = load_model_checkpoint(path)
    assert model is not None
    assert payload["training_config"]["train_matches"] == ["a"]
    assert payload["training_config"]["validation_matches"] == ["b"]
    assert payload["training_config"]["trial_number"] == 7
    assert set(payload["prepared_sha256"]) == {"a_half1.npz", "b_half1.npz"}
    assert payload["dataset_provenance"]["status"] == "unmanifested_local_cache"
    assert len(payload["prepared_inputs_hash"]) == 64
    with pytest.raises(ValueError, match="disjoint"):
        session.save_checkpoint(path, validation_matches=["a"])
    with pytest.raises(ValueError, match="reserved"):
        session.save_checkpoint(path, extra_training_config={"train_matches": ["c"]})


def test_half_cache_bounds_lru_and_shared_hits(tmp_path):
    data = _prepared(tmp_path)
    generous = HalfCache(data, max_bytes=10_000_000)
    a = generous.get("a_half1.npz")
    size = generous.bytes
    assert size > 0
    assert generous.get(data / "a_half1.npz") is a
    assert generous.hits == 1 and generous.misses == 1
    bounded = HalfCache(data, max_bytes=size + 1)
    bounded.get("a_half1.npz")
    bounded.get("b_half1.npz")
    assert bounded.bytes <= bounded.max_bytes
    assert bounded.peak_bytes <= bounded.max_bytes
    assert bounded.misses == 2
    bounded.get("a_half1.npz")
    assert bounded.misses == 3
    shared = HalfCache(data, max_bytes=10_000_000)
    one = _session(data, cache=shared)
    two = _session(data, cache=shared)
    one.advance_to(4)
    hits_before = shared.hits
    two.advance_to(4)
    assert shared.hits > hits_before
