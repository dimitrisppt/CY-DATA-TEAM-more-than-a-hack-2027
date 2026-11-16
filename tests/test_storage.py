import numpy as np
import pytest

from possession.schema import FrameTable, TrainingTargets
from possession.storage import load_half, save_half


def example_half(n=20, match_id="example", period=1):
    frames = FrameTable(
        np.arange(100, 100 + n), np.zeros((n, 2, 2, 2)),
        np.ones((n, 2, 2), bool), np.zeros((n, 2)), np.ones(n, bool),
        np.ones(n, bool), match_id=match_id, period=period,
    )
    state = np.resize([0, 1, 2], n).astype(np.int8)
    targets = TrainingTargets(state != 2, np.ones(n, bool), state == 0,
                              state != 2, state, np.ones(n, bool))
    return frames, targets


def test_cache_roundtrip_without_pickle(tmp_path):
    frames, targets = example_half()
    path = tmp_path / "nested" / "example_half1.npz"
    save_half(path, frames, targets)
    with np.load(path, allow_pickle=False) as data:
        assert all(data[name].dtype.kind != "O" for name in data.files)
    restored, labels = load_half(path)
    assert restored.match_id == "example"
    assert restored.pitch == frames.pitch
    assert restored.period == 1
    np.testing.assert_array_equal(restored.frame_id, frames.frame_id)
    np.testing.assert_array_equal(labels.state, targets.state)
    np.testing.assert_array_equal(restored.players_xy, frames.players_xy)


def test_reject_cache_with_wrong_target_length(tmp_path):
    frames, targets = example_half()
    with pytest.raises(ValueError, match="length"):
        save_half(tmp_path / "bad.npz", frames, targets.subset(0, 2))
    assert not (tmp_path / "bad.npz").exists()


def test_reject_unsupported_schema(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, metadata='{"schema_version": 99}')
    with pytest.raises(ValueError, match="schema"):
        load_half(path)
