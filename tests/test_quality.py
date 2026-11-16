"""Quality inspection uses observations, not possession targets."""

import json

import numpy as np

from experiments.quality import audit_prepared, inspect_half
from possession.schema import FrameTable
from possession.storage import FRAME_ARRAYS


def test_input_only_cache_and_graded_run_gaps(tmp_path):
    frames = FrameTable(
        np.arange(8), np.zeros((8, 2, 1, 2)), np.ones((8, 2, 1), dtype=bool),
        np.zeros((8, 2)), np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=bool),
        np.array([1, 1, 1, 0, 0, 1, 1, 1], dtype=bool), match_id="a", fps=2.)
    arrays = {key: getattr(frames, key) for key in FRAME_ARRAYS}
    metadata = {"schema_version": 1, "fps": 2., "pitch": [105.3, 68.],
                "match_id": "a", "period": 1}
    np.savez(tmp_path / "a_half1.npz", metadata=json.dumps(metadata), **arrays)
    report = audit_prepared(tmp_path)
    assert report["graded_frames"] == 6
    assert report["ball_missing_frames"] == 4
    half = report["halves"][0]
    assert len(half["graded_spans"]) == 2
    assert sum(item["count"] for item in half["gap_histogram_seconds"]) == 2
    assert half["player_observation_fraction"] == 1.


def test_no_graded_frames_are_explicitly_unmeasured():
    frames = FrameTable(np.arange(1), np.zeros((1, 2, 0, 2)),
                        np.zeros((1, 2, 0), dtype=bool), np.zeros((1, 2)),
                        np.zeros(1, dtype=bool), np.zeros(1, dtype=bool))
    result = inspect_half(frames)
    assert result["ball_missing_fraction"] is None
    assert result["player_observation_fraction"] is None
    assert result["graded_spans"] == []
