"""Synthetic-only tests for the public timing-cache access boundary."""

import json

import numpy as np
import pytest

from experiments.campaign_evaluation import PREDICTION_SCHEMA
from experiments.timing_cache import _check_hash, _fine_arrays, _input_arrays, _public_path
from possession.storage import sha256_file


def test_compact_reader_never_materializes_player_features(tmp_path):
    path = tmp_path / "public.npz"
    np.savez(
        path,
        frame_id=np.arange(10, 13),
        gt=np.array([0, 1, 2]),
        ball_valid=np.ones(3, dtype=bool),
        nearest_ball_seconds=np.zeros(3),
        ref_fps=np.array(25.0),
        player_tokens=np.array([object()], dtype=object),
    )
    arrays = _input_arrays({"input_path": str(path)})
    assert "player_tokens" not in arrays
    assert set(arrays) == {"frame_id", "gt", "ball_valid", "nearest_ball_seconds", "ref_fps"}


def test_compact_reader_rejects_noncontiguous_frame_ids(tmp_path):
    path = tmp_path / "public.npz"
    np.savez(
        path,
        frame_id=np.array([1, 3]),
        gt=np.zeros(2),
        ball_valid=np.ones(2, dtype=bool),
        nearest_ball_seconds=np.zeros(2),
    )
    with pytest.raises(ValueError, match="timeline"):
        _input_arrays({"input_path": str(path)})


def test_public_path_rejects_escape_and_symlink(tmp_path):
    root = tmp_path / "public"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("synthetic")
    (root / "link").symlink_to(outside)
    for path in (outside, root / "link"):
        with pytest.raises(ValueError, match="outside"):
            _public_path(path, root)


def test_checksum_rejects_changed_evidence(tmp_path):
    path = tmp_path / "evidence"
    path.write_text("before")
    checksum = sha256_file(path)
    path.write_text("after")
    with pytest.raises(ValueError, match="checksum"):
        _check_hash(path, checksum)


def test_fine_prediction_checks_ancestry_sidecar_and_hash(tmp_path):
    directory = tmp_path / "ema-stability-v1" / "predictions"
    directory.mkdir(parents=True)
    path = directory / "prediction.npz"
    values = {
        "neural_live": np.ones(2),
        "neural_home": np.zeros(2),
        "neural_boundary": np.zeros((2, 3, 3)),
    }
    np.savez(path, **values)
    source = {
        "input_path": str(tmp_path / "public.npz"),
        "input_sha256": "a" * 64,
        "input_identity": "b" * 64,
        "match": "DFL-MAT-SYNTHETIC",
        "period": 1,
        "scenario": "clean",
    }
    identity = {"train_matches": ["DFL-MAT-TRAIN"]}
    record = {
        **source,
        "model_identity": identity,
        "prediction_path": str(path),
        "prediction_sha256": sha256_file(path),
        "center_size": 2048,
    }
    metadata = {
        "identity": {
            "schema": PREDICTION_SCHEMA,
            "input_sha256": source["input_sha256"],
            "model_identity": identity,
            "center_size": 2048,
            "match": source["match"],
            "period": 1,
            "scenario": "clean",
        },
        "prediction_sha256": record["prediction_sha256"],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    actual = _fine_arrays(record, source, identity, tmp_path)
    assert all(np.array_equal(actual[name], values[name]) for name in values)
    with pytest.raises(ValueError, match="ancestry"):
        _fine_arrays(record, source, {"train_matches": [source["match"]]}, tmp_path)
    metadata["identity"]["model_identity"] = {"train_matches": [source["match"]]}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="sidecar"):
        _fine_arrays(record, source, identity, tmp_path)
