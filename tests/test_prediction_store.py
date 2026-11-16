"""Lossless prediction bundles verify both data bytes and half identity."""

import numpy as np
import pytest

from experiments.prediction_store import load_prediction_bundle, save_prediction_bundle


def _input(n=4):
    neural = {"live_probability": np.array([0., .25, .75, 1.][:n], np.float64),
              "home_probability": np.array([1., .6, .4, 0.][:n], np.float64),
              "boundary": np.zeros((n, 3, 3), np.float64)}
    reference = {"live_probability": np.array([.1, .2, .8, .9][:n], np.float64),
                 "home_probability": np.array([.9, .7, .3, .1][:n], np.float64),
                 "boundary": np.zeros((n, 3, 3), np.float64)}
    return {"neural": neural, "reference": reference,
            "gt": np.array([0, 1, 2, 3][:n], np.int8),
            "ball_valid": np.array([True, False, False, True][:n]),
            "nearest_ball_seconds": np.array([0., .04, .04, 0.][:n], np.float64),
            "frame_id": np.arange(100, 100 + n, dtype=np.int64),
            "match": "M1", "period": 1, "scenario": "clean"}


def test_round_trip_exact_arrays_and_identical_reuse(tmp_path):
    path = tmp_path / "clean" / "half1.npz"
    source = _input()
    record = save_prediction_bundle(path, **source)
    assert record["path"] == str(path.resolve())
    assert record["schema_version"] == 1
    assert path.with_suffix(".sha256.json").is_file()
    arrays = load_prediction_bundle(record)
    for name, expected in {
        "neural_live": source["neural"]["live_probability"],
        "neural_home": source["neural"]["home_probability"],
        "reference_live": source["reference"]["live_probability"],
        "reference_home": source["reference"]["home_probability"],
        "gt": source["gt"], "ball_valid": source["ball_valid"],
        "nearest_ball_seconds": source["nearest_ball_seconds"],
        "frame_id": source["frame_id"],
    }.items():
        np.testing.assert_array_equal(arrays[name], expected)
    assert {name: array.dtype for name, array in arrays.items()} == {
        "neural_live": np.dtype("float64"), "neural_home": np.dtype("float64"),
        "reference_live": np.dtype("float64"), "reference_home": np.dtype("float64"),
        "gt": np.dtype("int8"), "ball_valid": np.dtype("bool"),
        "nearest_ball_seconds": np.dtype("float64"), "frame_id": np.dtype("int64")}
    mtime = path.stat().st_mtime_ns
    assert save_prediction_bundle(path, **source) == record
    assert path.stat().st_mtime_ns == mtime


def test_corrupted_bundle_identity_and_cross_half_rejected(tmp_path):
    path = tmp_path / "half1.npz"
    source = _input()
    record = save_prediction_bundle(path, **source)
    for changed in ({"match": "M2"}, {"period": 2}, {"scenario": "block41_seed3101"}):
        with pytest.raises(ValueError, match="identity"):
            load_prediction_bundle({**record, **changed})
        with pytest.raises(ValueError, match="identity"):
            save_prediction_bundle(path, **{**source, **changed})
    altered = _input()
    altered["neural"]["home_probability"][0] = .5
    with pytest.raises(ValueError, match="arrays differ"):
        save_prediction_bundle(path, **altered)
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash differs"):
        load_prediction_bundle(record)


@pytest.mark.parametrize("change,error", [
    ({"gt": np.array([0, 1, 2, 4], np.int8)}, "official codes"),
    ({"gt": np.array([0., 1., 2., 3.])}, "integer"),
    ({"ball_valid": np.array([1, 0, 0, 1])}, "boolean"),
    ({"nearest_ball_seconds": np.array([0., -1., 1., 0.])}, "nonnegative"),
    ({"nearest_ball_seconds": np.array([0., np.inf, 1., 0.])}, "finite"),
    ({"frame_id": np.array([100, 101, 103, 104])}, "consecutive"),
    ({"frame_id": np.array([2**63, 2**63 + 1, 2**63 + 2, 2**63 + 3],
                            dtype=np.uint64)}, "without loss"),
    ({"frame_id": np.array([100, 101, 102])}, "shape"),
])
def test_invalid_arrays_rejected(tmp_path, change, error):
    with pytest.raises(ValueError, match=error):
        save_prediction_bundle(tmp_path / "half1.npz", **{**_input(), **change})


def test_sidecar_required_on_reuse(tmp_path):
    path = tmp_path / "half1.npz"
    source = _input()
    save_prediction_bundle(path, **source)
    path.with_suffix(".sha256.json").unlink()
    with pytest.raises(ValueError, match="sidecar"):
        save_prediction_bundle(path, **source)
