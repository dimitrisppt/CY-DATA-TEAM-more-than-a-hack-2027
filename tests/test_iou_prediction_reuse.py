from pathlib import Path

import pytest

from experiments.campaign_protocol import publish
from possession.storage import sha256_file
from scripts.reuse_iou_predictions import reuse_once


def _source(root):
    path = root / "inference/batch_1/control/seed_8101/mean_007/M/half1/clean.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"immutable prediction")
    publish(path.with_suffix(".json"), {"sha256": sha256_file(path),
                                        "identity": {"input_sha256": "same-input", "fit": "same-model"}})
    return path


def test_only_identical_inputs_reuse_without_overwriting_existing_files(tmp_path):
    source = _source(tmp_path)
    target = tmp_path / "historical-masks"
    index = {("M", 1, "same-input"): [(target, "batch_1", "clean")],
             ("M", 1, "different-input"): [(target, "batch_1", "short")]}
    seen = set()
    rows = reuse_once(tmp_path, index, seen)
    assert len(rows) == 1
    destination = Path(rows[0]["destination"])
    assert destination.stat().st_ino == source.stat().st_ino
    assert not destination.with_name("short.npz").exists()
    assert reuse_once(tmp_path, index, set()) == []


def test_corrupt_prediction_refused_before_reuse(tmp_path):
    source = _source(tmp_path)
    source.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash differs"):
        reuse_once(tmp_path, {("M", 1, "same-input"): [(tmp_path / "target", "batch_1", "clean")]}, set())
