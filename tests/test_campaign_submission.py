import builtins
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import campaign_submission as export
from experiments.campaign_decode import baseline_decoder, decode_arrays, reference_basis
from experiments.campaign_evaluation import _gap_seconds
from experiments.campaign_search import blend_records
from possession.features import build_features
from possession.model import PossessionModel
from possession.predict import predict_features
from possession.schema import FrameTable
from possession.storage import sha256_file, write_json


def _frames(n=36):
    rng = np.random.default_rng(9)
    return FrameTable(np.arange(n), rng.uniform(0, 40, (n, 2, 3, 2)),
                      np.ones((n, 2, 3), dtype=bool), rng.uniform(0, 30, (n, 2)),
                      rng.random(n) > .4, np.ones(n, dtype=bool))


def _selection():
    decoder = baseline_decoder({"ensemble": {"neural_weight": .512},
                                "calibration": {"live_temperature": 2.45},
                                "decoder": {"boundary_scale": 1.84, "switch_cost": 2}})
    decoder.update(switch_cost_type="typed",
                   switch_cost={"owner_switch": .003, "to_dead": 2.64, "from_dead": 2.62})
    return {"kind": "convex_pair", "blend_weight": .5,
            "members": [{"id": "mean"}, {"id": "attention"}], "decoder": decoder}


def _scores(seed, n=36):
    rng = np.random.default_rng(seed)
    return {"live_probability": rng.uniform(0, 1, n),
            "home_probability": rng.uniform(0, 1, n),
            "boundary": rng.normal(size=(n, 3, 3))}


@pytest.mark.parametrize("weight", [0., .25, .5, .75, 1.])
def test_label_free_blend_matches_search_exactly(weight):
    frames = _frames()
    left, right = _scores(3), _scores(4)
    left["live_probability"][:2] = [0, 1]
    records = []
    for scores in (left, right):
        records.append([{"match": "fixture", "period": 1, "scenario": "clean",
                         "input_sha256": "same", "arrays": {
                             "frame_id": frames.frame_id, "gt": np.zeros(36),
                             "ball_valid": frames.ball_valid,
                             "nearest_ball_seconds": _gap_seconds(frames.ball_valid, 25),
                             **reference_basis(frames),
                             "neural_live": scores["live_probability"],
                             "neural_home": scores["home_probability"],
                             "neural_boundary": scores["boundary"]}}])
    expected = blend_records(*records, weight)[0]["arrays"]
    actual = export.blend_neural(left, right, weight)
    for key, campaign_key in (("live_probability", "neural_live"),
                              ("home_probability", "neural_home"),
                              ("boundary", "neural_boundary")):
        np.testing.assert_array_equal(actual[key], expected[campaign_key])


@pytest.mark.parametrize("mutation", [
    lambda scores: scores.update(boundary=np.ones((36, 3))),
    lambda scores: scores.update(home_probability=np.full(36, np.nan)),
    lambda scores: scores.update(live_probability=np.full(36, 1.1)),
])
def test_blend_rejects_invalid_predictions(mutation):
    right = _scores(4)
    mutation(right)
    with pytest.raises(ValueError):
        export.blend_neural(_scores(3), right, .5)


def test_real_network_half_inference_matches_cached_campaign_path():
    torch.manual_seed(21)
    models = {name: PossessionModel(width=8, dilations=[1, 2], pooling=pooling).eval()
              for name, pooling in (("mean", "mean"), ("attention", "ball_attention"))}
    frames, selection = _frames(), _selection()
    features = build_features(frames)
    records = []
    for member in selection["members"]:
        scores = predict_features(features, models[member["id"]], center_size=2048)
        records.append([{"match": "fixture", "period": 1, "scenario": "clean",
                         "input_sha256": "same", "arrays": {
                             "frame_id": frames.frame_id, "gt": np.zeros(36),
                             "ball_valid": frames.ball_valid,
                             "nearest_ball_seconds": _gap_seconds(frames.ball_valid, frames.fps),
                             **reference_basis(frames, features),
                             "neural_live": scores["live_probability"],
                             "neural_home": scores["home_probability"],
                             "neural_boundary": scores["boundary"]}}])
    arrays = blend_records(*records, .5)[0]["arrays"]
    expected = decode_arrays(arrays, selection["decoder"])
    np.testing.assert_array_equal(export.predict_ensemble_half(frames, models, selection), expected)


def test_half_boundaries_reset_all_context_and_outside_frames_are_x(monkeypatch):
    calls = []

    def predictor(half, models, selection):
        calls.append((half.period, half.frame_id.tolist()))
        return np.full(len(half.frame_id), half.period - 1, dtype=np.int8)

    monkeypatch.setattr(export, "predict_ensemble_half", predictor)
    codes, _ = export.infer_spans(_frames(10), {}, {}, ((1, 4), (6, 9)))
    assert calls == [(1, [1, 2, 3]), (2, [6, 7, 8])]
    assert codes.tolist() == [3, 0, 0, 0, 3, 3, 1, 1, 1, 3]


def _target_fixture(tmp_path, monkeypatch):
    bundle, data = tmp_path / "bundle", tmp_path / "data"
    bundle.mkdir()
    write_json(bundle / "manifest.json", {"fixture": True})
    write_json(bundle / "recipe.json", {"fixture": True})
    tactical = data / "tactical"
    tactical.mkdir(parents=True)
    raw, metadata = tactical / "BAR-ATH_tactical_atd.txt", tactical / "BAR-ATH_tactical_atd.xml"
    raw.write_text("".join(f"{i+1}:0.2,0.5;0.8,0.5:0.3,0.5\n" for i in range(10)))
    metadata.write_text("<Root><Team id='h'/><Team id='a'/><Player teamId='h'/>"
                        "<Player teamId='a'/></Root>")
    write_json(data / "input-manifest.json", {"files": [
        {"path": path.relative_to(data).as_posix(), "sha256": sha256_file(path),
         "bytes": path.stat().st_size} for path in (raw, metadata)] + [
             {"path": "ground-truth.csv", "sha256": "must-not-open", "bytes": 0}]})
    monkeypatch.setattr(export, "verify_bundle", lambda path: {
        "manifest": {}, "recipe": {"selection": _selection()}, "member_paths": {}})
    monkeypatch.setattr(export, "_verify_execution", lambda manifest: None)
    monkeypatch.setattr(export, "N_FRAMES", 10)
    monkeypatch.setattr(export, "TARGET_SPANS", ((1, 4), (6, 9)))
    monkeypatch.setattr(export, "predict_ensemble_half", lambda half, *args: np.full(
        len(half.frame_id), half.period - 1, dtype=np.int8))
    return bundle, data, raw


def test_export_verifies_inputs_validates_format_and_never_opens_target_truth(tmp_path, monkeypatch):
    bundle, data, _ = _target_fixture(tmp_path, monkeypatch)
    original_open, original_path_open = builtins.open, Path.open
    allowed = {data / "input-manifest.json", data / "tactical" / "BAR-ATH_tactical_atd.txt",
               data / "tactical" / "BAR-ATH_tactical_atd.xml"}

    def guard(path):
        if isinstance(path, (str, Path)) and Path(path).is_relative_to(data):
            assert Path(path) in allowed, f"unexpected target asset read: {path}"

    def checked_open(path, *args, **kwargs):
        guard(path)
        return original_open(path, *args, **kwargs)

    def checked_path_open(path, *args, **kwargs):
        guard(path)
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", checked_open)
    monkeypatch.setattr(Path, "open", checked_path_open)
    output = tmp_path / "output.csv"
    report = export.infer_campaign_submission(bundle, data, output)
    assert report["ground_truth_used"] is False
    assert report["format_validated"] is True
    assert report["output"]["label_counts"] == {"H": 3, "A": 3, "D": 0, "X": 4}
    assert export.infer_campaign_submission(bundle, data, output) == report
    output.write_text("changed")
    with pytest.raises(ValueError, match="different or changed"):
        export.infer_campaign_submission(bundle, data, output)


def test_changed_target_input_is_rejected_before_inference(tmp_path, monkeypatch):
    bundle, data, raw = _target_fixture(tmp_path, monkeypatch)
    raw.write_text("changed")
    monkeypatch.setattr(export, "read_atd", lambda *args, **kwargs: pytest.fail("read input"))
    output = tmp_path / "output.csv"
    with pytest.raises(ValueError, match="hash or size differs"):
        export.infer_campaign_submission(bundle, data, output)
    assert not output.exists()


def test_changed_bundle_is_rejected_before_target_read(tmp_path, monkeypatch):
    bundle, data, _ = _target_fixture(tmp_path, monkeypatch)

    def invalid(path):
        raise ValueError("bundle file hash differs")

    monkeypatch.setattr(export, "verify_bundle", invalid)
    monkeypatch.setattr(export, "_verified_inputs", lambda *args: pytest.fail("read input"))
    with pytest.raises(ValueError, match="bundle file hash"):
        export.infer_campaign_submission(bundle, data, tmp_path / "output.csv")


def test_existing_baseline_is_not_overwritten(tmp_path, monkeypatch):
    bundle, data, _ = _target_fixture(tmp_path, monkeypatch)
    output = tmp_path / "baseline.csv"
    output.write_text("existing baseline")
    with pytest.raises(ValueError, match="incomplete"):
        export.infer_campaign_submission(bundle, data, output)
    assert output.read_text() == "existing baseline"


def test_bundle_cannot_be_overwritten(tmp_path, monkeypatch):
    bundle, data, _ = _target_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outside bundle"):
        export.infer_campaign_submission(bundle, data, bundle / "recipe.json")


def test_existing_sidecar_cannot_silently_change_inputs(tmp_path, monkeypatch):
    bundle, data, _ = _target_fixture(tmp_path, monkeypatch)
    output = tmp_path / "output.csv"
    export.infer_campaign_submission(bundle, data, output)
    document = json.loads((data / "input-manifest.json").read_text())
    document["extra"] = "different input manifest"
    write_json(data / "input-manifest.json", document)
    with pytest.raises(ValueError, match="different or changed"):
        export.infer_campaign_submission(bundle, data, output)
