import json

import numpy as np
import pytest
import torch

from evaluation import possession_metrics as official
from experiments import evaluate as external
from possession.features import build_features
from possession.model import PossessionModel
from possession.predict import load_model_checkpoint, predict_model
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half


def fixture_half(match_id, period, states):
    states = np.asarray(states, dtype=np.int8)
    n = len(states)
    frames = FrameTable(
        frame_id=np.arange(n), players_xy=np.zeros((n, 2, 1, 2)),
        player_valid=np.zeros((n, 2, 1), dtype=bool), ball_xy=np.zeros((n, 2)),
        ball_valid=np.zeros(n, dtype=bool), graded=np.ones(n, dtype=bool),
        match_id=match_id, period=period,
    )
    known = states >= 0
    alive = (states == 0) | (states == 1)
    targets = TrainingTargets(
        live=alive.astype(np.int8), live_valid=known,
        owner=(states == 0).astype(np.int8), owner_valid=alive,
        state=states, state_valid=known,
    )
    return frames, targets


def test_prepared_evaluation_preserves_unknown_spans_and_half_reset(tmp_path, monkeypatch):
    first = fixture_half("M1", 1, [0, 0, -1, 0, 0])
    second = fixture_half("M1", 2, [0, 0])
    save_half(tmp_path / "M1_half1.npz", *first)
    save_half(tmp_path / "M1_half2.npz", *second)
    monkeypatch.setattr(external, "reference_scores", lambda frames: {
        "live_probability": np.full(len(frames.frame_id), .99),
        "home_probability": np.full(len(frames.frame_id), .99),
        "boundary": np.zeros((len(frames.frame_id), 3, 3)),
    })
    report = external.evaluate_prepared(tmp_path, matches=["M1"])
    match = report["by_match"]["M1"]
    np.testing.assert_array_equal(match["official"]["confusion"],
                                  official.evaluate(np.array([0, 0, 3, 0, 0, 3, 0, 0]),
                                                    np.array([0, 0, 0, 0, 0, 3, 0, 0]))["confusion"])
    assert match["official"]["n_gt_spells"] == 3
    assert match["official"]["n_graded"] == 6
    assert match["n_frames_native"] == 7
    assert match["halves"][0]["official"]["n_gt_spells"] == 2
    assert match["halves"][1]["official"]["n_gt_spells"] == 1
    assert match["halves"][0]["n_unknown_state"] == 1
    json.dumps(report)


def test_cpu_chunked_prediction_matches_full_half_forward():
    torch.manual_seed(5)
    frames, _ = fixture_half("M1", 1, [2] * 8)
    frames.player_valid[:, 0, 0] = True
    frames.players_xy[:, 0, 0, 0] = np.arange(8)
    frames.ball_valid[:] = True
    frames.ball_xy[:, 0] = np.arange(8) + 0.5
    model = PossessionModel(width=8, dilations=(1,)).eval()
    features = build_features(frames)
    with torch.inference_mode():
        full = model(
            torch.from_numpy(features.player_tokens)[None],
            torch.from_numpy(features.player_mask)[None],
            torch.from_numpy(features.global_tokens)[None],
            torch.from_numpy(features.time_valid)[None],
        )
    chunked = predict_model(frames, model, center_size=3, halo=1)
    np.testing.assert_allclose(chunked["live_probability"],
                               torch.sigmoid(full["live_logits"][0]).numpy(), atol=1e-6)
    np.testing.assert_allclose(chunked["home_probability"],
                               torch.sigmoid(full["owner_logits"][0]).numpy(), atol=1e-6)
    np.testing.assert_allclose(chunked["boundary"], full["boundary"][0].numpy(), atol=1e-6)


def test_checkpoint_cannot_score_a_training_match_by_default(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {}, "model_kwargs": {},
                "training_config": {"train_matches": ["J03WMX"]}}, checkpoint)
    with pytest.raises(ValueError, match="training match"):
        external.evaluate_prepared(tmp_path, matches=["DFL-MAT-J03WMX"], checkpoint=checkpoint)


@pytest.mark.parametrize("declared", ["DFL-MAT-J03WMX", [""], [3], {"J03WMX": True}])
def test_checkpoint_rejects_malformed_training_provenance(tmp_path, declared):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {}, "model_kwargs": {},
                "training_config": {"train_matches": declared}}, checkpoint)
    with pytest.raises(ValueError, match="train_matches"):
        external.evaluate_prepared(tmp_path, matches=["J03WMX"], checkpoint=checkpoint)


def test_checkpoint_path_runs_and_discloses_validation_overlap(tmp_path):
    save_half(tmp_path / "M2_half1.npz", *fixture_half("M2", 1, [2, 2, 2]))
    model = PossessionModel(width=8, dilations=(1,))
    checkpoint = tmp_path / "model.pt"
    torch.save({
        "state_dict": model.state_dict(), "model_kwargs": {"width": 8, "dilations": [1]},
        "training_config": {"train_matches": ["M1"], "validation_match": "M2"},
    }, checkpoint)
    loaded, payload = load_model_checkpoint(checkpoint)
    assert isinstance(loaded, PossessionModel)
    assert payload["training_config"]["train_matches"] == ["M1"]
    report = external.evaluate_prepared(tmp_path, matches=["M2"], checkpoint=checkpoint)
    assert report["predictor"] == "checkpoint"
    assert report["validation_overlap"] == ["M2"]
    assert report["training_overlap"] == []
    assert report["by_match"]["M2"]["official"]["n_graded"] == 3


def test_short_match_id_resolves_to_canonical_cache(tmp_path, monkeypatch):
    match = "DFL-MAT-J03WMX"
    save_half(tmp_path / f"{match}_half1.npz", *fixture_half(match, 1, [2, 2]))
    monkeypatch.setattr(external, "reference_scores", lambda frames: {
        "live_probability": np.full(len(frames.frame_id), .01),
        "home_probability": np.full(len(frames.frame_id), .5),
        "boundary": np.zeros((len(frames.frame_id), 3, 3)),
    })
    report = external.evaluate_prepared(tmp_path, matches=["J03WMX"])
    assert list(report["by_match"]) == [match]
    assert report["by_match"][match]["official"]["n_graded"] == 2


def test_split_manifest_has_disjoint_fixed_seven_fold_protocol():
    with open("configs/splits.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    all_matches = set(manifest["matches"])
    assert len(all_matches) == 7
    assert len(manifest["outer_folds"]) == 7
    for fold in manifest["outer_folds"]:
        train = set(fold["train_matches"])
        valid = fold["validation_match"]
        test = fold["test_match"]
        assert len(train) == 5
        assert train | {valid, test} == all_matches
        assert not train & {valid, test}


@pytest.mark.parametrize("command", ["train", "evaluate"])
def test_entry_points_reject_an_invalid_prepared_manifest(tmp_path, command):
    save_half(tmp_path / "M1_half1.npz", *fixture_half("M1", 1, [2, 2, 2]))
    (tmp_path / "manifest.json").write_text('{"version": 99, "matches": {}}')
    with pytest.raises(ValueError, match="manifest"):
        if command == "train":
            from possession.train import train_model
            train_model(tmp_path, tmp_path / "run", train_matches=["M1"], steps=1,
                        width=8, dilations=(1,), center_size=2, halo=1)
        else:
            external.evaluate_prepared(tmp_path, matches=["M1"])
