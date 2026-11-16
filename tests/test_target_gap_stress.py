"""Small target-pattern replay and post-selection provenance contracts."""

import copy
import json

import numpy as np
import pytest

from experiments import target_gap_stress as stress
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half, sha256_file, write_json


def test_cyclic_masks_are_deterministic_and_report_clipped_seams(monkeypatch):
    pattern = np.array([True, True, False, False, True])
    monkeypatch.setattr(stress, "_offset", lambda *args: 0)
    first, evidence = stress._replay_mask(pattern, 12, "M1", 1, 5101)
    second, again = stress._replay_mask(pattern, 12, "M1", 1, 5101)
    np.testing.assert_array_equal(first, second)
    assert evidence == again
    assert evidence["cycle_seams"] == 2
    assert evidence["cycle_seam_clipped_missing_frames"] == 2
    np.testing.assert_array_equal(first, [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    assert evidence["half_start_clipped_source_context_frames"] == 1
    assert evidence["half_end_clipped_source_context_frames"] == 0
    assert stress._offset("M1", 1, 5101, 5) == stress._offset("M1", 1, 5101, 5)
    assert 0 <= stress._offset("M1", 1, 5101, 5) < 5
    with pytest.raises(ValueError):
        stress._replay_mask(pattern, 5, "M1", 1, 9999)


def test_target_reader_uses_only_published_spans_and_checks_hash(tmp_path, monkeypatch):
    target = tmp_path / "target"
    raw = target / "tactical" / "BAR-ATH_tactical_atd.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("".join(
        f"{index + 1}:ignored:{'NaN,NaN,NaN' if index in (1, 5) else '0.2,0.3,0'}\n"
        for index in range(8)))
    write_json(target / "input-manifest.json", {"files": [{
        "path": "tactical/BAR-ATH_tactical_atd.txt", "bytes": raw.stat().st_size,
        "sha256": sha256_file(raw)}]})
    monkeypatch.setattr(stress, "GAMES", {"BAR-ATH": {"n_video_frames": 8}})
    monkeypatch.setattr(stress, "half_video_blocks", lambda game: [(1, 1, 2), (2, 5, 6)])
    patterns, evidence = stress._target_patterns(target)
    np.testing.assert_array_equal(patterns[1], [True, False])
    np.testing.assert_array_equal(patterns[2], [True, False])
    assert evidence["patterns"]["1"]["span"] == [1, 3]
    assert evidence["patterns"]["2"]["missing_frames"] == 1
    raw.write_text(raw.read_text() + "tamper")
    with pytest.raises(ValueError, match="hash or size"):
        stress._target_patterns(target)


def _prepared_half(path):
    n = 12
    state = np.array([0] * 5 + [2] * 2 + [1] * 5, dtype=np.int8)
    frames = FrameTable(
        frame_id=np.arange(100, 112),
        players_xy=np.zeros((n, 2, 1, 2), np.float32),
        player_valid=np.ones((n, 2, 1), bool),
        ball_xy=np.ones((n, 2), np.float32), ball_valid=np.ones(n, bool),
        graded=np.ones(n, bool), match_id="M1", period=1,
    )
    targets = TrainingTargets(
        live=(state != 2).astype(np.int8), live_valid=np.ones(n, bool),
        owner=(state == 0).astype(np.int8), owner_valid=(state != 2),
        state=state, state_valid=np.ones(n, bool),
    )
    save_half(path, frames, targets)
    return state


def test_ball_only_replay_preserves_labels_and_reuses_same_checkpoint(tmp_path, monkeypatch):
    data = tmp_path / "prepared"
    data.mkdir()
    original_gt = _prepared_half(data / "M1_half1.npz")
    manifest = {"prepared_root": str(data),
                "recipes": {role: {"training_args": {"center_size": 4}}
                            for role in ("baseline", "selected")}}
    fold = {"checkpoint_sha256": "a" * 64}
    model = type("Stub", (), {"radius": 1})()
    calls = []

    def predict(features, model, **kwargs):
        calls.append(True)
        assert not features.global_tokens[[1, 2], 2].any()
        return {"live_probability": np.full(12, .8),
                "home_probability": np.full(12, .6),
                "boundary": np.zeros((12, 3, 3))}

    def reference(frames):
        assert not frames.ball_valid[[1, 2]].any()
        assert np.all(frames.ball_xy[[1, 2]] == 0)
        return {"live_probability": np.full(12, .7),
                "home_probability": np.full(12, .5),
                "boundary": np.zeros((12, 3, 3))}

    monkeypatch.setattr(stress, "_replay_mask", lambda *args: (
        np.array([False, True, True] + [False] * 9),
        {"replayed_missing_frames": 2, "replayed_mask_sha256": "b" * 64}))
    monkeypatch.setattr(stress, "predict_features", predict)
    monkeypatch.setattr(stress, "reference_scores", reference)
    first, evidence = stress._prediction(tmp_path / "out", manifest, "baseline", fold,
                                         "M1", 1, 5101, np.ones(4, bool), model)
    second, replayed = stress._prediction(tmp_path / "out", manifest, "selected", fold,
                                          "M1", 1, 5101, np.ones(4, bool), model)
    assert len(calls) == 1
    np.testing.assert_array_equal(first["gt"], original_gt)
    np.testing.assert_array_equal(second["gt"], original_gt)
    assert replayed == evidence
    assert evidence["additional_missing_frames"] == 2
    assert evidence["resulting_gap_statistics"]["missing_frames"] == 2
    assert json.dumps(evidence, allow_nan=False)


def test_frame_gap_bins_stop_at_unknown_and_count_raw_errors():
    gt = np.array([0, 0, 3, 0, 0, 0, 0], dtype=np.int8)
    pred = np.array([1, 0, 0, 0, 1, 0, 0], dtype=np.int8)
    valid = np.array([False, False, False, False, False, True, True])
    bins = stress._frame_errors(gt, pred, valid, fps=1)
    assert bins["(0,10]"]["n_frames"] == 4
    assert bins["(0,10]"]["n_errors"] == 2
    assert bins["0"]["n_frames"] == 2
    assert bins["0"]["n_errors"] == 0
    assert sum(bucket["n_frames"] for bucket in bins.values()) == 6


def test_paired_half_scoring_keeps_spell_boundaries(monkeypatch, tmp_path):
    gt = np.array([0, 0, 3, 1, 1], np.int8)
    arrays = {"gt": gt, "ball_valid": np.array([True, False, False, False, True])}
    recipe = {"ensemble": {"neural_weight": 1.0},
              "calibration": {"live_temperature": 1.0, "owner_temperature": 1.0},
              "decoder": {"switch_cost": 0.0, "boundary_scale": 0.0}}
    manifest = {"recipes": {"baseline": recipe}}
    seen = []

    def prediction(*args):
        seen.append(args[5])
        return arrays, {"replayed_mask_sha256": "a" * 64}

    def decode(values, frozen):
        assert frozen == recipe
        return np.array([0, 0, 0, 1, 1], np.int8)

    monkeypatch.setattr(stress, "_prediction", prediction)
    monkeypatch.setattr(stress, "decode_oof_half", decode)
    result = stress._score_match(tmp_path, manifest, "baseline", {}, "M1", 5101,
                                 {1: np.zeros(2, bool), 2: np.ones(2, bool)}, object())
    assert seen == [1, 2]
    assert result["official"]["n_gt_spells"] == 4
    assert result["spell_diagnostics"]["n_halves"] == 2
    assert result["frame_errors_by_gap_duration"]["(0,10]"]["n_frames"] == 4
    assert result["frame_errors_by_gap_duration"]["0"]["n_frames"] == 4


def test_thread_count_comes_from_frozen_runtime_and_is_capped(monkeypatch, tmp_path):
    plan = {"payload": {"environment": {"torch_num_threads": 2}}}
    monkeypatch.setattr(stress, "_records", lambda *args: (plan,))
    called = []
    monkeypatch.setattr(stress.torch, "set_num_threads", called.append)
    stress._set_frozen_threads(tmp_path)
    assert called == [2]
    plan["payload"]["environment"]["torch_num_threads"] = 8
    with pytest.raises(ValueError, match="at most four"):
        stress._set_frozen_threads(tmp_path)


def test_frozen_gate_and_reused_confirmation_checkpoint(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="selection_frozen"):
        stress.run(tmp_path, tmp_path, tmp_path, tmp_path / "out")

    focused = tmp_path / "focused"
    directory = focused / "confirmation" / "baseline" / "seed_13" / "fold_0"
    directory.mkdir(parents=True)
    (directory / "model.pt").write_bytes(b"checkpoint")
    sha = sha256_file(directory / "model.pt")
    fold = {"train": ["M2", "M3"], "validation": ["M0", "M1"]}
    job = {"seed": 13, "fold_index": 0, "train_matches": fold["train"],
           "validation_matches": fold["validation"], "protocol_sha256": "a" * 64}
    entry = {"fit": {"job": job, "checkpoint_sha256": sha}}
    stress._save_record(directory / "fit.json", entry["fit"])
    confirmation = {role: {"entries": {"13": [copy.deepcopy(entry)]}}
                    for role in ("baseline", "selected")}
    plan = {"sha256": "a" * 64,
            "payload": {"resources": {"confirmation_seed": 13},
                        "fold": {"inner": [fold]}}}
    monkeypatch.setattr(stress, "_verify_fit", lambda *args: None)
    checked = stress._checkpoint_entries(focused, plan, confirmation)
    assert checked["selected"][0]["checkpoint"] == str(directory / "model.pt")
    confirmation["selected"]["entries"]["13"][0]["fit"]["job"]["validation_matches"] = ["M0"]
    with pytest.raises(ValueError, match="fold ancestry"):
        stress._checkpoint_entries(focused, plan, confirmation)


def test_existing_manifest_change_fails_before_inference(tmp_path, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    write_json(output / "frozen-manifest.json", {"old": True})
    monkeypatch.setattr(stress, "_set_frozen_threads", lambda *args: None)
    monkeypatch.setattr(stress, "_manifest", lambda *args: ({"new": True}, {}))
    monkeypatch.setattr(stress, "load_model_checkpoint", lambda *args: pytest.fail("inference reached"))
    with pytest.raises(ValueError, match="provenance changed"):
        stress.run(tmp_path / "focused", tmp_path / "data", tmp_path / "target", output,
                   selection_frozen=True)
