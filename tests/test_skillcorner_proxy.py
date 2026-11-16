import csv
import json

import numpy as np
import pytest

from experiments.skillcorner_proxy import (
    _labels,
    _mapping,
    prepare_match,
    resample_observed,
    score_native_proxy,
)


def test_agreement_labels_only_and_phase_gaps_stay_unknown():
    # Home and away IDs are deliberately nonzero and nonconsecutive.
    groups = np.array([0, 1, 0, -1, 1, 0], np.int8)
    phase = np.array([1803, 866, 866, 866, -1, -1], np.int64)
    labels, audit = _labels(groups, phase, home=1803, away=866)
    assert labels.state.tolist() == [0, 1, -1, -1, -1, -1]
    assert labels.state_valid.tolist() == [True, True, False, False, False, False]
    assert labels.owner.tolist()[:2] == [1, 0]
    assert labels.live_valid.tolist() == labels.state_valid.tolist()
    assert audit["phase_and_group_disagree"] == 1
    assert audit["phase_group_null"] == 1
    assert audit["phase_gap_unknown"] == 2
    assert audit["dead_truth_frames"] == 0
    assert not np.any(labels.state == 2)


def test_resampling_requires_detected_neighbors_and_never_fills_gaps():
    xy = np.array([[0., 0.], [10., 0.], [20., 0.]], np.float32)
    out, valid = resample_observed(xy, np.array([True, False, True]))
    assert len(out) == 6  # source t=0..0.2 s, model t=0..0.2 s
    assert valid.tolist() == [True, False, False, False, False, True]
    assert out[1:5].tolist() == [[0., 0.]] * 4
    out2, valid2 = resample_observed(xy, np.ones(3, bool))
    assert valid2.all()
    np.testing.assert_allclose(out2[:, 0], [0, 4, 8, 12, 16, 20])
    assert _mapping(3).tolist() == [0, 3, 5]
    assert _mapping(4).tolist() == [0, 3, 5, 7]
    assert np.max(np.abs(_mapping(20) / 25 - np.arange(20) / 10)) <= 0.02 + 1e-9


def _fixture_files(tmp_path):
    match = "2016236"
    metadata = {
        "id": int(match), "home_team": {"id": 1803}, "away_team": {"id": 866},
        "pitch_length": 100, "pitch_width": 60,
        "players": [{"id": 11, "team_id": 1803}, {"id": 22, "team_id": 866}],
        "match_periods": [
            {"period": 1, "start_frame": 100, "end_frame": 103},
            {"period": 2, "start_frame": 200, "end_frame": 201},
        ],
    }
    (tmp_path / f"{match}_match.json").write_text(json.dumps(metadata))
    with (tmp_path / f"{match}_phases_of_play.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["match_id", "period", "frame_start",
                                                  "frame_end", "team_in_possession_id"])
        writer.writeheader()
        writer.writerows([
            {"match_id": match, "period": 1, "frame_start": 100, "frame_end": 102,
             "team_in_possession_id": 1803},
            {"match_id": match, "period": 1, "frame_start": 102, "frame_end": 104,
             "team_in_possession_id": 866},
            {"match_id": match, "period": 2, "frame_start": 200, "frame_end": 202,
             "team_in_possession_id": 866},
        ])
    rows = []
    for frame, period, group, detected in [
        (100, 1, "home team", True), (101, 1, "home team", False),
        (102, 1, "home team", True), (103, 1, None, True),
        (200, 2, "away team", True), (201, 2, None, True),
    ]:
        rows.append({"frame": frame, "period": period,
                     "possession": {"player_id": None, "group": group},
                     "ball_data": {"x": frame - 100, "y": 0, "is_detected": detected},
                     "player_data": [
                         {"player_id": 11, "x": frame - 100, "y": 0,
                          "is_detected": detected},
                         {"player_id": 22, "x": 0, "y": 1, "is_detected": True},
                     ]})
    (tmp_path / f"{match}_tracking_extrapolated.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n")
    return match


def test_adapter_keeps_inputs_and_native_labels_separate(tmp_path):
    match = _fixture_files(tmp_path)
    halves = prepare_match(tmp_path, match)
    assert len(halves) == 2
    first, second = halves
    assert first.source_frame_10hz.tolist() == [100, 101, 102, 103]
    assert first.source_to_model.tolist() == [0, 3, 5, 7]
    assert first.proxy_10hz.state.tolist() == [0, 0, -1, -1]
    assert first.audit["phase_and_group_disagree"] == 1
    assert first.audit["phase_group_null"] == 1
    assert first.audit["dead_truth_frames"] == 0
    assert second.proxy_10hz.state.tolist() == [1, -1]
    assert first.input_25hz.players_xy.shape[:2] == (8, 2)
    assert not first.input_25hz.ball_valid[1:5].any()
    assert not hasattr(first.input_25hz, "state")
    assert not hasattr(first.input_25hz, "owner")
    np.testing.assert_allclose(first.input_25hz.players_xy[0, 0, 0], [50, 30])
    # Changing provider label fields cannot alter geometry, validity, or timestamps.
    path = tmp_path / f"{match}_tracking_extrapolated.jsonl"
    path.write_text(path.read_text().replace('"home team"', '"away team"'))
    again = prepare_match(tmp_path, match)[0]
    np.testing.assert_array_equal(first.input_25hz.players_xy, again.input_25hz.players_xy)
    np.testing.assert_array_equal(first.input_25hz.ball_valid, again.input_25hz.ball_valid)
    np.testing.assert_array_equal(first.source_to_model, again.source_to_model)


def test_native_scorer_requires_frozen_selection():
    labels, _ = _labels(np.array([0, 0, 1], np.int8),
                         np.array([1803, 1803, 866]), 1803, 866)
    with pytest.raises(ValueError, match="freeze model selection"):
        score_native_proxy(labels, np.array([0, 0, 1], np.int8), np.array([0, 1, 2]))


def test_native_scorer_uses_ten_hz_durations_and_five_ten_frame_tolerances():
    labels, _ = _labels(np.array([0] * 5 + [1] * 5, np.int8),
                         np.array([1803] * 5 + [866] * 5), 1803, 866)
    result = score_native_proxy(labels, np.array([0] * 5 + [1] * 5, np.int8),
                                np.arange(10), selection_frozen=True)
    assert result["native_fps"] == 10
    assert result["gt_mean_spell_duration_s"] == 0.5
    assert result["pred_mean_spell_duration_s"] == 0.5
    assert set(result["transitions"]) == {5, 10}
    assert result["spell_f1_mean_conditional"] == 1.0


def test_unknown_team_id_rejected(tmp_path):
    match = _fixture_files(tmp_path)
    path = tmp_path / f"{match}_phases_of_play.csv"
    path.write_text(path.read_text().replace(",1803\n", ",9999\n"))
    with pytest.raises(ValueError, match="unknown owning team"):
        prepare_match(tmp_path, match)
