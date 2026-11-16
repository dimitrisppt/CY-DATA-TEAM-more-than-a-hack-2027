import numpy as np

from possession.reference import ReferenceConfig, reference_scores
from possession.schema import FrameTable


def scene(ball_valid=None, stationary=False, n=15):
    t = np.arange(n, dtype=float)
    drift = np.zeros(n) if stationary else .8 * t
    players = np.zeros((n, 2, 2, 2))
    players[:, 0, 0] = np.column_stack((12 + drift, np.full(n, 20.)))
    players[:, 0, 1] = np.column_stack((16 + drift, np.full(n, 25.)))
    players[:, 1, 0] = np.column_stack((85 - drift, np.full(n, 20.)))
    players[:, 1, 1] = np.column_stack((89 - drift, np.full(n, 25.)))
    ball = np.column_stack((13 + drift, np.full(n, 20.)))
    visible = np.ones(n, dtype=bool) if ball_valid is None else np.asarray(ball_valid)
    return FrameTable(
        frame_id=np.arange(n), players_xy=players,
        player_valid=np.ones((n, 2, 2), dtype=bool),
        ball_xy=ball, ball_valid=visible, graded=np.ones(n, dtype=bool),
        fps=1., pitch=(105.3, 68.),
    )


def test_clear_geometry_home_away_swap_and_player_permutation():
    sample = scene()
    base = reference_scores(sample)
    assert np.all(base["home_probability"][2:-2] > .75)
    sample.players_xy = sample.players_xy[:, ::-1, ::-1].copy()
    sample.player_valid = sample.player_valid[:, ::-1, ::-1].copy()
    swapped = reference_scores(sample)
    np.testing.assert_allclose(swapped["home_probability"], 1 - base["home_probability"])
    np.testing.assert_allclose(swapped["live_probability"], base["live_probability"])
    assert base["boundary"].shape == (15, 3, 3)


def test_missing_ball_is_neutral_owner_evidence_and_not_dead_by_itself():
    visible = np.zeros(15, dtype=bool)
    sample = scene(visible)
    scores = reference_scores(sample)
    assert np.isfinite(scores["home_probability"]).all()
    assert np.isfinite(scores["live_probability"]).all()
    np.testing.assert_allclose(scores["home_probability"], .5)
    assert scores["live_probability"][5:10].mean() > .5


def test_player_activity_distinguishes_stoppage_even_with_missing_ball():
    visible = np.zeros(15, dtype=bool)
    active = reference_scores(scene(visible))["live_probability"]
    stopped = reference_scores(scene(visible, stationary=True))["live_probability"]
    assert stopped[5:10].mean() < active[5:10].mean()
    assert stopped[5:10].mean() < .5


def test_occluded_interception_can_use_future_team_evidence():
    sample = scene(n=15)
    sample.ball_xy[9:] = sample.players_xy[9:, 1, 0] + [1., 0.]
    sample.ball_valid[5:9] = False
    scores = reference_scores(sample)
    assert scores["home_probability"][2] > .5
    assert scores["home_probability"][12] < .5
    assert scores["home_probability"][8] < scores["home_probability"][5]


def test_outside_pitch_is_weak_dead_clue_and_config_is_serializable():
    sample = scene()
    sample.ball_xy[7] = [106., 20.]
    scores = reference_scores(sample)
    assert 0 < scores["live_probability"][7] < 1
    assert scores["live_probability"][7] < scores["live_probability"][6]
    assert ReferenceConfig().to_dict()["owner_smoothing_seconds"] > 0
