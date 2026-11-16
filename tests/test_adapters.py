import numpy as np
import pytest

from possession.io_atd import read_atd
from possession.io_dfl import audit_dfl, read_dfl

ATD_META = """<Root xmlns='urn:epts'><Teams><Team id='H'/><Team id='A'/></Teams>
<Players><Player teamId='A'/><Player teamId='H'/><Player teamId='H'/></Players></Root>"""


def atd_files(tmp_path, lines):
    meta = tmp_path / "metadata.xml"
    raw = tmp_path / "raw.txt"
    meta.write_text(ATD_META)
    raw.write_text("\n".join(lines) + "\n")
    return raw, meta


def test_atd_maps_one_based_ids_team_slots_missing_and_graded_spans(tmp_path):
    raw, meta = atd_files(tmp_path, [
        "1:0.5,0.25,1;NaN,NaN,NaN;1.1,-0.1,0:0.2,0.5,0",
        "2:NaN,NaN,NaN;0,1,0;NaN,NaN,NaN:NaN,NaN,NaN",
    ])
    frames = read_atd(raw, meta, n_frames=2, graded_spans=[(0, 1)])
    assert frames.frame_id.tolist() == [0, 1]
    assert frames.graded.tolist() == [True, False]
    assert frames.player_valid[:, 0, :].tolist() == [[False, True], [True, False]]
    assert frames.player_valid[:, 1, :].tolist() == [[True, False], [False, False]]
    np.testing.assert_allclose(frames.players_xy[0, 1, 0], [52.65, 17.0])
    np.testing.assert_allclose(frames.players_xy[0, 0, 1], [115.83, -6.8])
    assert frames.ball_valid.tolist() == [True, False]
    np.testing.assert_allclose(frames.ball_xy[0], [21.06, 34.0])
    assert frames.ball_xy[1].tolist() == [0.0, 0.0]


def test_atd_accepts_standalone_missing_markers(tmp_path):
    raw, meta = atd_files(tmp_path, ["1:*;NaN,NaN,NaN;0,0,0:*"])
    frames = read_atd(raw, meta)
    assert frames.player_valid[0].tolist() == [[False, True], [False, False]]
    assert not frames.ball_valid[0]


@pytest.mark.parametrize("lines", [
    ["1:0,0;0,0;0,0:0,0", "1:0,0;0,0;0,0:0,0"],
    ["1:0,0;0,0;0,0:0,0", "3:0,0;0,0;0,0:0,0"],
    ["2:0,0;0,0;0,0:0,0"],
    ["1:0,0;0,0:0,0"],
    ["1:inf,0;0,0;0,0:0,0"],
])
def test_atd_rejects_broken_alignment_or_observations(tmp_path, lines):
    raw, meta = atd_files(tmp_path, lines)
    with pytest.raises(ValueError):
        read_atd(raw, meta)


DFL_META = """<PutDataRequest><MatchInformation>
<General MatchId='MATCH' HomeTeamId='H' GuestTeamId='A'/>
<Environment PitchX='100' PitchY='60'/>
<Teams><Team TeamId='A' Role='guest'><Players><Player PersonId='away'/></Players></Team>
<Team TeamId='H' Role='home'><Players><Player PersonId='home'/></Players></Team></Teams>
</MatchInformation></PutDataRequest>"""


def dfl_files(tmp_path, ball_rows, player_rows=None):
    meta = tmp_path / "match.xml"
    pos = tmp_path / "positions.xml"
    meta.write_text(DFL_META)
    player_rows = player_rows or [(100, 0, 0), (101, 1, 2), (102, 2, 3), (103, 3, 4)]
    def frames(rows, extras=False):
        return "".join(
            f"<Frame N='{row[0]}' X='{row[1]}' Y='{row[2]}'"
            + (f" BallStatus='{row[3]}' BallPossession='{row[4]}'" if extras else "")
            + "/>" for row in rows
        )
    pos.write_text(
        "<PutDataRequest><Positions><MetaData MatchId='MATCH'><PitchSize X='100' Y='60'/></MetaData>"
        f"<FrameSet GameSection='firstHalf' TeamId='H' PersonId='home'>{frames(player_rows)}</FrameSet>"
        f"<FrameSet GameSection='firstHalf' TeamId='BALL' PersonId='ball'>{frames(ball_rows, True)}</FrameSet>"
        f"<FrameSet GameSection='secondHalf' TeamId='A' PersonId='away'><Frame N='200' X='-1' Y='-2'/></FrameSet>"
        "<FrameSet GameSection='secondHalf' TeamId='BALL' PersonId='ball'><Frame N='200' X='0' Y='0' BallStatus='0' BallPossession='1'/></FrameSet>"
        "</Positions></PutDataRequest>"
    )
    return pos, meta


def test_dfl_separates_geometry_from_partial_labels_and_halves(tmp_path):
    pos, meta = dfl_files(tmp_path, [
        (100, 0, 0, "1", "1"),
        (101, 1, 2, "1", "null"),
        (102, 2, 3, "0", "2"),
        (103, 3, 4, "null", "1"),
    ])
    pairs = read_dfl(pos, meta)
    assert len(pairs) == 2
    frames, labels = pairs[0]
    assert frames.frame_id.tolist() == [100, 101, 102, 103]
    np.testing.assert_allclose(frames.players_xy[0, 0, 0], [50, 30])
    np.testing.assert_allclose(frames.ball_xy[1], [51, 32])
    assert labels.live_valid.tolist() == [True, True, True, False]
    assert labels.live.tolist() == [1, 1, 0, 0]
    assert labels.owner_valid.tolist() == [True, False, False, False]
    assert labels.state.tolist() == [0, -1, 2, -1]
    assert labels.state_valid.tolist() == [True, False, True, False]
    assert pairs[1][0].frame_id.tolist() == [200]
    assert pairs[1][1].state.tolist() == [2]
    assert not frames.player_valid[:, 1, :].any()


def test_dfl_audit_counts_restart_and_source_label_gaps(tmp_path):
    pos, meta = dfl_files(tmp_path, [
        (100, 0, 0, "0", "null"),
        (101, 1, 0, "1", "1"),
        (102, 2, 0, "1", "null"),
        (103, 3, 0, "1", "2"),
    ])
    audit = audit_dfl(pos, meta)["halves"][0]
    assert audit["dead_to_active"] == 1
    assert audit["active_unknown_owner"] == 1
    assert audit["active_owner_switch"] == 0
    assert audit["status_null"] == 0
    assert audit["mean_players_per_frame"] == 1.0


def test_dfl_label_mutation_does_not_change_geometry(tmp_path):
    a, meta = dfl_files(tmp_path, [(100, 0, 0, "1", "1")], [(100, -3, 4)])
    first = read_dfl(a, meta)[0][0]
    a.write_text(a.read_text().replace("BallPossession='1'", "BallPossession='2'").replace("BallStatus='1'", "BallStatus='0'"))
    second = read_dfl(a, meta)[0][0]
    np.testing.assert_array_equal(first.players_xy, second.players_xy)
    np.testing.assert_array_equal(first.ball_xy, second.ball_xy)


def test_dfl_rejects_duplicate_person_frame(tmp_path):
    pos, meta = dfl_files(tmp_path, [(100, 0, 0, "1", "1")], [(100, 0, 0), (100, 1, 1)])
    with pytest.raises(ValueError, match="duplicate|order"):
        read_dfl(pos, meta)


def test_dfl_rejects_duplicate_person_frame_across_framesets(tmp_path):
    pos, meta = dfl_files(tmp_path, [(100, 0, 0, "1", "1")], [(100, "null", "null")])
    text = pos.read_text()
    duplicate = "<FrameSet GameSection='firstHalf' TeamId='H' PersonId='home'><Frame N='100' X='1' Y='1'/></FrameSet>"
    pos.write_text(text.replace("</Positions>", duplicate + "</Positions>"))
    with pytest.raises(ValueError, match="duplicate"):
        read_dfl(pos, meta)


def test_dfl_player_id_rename_only_changes_slot_order(tmp_path):
    pos, meta = dfl_files(tmp_path, [(100, 0, 0, "1", "1")], [(100, -3, 4)])
    meta.write_text(meta.read_text().replace(
        "<Player PersonId='home'/>", "<Player PersonId='home'/><Player PersonId='other'/>"))
    pos.write_text(pos.read_text().replace(
        "</Positions>",
        "<FrameSet GameSection='firstHalf' TeamId='H' PersonId='other'>"
        "<Frame N='100' X='4' Y='-3'/></FrameSet></Positions>"))
    before = read_dfl(pos, meta)[0][0]
    meta.write_text(meta.read_text().replace(
        "<Player PersonId='home'/><Player PersonId='other'/>",
        "<Player PersonId='renamed'/><Player PersonId='home'/>"))
    pos.write_text(pos.read_text().replace("PersonId='other'", "PersonId='renamed'"))
    after = read_dfl(pos, meta)[0][0]
    np.testing.assert_array_equal(before.players_xy[0, 0, 0], after.players_xy[0, 0, 1])
    np.testing.assert_array_equal(before.players_xy[0, 0, 1], after.players_xy[0, 0, 0])
    np.testing.assert_array_equal(before.ball_xy, after.ball_xy)
