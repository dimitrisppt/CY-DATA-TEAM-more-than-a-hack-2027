import json

import pytest

from experiments.prepare import prepare_dfl
from possession.storage import load_half, sha256_file


def source_pair(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    meta = raw / "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-TEST.xml"
    pos = raw / "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-TEST.xml"
    meta.write_text("""<PutDataRequest><MatchInformation>
<General MatchId='DFL-MAT-TEST' HomeTeamId='H' GuestTeamId='A'/>
<Environment PitchX='100' PitchY='60'/>
<Teams><Team TeamId='H'><Players><Player PersonId='home'/></Players></Team>
<Team TeamId='A'><Players><Player PersonId='away'/></Players></Team></Teams>
</MatchInformation></PutDataRequest>""")
    pos.write_text("""<PutDataRequest><Positions><MetaData MatchId='DFL-MAT-TEST'>
<PitchSize X='100' Y='60'/></MetaData>
<FrameSet GameSection='firstHalf' TeamId='H' PersonId='home'>
<Frame N='100' X='-1' Y='2'/></FrameSet>
<FrameSet GameSection='firstHalf' TeamId='BALL' PersonId='ball'>
<Frame N='100' X='0' Y='0' BallStatus='1' BallPossession='1'/></FrameSet>
<FrameSet GameSection='secondHalf' TeamId='A' PersonId='away'>
<Frame N='200' X='1' Y='-2'/></FrameSet>
<FrameSet GameSection='secondHalf' TeamId='BALL' PersonId='ball'>
<Frame N='200' X='0' Y='0' BallStatus='0' BallPossession='2'/></FrameSet>
</Positions></PutDataRequest>""")
    (raw / "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-PART.xml.part").write_text("partial")
    return raw, pos, meta


def test_prepare_writes_verified_cache_and_manifest(tmp_path):
    raw, pos, meta = source_pair(tmp_path)
    out = tmp_path / "processed"
    manifest = prepare_dfl(raw, out)
    cache = out / "DFL-MAT-TEST_half1.npz"
    frames, targets = load_half(cache)
    assert frames.frame_id.tolist() == [100]
    assert targets.state.tolist() == [0]
    assert manifest == json.loads((out / "manifest.json").read_text())
    entry = manifest["matches"]["DFL-MAT-TEST"]
    assert entry["sources"]["positions"]["sha256"] == sha256_file(pos)
    assert entry["sources"]["match_info"]["sha256"] == sha256_file(meta)
    assert entry["halves"][0]["cache_sha256"] == sha256_file(cache)
    assert entry["halves"][0]["pitch"] == [100.0, 60.0]
    assert entry["halves"][0]["first_frame"] == 100


def test_prepare_reuses_valid_cache_but_repairs_corruption(tmp_path):
    raw, _, _ = source_pair(tmp_path)
    out = tmp_path / "processed"
    prepare_dfl(raw, out)
    cache = out / "DFL-MAT-TEST_half1.npz"
    inode = cache.stat().st_ino
    prepare_dfl(raw, out)
    assert cache.stat().st_ino == inode
    cache.write_bytes(b"corrupt")
    prepare_dfl(raw, out)
    assert load_half(cache)[1].state.tolist() == [0]
    assert cache.stat().st_ino != inode


def test_prepare_rebuilds_when_source_hash_changes(tmp_path):
    raw, pos, _ = source_pair(tmp_path)
    out = tmp_path / "processed"
    prepare_dfl(raw, out)
    pos.write_text(pos.read_text().replace("BallPossession='1'", "BallPossession='2'"))
    prepare_dfl(raw, out)
    assert load_half(out / "DFL-MAT-TEST_half1.npz")[1].state.tolist() == [1]


def test_prepare_rejects_missing_requested_match(tmp_path):
    raw, _, _ = source_pair(tmp_path)
    with pytest.raises(ValueError, match="J03WMX"):
        prepare_dfl(raw, tmp_path / "processed", matches=["J03WMX"])


def test_prepare_ignores_incomplete_manifest_entry(tmp_path):
    raw, _, _ = source_pair(tmp_path)
    out = tmp_path / "processed"
    manifest = prepare_dfl(raw, out)
    del manifest["matches"]["DFL-MAT-TEST"]["match_id"]
    (out / "manifest.json").write_text(json.dumps(manifest))
    repaired = prepare_dfl(raw, out)
    assert repaired["matches"]["DFL-MAT-TEST"]["match_id"] == "DFL-MAT-TEST"


def test_subset_preparation_keeps_other_verified_manifest_entries(tmp_path):
    raw, pos, meta = source_pair(tmp_path)
    (raw / pos.name.replace("TEST", "OTHER")).write_text(pos.read_text().replace("TEST", "OTHER"))
    (raw / meta.name.replace("TEST", "OTHER")).write_text(meta.read_text().replace("TEST", "OTHER"))
    out = tmp_path / "processed"
    before = prepare_dfl(raw, out)
    other_cache = out / "DFL-MAT-OTHER_half1.npz"
    other_inode = other_cache.stat().st_ino
    after = prepare_dfl(raw, out, matches=["TEST"])
    assert set(after["matches"]) == {"DFL-MAT-TEST", "DFL-MAT-OTHER"}
    assert after["matches"]["DFL-MAT-OTHER"] == before["matches"]["DFL-MAT-OTHER"]
    assert other_cache.stat().st_ino == other_inode


@pytest.mark.parametrize("damage", ["missing_second", "duplicate_first"])
def test_prepare_repairs_incomplete_or_duplicate_half_manifest(tmp_path, damage):
    raw, _, _ = source_pair(tmp_path)
    out = tmp_path / "processed"
    manifest = prepare_dfl(raw, out)
    halves = manifest["matches"]["DFL-MAT-TEST"]["halves"]
    if damage == "missing_second":
        halves.pop()
    else:
        halves[1] = halves[0].copy()
    (out / "manifest.json").write_text(json.dumps(manifest))
    repaired = prepare_dfl(raw, out)
    assert [h["period"] for h in repaired["matches"]["DFL-MAT-TEST"]["halves"]] == [1, 2]
