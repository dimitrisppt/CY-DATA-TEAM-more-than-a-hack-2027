import json

import pytest

from experiments.audit import main
from possession.storage import sha256_file


def audit_pair(raw):
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
</Positions></PutDataRequest>""")
    return pos, meta


def test_audit_cli_writes_current_layout_and_source_provenance(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pos, meta = audit_pair(raw)
    (raw / "download-manifest.json").write_text(json.dumps({
        "article": "https://example.org/article", "license": {"name": "CC BY"},
        "files": [{"name": pos.name, "sha256": sha256_file(pos), "download_url": "https://example.org/positions"},
                  {"name": meta.name, "sha256": sha256_file(meta), "download_url": "https://example.org/meta"}],
    }))
    report = tmp_path / "audit.json"
    main(["--raw-dir", str(raw), "--report", str(report)])
    audit = json.loads(report.read_text())
    assert audit["total_frames"] == 1
    assert audit["matches"][0]["match_id"] == "DFL-MAT-TEST"
    assert audit["matches"][0]["halves"][0]["status_1"] == 1
    assert audit["matches"][0]["source_positions"] == pos.name
    assert audit["matches"][0]["source_fingerprints"]["positions"]["sha256"] == sha256_file(pos)
    assert audit["source"]["article"] == "https://example.org/article"


def test_audit_cli_rejects_unpaired_xml(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pos, meta = audit_pair(raw)
    meta.unlink()
    with pytest.raises(ValueError, match="pair|missing"):
        main(["--raw-dir", str(raw), "--report", str(tmp_path / "audit.json")])
    assert pos.exists()
    assert not (tmp_path / "audit.json").exists()
