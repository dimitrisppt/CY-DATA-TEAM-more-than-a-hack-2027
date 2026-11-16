import hashlib
import io

import pytest

from experiments.fetch_dfl import download_file, select_files


def entry(name='DFL_positions_J03WMX.xml', data=b'public XML'):
    return {'name': name, 'size': len(data), 'download_url': 'https://ndownloader.figshare.com/files/1',
            'computed_md5': hashlib.md5(data).hexdigest()}


def test_selects_requested_match_inputs_only():
    files = [entry('DFL_positions_J03WMX.xml'), entry('DFL_matchinformation_J03WMX.xml'),
             entry('DFL_events_J03WMX.xml'), entry('DFL_positions_J03WN1.xml')]
    assert [f['name'] for f in select_files(files, ['J03WMX'])] == [f['name'] for f in files[:2]]
    with pytest.raises(ValueError):
        select_files(files, ['UNKNOWN'])


def test_rejects_path_escape_before_network(tmp_path):
    with pytest.raises(ValueError):
        download_file(entry('../outside.xml'), tmp_path)
    assert not (tmp_path.parent / 'outside.xml').exists()


def test_checksum_failure_never_publishes_file(tmp_path, monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kw: io.BytesIO(b'wrong data'))
    with pytest.raises(ValueError, match='checksum|size'):
        download_file(entry(), tmp_path)
    assert not (tmp_path / entry()['name']).exists()


def test_verified_download_is_reusable(tmp_path, monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kw: io.BytesIO(b'public XML'))
    result = download_file(entry(), tmp_path)
    assert result['sha256'] == hashlib.sha256(b'public XML').hexdigest()
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kw: pytest.fail('redownload'))
    assert download_file(entry(), tmp_path)['sha256'] == result['sha256']
