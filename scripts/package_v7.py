"""Build a portable v7 source/model archive without participant assets or run caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

from experiments.campaign_submission import _validate_output
from possession.storage import sha256_file
from scripts.evaluate_v7 import evaluate
from scripts.v7_bundle import ROOT, verify_bundle


def package(output):
    output = Path(output).resolve()
    verify_bundle(ROOT / 'models/v7')
    evaluate(ROOT / 'benchmarks/v7')
    manifest = json.loads((ROOT / 'submission/manifest.json').read_text())
    csv = ROOT / 'submission/CY-DATA-TEAM.csv'
    if manifest['version'] != 'v7' or sha256_file(csv) != manifest['sha256']:
        raise ValueError('current submission is not the verified v7 CSV')
    _validate_output(csv)
    receipt = json.loads((ROOT / 'submission/verification-v7.json').read_text())
    if receipt['csv_sha256'] != manifest['sha256'] or not receipt['verified']:
        raise ValueError('release verification is missing or stale')
    if sha256_file(ROOT / 'submission/CY-DATA-TEAM.pdf') != receipt['pdf_sha256']:
        raise ValueError('PDF differs from release verification')
    paths = []
    for directory, pattern in [('possession', '*.py'), ('experiments', '*.py'),
                               ('evaluation', '*.py'), ('scripts', '*.py'), ('tests', '*.py'),
                               ('docs', '*.md'), ('configs', '*.json'), ('models/v7', '*'),
                               ('benchmarks', '*'), ('submission', '*')]:
        paths.extend(p for p in (ROOT / directory).rglob(pattern) if p.is_file())
    paths.extend(ROOT / name for name in ['README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
                 'CHALLENGE.md', 'DATA.md', 'EVALUATION.md', 'FAQ.md', 'OVERVIEW.md',
                 'RULES.md', 'SUBMISSION.md', 'CITATION.cff', 'games.py', 'pyproject.toml',
                 'uv.lock', '.python-version', '.gitignore', 'requirements.txt',
                 'requirements-starter.txt', 'data/README.md'])
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(set(paths))}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    payload['package-manifest.json'] = (json.dumps({
        'schema': 'v7_source_model_package_v1', 'files': hashes,
        'csv_sha256': manifest['sha256'], 'participant_assets_included': False,
    }, indent=2, sort_keys=True) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, suffix='.zip')
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(payload.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 11, 17, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('release archive CRC check failed')
            for name, digest in hashes.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError(f'archive entry differs: {name}')
        if output.exists() and sha256_file(output) != sha256_file(temporary):
            raise ValueError('refusing to replace a different release archive')
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'output': str(output), 'files': len(payload), 'sha256': sha256_file(output),
            'bytes': output.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/CY-DATA-TEAM-v7.zip')
    args = parser.parse_args()
    print(json.dumps(package(args.output), indent=2))


if __name__ == '__main__':
    main()
