"""Download the public Sportec release with publisher checksums.

Only tracking and match metadata are needed. No challenge assets are downloaded.
"""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ARTICLE = "https://api.figshare.com/v2/articles/28196177"


def select_files(files, matches):
    selected = [f for f in files if any(kind in f['name'] for kind in
                ('positions', 'matchinformation')) and
                (not matches or any(f['name'].endswith(f'{m}.xml') for m in matches))]
    for match in matches:
        if not any('positions' in f['name'] and f['name'].endswith(f'{match}.xml')
                   for f in selected):
            raise ValueError(f'No position file for match {match}')
    return selected


def _verify(path, entry):
    md5, sha = hashlib.md5(), hashlib.sha256()
    size = 0
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            size += len(block)
            md5.update(block)
            sha.update(block)
    if size != entry['size'] or md5.hexdigest() != entry['computed_md5']:
        raise ValueError(f"size or checksum mismatch for {entry['name']}")
    return {**entry, 'sha256': sha.hexdigest()}


def download_file(entry, destination):
    name = entry['name']
    if Path(name).name != name or '/' in name or '\\' in name or name in ('.', '..'):
        raise ValueError('Unsafe download filename')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / name
    if final.exists():
        return _verify(final, entry)
    partial = final.with_suffix(final.suffix + '.part')
    try:
        with urllib.request.urlopen(entry['download_url'], timeout=60) as response, partial.open('wb') as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        result = _verify(partial, entry)
        partial.replace(final)
        return result
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/external/dfl'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--match', action='append', dest='matches')
    group.add_argument('--all', action='store_true')
    args = parser.parse_args()
    with urllib.request.urlopen(ARTICLE, timeout=30) as response:
        article = json.load(response)
    selected = select_files(article['files'], args.matches or [])
    args.output.mkdir(parents=True, exist_ok=True)
    needed = sum(f['size'] for f in selected if not (args.output / f['name']).exists())
    if shutil.disk_usage(args.output).free < needed + 1024**3:
        raise SystemExit('Insufficient space: download needs a 1 GiB working reserve')
    manifest = {'article': ARTICLE, 'license': article['license'], 'files': []}
    for entry in selected:
        print(f"Downloading/verifying {entry['name']} ({entry['size']:,} bytes)", flush=True)
        manifest['files'].append(download_file(entry, args.output))
        (args.output / 'download-manifest.json').write_text(
            json.dumps(manifest, indent=2) + '\n')
    print(f"Verified {len(selected)} files under {args.output}", flush=True)


if __name__ == '__main__':
    main()
