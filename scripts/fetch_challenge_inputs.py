"""Import only authorised tactical input files from Google Drive ZIP downloads.

The participant folder's assets are licensed separately from this repository.
This script never extracts videos, corrected labels, public GT, or broadcast files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

ALLOWED = {
    "BAR-ATH_tactical_atd.txt",
    "BAR-ATH_tactical_atd.xml",
    "BAR-ATH_tactical_homography.csv",
    "BAR-ATH_tactical_smart_tagging.xml",
}
ZIP_PREFIX = "More Than A Hack 2027 - Initial Challenge - Dataset/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_from_zips(archives: list[Path], destination: Path) -> dict:
    """Verify unique safe archive members, preserve existing files, write SHA manifest."""
    found = {}
    for path in map(Path, archives):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                prefix = ZIP_PREFIX + "tactical/"
                if not member.filename.startswith(prefix):
                    continue
                name = member.filename[len(prefix):]
                if name not in ALLOWED:
                    continue
                if name in found:
                    raise ValueError(f"duplicate tactical asset {name}")
                found[name] = (path, member.filename, member.file_size)
    if set(found) != ALLOWED:
        raise ValueError(f"tactical inputs missing: {sorted(ALLOWED - set(found))}")
    if any("/" in name or "\\" in name or Path(name).name != name for name in found):
        raise ValueError("unsafe archive member")
    tactical = Path(destination) / "tactical"
    tactical.mkdir(parents=True, exist_ok=True)
    largest_working_file = max(size for _, _, size in found.values())
    if shutil.disk_usage(tactical).free < 5 * 1024**3 + largest_working_file:
        raise ValueError("insufficient disk space for 5 GiB reserve and temporary extraction")
    records = []
    for name in sorted(found):
        archive_path, member_name, expected_size = found[name]
        output = tactical / name
        if output.exists() and (not output.is_file() or output.stat().st_size != expected_size):
            raise ValueError(f"existing tactical file differs in size: {name}")
        # Preserve existing bytes; compare their hash to the same ZIP member below.
        partial = output.with_suffix(output.suffix + ".part")
        try:
            digest = hashlib.sha256()
            size = 0
            with (zipfile.ZipFile(archive_path) as archive,
                  archive.open(member_name) as source,
                  partial.open("wb") as target):
                while block := source.read(1024 * 1024):
                    target.write(block)
                    digest.update(block)
                    size += len(block)
            if size != expected_size:
                raise ValueError(f"archive member size mismatch: {name}")
            if output.exists():
                if sha256(output) != digest.hexdigest():
                    raise ValueError(f"existing tactical file differs in content: {name}")
            else:
                os.replace(partial, output)
        finally:
            partial.unlink(missing_ok=True)
        records.append({"path": f"tactical/{name}", "bytes": expected_size,
                        "sha256": digest.hexdigest(),
                        "source_archive": archive_path.name,
                        "source_archive_member": member_name})
    manifest = {"source": "participant-supplied local ZIP archives",
                "scope": "tactical automatic inputs only, no video or labels",
                "files": records}
    tmp = Path(destination) / "input-manifest.json.part"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, Path(destination) / "input-manifest.json")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/target/2027"))
    args = parser.parse_args(argv)
    result = import_from_zips(args.archives, args.output)
    print(json.dumps({"files": len(result["files"]),
                      "bytes": sum(row["bytes"] for row in result["files"]),
                      "manifest": str(args.output / "input-manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
