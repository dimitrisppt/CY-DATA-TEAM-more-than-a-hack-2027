"""Regenerate the raw DFL status, owner, and tracking coverage audit."""

import argparse
import json
from pathlib import Path

from possession.io_dfl import audit_dfl
from possession.storage import sha256_file, write_json

from .prepare import _discover


def _publisher_manifest(raw_dir):
    path = raw_dir / "download-manifest.json"
    if not path.exists():
        return {}, {}
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("invalid download manifest JSON") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise TypeError("invalid download manifest")
    files = {entry["name"]: entry for entry in manifest["files"]}
    source = {key: manifest[key] for key in ("article", "license") if key in manifest}
    return files, source


def _fingerprint(path, publisher_files):
    digest = sha256_file(path)
    fingerprint = {"sha256": digest}
    publisher = publisher_files.get(path.name)
    if publisher is not None:
        if publisher.get("sha256") != digest:
            raise ValueError(f"download manifest hash mismatch for {path.name}")
        for source_key, output_key in (("download_url", "download_url"),
                                       ("id", "publisher_file_id"),
                                       ("computed_md5", "md5")):
            if source_key in publisher:
                fingerprint[output_key] = publisher[source_key]
    return fingerprint


def generate_audit(raw_dir, report):
    """Audit every complete XML pair and publish one atomic JSON report."""
    raw_dir = Path(raw_dir)
    positions = _discover(raw_dir, "positions")
    metadata = _discover(raw_dir, "matchinformation")
    if not positions and not metadata:
        raise ValueError("no DFL XML pairs found")
    unmatched = sorted(set(positions) ^ set(metadata))
    if unmatched:
        raise ValueError(f"missing DFL XML pair for {', '.join(unmatched)}")
    publisher_files, attribution = _publisher_manifest(raw_dir)
    matches = []
    for match_id in sorted(positions):
        pos, meta = positions[match_id], metadata[match_id]
        entry = audit_dfl(pos, meta)
        if entry["match_id"] != match_id:
            raise ValueError(f"DFL XML match ID mismatch for {match_id}")
        entry["source_positions"] = pos.name
        entry["source_match_info"] = meta.name
        entry["source_fingerprints"] = {
            "positions": _fingerprint(pos, publisher_files),
            "match_info": _fingerprint(meta, publisher_files),
        }
        matches.append(entry)
    result = {"matches": matches,
              "total_frames": sum(half["frames"] for entry in matches for half in entry["halves"])}
    if attribution:
        result["source"] = attribution
    write_json(report, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/external/dfl"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/dfl-audit.json"))
    args = parser.parse_args(argv)
    result = generate_audit(args.raw_dir, args.report)
    print(f"Audited {len(result['matches'])} matches, {result['total_frames']} frames: {args.report}")
    return result


if __name__ == "__main__":
    main()
