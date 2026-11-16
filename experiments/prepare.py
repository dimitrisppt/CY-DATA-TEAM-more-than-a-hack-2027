"""Prepare provenance-checked local DFL half caches from verified source XML."""

import json
import re
from pathlib import Path

from possession.io_dfl import read_dfl
from possession.prepared import _checked_entry
from possession.storage import save_half, sha256_file, write_json

_MATCH_ID = re.compile(r"(DFL-MAT-[^.]+)\.xml$")


def _discover(raw_dir, marker):
    found = {}
    for path in Path(raw_dir).glob(f"*{marker}*DFL-MAT-*.xml"):
        match = _MATCH_ID.search(path.name)
        if match is None:
            continue
        match_id = match.group(1)
        if match_id in found:
            raise ValueError(f"duplicate {marker} XML for {match_id}")
        found[match_id] = path
    return found


def _source(path):
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _reusable(entry, match_id, sources, output_dir):
    try:
        _checked_entry(entry, match_id, output_dir, expected_sources=sources)
        return True
    except ValueError:
        return False


def prepare_dfl(raw_dir, output_dir, *, matches=None):
    """Build per-half NPZs, then atomically publish hashes and timeline metadata."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    positions = _discover(raw_dir, "positions")
    metadata = _discover(raw_dir, "matchinformation")
    requested = ({f"DFL-MAT-{name}" if not name.startswith("DFL-MAT-") else name
                  for name in matches} if matches is not None else set(positions) | set(metadata))
    if not requested:
        raise ValueError("no DFL matches found")
    absent = sorted(match for match in requested if match not in positions or match not in metadata)
    if absent:
        raise ValueError(f"DFL XML pair missing for {', '.join(absent)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    try:
        old = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        old = {}
    old_matches = old.get("matches", {}) if isinstance(old, dict) and old.get("version") == 1 else {}
    if not isinstance(old_matches, dict):
        old_matches = {}
    result = {"version": 1, "matches": {}}
    for match_id, entry in old_matches.items():
        if match_id in requested:
            continue
        try:
            _checked_entry(entry, match_id, output_dir)
        except ValueError:
            continue
        result["matches"][match_id] = entry
    for match_id in sorted(requested):
        sources = {"positions": _source(positions[match_id]),
                   "match_info": _source(metadata[match_id])}
        previous = old_matches.get(match_id)
        if _reusable(previous, match_id, sources, output_dir):
            result["matches"][match_id] = previous
            continue
        parsed = read_dfl(positions[match_id], metadata[match_id])
        if ([frames.period for frames, _ in parsed] != [1, 2]
                or any(frames.match_id != match_id for frames, _ in parsed)):
            raise ValueError(f"DFL XML has missing halves or match ID mismatch for {match_id}")
        entry = {"match_id": match_id, "sources": sources, "halves": []}
        for frames, targets in parsed:
            cache = f"{match_id}_half{frames.period}.npz"
            cache_path = output_dir / cache
            save_half(cache_path, frames, targets)
            entry["halves"].append({
                "period": frames.period,
                "cache": cache,
                "cache_sha256": sha256_file(cache_path),
                "frames": len(frames.frame_id),
                "first_frame": int(frames.frame_id[0]),
                "last_frame": int(frames.frame_id[-1]),
                "pitch": list(frames.pitch),
                "fps": frames.fps,
            })
        result["matches"][match_id] = entry
    write_json(manifest_path, result)
    return result
