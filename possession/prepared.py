"""Verify local prepared-cache provenance before training or evaluation."""

import json
import re
from pathlib import Path

from .storage import sha256_file

_CACHE = re.compile(r"(DFL-MAT-[A-Za-z0-9]+)_half([12])\.npz$")
_SHA256 = re.compile(r"[0-9a-f]{64}$")


def _checked_entry(entry, match_id, data_dir, expected_sources=None):
    if not isinstance(entry, dict) or entry.get("match_id") != match_id:
        raise ValueError(f"invalid manifest entry for {match_id}")
    sources = entry.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"positions", "match_info"}:
        raise ValueError(f"missing manifest source fingerprints for {match_id}")
    for source in sources.values():
        if (not isinstance(source, dict) or not isinstance(source.get("path"), str)
                or not source["path"] or not isinstance(source.get("sha256"), str)
                or not _SHA256.fullmatch(source["sha256"])):
            raise ValueError(f"invalid manifest source fingerprint for {match_id}")
    if expected_sources is not None and sources != expected_sources:
        raise ValueError(f"source hash changed for {match_id}")
    halves = entry.get("halves")
    if not isinstance(halves, list) or len(halves) != 2:
        raise ValueError(f"manifest must declare both halves for {match_id}")
    periods = [half.get("period") if isinstance(half, dict) else None for half in halves]
    if set(periods) != {1, 2} or len(set(periods)) != 2:
        raise ValueError(f"manifest has duplicate or missing half period for {match_id}")
    caches = []
    for half in sorted(halves, key=lambda value: value["period"]):
        name = f"{match_id}_half{half['period']}.npz"
        digest = half.get("cache_sha256")
        if half.get("cache") != name or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid manifest cache name or hash for {match_id}")
        path = data_dir / name
        if not path.is_file():
            raise ValueError(f"manifest cache is missing: {name}")
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"cache hash changed for {name}")
        caches.append({"cache": name, "sha256": actual, "match_id": match_id,
                       "period": half["period"]})
    return {"sources": sources, "caches": caches}


def verify_prepared(data_dir, paths):
    """Return JSONable evidence for selected caches, or reject a broken manifest."""
    data_dir = Path(data_dir).resolve()
    selected = [Path(path) for path in paths]
    if not selected:
        raise ValueError("no prepared caches selected")
    selected = [(path if path.is_absolute() or path.exists() else data_dir / path).resolve()
                for path in selected]
    for path in selected:
        if path.parent != data_dir or not path.is_file():
            raise ValueError(f"prepared cache is outside data directory or missing: {path}")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        return {"status": "unmanifested_local_cache",
                "caches": [{"cache": path.name, "sha256": sha256_file(path)} for path in selected]}
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("invalid prepared manifest JSON") from exc
    if (not isinstance(manifest, dict) or manifest.get("version") != 1
            or not isinstance(manifest.get("matches"), dict)):
        raise ValueError("invalid prepared manifest version or matches")
    by_match = {}
    for path in selected:
        match = _CACHE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"noncanonical cache filename in manifest dataset: {path.name}")
        by_match.setdefault(match.group(1), set()).add(path.name)
    sources = {}
    caches = []
    for match_id in sorted(by_match):
        entry = manifest["matches"].get(match_id)
        checked = _checked_entry(entry, match_id, data_dir)
        expected = {cache["cache"] for cache in checked["caches"]}
        if by_match[match_id] != expected:
            raise ValueError(f"both half periods must be selected for {match_id}")
        sources[match_id] = checked["sources"]
        caches.extend(checked["caches"])
    return {"status": "verified_manifest", "manifest_version": 1,
            "sources": sources, "caches": caches}
