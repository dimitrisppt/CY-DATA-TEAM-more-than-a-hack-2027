"""Pinned SkillCorner proxy inputs and labels; reserved matches are never fitted here.

These provider-generated labels are not the challenge's corrected tracking truth.
No model is invoked by fetch, audit, or prepare. The evaluate command only consumes
predictions supplied after model selection has been frozen elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as official
from possession.schema import FrameTable, TrainingTargets

COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
MATCHES = ("2016236", "2017461")
DATA_DIR = Path("data/external/skillcorner-proxy")
TRACKING_SIZE = {"2016236": 94737872, "2017461": 89543442}
SMALL_SIZES = {
    "2016236": {"match": 33791, "phases_of_play": 133630},
    "2017461": {"match": 28545, "phases_of_play": 118313},
}
SOURCE_SHA256 = {
    "2016236_tracking_extrapolated.jsonl": "d78555f9dfd7d1ce9de38066d3c98e6d54417c6d45d73b1f1ce7e26f07263d8f",
    "2016236_match.json": "93ccf67cead82def1acda87d3cf550fd42f96cb0e1adbff9eeb8824077980495",
    "2016236_phases_of_play.csv": "cb162b42bd95306404da15071df07f70cb0458db9b532930dac2a1b0092242e9",
    "2017461_tracking_extrapolated.jsonl": "59cfa5588efee041da439f44df5b5b92f2b7305571dc831164e3d05f7415787a",
    "2017461_match.json": "db27759bf734dc8e3d9e56f047965cc7c74833bc3d0345ee3821ab97681ce4b9",
    "2017461_phases_of_play.csv": "5331495ecea48056df56dc843864bf36f366f7bca3898d82273eba7a3a54eba5",
}


def source_files():
    """Return only the six named, commit-pinned public source files."""
    base = f"https://{{host}}/SkillCorner/opendata/{COMMIT}/data/matches"
    for match in MATCHES:
        for kind, suffix, size in (
            ("tracking", "tracking_extrapolated.jsonl", TRACKING_SIZE[match]),
            ("match", "match.json", SMALL_SIZES[match]["match"]),
            ("phases_of_play", "phases_of_play.csv", SMALL_SIZES[match]["phases_of_play"]),
        ):
            name = f"{match}_{suffix}"
            host = "media.githubusercontent.com/media" if kind == "tracking" else "raw.githubusercontent.com"
            yield {"match_id": match, "kind": kind, "filename": name,
                   "url": base.format(host=host) + f"/{match}/{name}", "bytes": size,
                   "expected_sha256": SOURCE_SHA256[name]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_destination(directory: Path, filename: str) -> Path:
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("unsafe source filename")
    return directory / filename


def fetch(directory: Path = DATA_DIR) -> dict:
    """Download only pinned files; verify size and LFS object SHA-256 before publication."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in source_files():
        final = _safe_destination(directory, entry["filename"])
        if final.exists() and final.stat().st_size == entry["bytes"]:
            digest = sha256_file(final)
        else:
            partial = final.with_suffix(final.suffix + ".part")
            try:
                with urllib.request.urlopen(entry["url"], timeout=120) as response, partial.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                if partial.stat().st_size != entry["bytes"]:
                    raise ValueError(f"size mismatch for {entry['filename']}")
                digest = sha256_file(partial)
                if digest != entry["expected_sha256"]:
                    raise ValueError(f"hash mismatch for {entry['filename']}")
                os.replace(partial, final)
            finally:
                partial.unlink(missing_ok=True)
        if final.stat().st_size != entry["bytes"]:
            raise ValueError(f"size mismatch for {entry['filename']}")
        if digest != entry["expected_sha256"]:
            raise ValueError(f"hash mismatch for {entry['filename']}")
        if entry["kind"] == "tracking" and digest != _lfs_sha256(entry):
            raise ValueError(f"LFS pointer mismatch for {entry['filename']}")
        records.append({**entry, "sha256": digest})
    manifest = {"provider": "SkillCorner/opendata", "commit": COMMIT,
                "label_status": "provider-generated proxy; no corrected dead-ball ground truth",
                "files": records}
    temp = directory / "manifest.json.part"
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temp, directory / "manifest.json")
    return manifest


def verify_sources(directory: Path) -> None:
    """Refuse to prepare proxy data if any pinned source bytes have changed."""
    for entry in source_files():
        path = _safe_destination(Path(directory), entry["filename"])
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"missing or wrong-sized source: {entry['filename']}")
        if sha256_file(path) != entry["expected_sha256"]:
            raise ValueError(f"source hash mismatch: {entry['filename']}")


def _lfs_sha256(entry: dict) -> str:
    """Read the tiny immutable Git LFS pointer rather than trusting a moving branch."""
    path = f"data/matches/{entry['match_id']}/{entry['filename']}"
    url = f"https://raw.githubusercontent.com/SkillCorner/opendata/{COMMIT}/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        pointer = response.read(512).decode("ascii")
    lines = dict(line.split(" ", 1) for line in pointer.splitlines() if " " in line)
    if lines.get("version") != "https://git-lfs.github.com/spec/v1":
        raise ValueError("expected LFS pointer")
    if int(lines.get("size", -1)) != entry["bytes"]:
        raise ValueError("LFS pointer size mismatch")
    oid = lines.get("oid", "")
    if not oid.startswith("sha256:") or len(oid) != 71:
        raise ValueError("invalid LFS object ID")
    return oid[7:]


@dataclass
class ProxyHalf:
    """Model input and native proxy labels are deliberately different objects/timelines."""

    input_25hz: FrameTable
    source_frame_10hz: np.ndarray
    source_to_model: np.ndarray
    proxy_10hz: TrainingTargets
    audit: dict


def _metadata(directory: Path, match: str) -> tuple[dict, dict, list]:
    metadata = json.loads((directory / f"{match}_match.json").read_text())
    if str(metadata.get("id")) != match:
        raise ValueError("match metadata ID mismatch")
    home, away = metadata["home_team"]["id"], metadata["away_team"]["id"]
    if not isinstance(home, int) or not isinstance(away, int) or home == away:
        raise ValueError("invalid home/away team IDs")
    players = {}
    for row in metadata["players"]:
        pid, tid = row["id"], row["team_id"]
        if pid in players or tid not in (home, away):
            raise ValueError("duplicate player or unknown team ID")
        players[pid] = tid
    periods = sorted(metadata["match_periods"], key=lambda row: row["period"])
    if [p["period"] for p in periods] != [1, 2]:
        raise ValueError("expected two metadata periods")
    if any(p["end_frame"] < p["start_frame"] for p in periods):
        raise ValueError("invalid period bounds")
    return metadata, players, periods


def _phase_owners(directory: Path, match: str, metadata: dict,
                  periods: list) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    owners = {p["period"]: np.full(p["end_frame"] - p["start_frame"] + 1, -1, np.int64)
              for p in periods}
    home, away = metadata["home_team"]["id"], metadata["away_team"]["id"]
    overhang = {p["period"]: 0 for p in periods}
    with (directory / f"{match}_phases_of_play.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row["match_id"]) != match:
                raise ValueError("phase match ID mismatch")
            period = int(row["period"])
            if period not in owners:
                raise ValueError("phase has unknown period")
            team = int(row["team_in_possession_id"])
            if team not in (home, away):
                raise ValueError("phase has unknown owning team")
            start, end = int(row["frame_start"]), int(row["frame_end"])
            base = periods[period - 1]["start_frame"]
            # Provider duration = (end - start) / 10: phase end is exclusive.
            limit = base + len(owners[period])
            if not base <= start < limit or end <= start or end > limit + 10:
                raise ValueError("phase lies outside its period")
            overhang[period] += max(0, end - limit)
            section = owners[period][start - base:min(end, limit) - base]
            if np.any((section != -1) & (section != team)):
                raise ValueError("conflicting overlapping phases")
            section[:] = team
    return owners, overhang


def _source_index_and_weights(n_source: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """25 Hz grid inside a 10 Hz half; never extrapolate past the last sample."""
    if n_source < 1:
        raise ValueError("empty source half")
    n_model = (n_source - 1) * 5 // 2 + 1
    source_fifths = np.arange(n_model, dtype=np.int64) * 2
    left = source_fifths // 5
    remainder = source_fifths % 5
    right = left + (remainder > 0)
    weight = remainder.astype(np.float32) / 5
    return left, right, weight


def resample_observed(xy: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate only across adjacent detected source samples, slot by slot."""
    xy = np.asarray(xy, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if xy.shape[-1] != 2 or valid.shape != xy.shape[:-1]:
        raise ValueError("xy/valid shape mismatch")
    left, right, weight = _source_index_and_weights(len(xy))
    exact = left == right
    keep = valid[left] & (exact.reshape((-1,) + (1,) * (valid.ndim - 1)) | valid[right])
    w = weight.reshape((-1,) + (1,) * (xy.ndim - 1)).astype(np.float32)
    out = xy[left] * (1 - w) + xy[right] * w
    out[~keep] = 0
    return out, keep


def _mapping(n_source: int) -> np.ndarray:
    """Nearest 25 Hz model instant for each 10 Hz source frame (<= 0.02 s error)."""
    n_model = len(_source_index_and_weights(n_source)[0])
    index = (np.arange(n_source, dtype=np.int64) * 5 + 1) // 2
    return np.minimum(index, n_model - 1)


def _init_period(period: dict, p: int) -> dict:
    n = period["end_frame"] - period["start_frame"] + 1
    return {"xy": np.zeros((n, 2, p, 2), np.float32),
            "valid": np.zeros((n, 2, p), bool),
            "ball": np.zeros((n, 2), np.float32), "ball_valid": np.zeros(n, bool),
            "group": np.full(n, -1, np.int8), "seen": np.zeros(n, bool),
            "player_rows": 0, "player_detected": 0, "ball_coordinates": 0,
            "ball_extrapolated": 0}


def _finite_pair(row: dict) -> np.ndarray | None:
    x, y = row.get("x"), row.get("y")
    if x is None and y is None:
        return None
    if x is None or y is None or not np.isfinite([x, y]).all():
        raise ValueError("partial or nonfinite coordinates")
    return np.array([x, y], dtype=np.float32)


def _read_tracking(directory: Path, match: str, metadata: dict, players: dict,
                   periods: list) -> tuple[dict[int, dict], dict]:
    home, away = metadata["home_team"]["id"], metadata["away_team"]["id"]
    sides = ([sorted(pid for pid, team in players.items() if team == tid)
              for tid in (home, away)])
    p = max(map(len, sides))
    slots = {pid: (side, slot) for side, ids in enumerate(sides)
             for slot, pid in enumerate(ids)}
    buffers = {row["period"]: _init_period(row, p) for row in periods}
    outside = 0
    with (directory / f"{match}_tracking_extrapolated.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            period = row["period"]
            if period is None:
                outside += 1
                continue
            if period not in buffers:
                raise ValueError("tracking row has unknown period")
            buf = buffers[period]
            base = periods[period - 1]["start_frame"]
            frame = row["frame"]
            if not isinstance(frame, int) or not base <= frame < base + len(buf["seen"]):
                raise ValueError("tracking frame outside metadata period")
            i = frame - base
            if buf["seen"][i]:
                raise ValueError("duplicate tracking frame")
            buf["seen"][i] = True
            group = row["possession"]["group"]
            if group not in (None, "home team", "away team"):
                raise ValueError("unknown possession group")
            buf["group"][i] = {None: -1, "home team": 0, "away team": 1}[group]
            ball = row["ball_data"]
            bxy = _finite_pair(ball)
            if bxy is not None:
                buf["ball_coordinates"] += 1
            if ball["is_detected"] is True:
                if bxy is None:
                    raise ValueError("detected ball lacks coordinates")
                buf["ball"][i] = bxy
                buf["ball_valid"][i] = True
            elif bxy is not None:
                buf["ball_extrapolated"] += 1
            used = set()
            for item in row["player_data"]:
                pid = item["player_id"]
                if pid in used or pid not in slots:
                    raise ValueError("duplicate or unknown player ID")
                used.add(pid)
                buf["player_rows"] += 1
                xy = _finite_pair(item)
                if item["is_detected"] is True:
                    if xy is None:
                        raise ValueError("detected player lacks coordinates")
                    side, slot = slots[pid]
                    buf["xy"][i, side, slot] = xy
                    buf["valid"][i, side, slot] = True
                    buf["player_detected"] += 1
    if any(not b["seen"].all() for b in buffers.values()):
        raise ValueError("source tracking omits a metadata-period frame")
    return buffers, {"outside_period_rows": outside, "slots_per_side": [len(s) for s in sides]}


def _labels(groups: np.ndarray, phase_team: np.ndarray, home: int, away: int) -> tuple[TrainingTargets, dict]:
    if groups.shape != phase_team.shape:
        raise ValueError("group and phase arrays differ")
    candidate_team = np.where(groups == 0, home, np.where(groups == 1, away, -1))
    in_phase = phase_team != -1
    nonnull = groups != -1
    agreement = in_phase & nonnull & (candidate_team == phase_team)
    disagreement = in_phase & nonnull & (candidate_team != phase_team)
    n = len(groups)
    live = np.zeros(n, np.int8)
    live[agreement] = 1
    state = np.full(n, -1, np.int8)
    state[agreement] = groups[agreement]
    owner = np.full(n, -1, np.int8)
    owner[agreement] = 1 - groups[agreement]
    targets = TrainingTargets(live, agreement.copy(), owner, agreement.copy(),
                              state, agreement.copy())
    audit = {"frames": n, "phase_frames": int(in_phase.sum()),
             "phase_gap_unknown": int((~in_phase).sum()),
             "group_nonnull": int(nonnull.sum()),
             "phase_and_group_agree": int(agreement.sum()),
             "phase_and_group_disagree": int(disagreement.sum()),
             "phase_group_null": int((in_phase & ~nonnull).sum()),
             "proxy_label_coverage": float(agreement.mean()),
             "dead_truth_frames": 0}
    return targets, audit


def prepare_match(directory: Path, match: str) -> list[ProxyHalf]:
    """Build conservative 25 Hz inputs and separate 10 Hz H/A-only proxy labels."""
    directory = Path(directory)
    if match not in MATCHES:
        raise ValueError("match is not a reserved proxy holdout")
    metadata, players, periods = _metadata(directory, match)
    phase_teams, phase_overhang = _phase_owners(directory, match, metadata, periods)
    buffers, common_audit = _read_tracking(directory, match, metadata, players, periods)
    pitch = (float(metadata["pitch_length"]), float(metadata["pitch_width"]))
    home, away = metadata["home_team"]["id"], metadata["away_team"]["id"]
    output = []
    for period_meta in periods:
        period = period_meta["period"]
        buf = buffers[period]
        source_frames = np.arange(period_meta["start_frame"], period_meta["end_frame"] + 1)
        labels, audit = _labels(buf["group"], phase_teams[period], home, away)
        # SkillCorner and DFL coordinates are both centred in metres; FrameTable uses [0,pitch].
        offset = np.array([pitch[0] / 2, pitch[1] / 2], dtype=np.float32)
        player_xy = buf["xy"].copy()
        ball_xy = buf["ball"].copy()
        player_xy[buf["valid"]] += offset
        ball_xy[buf["ball_valid"]] += offset
        players25, player_valid25 = resample_observed(player_xy, buf["valid"])
        ball25, ball_valid25 = resample_observed(ball_xy, buf["ball_valid"])
        frames25 = FrameTable(np.arange(len(ball25)), players25, player_valid25,
                              ball25, ball_valid25, np.ones(len(ball25), bool),
                              25.0, pitch, match, period)
        audit.update({"period": period, "source_start_frame": int(source_frames[0]),
                      "source_end_frame": int(source_frames[-1]),
                      "phase_frames_clipped_after_period": phase_overhang[period],
                      "model_25hz_frames": len(ball25),
                      "max_source_to_model_time_error_s": float(np.max(np.abs(
                          _mapping(len(source_frames)) / 25 - np.arange(len(source_frames)) / 10))),
                      "detected_ball_frames": int(buf["ball_valid"].sum()),
                      "detected_player_observations": buf["player_detected"],
                      "all_player_rows": buf["player_rows"],
                      "ball_extrapolated_masked": buf["ball_extrapolated"],
                      **common_audit})
        output.append(ProxyHalf(frames25, source_frames, _mapping(len(source_frames)), labels, audit))
    return output


def save_prepared(directory: Path = DATA_DIR) -> dict:
    """Save separate input, proxy-label, and timestamp-map files; never mix labels into inputs."""
    directory = Path(directory)
    verify_sources(directory)
    audit = {"provider_commit": COMMIT, "matches": {},
             "warning": "H/A agreement subset only; D is unknown; no model scores computed"}
    for match in MATCHES:
        audit["matches"][match] = []
        for half in prepare_match(directory, match):
            period = half.input_25hz.period
            prefix = directory / f"{match}_half{period}"
            f = half.input_25hz
            np.savez_compressed(str(prefix) + "_input25.npz",
                                frame_id=f.frame_id, players_xy=f.players_xy,
                                player_valid=f.player_valid, ball_xy=f.ball_xy,
                                ball_valid=f.ball_valid, graded=f.graded,
                                fps=np.array(f.fps), pitch=np.array(f.pitch),
                                match_id=np.array(f.match_id), period=np.array(f.period))
            y = half.proxy_10hz
            np.savez_compressed(str(prefix) + "_proxy10_labels.npz",
                                live=y.live, live_valid=y.live_valid,
                                owner=y.owner, owner_valid=y.owner_valid,
                                state=y.state, state_valid=y.state_valid)
            np.savez_compressed(str(prefix) + "_source_map.npz",
                                source_frame_10hz=half.source_frame_10hz,
                                model_index_25hz=half.source_to_model,
                                source_fps=np.array(10.0), model_fps=np.array(25.0))
            audit["matches"][match].append(half.audit)
    path = directory / "coverage-audit.json"
    path.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def score_native_proxy(labels: TrainingTargets, model_codes_25hz: np.ndarray,
                       source_to_model: np.ndarray, *, selection_frozen: bool = False) -> dict:
    """Conditional H/A proxy score at native 10 Hz, using official spell/transition matching.

    Caller must attest selection was frozen before looking at these holdouts. No dead-ball
    F1 or challenge score is computed because this proxy has no reliable D truth.
    """
    if not selection_frozen:
        raise ValueError("freeze model selection before viewing proxy holdout scores")
    model_codes_25hz = np.asarray(model_codes_25hz)
    source_to_model = np.asarray(source_to_model)
    if source_to_model.shape != labels.state.shape or not np.issubdtype(source_to_model.dtype, np.integer):
        raise ValueError("source-to-model mapping shape/type mismatch")
    if np.any(source_to_model < 0) or np.any(source_to_model >= len(model_codes_25hz)):
        raise ValueError("source-to-model mapping outside model timeline")
    pred = official.label_codes(model_codes_25hz) if model_codes_25hz.dtype.kind in "USO" else model_codes_25hz
    if not np.isin(pred, range(4)).all():
        raise ValueError("prediction label outside H/A/D/X")
    pred10 = pred[source_to_model]
    gt10 = np.where(labels.state_valid, labels.state, official.X).astype(np.int8)
    runs = official.valid_runs(gt10)
    filled = official.fill_abstentions(pred10, runs)
    cm = official.scored_confusion(official.confusion(gt10, pred10))
    spells = official.spell_scores(gt10, filled, runs)
    gt_spells = official.possession_spells(gt10, runs)
    pred_spells = official.possession_spells(filled, runs)
    return {"label_kind": "SkillCorner H/A agreement proxy; D unverified",
            "native_fps": 10, "graded_HA_frames": int(labels.state_valid.sum()),
            "total_native_frames": len(gt10),
            "coverage": float(labels.state_valid.mean()),
            "team_macro_f1_conditional": float(np.mean([
                official.per_class_prf(cm, k)[2] for k in (official.H, official.A)])),
            "spell_f1_mean_conditional": spells["mean_f1"],
            "spell_f1_by_iou": spells["per_iou"],
            "n_gt_spells": spells["n_gt"], "n_pred_spells": spells["n_pred"],
            "gt_mean_spell_duration_s": float(np.mean([b-a for a,b,_ in gt_spells])/10) if gt_spells else 0.0,
            "pred_mean_spell_duration_s": float(np.mean([b-a for a,b,_ in pred_spells])/10) if pred_spells else 0.0,
            "transitions": {tol: official.transition_scores(
                official.transitions(gt10, runs), official.transitions(filled, runs), tol)
                for tol in (5, 10)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "prepare"])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        result = fetch(args.data_dir)
        print(json.dumps({"commit": result["commit"], "files": len(result["files"]),
                          "manifest": str(args.data_dir / "manifest.json")}, indent=2))
    elif args.command == "prepare":
        result = save_prepared(args.data_dir)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
