"""Strict reader for the participant Metrica EPTS export."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .schema import FrameTable


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _slot_teams(metadata_path):
    root = ET.parse(metadata_path).getroot()
    teams = [element.get("id") for element in root.iter() if _tag(element) == "Team"]
    if len(teams) != 2 or not all(teams) or teams[0] == teams[1]:
        raise ValueError("ATD metadata must contain exactly two distinct teams")
    slots = [element.get("teamId") for element in root.iter() if _tag(element) == "Player"]
    if not slots or any(team not in teams for team in slots):
        raise ValueError("ATD metadata has no players or an unknown player team")
    return np.array([teams.index(team) for team in slots], dtype=np.int8)


def _xy(group, scale):
    group = group.strip()
    if group.lower() in {"*", "nan"}:
        return None
    values = group.split(",")
    if len(values) < 2:
        raise ValueError(f"malformed ATD coordinate group: {group!r}")
    x_text, y_text = values[:2]
    if x_text.lower() in {"nan", "*"} and y_text.lower() in {"nan", "*"}:
        return None
    x, y = float(x_text), float(y_text)
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError(f"nonfinite observed ATD coordinate: {group!r}")
    return x * scale[0], y * scale[1]


def read_atd(raw_path, metadata_path, *, n_frames=None, pitch=(105.3, 68), fps=25,
             graded_spans=None) -> FrameTable:
    """Read one-based EPTS rows as zero-based video-aligned frames."""
    teams = _slot_teams(metadata_path)
    slots = [np.flatnonzero(teams == team) for team in (0, 1)]
    p = max(map(len, slots))
    slot_index = [(int(teams[i]), int(np.searchsorted(slots[int(teams[i])], i)))
                  for i in range(len(teams))]
    ids, all_xy, all_valid, ball_xy, ball_valid = [], [], [], [], []
    with open(raw_path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank ATD row at line {lineno}")
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                raise ValueError(f"ATD row {lineno} must have frame:players:ball")
            source_id = int(parts[0])
            if source_id != lineno:
                raise ValueError(f"ATD frame_id at line {lineno} is not ordered 1-based contiguous")
            groups = parts[1].split(";")
            if len(groups) != len(teams):
                raise ValueError(f"ATD frame {source_id} has incorrect slot count")
            xy = np.zeros((2, p, 2), dtype=np.float32)
            valid = np.zeros((2, p), dtype=bool)
            for group, (team, index) in zip(groups, slot_index):
                point = _xy(group, pitch)
                if point is not None:
                    xy[team, index] = point
                    valid[team, index] = True
            ball_point = _xy(parts[2], pitch)
            ids.append(source_id - 1)
            all_xy.append(xy)
            all_valid.append(valid)
            ball_xy.append(ball_point if ball_point is not None else (0, 0))
            ball_valid.append(ball_point is not None)
    if not ids:
        raise ValueError("empty ATD file")
    if n_frames is not None and len(ids) != n_frames:
        raise ValueError(f"ATD has {len(ids)} frames, expected {n_frames}")
    graded = np.zeros(len(ids), dtype=bool)
    if graded_spans is None:
        graded[:] = True
    else:
        for start, stop in graded_spans:
            if not (0 <= start <= stop <= len(ids)):
                raise ValueError("graded span is outside ATD timeline")
            graded[start:stop] = True
    return FrameTable(np.asarray(ids), np.stack(all_xy), np.stack(all_valid),
                      np.asarray(ball_xy), np.asarray(ball_valid), graded,
                      float(fps), tuple(pitch), Path(raw_path).stem, 1)
