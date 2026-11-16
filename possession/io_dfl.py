"""Streaming reader and source-field audit for Sportec DFL position XML."""

import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

from .schema import FrameTable, TrainingTargets


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _metadata(path):
    root = ET.parse(path).getroot()
    general = next((e for e in root.iter() if _tag(e) == "General"), None)
    if general is None:
        raise ValueError("DFL match information has no General element")
    home, away = general.get("HomeTeamId"), general.get("GuestTeamId")
    if not home or not away or home == away:
        raise ValueError("DFL match information lacks distinct home and away IDs")
    players = [[], []]
    seen = set()
    for team in (e for e in root.iter() if _tag(e) == "Team"):
        team_id = team.get("TeamId")
        if team_id not in (home, away):
            raise ValueError(f"unknown DFL team {team_id}")
        side = 0 if team_id == home else 1
        for player in (e for e in team.iter() if _tag(e) == "Player"):
            person = player.get("PersonId")
            if not person or person in seen:
                raise ValueError(f"missing or duplicate DFL player ID: {person}")
            seen.add(person)
            players[side].append(person)
    if not all(players):
        raise ValueError("DFL match information lacks players on a side")
    slots = {(person): (side, i) for side, side_players in enumerate(players)
             for i, person in enumerate(side_players)}
    environment = next((e for e in root.iter() if _tag(e) == "Environment"), None)
    pitch = (float(environment.get("PitchX")), float(environment.get("PitchY"))) if environment is not None else None
    return general.get("MatchId", ""), (home, away), slots, max(map(len, players)), pitch


def _float_xy(attrs):
    x, y = attrs.get("X"), attrs.get("Y")
    missing = {None, "", "null", "NaN", "nan"}
    if x in missing and y in missing:
        return None
    if x in missing or y in missing:
        raise ValueError("partially missing DFL coordinate")
    xy = (float(x), float(y))
    if not all(np.isfinite(xy)):
        raise ValueError("nonfinite observed DFL coordinate")
    return xy


class _Half:
    def __init__(self, period, p):
        self.period = period
        self.p = p
        self.base = None
        self.capacity = 0
        self.last = None
        self.stats = Counter()

    def _allocate(self, capacity):
        self.players_xy = np.zeros((capacity, 2, self.p, 2), dtype=np.float32)
        self.player_valid = np.zeros((capacity, 2, self.p), dtype=bool)
        self.player_recorded = np.zeros((capacity, 2, self.p), dtype=bool)
        self.ball_xy = np.zeros((capacity, 2), dtype=np.float32)
        self.ball_valid = np.zeros(capacity, dtype=bool)
        self.ball_recorded = np.zeros(capacity, dtype=bool)
        self.seen = np.zeros(capacity, dtype=bool)
        self.live = np.zeros(capacity, dtype=np.int8)
        self.live_valid = np.zeros(capacity, dtype=bool)
        self.owner = np.full(capacity, -1, dtype=np.int8)
        self.owner_valid = np.zeros(capacity, dtype=bool)
        self.state = np.full(capacity, -1, dtype=np.int8)
        self.state_valid = np.zeros(capacity, dtype=bool)

    def _extend(self, n):
        if self.base is None:
            self.base = n
            self.capacity = 1024
            self._allocate(self.capacity)
        if n < self.base:
            shift = self.base - n
            old = self._arrays()
            old_cap = self.capacity
            self.base = n
            self.capacity = max(1 << (old_cap + shift - 1).bit_length(), 1024)
            self._allocate(self.capacity)
            for name, values in old.items():
                getattr(self, name)[shift:shift + old_cap] = values
        elif n - self.base >= self.capacity:
            old = self._arrays()
            old_cap = self.capacity
            self.capacity = 1 << (n - self.base).bit_length()
            self._allocate(self.capacity)
            for name, values in old.items():
                getattr(self, name)[:old_cap] = values
        self.last = n if self.last is None else max(self.last, n)
        i = n - self.base
        self.seen[i] = True
        return i

    def _arrays(self):
        return {name: getattr(self, name) for name in (
            "players_xy", "player_valid", "player_recorded", "ball_xy", "ball_valid",
            "ball_recorded", "seen", "live",
            "live_valid", "owner", "owner_valid", "state", "state_valid")}

    def add(self, frame, kind, slot=None):
        n = int(frame["N"])
        i = self._extend(n)
        xy = _float_xy(frame)
        if kind == "ball":
            if self.ball_recorded[i]:
                raise ValueError(f"duplicate ball frame {n}")
            self.ball_recorded[i] = True
            self.stats["ball_frames"] += 1
            if xy is not None:
                self.ball_xy[i] = xy
                self.ball_valid[i] = True
                self.stats["ball_observed"] += 1
            if frame.get("Z") not in (None, "", "null", "NaN", "nan"):
                self.stats["z_observed"] += 1
            status = frame.get("BallStatus")
            possession = frame.get("BallPossession")
            status_key = status if status in {"0", "1"} else ("null" if status in {None, "", "null"} else "other")
            owner_key = possession if possession in {"1", "2"} else ("null" if possession in {None, "", "null"} else "other")
            self.stats[f"status_{status_key}"] += 1
            self.stats[f"owner_given_status_{status_key}_{owner_key}"] += 1
            if status == "0":
                self.live_valid[i] = True
                self.state[i] = 2
                self.state_valid[i] = True
            elif status == "1":
                self.live[i] = 1
                self.live_valid[i] = True
                if possession in {"1", "2"}:
                    self.owner[i] = 1 if possession == "1" else 0
                    self.owner_valid[i] = True
                    self.state[i] = 0 if possession == "1" else 1
                    self.state_valid[i] = True
        else:
            side, player = slot
            if self.player_recorded[i, side, player]:
                raise ValueError(f"duplicate player frame {n}")
            self.player_recorded[i, side, player] = True
            if xy is not None:
                self.players_xy[i, side, player] = xy
                self.player_valid[i, side, player] = True
                self.stats["player_observed"] += 1

    def finish(self, match_id, pitch):
        if self.base is None:
            raise ValueError("empty DFL half")
        length = self.last - self.base + 1
        if not self.seen[:length].all():
            raise ValueError("DFL half has noncontiguous source frame IDs")
        offset = np.array((pitch[0] / 2, pitch[1] / 2), dtype=np.float32)
        players_xy = self.players_xy[:length].copy()
        ball_xy = self.ball_xy[:length].copy()
        players_xy[self.player_valid[:length]] += offset
        ball_xy[self.ball_valid[:length]] += offset
        frames = FrameTable(
            np.arange(self.base, self.last + 1, dtype=np.int64),
            players_xy, self.player_valid[:length].copy(),
            ball_xy, self.ball_valid[:length].copy(),
            np.ones(length, dtype=bool), 25.0, pitch, match_id, self.period,
        )
        targets = TrainingTargets(*(getattr(self, name)[:length].copy() for name in
                                    ("live", "live_valid", "owner", "owner_valid", "state", "state_valid")))
        audit = dict(self.stats)
        live = self.live[:length]
        live_valid = self.live_valid[:length]
        owner = self.owner[:length]
        owner_valid = self.owner_valid[:length]
        adjacent_valid = live_valid[:-1] & live_valid[1:]
        audit["dead_to_active"] = int(np.sum(adjacent_valid & (live[:-1] == 0) & (live[1:] == 1)))
        audit["active_to_dead"] = int(np.sum(adjacent_valid & (live[:-1] == 1) & (live[1:] == 0)))
        audit["active_owner_switch"] = int(np.sum(
            adjacent_valid & owner_valid[:-1] & owner_valid[1:]
            & (live[:-1] == 1) & (live[1:] == 1) & (owner[:-1] != owner[1:])))
        audit["active_unknown_owner"] = int(np.sum(live_valid & (live == 1) & ~owner_valid))
        audit["unknown_status_frames"] = int(np.sum(~live_valid))
        for key in ("status_0", "status_1", "status_null", "status_other"):
            audit.setdefault(key, 0)
        audit.update(period=self.period, frames=length, first_frame=self.base, last_frame=self.last,
                     player_coverage=float(np.mean(self.player_valid[:length])),
                     mean_players_per_frame=float(np.mean(np.sum(self.player_valid[:length], axis=(1, 2)))),
                     ball_coverage=float(np.mean(self.ball_valid[:length])),
                     live_coverage=float(np.mean(self.live_valid[:length])),
                     owner_coverage=float(np.mean(self.owner_valid[:length])),
                     state_coverage=float(np.mean(self.state_valid[:length])))
        return frames, targets, audit


def _parse(position_path, match_info_path):
    match_id, team_ids, slots, p, meta_pitch = _metadata(match_info_path)
    halves = {"firstHalf": _Half(1, p), "secondHalf": _Half(2, p)}
    pitch = None
    source_match = None
    current = None
    last_n = None
    for event, element in ET.iterparse(position_path, events=("start", "end")):
        tag = _tag(element)
        if event == "start":
            if tag == "MetaData":
                source_match = element.get("MatchId")
            elif tag == "FrameSet":
                section, team, person = (element.get(k) for k in ("GameSection", "TeamId", "PersonId"))
                if section not in halves:
                    raise ValueError(f"unknown DFL GameSection {section}")
                if team == "BALL":
                    current = (halves[section], "ball", None)
                elif team in team_ids:
                    if person not in slots or slots[person][0] != team_ids.index(team):
                        raise ValueError(f"DFL person {person} is not on team {team}")
                    current = (halves[section], "player", slots[person])
                elif team == "referee":
                    current = (halves[section], "referee", None)
                else:
                    raise ValueError(f"unknown DFL frame team {team}")
                last_n = None
        elif tag == "PitchSize":
            pitch = (float(element.get("X")), float(element.get("Y")))
            element.clear()
        elif tag == "Frame":
            if current is None:
                raise ValueError("DFL Frame outside FrameSet")
            n = int(element.get("N"))
            if last_n is not None and n <= last_n:
                raise ValueError(f"duplicate or out-of-order DFL frame {n}")
            last_n = n
            half, kind, slot = current
            if kind == "referee":
                half._extend(n)
            else:
                half.add(element.attrib, kind, slot)
            element.clear()
        elif tag == "FrameSet":
            current = None
            element.clear()
    if source_match and match_id and source_match != match_id:
        raise ValueError("DFL match ID differs between positions and match information")
    pitch = pitch or meta_pitch
    if pitch is None or not all(np.isfinite(pitch)) or min(pitch) <= 0:
        raise ValueError("DFL pitch dimensions unavailable")
    return [halves[section].finish(match_id, pitch) for section in ("firstHalf", "secondHalf")
            if halves[section].base is not None]


def read_dfl(position_path, match_info_path) -> list[tuple[FrameTable, TrainingTargets]]:
    """Return each available half, preserving source frame IDs and masking unknown labels."""
    return [(frames, targets) for frames, targets, _ in _parse(position_path, match_info_path)]


def audit_dfl(position_path, match_info_path) -> dict:
    """Count raw label values and observation coverage for each half."""
    parsed = _parse(position_path, match_info_path)
    return {"match_id": parsed[0][0].match_id if parsed else "",
            "halves": [audit for _, _, audit in parsed]}
