"""Deterministic nested match splits for external possession evaluation."""

from __future__ import annotations

import random
import re


def _match_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"invalid match ID: {value!r}")
    return "DFL-MAT-" + value if re.fullmatch(r"J[0-9A-Z]{5}", value) else value


def nested_splits(matches, inner_folds: int = 3, seed: int = 0) -> list[dict]:
    """Leave each match out, then partition the remaining matches for validation.

    Every inner-validation match appears exactly once per outer fold. The two
    specified non-Düsseldorf Bundesliga matches are put into different
    validation groups whenever both are in the outer training set.
    """
    ids = [_match_id(value) for value in matches]
    if len(ids) < 3 or len(ids) != len(set(ids)):
        raise ValueError("at least three distinct match IDs are required")
    if not isinstance(inner_folds, int) or isinstance(inner_folds, bool) or inner_folds < 2:
        raise ValueError("inner_folds must be at least two")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    ids = sorted(ids)
    result = []
    pair = ("DFL-MAT-J03WMX", "DFL-MAT-J03WN1")
    for outer in ids:
        remaining = [match for match in ids if match != outer]
        n_groups = min(inner_folds, len(remaining))
        groups: list[list[str]] = [[] for _ in range(n_groups)]
        rng = random.Random(f"{seed}:{outer}")
        shuffled = remaining.copy()
        rng.shuffle(shuffled)
        if all(match in remaining for match in pair):
            groups[0].append(pair[0])
            groups[1].append(pair[1])
            shuffled = [match for match in shuffled if match not in pair]
        for match in shuffled:
            smallest = min(len(group) for group in groups)
            candidates = [index for index, group in enumerate(groups) if len(group) == smallest]
            groups[rng.choice(candidates)].append(match)
        if any(not group for group in groups):
            raise AssertionError("empty validation group")
        inner = []
        for group in groups:
            validation = sorted(group)
            train = sorted(set(remaining) - set(validation))
            if not train or set(train) & set(validation) or outer in train + validation:
                raise AssertionError("invalid nested split")
            inner.append({"train": train, "validation": validation})
        if sorted(match for fold in inner for match in fold["validation"]) != remaining:
            raise AssertionError("inner validation is not a partition")
        result.append({"outer_test": [outer], "outer_train": remaining, "inner": inner})
    return result
