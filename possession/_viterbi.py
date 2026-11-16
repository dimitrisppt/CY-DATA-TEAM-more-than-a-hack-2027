"""Exact three-state Viterbi loop with an optional cached Numba specialization."""

import numpy as np


def viterbi_python(emit, transition, cost):
    """Return (path, score, switches, ties) for validated float64 inputs.

    The comparison order is deliberate: count every exact score tie before
    applying the fewer-switches and lower-index predecessor preferences.
    """
    n = len(emit)
    if n == 0:
        return np.empty(0, dtype=np.int8), 0.0, 0, 0
    prior_score = emit[0].copy()
    prior_switches = np.zeros(3, dtype=np.int64)
    back = np.zeros((n, 3), dtype=np.int8)
    ties = 0
    for t in range(1, n):
        score = np.empty(3, dtype=np.float64)
        switches = np.empty(3, dtype=np.int64)
        for dest in range(3):
            best_score = -np.inf
            best_switches = np.iinfo(np.int64).max
            best_prev = 0
            for prev in range(3):
                change = int(prev != dest)
                candidate = prior_score[prev] + emit[t, dest]
                if change:
                    candidate += transition[t, prev, dest] - cost[prev, dest]
                count = int(prior_switches[prev]) + change
                if candidate == best_score:
                    ties += 1
                if candidate > best_score or (candidate == best_score and count < best_switches):
                    best_score, best_switches, best_prev = candidate, count, prev
            score[dest], switches[dest], back[t, dest] = best_score, best_switches, best_prev
        prior_score, prior_switches = score, switches

    final = 0
    for state in (1, 2):
        if prior_score[state] == prior_score[final]:
            ties += 1
        if prior_score[state] > prior_score[final] or (
            prior_score[state] == prior_score[final]
            and prior_switches[state] < prior_switches[final]
        ):
            final = state
    path = np.empty(n, dtype=np.int8)
    path[-1] = final
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, float(prior_score[final]), int(prior_switches[final]), ties


try:
    from numba import njit
except ImportError:
    viterbi_compiled = None
else:
    viterbi_compiled = njit(cache=True, fastmath=False)(viterbi_python)

viterbi = viterbi_compiled if viterbi_compiled is not None else viterbi_python
