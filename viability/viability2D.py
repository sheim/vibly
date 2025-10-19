"""
Legacy 2D viability routines.

These helpers are kept for reference to understand the basic idea behind the algorithms, but may contain bugs and are not maintained.
"""

import itertools as it
import numpy as np


def compute_Q_2D(s_grid, a_grid, p_map):
    """Compute the transition map of a system with 1D state and 1D action."""

    Q_map = np.zeros((s_grid.size * a_grid.size, 1))
    Q_F = np.zeros((s_grid.size * a_grid.size, 1))

    n = len(s_grid) * len(a_grid)
    for idx, state_action in enumerate(it.product(s_grid, a_grid)):
        if idx % (n / 10) == 0:
            print(".", end=" ")
        x, p = p_map.sa2xp(state_action, p_map.p)
        x_next, failed = p_map(x, p)
        if not failed:
            s_next = p_map.xp2s(x_next, p)
            # note: Q_map is implicitly already excluding transitions that
            # move straight to a failure. While this is not equivalent to the
            # algorithm in the paper, for our systems it is a bit faster
            Q_map[idx] = s_next
        else:
            Q_F[idx] = 1

    return (
        Q_map.reshape((s_grid.size, a_grid.size)),  # only 2D
        Q_F.reshape((s_grid.size, a_grid.size)),
    )


def project_Q2S_2D(Q):
    S = np.zeros((Q.shape[0], 1))
    for sdx, _ in enumerate(S):
        if sum(Q[sdx, :]) > 0:
            S[sdx] = 1
    return S


def is_outside_2D(s, S_V, s_grid):
    """Given a level set S, check if s is inside S or not."""
    if sum(S_V) <= 1:
        return True

    s_min, s_max = s_grid[S_V > 0][[0, -1]]
    if s > s_max or s < s_min:
        return True
    return False


def compute_QV_2D(Q_map, grids, Q_V=None):
    """Compute viable sets for the 2D specialisation."""

    if Q_V is None:
        Q_V = np.copy(Q_map)
        Q_V[Q_V > 0] = 1

    S_old = np.zeros((Q_V.shape[0], 1))
    S_V = project_Q2S_2D(Q_V)
    while np.array_equal(S_V, S_old):
        for qdx, is_viable in enumerate(np.nditer(Q_V)):
            if is_viable:
                if is_outside_2D(Q_map[qdx], S_V, grids["states"]):
                    Q_V[qdx] = 0
        S_old = S_V
        S_V = project_Q2S_2D(Q_V)

    return Q_V, S_V
