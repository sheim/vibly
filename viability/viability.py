import itertools as it
import multiprocessing as mp
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np

"""
Tools for computing the viable set (in state-action space) and viability kernel (in
state space) of a dynamical system in ND.
"""


@dataclass
class TransitionResult:
    q_map: np.ndarray
    q_fail: np.ndarray
    q_on_grid: Optional[np.ndarray] = None
    q_reached: Optional[np.ndarray] = None


def digitize_s(s, s_grid, shape=None, to_bin=True):
    """
    s_grid is a tuple/list of 1-D grids. digitize_s finds the corresponding
    index of the bin overlaid on the N-dimensional grid S.

    Assumes you wat to bin things by default, turn to_bin to false to return
    closest grid coordinates.

    output:
    - either an array of indices of N dimensions
    - a raveled index
    """
    # assert type(s_grid) is tuple or type(s_grid) is list
    s = np.atleast_1d(s)  # make this indexable even if scalar
    s_idx = np.zeros((len(s_grid)), dtype=int)
    if to_bin:
        for dim_idx, grid in enumerate(s_grid):  # TODO: can zip this with s
            s_idx[dim_idx] = np.digitize(s[dim_idx], grid)
    else:
        for dim_idx, grid in enumerate(s_grid):
            s_idx[dim_idx] = np.argmin(np.abs(grid - s[dim_idx]))

    if shape is None:
        return s_idx
    else:
        return np.ravel_multi_index(s_idx, shape)


def project_Q2S(Q, grids, proj_opt=None):
    if proj_opt is None:
        proj_opt = np.any
    a_axes = tuple(range(Q.ndim - len(grids["actions"]), Q.ndim))
    return proj_opt(Q, a_axes)


def compute_QV(Q_map, grids, Q_V=None, Q_on_grid=None):
    """
    Starting from the transition map and set of non-failing state-action
    pairs, compute the viable sets. The input Q_V is referred to as Q_N in the
    paper when passing it in, but since it is immediately copied to Q_V, we
    directly use this naming.
    """

    # initialize estimate of Q_V
    if Q_V is None:
        Q_V = np.copy(Q_map)
        Q_V = Q_V.astype(bool)
    # if you have no info, treat everything as if in a bin
    if Q_on_grid is None:
        Q_on_grid = np.zeros(Q_V.shape, dtype=bool)
    # initialize empty of S_old
    # initialize estimate of S_V
    S_V = project_Q2S(Q_V, grids)
    S_old = np.zeros_like(S_V)

    # while S_V != S_old
    while not np.array_equal(S_V, S_old):
        # iterate over all s in S_V
        for qdx, is_viable in np.ndenumerate(Q_V):
            # iterate over all a
            if is_viable:
                # if s_k isOutside S_V:
                if is_outside(Q_map[qdx], grids["states"], S_V, on_grid=Q_on_grid[qdx]):
                    Q_V[qdx] = False
                    # remove (s,a) from Q_V
        S_old = S_V
        S_V = project_Q2S(Q_V, grids)

    return Q_V, S_V


def is_outside(s, s_grid, S_V, already_binned=True, on_grid=False):
    """
    given a level set S, check if s lands in a bin inside of S or not
    """

    # only checking 1 state vector
    if not already_binned:
        bin_idx = digitize_s(s, s_grid)  # get unraveled indices
    else:
        bin_idx = np.unravel_index(s, tuple(x + 1 for x in map(np.size, s_grid)))

    if on_grid:  # s is already binned in the flat
        s_grid_shape = list(map(np.size, s_grid))
        sdx = np.unravel_index(s, s_grid_shape)
        if S_V[sdx]:
            return False
        else:
            return True

    edge_indices = get_grid_indices(bin_idx, s_grid)

    if len(edge_indices) > 0:
        for idx in edge_indices:
            if not S_V[idx]:
                return True
    else:
        return True

    return False


def get_grid_indices(bin_idx, s_grid):
    """
    from a bin index (unraveled), get surrounding grid indices. Returns a list
    of tuples.
    """

    grid_coords = list()
    for neighbor in it.product(*[(x - 1, x) for x in bin_idx]):
        # check if neighbor is out of bounds
        for idx, x in enumerate(neighbor):
            if x < 0 or x >= s_grid[idx].size:
                break
        else:  # if for loop completes (not out of bounds)
            grid_coords.append(neighbor)

    return grid_coords


def map_S2Q(Q_map, S_M, s_grid, Q_V=None, Q_on_grid=None):
    """
    map the measure of robustness of S to the state action space Q, via
    inverse dynamics (using the lookup table).
    """

    # TODO s_grid isn't strictly needed, can be done without it

    if Q_on_grid is None:
        Q_on_grid = np.zeros_like(Q_map, dtype=bool)

    if Q_V is None:
        Q_V = Q_map.astype(bool)

    Q_M = np.zeros(Q_map.shape)
    # iterate over viable state-action pairs in Q_V
    for qdx, is_viable in np.ndenumerate(Q_V):
        if is_viable:  # only check states-action pairs that are viable
            if Q_on_grid[qdx]:
                sdx = np.unravel_index(Q_map[qdx], S_M.shape)
            else:
                sdx = np.unravel_index(Q_map[qdx], list(x + 1 for x in S_M.shape))
            edge_indices = get_grid_indices(sdx, s_grid)

            # TODO check that it actually has the right size
            measure = 0
            # taking the average measure of all enclosing grid-points
            if len(edge_indices) > 0:
                for idx in edge_indices:
                    measure += S_M[(idx)]
                measure /= len(edge_indices)

            # If sdx is on the outer edge, don't map it

            Q_M[qdx] = measure

    return Q_M


def get_feasibility_mask(feasible, sa2xp, grids, x0, p0):
    """
    cycle through the state and action grids, and check if that state-action
    pair is feasible or nay. Returns an ND array of booleans.

    feasible: function to check if a state and parameter are feasible
    sa2xp: mapping to go from state-action to x and p
    grids: grids of states and actions

    x0: a default x0, used to fill out x0 (used by sa2xp)

    p: a default parameter dict, used to fill out p
    """
    # initialize 1D, reshape later
    s_shape = list(map(np.size, grids["states"]))  # shape of state-space grid
    a_shape = list(map(np.size, grids["actions"]))
    Q_feasible = np.zeros(np.prod(s_shape) * np.prod(a_shape), dtype=bool)

    # state_action_iter = it.product(*grids["states"], *grids["actions"])
    for idx, state_action in enumerate(it.product(*grids["states"], *grids["actions"])):
        x, p = sa2xp(state_action, p0)
        Q_feasible[idx] = feasible(x, p)

    return Q_feasible.reshape(s_shape + a_shape)


def _grid_shapes(grids):
    s_shape = tuple(map(np.size, grids["states"]))
    a_shape = tuple(map(np.size, grids["actions"]))
    return s_shape, a_shape


def _total_gridpoints(grids):
    s_shape, a_shape = _grid_shapes(grids)
    return int(np.prod(s_shape) * np.prod(a_shape))


def _assemble_transition(
    grids,
    records: Iterable[Tuple[np.ndarray, bool]],
    total_count: int,
    *,
    check_grid: bool,
    keep_coords: bool,
    bin_mode: str,
) -> TransitionResult:
    s_grid_shape, a_grid_shape = _grid_shapes(grids)
    s_bin_shape = tuple(dim + 1 for dim in s_grid_shape)

    q_map_flat = np.zeros(total_count, dtype=int)
    q_fail_flat = np.zeros(total_count, dtype=bool)
    q_on_grid_flat = np.zeros(total_count, dtype=bool) if check_grid else None
    q_reached = np.zeros((len(grids["states"]), total_count)) if keep_coords else None

    if bin_mode == "nearest":

        def encode(point):
            return digitize_s(point, grids["states"], shape=s_grid_shape, to_bin=False)

        encode_on_grid = encode
    else:

        def encode(point):
            return digitize_s(point, grids["states"], s_bin_shape)

        def encode_on_grid(point):
            return digitize_s(point, grids["states"], s_grid_shape, to_bin=False)

    for idx, (s_next, failed) in enumerate(records):
        s_vec = np.atleast_1d(s_next)
        if keep_coords and q_reached is not None:
            q_reached[:, idx] = s_vec
        if failed:
            q_fail_flat[idx] = True
            continue
        if check_grid and q_on_grid_flat is not None:
            for dim_idx, sval in enumerate(s_vec):
                if not np.isin(sval, grids["states"][dim_idx]):
                    q_map_flat[idx] = encode(s_vec)
                    break
            else:
                q_on_grid_flat[idx] = True
                q_map_flat[idx] = encode_on_grid(s_vec)
        else:
            q_map_flat[idx] = encode(s_vec)

    q_map = q_map_flat.reshape(s_grid_shape + a_grid_shape)
    q_fail = q_fail_flat.reshape(s_grid_shape + a_grid_shape)
    q_on_grid = (
        q_on_grid_flat.reshape(s_grid_shape + a_grid_shape)
        if check_grid and q_on_grid_flat is not None
        else None
    )

    return TransitionResult(
        q_map=q_map,
        q_fail=q_fail,
        q_on_grid=q_on_grid,
        q_reached=q_reached,
    )


def compute_Q_map(
    grids,
    p_map,
    verbose=0,
    check_grid=False,
    keep_coords=False,
    parallel=False,
    bin_mode="bin",
):
    """Compute the transition map of a system."""

    if bin_mode not in {"bin", "nearest"}:
        raise ValueError(f"Unsupported bin_mode '{bin_mode}'")

    total_gridpoints = _total_gridpoints(grids)
    if verbose > 0:
        print("computing a total of " + str(total_gridpoints) + " points.")

    progress_mod = None
    if verbose > 1 and total_gridpoints >= 10:
        progress_mod = max(total_gridpoints // 10, 1)

    if parallel:
        state_actions = list(it.product(*grids["states"], *grids["actions"]))
        base_params = p_map.p.copy()
        args = [p_map.sa2xp(sa, base_params) for sa in state_actions]
        with mp.Pool() as pool:
            results = pool.starmap(p_map, args)

        records = []
        for idx, ((x_next, failed), (_, params)) in enumerate(zip(results, args)):
            if progress_mod and idx % progress_mod == 0:
                print(".", end=" ")
            s_next = p_map.xp2s(x_next, params)
            records.append((np.atleast_1d(s_next), bool(failed)))
    else:
        state_actions = it.product(*grids["states"], *grids["actions"])

        def record_iter():
            for idx, state_action in enumerate(state_actions):
                if progress_mod and idx % progress_mod == 0:
                    print(".", end=" ")
                x, params = p_map.sa2xp(state_action, p_map.p)
                x_next, failed = p_map(x, params)
                s_next = p_map.xp2s(x_next, params)
                yield np.atleast_1d(s_next), bool(failed)

        records = record_iter()

    result = _assemble_transition(
        grids,
        records,
        total_gridpoints,
        check_grid=check_grid,
        keep_coords=keep_coords,
        bin_mode=bin_mode,
    )
    return result
