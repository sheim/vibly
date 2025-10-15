import numpy as np

from viability import viability as vibly


def test_digitize_s_handles_bins_and_coordinates():
    s_grid = (np.array([0.0, 0.5, 1.0]), np.array([-1.0, 0.0, 1.0]))

    s = np.array([0.5, 0.0])
    bin_indices = vibly.digitize_s(s, s_grid)
    assert tuple(bin_indices) == (2, 2)

    ravelled = vibly.digitize_s(s, s_grid, shape=(3, 3))
    assert ravelled == np.ravel_multi_index((2, 2), (3, 3))

    coord_indices = vibly.digitize_s(s, s_grid, to_bin=False)
    assert tuple(coord_indices) == (1, 1)


def test_get_grid_indices_returns_enclosing_vertices():
    s_grid = (np.arange(3), np.arange(4))

    neighbors = vibly.get_grid_indices((1, 2), s_grid)
    assert set(neighbors) == {(0, 1), (0, 2), (1, 1), (1, 2)}

    # Edge bins should only report in-bounds vertices
    edge_neighbors = vibly.get_grid_indices((0, 0), s_grid)
    assert edge_neighbors == [(0, 0)]


def test_project_Q2S_respects_projection_operator():
    grids = {
        "states": (np.array([0, 1]), np.array([0, 1])),
        "actions": (np.array([0]),),
    }
    q_values = np.array([[[True], [False]], [[False], [True]]])

    default_projection = vibly.project_Q2S(q_values, grids)
    assert default_projection.dtype == bool
    assert np.array_equal(default_projection, np.array([[True, False], [False, True]]))

    summed = vibly.project_Q2S(q_values.astype(int), grids, proj_opt=np.sum)
    assert np.array_equal(summed, np.array([[1, 0], [0, 1]]))


class DummyMap:
    def __init__(self):
        self.p = {}

    def sa2xp(self, state_action, params):
        state, action = state_action
        return np.array([state, action]), params

    def xp2s(self, x_next, params):
        return np.array([x_next[0]])

    def __call__(self, x, params):
        state, action = x
        next_state = state + action
        failed = (next_state < 0.0) or (next_state > 2.0)
        return np.array([next_state, action]), failed


def test_compute_Q_map_and_compute_QV_simple_system():
    grids = {
        "states": (np.array([0.0, 1.0, 2.0]),),
        "actions": (np.array([-1.0, 1.0]),),
    }
    p_map = DummyMap()

    q_map, q_fail = vibly.compute_Q_map(grids, p_map)
    assert q_map.shape == (3, 2)
    assert q_fail.shape == (3, 2)

    # state 0 with action -1 should fail, state 2 with action +1 should fail
    assert q_fail[0, 0]
    assert q_fail[2, 1]
    assert not q_fail[1, 0]

    q_v, s_v = vibly.compute_QV(q_map, grids, Q_V=~q_fail)
    # All states remain viable because each has at least one safe action
    assert np.array_equal(s_v.astype(int), np.array([1, 1, 1]))

    # Map a simple state-space measure back into Q
    state_measure = np.array([1.0, 0.5, 0.0])
    q_m = vibly.map_S2Q(q_map, state_measure, grids["states"], Q_V=q_v)
    assert q_m.shape == q_map.shape
    expected_q_m = np.array([[0.0, 0.25], [0.75, 0.0], [0.25, 0.0]])
    assert np.allclose(q_m, expected_q_m)


def test_compute_Q_map_with_check_grid_reports_on_grid_hits():
    grids = {
        "states": (np.array([0.0, 1.0, 2.0]),),
        "actions": (np.array([-1.0, 1.0]),),
    }
    p_map = DummyMap()

    q_map, q_fail, q_on_grid, q_reached = vibly.compute_Q_map(
        grids, p_map, check_grid=True, keep_coords=True
    )

    assert q_map.shape == (3, 2)
    assert q_on_grid.shape == q_map.shape
    assert q_on_grid.dtype == bool
    assert np.any(q_on_grid)

    total_points = np.prod([grid.size for grid in grids["states"]]) * np.prod(
        [grid.size for grid in grids["actions"]]
    )
    assert q_reached.shape == (len(grids["states"]), total_points)


def test_compute_QV_same_with_and_without_on_grid_flag():
    grids = {
        "states": (np.array([0.0, 1.0, 2.0]),),
        "actions": (np.array([-1.0, 1.0]),),
    }
    p_map = DummyMap()

    q_map, q_fail, q_on_grid, _ = vibly.compute_Q_map(
        grids, p_map, check_grid=True, keep_coords=True
    )

    q_v_default, s_v_default = vibly.compute_QV(q_map, grids, Q_V=~q_fail)
    q_v_on_grid, s_v_on_grid = vibly.compute_QV(
        q_map, grids, Q_V=~q_fail, Q_on_grid=q_on_grid
    )

    assert np.array_equal(q_v_default, q_v_on_grid)
    assert np.array_equal(s_v_default, s_v_on_grid)


def test_is_outside_handles_on_grid_points():
    s_grid = (np.array([0.0, 1.0, 2.0]),)
    S_V = np.array([True, False, True])

    idx_viable = np.ravel_multi_index((0,), (3,))
    assert not vibly.is_outside(idx_viable, s_grid, S_V, on_grid=True)

    idx_outside = np.ravel_multi_index((1,), (3,))
    assert vibly.is_outside(idx_outside, s_grid, S_V, on_grid=True)

    # also exercise the "not already binned" path (currently treats boundary points as outside)
    assert vibly.is_outside([0.0], s_grid, S_V, already_binned=False)
    assert vibly.is_outside([1.5], s_grid, S_V, already_binned=False)


def test_map_S2Q_uses_on_grid_lookup():
    grids = {
        "states": (np.array([0.0, 1.0, 2.0]),),
        "actions": (np.array([-1.0, 1.0]),),
    }
    p_map = DummyMap()

    q_map, q_fail, q_on_grid, _ = vibly.compute_Q_map(
        grids, p_map, check_grid=True, keep_coords=True
    )
    q_v = ~q_fail

    state_measure = np.array([1.0, 0.5, 0.0])
    q_m = vibly.map_S2Q(
        q_map,
        state_measure,
        grids["states"],
        Q_V=q_v,
        Q_on_grid=q_on_grid,
    )

    assert q_m.shape == q_map.shape

    mask = q_v & q_on_grid
    assert np.any(mask)
    for idx in zip(*np.where(mask)):
        s_idx = np.unravel_index(q_map[idx], state_measure.shape)
        neighbors = vibly.get_grid_indices(s_idx, grids["states"])
        expected = np.mean([state_measure[n] for n in neighbors])
        assert np.isclose(q_m[idx], expected)
