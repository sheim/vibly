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
