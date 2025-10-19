from pathlib import Path

import numpy as np


FIXTURE_DIR = Path(__file__).resolve().parent / "data"


def test_slip_demo_reference_solution():
    path = FIXTURE_DIR / "slip_demo_sol.npz"
    assert path.exists(), "Expected slip demo reference output to be present."

    data = np.load(path)
    t = data["t"]
    y = data["y"]

    diffs = np.diff(t)
    assert np.all(diffs >= 0), "Solution time steps should be non-decreasing."
    assert np.any(diffs > 0), "Solution should contain at least one positive timestep."
    assert y.shape == (7, t.size)


def test_hover_map_reference_contents():
    path = FIXTURE_DIR / "hover_map.npz"
    assert path.exists(), "Expected hover_map.npz to be present."

    data = np.load(path)
    q_map = data["Q_map"]
    q_fail = data["Q_F"]
    q_viable = data["Q_V"]
    q_measure = data["Q_M"]
    s_measure = data["S_M"]
    s_grid = data["s_grid"]
    a_grid = data["a_grid"]

    assert q_map.shape == (201, 161)
    assert q_map.dtype == np.int64
    assert q_fail.shape == q_map.shape
    assert q_viable.dtype == bool
    assert q_measure.shape == q_map.shape
    assert s_measure.shape == (201,)
    assert np.isclose(a_grid.max(), 0.8)
    assert np.isclose(s_grid.min(), 0.0)
    assert np.any(q_fail)
    assert np.any(~q_fail)


def test_slip_map_reference_contents():
    path = FIXTURE_DIR / "slip_map.npz"
    assert path.exists(), "Expected slip_map.npz to be present."

    data = np.load(path)
    q_map = data["Q_map"]
    q_fail = data["Q_F"]
    q_viable = data["Q_V"]
    q_measure = data["Q_M"]
    s_measure = data["S_M"]
    s_grid = data["s_grid"]
    a_grid = data["a_grid"]

    assert q_map.shape == (180, 161)
    assert q_map.dtype == np.int64
    assert q_fail.shape == q_map.shape
    assert q_viable.dtype == bool
    assert q_measure.shape == q_map.shape
    assert s_measure.shape == (180,)
    assert np.isclose(s_grid[0], 0.1)
    assert np.isclose(a_grid[0], -10 / 180 * np.pi)
    assert np.all(q_measure[q_viable] >= 0.0)
    assert np.any(q_measure[q_viable] > 0.0)

