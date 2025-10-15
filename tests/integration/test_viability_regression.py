from pathlib import Path
import warnings
import multiprocessing as mp

import numpy as np
import pytest

import models.hovership as hovership
import models.slip as slip
from models.hovership import p_map as hovership_p_map
import viability as vibly
import control


FIXTURE_DIR = Path(__file__).resolve().parent / "data"

if mp.get_start_method(allow_none=True) != "fork":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass


@pytest.mark.slow
def test_hovership_viability_matches_reference():
    reference = np.load(FIXTURE_DIR / "hover_map.npz")

    p = {
        "n_states": 1,
        "base_gravity": 0.1,
        "gravity": 1,
        "thrust": 0,
        "max_thrust": 0.8,
        "ceiling": 2,
        "control_frequency": 1,
    }
    x0 = np.array([0.5])

    s_grid = (np.linspace(-0.0, p["ceiling"], 201),)
    a_grid = (np.linspace(0.0, p["max_thrust"], 161),)
    grids = {"states": s_grid, "actions": a_grid}

    p_map = hovership_p_map
    p_map.p = p
    p_map.x = x0
    p_map.sa2xp = hovership.sa2xp
    p_map.xp2s = hovership.xp2s

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="Conversion of an array with ndim > 0 to a scalar is deprecated",
        )
        q_map, q_fail, q_on_grid = vibly.compute_Q_map(grids, p_map, check_grid=True)
        q_v, s_v = vibly.compute_QV(q_map, grids, Q_V=~q_fail, Q_on_grid=q_on_grid)
        s_m = vibly.project_Q2S(q_v, grids, proj_opt=np.mean)
        q_m = vibly.map_S2Q(q_map, s_m, s_grid, Q_V=q_v, Q_on_grid=q_on_grid)

    assert np.array_equal(q_map, reference["Q_map"])
    assert np.array_equal(q_fail, reference["Q_F"])
    assert np.array_equal(q_v, reference["Q_V"])
    assert np.allclose(s_m, reference["S_M"])
    assert np.allclose(q_m, reference["Q_M"])


@pytest.mark.slow
def test_slip_viability_matches_reference():
    reference = np.load(FIXTURE_DIR / "slip_map.npz")

    p = {
        "mass": 80.0,
        "stiffness": 8200.0,
        "resting_length": 1.0,
        "gravity": 9.81,
        "angle_of_attack": 1 / 5 * np.pi,
        "actuator_resting_length": 0,
    }
    x0 = np.array([0, 0.85, 5.5, 0, 0, 0, 0], dtype=float)
    x0 = slip.reset_leg(x0, p)
    p["x0"] = x0
    p["total_energy"] = slip.compute_total_energy(x0, p)

    s_grid = np.linspace(0.1, 1, 181)
    s_grid = (s_grid[:-1],)
    a_grid = (np.linspace(-10 / 180 * np.pi, 70 / 180 * np.pi, 161),)
    grids = {"states": s_grid, "actions": a_grid}

    p_map = slip.p_map
    p_map.p = p
    p_map.x = x0
    p_map.sa2xp = slip.sa2xp
    p_map.xp2s = slip.xp2s

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="Conversion of an array with ndim > 0 to a scalar is deprecated",
        )
        q_map, q_fail = vibly.parcompute_Q_map(grids, p_map)
    q_v, s_v = vibly.compute_QV(q_map, grids, Q_V=~q_fail)
    s_m = vibly.project_Q2S(q_v, grids, proj_opt=np.mean)
    q_m = vibly.map_S2Q(q_map, s_m, grids["states"], Q_V=q_v)

    assert np.array_equal(q_map, reference["Q_map"])
    assert np.array_equal(q_fail, reference["Q_F"])
    assert np.array_equal(q_v, reference["Q_V"])
    assert np.allclose(s_m, reference["S_M"])
    assert np.allclose(q_m, reference["Q_M"])


@pytest.mark.slow
def test_satellite_parcompute_matches_reference():
    fixture = np.load(FIXTURE_DIR / "closed_satellite11.npz")
    s_grid = (fixture["s_grid_0"], fixture["s_grid_1"])
    a_grid = (fixture["a_grid"],)
    grids = {"states": s_grid, "actions": a_grid}

    p = {
        "n_states": 2,
        "geocentric_constant": 10.0,
        "geocentric_radius": 10.0,
        "angular_speed": 0.1,
        "mass": 1.0,
        "control_frequency": 1,
        "thrust": 1.0,
        "radius": 1.0,
        "radio_range": 15.0,
    }
    x0 = fixture["x0"]

    import models.satellite as satellite
    from models.satellite import p_map as satellite_p_map

    p_map = satellite_p_map
    p_map.p = p
    p_map.x = x0
    p_map.sa2xp = satellite.sa2xp
    p_map.xp2s = satellite.xp2s

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="Conversion of an array with ndim > 0 to a scalar is deprecated",
        )
        q_map, q_fail, _ = vibly.parcompute_Q_mapC(
            grids, p_map, verbose=1, check_grid=False, keep_coords=True
        )

    assert np.array_equal(q_map, fixture["Q_map"])
    assert np.array_equal(q_fail, fixture["Q_F"])


@pytest.mark.slow
def test_satellite_value_iteration_matches_reference():
    fixture = np.load(FIXTURE_DIR / "closed_satellite11.npz")
    q_map = fixture["Q_map"]
    s_grid = (fixture["s_grid_0"], fixture["s_grid_1"])
    a_grid = (fixture["a_grid"],)
    grids = {"states": s_grid, "actions": a_grid}
    q_on_grid = np.ones(q_map.shape, dtype=bool)

    reference = np.load(FIXTURE_DIR / "satellite_q_value.npz")
    expected_q_value = reference["Q_value"]
    expected_r_value = reference["R_value"]
    failure_penalty = float(reference["failure_penalty"])

    radius = float(fixture["p_radius"])
    radio_range = float(fixture["p_radio_range"])

    def parsimonious_reward(s, a):
        if np.allclose(s, [10.0, 0.0]):
            return 1.0
        return 0.0

    def make_penalty(penalty_scale):
        def penalty(s, a):
            if s[0] >= radio_range or s[0] <= radius:
                return -penalty_scale
            return 0.0

        return penalty

    reward_functions = (parsimonious_reward, make_penalty(failure_penalty))

    # Warm-start with zero-penalty run as in VIS_basic.py
    zero_penalty_reward = (parsimonious_reward, make_penalty(0.0))
    q_value, _ = control.Q_value_iteration(
        q_map,
        grids,
        zero_penalty_reward,
        0.6,
        Q_on_grid=q_on_grid,
        stopping_threshold=1e-6,
        max_iter=1000,
        output_R=True,
        Q_values=None,
    )

    q_value, r_value = control.Q_value_iteration(
        q_map,
        grids,
        reward_functions,
        0.6,
        Q_on_grid=q_on_grid,
        stopping_threshold=1e-6,
        max_iter=1000,
        output_R=True,
        Q_values=q_value,
    )

    assert np.allclose(q_value, expected_q_value)
    assert np.allclose(r_value, expected_r_value)
