import numpy as np

from models import slip


def _build_default_params():
    params = {
        "mass": 80.0,
        "stiffness": 8200.0,
        "resting_length": 1.0,
        "gravity": 9.81,
        "angle_of_attack": 1 / 5 * np.pi,
        "actuator_resting_length": 0.0,
    }
    x0 = np.array([0, 0.85, 5.5, 0, 0, 0, 0], dtype=float)
    x0 = slip.reset_leg(x0, params)
    params["x0"] = x0
    params["total_energy"] = slip.compute_total_energy(x0, params)
    return x0, params


def test_step_analytic_matches_numeric_flight():
    x0, params = _build_default_params()

    analytic = slip.step(x0, params)
    numeric = slip.step(x0, params, flight_mode="numeric")

    np.testing.assert_allclose(analytic.y[:, -1], numeric.y[:, -1], atol=1e-9)

    assert len(analytic.t_events) == len(numeric.t_events)
    for idx, (t_a, t_b) in enumerate(zip(analytic.t_events, numeric.t_events)):
        np.testing.assert_allclose(
            t_a,
            t_b,
            atol=1e-9,
            err_msg=f"event index {idx} diverged",
        )
