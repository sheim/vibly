import numpy as np
import matplotlib.pyplot as plt
from models import nslip
import viability as vibly


if __name__ == "__main__":
    p = {
        "mass": 80.0,
        "stiffness": 705.0,
        "resting_angle": 17 / 18 * np.pi,
        "gravity": 9.81,
        "angle_of_attack": 1 / 5 * np.pi,
        "upper_leg": 0.5,
        "lower_leg": 0.5,
    }

    x0 = np.array([0.0, 0.85, 5.5, 0.0, 0.0, 0.0, 0.0])
    x0 = nslip.reset_leg(x0, p)
    p["x0"] = x0
    p["total_energy"] = nslip.compute_total_energy(x0, p)

    p_map = nslip.p_map
    p_map.p = p
    p_map.x = x0
    p_map.sa2xp = nslip.sa2xp
    p_map.xp2s = nslip.xp2s

    s_grid = np.linspace(0.1, 1.0, 61)
    s_grid = (s_grid[:-1],)
    a_grid = (np.linspace(-10 / 180 * np.pi, 70 / 180 * np.pi, 61),)
    grids = {"states": s_grid, "actions": a_grid}

    Q_map, Q_F = vibly.compute_Q_map(grids, p_map, parallel=True)
    Q_V, S_V = vibly.compute_QV(Q_map, grids)
    S_M = vibly.project_Q2S(Q_V, grids, proj_opt=np.mean)
    Q_M = vibly.map_S2Q(Q_map, S_M, s_grid, Q_V=Q_V)

    import pickle
    import os

    filename = "nslip_map.pickle"

    if os.path.exists("data"):
        path_to_file = "data/dynamics/"
    else:
        path_to_file = "../data/dynamics/"
    if not os.path.exists(path_to_file):
        os.makedirs(path_to_file)

    data2save = {
        "grids": grids,
        "Q_map": Q_map,
        "Q_F": Q_F,
        "Q_V": Q_V,
        "Q_M": Q_M,
        "S_M": S_M,
        "p": p,
        "x0": x0,
    }
    with open(path_to_file + filename, "wb") as outfile:
        pickle.dump(data2save, outfile)

    plt.imshow(Q_map, origin="lower")
    plt.show()
