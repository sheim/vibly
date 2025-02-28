import numpy as np
import viability as vibly  # algorithms for brute-force viability
import control

import pickle

# * here we choose the parameters to use
# * we also put in a place-holder
#  action (thrust)

pathname = "../../../data/dynamics/"
basename = "closed_satellite11"
# basename = 'test_satellite'
filename = pathname + basename
infile = open(filename + ".pickle", "rb")
data = pickle.load(infile)
infile.close()

# * unpack
grids = data["grids"]
Q_map = data["Q_map"]
Q_F = data["Q_F"]
Q_V = data["Q_V"]
Q_M = data["Q_M"]
S_M = data["S_M"]
p = data["p"]
Q_on_grid = np.ones(Q_map.shape, dtype=bool)

XV = S_M > 0.0

good_lb_position = p["geocentric_radius"] * 0.9
good_ub_position = p["geocentric_radius"] * 1.1


def proxy_reward_position(s, a):
    # close to geocentric orbit
    if s[0] >= good_lb_position and s[0] <= good_ub_position:
        return 1.0
    else:
        return 0.0


actuator_penalty = 1.0


def actuator_cost(s, a):
    return -actuator_penalty * a[0] ** 2


def proxy_penalty_speed(s, a):
    if s[1] >= -2.0 and s[1] <= 2:
        return 0.0
    else:
        return -1.0


def L2diverge(s, a):
    r = np.hypot(s0_weight * (s[0] - 10.0), s1_weight * s[1])
    return np.exp2(-(r**2))


failure_penalty = 1.0


def penalty(s, a):
    if s[0] >= p["radio_range"]:
        return -failure_penalty
    elif s[0] <= p["radius"]:
        return -failure_penalty
    else:
        return 0.0


reward_functions = (proxy_penalty_speed, proxy_reward_position, actuator_cost, penalty)

# gammas = [np.arange(0.3, 0.95, 0.05)].reverse()
# gammas = np.linspace(0.95, 0.4, 12)
gammas = [0.95, 0.9, 0.85]

# * implement binary search
Q_value = None

gamma_list = list()
penalty_list = list()
for gamma in gammas:
    gamma_list.append(gamma)
    p_left = 1e-6
    if gamma > 0.8:
        p_right = 3.0
    elif gamma > 0.5:
        p_right = 1000.0
    else:
        p_right = 1e4
    # * determine p_left
    # failure_penalty = p_left
    # Q_value, R_value = control.Q_value_iteration(Q_map, grids,
    #                             reward_functions, gamma=gamma, Q_on_grid=Q_on_grid,
    #                             stopping_threshold=1e-6, max_iter=1000,
    #                             output_R=True, Q_values=Q_value)
    # X_value = vibly.project_Q2S(Q_value, grids, proj_opt=np.max)
    # if X_value[XV].min() > X_value[~XV].max():
    #     print("FINISHED penalty of 1 is fine")
    #     break
    # * determine p_right
    failure_penalty = p_right
    trial = 1
    while True:
        Q_value, R_value = control.Q_value_iteration(
            Q_map,
            grids,
            reward_functions,
            gamma=gamma,
            Q_on_grid=Q_on_grid,
            stopping_threshold=1e-5,
            max_iter=1000,
            output_R=True,
        )
        X_value = vibly.project_Q2S(Q_value, grids, proj_opt=np.max)
        if X_value[XV].min() > X_value[~XV].max():
            break
        failure_penalty *= 10.0

        trial += 1
        # * plot
        print("============")
        print("PENALTY: ", failure_penalty)
        print("Min viable value: ", X_value[XV].min())
        print("Max unviable value: ", X_value[~XV].max())
        XV0 = X_value >= 0.0
        print("XV0 min value: ", X_value[XV0].min())
        if np.any(~XV0):
            print("Outside XV0 max value: ", X_value[~XV0].max())

    p_right = failure_penalty
    last_safe_penalty = p_right

    # * start binary search
    print("STARTING BINARY SEARCH FOR GAMMA:", gamma)
    while True:
        # failure_penalty = np.round(p_left + (p_right - p_left) / 2.)
        failure_penalty = p_left + (p_right - p_left) / 2.0
        print("l: ", p_left, "r: ", p_right, "mid: ", failure_penalty)
        Q_value, R_value = control.Q_value_iteration(
            Q_map,
            grids,
            reward_functions,
            gamma=gamma,
            Q_on_grid=Q_on_grid,
            stopping_threshold=1e-5,
            max_iter=1000,
            output_R=True,
            Q_values=Q_value,
        )
        X_value = vibly.project_Q2S(Q_value, grids, proj_opt=np.max)
        disc = X_value[XV].min() - X_value[~XV].max()
        print("inf - sup:", disc)
        if disc > 0:
            p_right = failure_penalty
            last_safe_penalty = p_right
        else:
            p_left = failure_penalty
        if p_right - p_left < 0.001:
            penalty_list.append(last_safe_penalty)
            break
    # * handle data and plot
    data2save = {"penalty_list": penalty_list, "gamma_list": gamma_list}
    outfile = open("COMBO_gamma_pstar.pickle", "wb")
    pickle.dump(data2save, outfile)
    outfile.close()
    # * plot
    print("******************************************************")
    print("******************************************************")
