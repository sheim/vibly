import numpy as np
import matplotlib.pyplot as plt
import viability as vibly  # algorithms for brute-force viability
from ttictoc import tictoc
import control
import plotting.value_plotters as vplot

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
        return 10.0
    else:
        return 0.0


good_lb_position = p["geocentric_radius"] * 0.9
good_ub_position = p["geocentric_radius"] * 1.1
s0 = 0.5 * p["geocentric_radius"]

actuator_penalty = 1.0


def actuator_cost(s, a):
    return 1 - actuator_penalty * a[0] ** 2


l2_boost = 5.0
s0_weight = 1 / (grids["states"][0][-1] - grids["states"][0][0])
s1_weight = 1 / (grids["states"][1][-1] - grids["states"][1][0])


def smooth_l2(s, a):
    # close to geocentric orbit
    r = np.hypot(s0_weight * (s[0] - s0), s1_weight * s[1])
    return -(r**2) + l2_boost


def independent_l2(s, a):
    return -s0_weight * (s[0] - s0) ** 2 - s1_weight * s[1] ** 2 + l2_boost


def wacky(s, a):
    r1 = np.hypot(s0_weight * (s[0] - 5.0), s1_weight * (s[1] + 3.0))
    r2 = np.hypot(s0_weight * (s[0] - 10.0), s1_weight * (s[1] - 3.0))
    r3 = np.hypot(s0_weight * (s[0] - 10.0), s1_weight * s[1])
    reward = 0.0
    if r1 <= 0.05:
        reward += 3.0
    if r2 <= 0.1:
        reward += 3.0
    if r3 <= 0.05:
        reward += 1.0

    return reward


def L2wacky(s, a):
    r1 = np.hypot(s0_weight * (s[0] - 5.0), s1_weight * (s[1] + 3.0))
    r2 = np.hypot(s0_weight * (s[0] - 10.0), s1_weight * (s[1] - 4.0))
    # r3 = np.hypot(s0_weight*(s[0]-10.), s1_weight*s[1])
    reward = 0.0
    reward += 2 * np.exp2(-25 * r1**2) + 2 * np.exp2(-25 * r2**2)
    # reward += np.exp2(-50*r3**2)

    return reward


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


reward_functions = (L2diverge, actuator_cost, penalty)
savename = "./wacker"
# savename = './closed21/orti/TEST'
# reward_schemes = ((target,),
#                   (penalty,))

###########################################################################
# * save data as pickle
###########################################################################

mymap = vplot.get_vmap()
# mynorm = colors.CenteredNorm()

# failure_penalties = [1e6, 1e9]
# failure_penalties = [100.,]
# failure_penalties = [0, 1., 10., 1e2, 500., 1e3]
failure_penalties = np.arange(20.0, 100.0, 1.0)


# for rdx, reward_functions in enumerate(reward_schemes):
Q_value = None
for failure_penalty in failure_penalties:
    tictoc.tic()
    if Q_value is not None:
        Q_value *= failure_penalty
    Q_value, R_value = control.Q_value_iteration(
        Q_map,
        grids,
        reward_functions,
        0.6,
        Q_on_grid=Q_on_grid,
        stopping_threshold=1e-6,
        max_iter=1000,
        output_R=True,
        Q_values=Q_value,
    )
    X_value = vibly.project_Q2S(Q_value, grids, proj_opt=np.max)

    time_elapsed = tictoc.toc()
    print("time elapsed (minutes): " + str(time_elapsed / 60.0))

    data2save = {
        "Q_value": Q_value,
        "R_value": R_value,
        "grids": grids,
        "Q_map": Q_map,
        "Q_F": Q_F,
        "Q_V": Q_V,
        "Q_M": Q_M,
        "S_M": S_M,
        "p": p,
        "penalty": failure_penalty,
    }
    outfile = open(savename + str(int(failure_penalty)) + ".pickle", "wb")
    pickle.dump(data2save, outfile)
    outfile.close()

    # * plot
    print("============")
    print("PENALTY: ", failure_penalty)
    print("Min viable value: ", X_value[XV].min())
    print("Max unviable value: ", X_value[~XV].max())
    XV0 = X_value >= 0.0
    print("XV0 min value: ", X_value[XV0].min())
    if np.any(~XV0):
        print("Outside XV0 max value: ", X_value[~XV0].max())
    print("Size of XV0: ", XV0.astype(int).sum())

    # viability_threshold = X_value[XV].min()

    # mynorm = colors.CenteredNorm()
    # mymap = plt.get_cmap('bwr_r')
    # shrunk_cmap = vplot.shiftedColorMap(
    #     mymap, start=0.2, midpoint=0.5, stop=0.8, name="shrunk"
    # )

    extent = [
        grids["states"][1][0],
        grids["states"][1][-1],
        grids["states"][0][0],
        grids["states"][0][-1],
    ]

    fig, axs = plt.subplots(2, 1)

    # * Reward function
    RX_value = vibly.project_Q2S(R_value, grids, proj_opt=np.mean)

    pc0 = vplot.reward_function(axs[0], RX_value, grids, XV=XV, mymap="PRGn")
    fig.colorbar(pc0, ax=axs[0])

    # * Value Function

    pc1 = vplot.value_function(axs[1], X_value, grids, XV=XV, mymap=mymap)
    fig.colorbar(pc1, ax=axs[1])
    plt.savefig(savename + str(failure_penalty) + ".pdf", format="pdf")

    print("******************")
    print("******************")
    # plt.imshow(np.transpose(S_M), origin='lower')  # visualize the S-safety measure
    # plt.show()
    # plt.imshow(Q_V) # visualize the viable set
    # plt.show()
