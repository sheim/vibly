import numpy as np
import matplotlib.pyplot as plt
import viability as vibly  # algorithms for brute-force viability
from ttictoc import tictoc
import control
import plotting.value_plotters as vplot

import pickle
import os

# * here we choose the parameters to use
# * we also put in a place-holder
#  action (thrust)

pathname = '../../../data/dynamics/'
basename = 'closed_satellite11'
# basename = 'test_satellite'
filename = pathname+basename
infile = open(filename+'.pickle', 'rb')
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

XV = S_M>0.0

failure_penalty = 1.0
def penalty(s, a):
    if s[0] >= p['radio_range']:
        return -failure_penalty
    elif s[0] <= p['radius']:
        return -failure_penalty
    else:
        return 0.

reward_functions = (penalty,)
savename = './viability'

###########################################################################
# * save data as pickle
###########################################################################

failure_penalties = [1.,]

# for rdx, reward_functions in enumerate(reward_schemes):
Q_value = None
for failure_penalty in failure_penalties:
    tictoc.tic()
    if Q_value is not None:
        Q_value *= failure_penalty
    Q_value, R_value = control.Q_value_iteration(Q_map, grids,
                            reward_functions, 0.6, Q_on_grid=Q_on_grid,
                            stopping_threshold=1e-6, max_iter=1000,
                            output_R=True, Q_values=Q_value)
    X_value = vibly.project_Q2S(Q_value, grids, proj_opt=np.max)

    time_elapsed = tictoc.toc()
    print("time elapsed (minutes): " + str(time_elapsed/60.0))

    data2save = {"Q_value": Q_value, "R_value": R_value,
                "grids": grids, "Q_map": Q_map, "Q_F": Q_F, "Q_V": Q_V,
                "Q_M": Q_M, "S_M": S_M, "p": p}
    outfile = open(savename+str(int(failure_penalty))+
                   '.pickle', 'wb')
    pickle.dump(data2save, outfile)
    outfile.close()


    # * plot
    print("============")
    print("PENALTY: ", failure_penalty)
    print("Min viable value: ", X_value[XV].min())
    print("Max unviable value: ", X_value[~XV].max())
    XV0 = X_value>=0.
    print("XV0 min value: ", X_value[XV0].min())
    if np.any(~XV0):
        print("Outside XV0 max value: ", X_value[~XV0].max())
    print("Size of XV0: ", XV0.astype(int).sum())

    # * basic plots (for better plots, use the dedicated script)

    mymap = vplot.get_vmap()
    extent = [grids['states'][1][0],
            grids['states'][1][-1],
            grids['states'][0][0],
            grids['states'][0][-1]]

    fig, axs = plt.subplots(2, 1)

    # * Reward function
    RX_value = vibly.project_Q2S(R_value, grids, proj_opt=np.mean)

    pc0 = vplot.reward_function(axs[0], RX_value, grids, XV=XV,
                                mymap="PRGn")
    fig.colorbar(pc0, ax=axs[0])

    # * Value Function

    pc1 = vplot.value_function(axs[1], X_value, grids, XV=XV,
                               mymap=mymap)
    fig.colorbar(pc1, ax=axs[1])
    plt.savefig(savename+str(failure_penalty)+'.pdf',
                format='pdf')

    print("******************")
    print("******************")