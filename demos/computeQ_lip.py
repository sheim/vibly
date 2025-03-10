import numpy as np
import matplotlib.pyplot as plt
import models.lip as sys
from models.lip import p_map
import viability as vibly  # algorithms for brute-force viability
from ttictoc import tictoc

# * here we choose the parameters to use
# * we also put in a place-holder action (thrust)
if __name__ == "__main__":
    p = {
        "n_states": 1,
        "max_step": np.sqrt(1.2**2 - 1),
        "step_timing": 0.1,
        "step_location": 0.2,
    }

    # leg_length = 1.
    # p['max_step'] = np.sqrt(leg_length**2 - 1)
    # * Choose an initial condition
    x0 = np.array([0.5])

    # * For convenience, helper functions, a default parameter dict and initial
    # * condition are attached to the transition map.
    p_map.p = p
    p_map.x = x0
    p_map.sa2xp = sys.sa2xp
    p_map.xp2s = sys.xp2s
    # p_map.sa2xp = sys.sa2xp_num
    # p_map.xp2s = sys.xp2s_num

    # * determine the bounds and resolution of your grids
    # * note, the s_grid is a tuple of grids, such that each dimension can have
    # * different resolution, and we do not need to initialize the entire array
    s_grid = (np.linspace(0.0, 1.5, 151),)
    a_grid = (
        np.linspace(0.0, 5.0, 251),  # timing
        np.linspace(0.0, 1.5, 151),
    )  # location
    # s_grid = (np.linspace(0.0, 1.5, 201),)
    # a_grid = (np.linspace(0.0, 5.5, 451),  # timingk
    #           np.linspace(0.0, 2, 401))  # location

    # * for convenience, both grids are placed in a dictionary
    grids = {"states": s_grid, "actions": a_grid}

    tictoc.tic()
    Q_map, Q_F = vibly.parcompute_Q_map(grids, p_map, verbose=1)
    time_elapsed = tictoc.toc()
    print("time elapsed: " + str(time_elapsed / 60.0))
    # * compute_QV computes the viable set and viability kernel
    Q_V, S_V = vibly.compute_QV(Q_map, grids, ~Q_F)

    # * project_Q2S takens a projection of the viable set onto state-space
    # * for the computing the measure, you can use either `np.mean` or `np.sum`
    # * as the projection operator
    S_M = vibly.project_Q2S(Q_V, grids, proj_opt=np.mean)
    # * map_S2Q maps the measure back into state-action space using the gridded
    # * transition map
    Q_M = vibly.map_S2Q(Q_map, S_M, s_grid, Q_V=Q_V)

    ###########################################################################
    # * save data as pickle
    ###########################################################################
    import pickle
    import os

    filename = "lip_map.pickle"
    # if we are in the vibly root folder:
    if os.path.exists("data"):
        path_to_file = "data/dynamics/"
    else:  # else we assume this is being run from the /demos folder.
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
    outfile = open(path_to_file + filename, "wb")
    pickle.dump(data2save, outfile)
    outfile.close()
    # # to load this data, do:
    # infile = open(filename, 'rb')
    # data = pickle.load(infile)
    # infile.close()
    ###########################################################################
    # plt.imshow(S_M)  # visualize the Q-safety measure
    # plt.show()
    # plt.imshow(Q_V) # visualize the viable set
    # plt.show()
    VLoc = np.mean(Q_V, axis=2).T
    # extent = [grids['actions'][1][0],
    #           grids['action'][1][-1],
    #           grids['states'][0][0],
    #           grids['states'][0][-1]]
    plt.imshow(VLoc, origin="lower")
    plt.xlabel("velocity at mid-stance")
    plt.ylabel("Time of stance")
    plt.show()
