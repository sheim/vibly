import numpy as np
import matplotlib.pyplot as plt
import models.pendulum as sys
from models.pendulum import p_map
import viability as vibly  # algorithms for brute-force viability
from ttictoc import tictoc

# * here we choose the parameters to use
# * we also put in a place-holder action (thrust)
if __name__ == '__main__':
    p = {'n_states': 2,
         'mass': 1.,
         'gravity': 1,
         'length': 1.,
         'torque': 0.,
         'control_frequency': 10.,
         'simulation_frequency': 100.
         }

    x0 = np.array([0.])

    p_map.p = p
    p_map.x = x0
    p_map.sa2xp = sys.sa2xp
    p_map.xp2s = sys.xp2s

    # * determine the bounds and resolution of your grids
    # * note, the s_grid is a tuple of grids, such that each dimension can have
    # * different resolution, and we do not need to initialize the entire array
    s_grid = (np.linspace(-np.pi, np.pi, 181), np.linspace(-1.5, 1.5, 101))
    # * same thing for the actions
    a_grid = (np.linspace(-1., 1., 51),)

    # * for convenience, both grids are placed in a dictionary
    grids = {'states': s_grid, 'actions': a_grid}

    tictoc.tic()
    Q_map, Q_F, Q_coords = vibly.parcompute_Q_mapC(grids, p_map, verbose=1,
                                                   check_grid=False,
                                                   keep_coords=True)
    time_elapsed = tictoc.toc()
    print("time elapsed (minutes) for pmap: " + str(time_elapsed/60.0))
    time_elapsed = tictoc.tic()
    Q_V, S_V = vibly.compute_QV(Q_map, grids, ~Q_F,
                                Q_on_grid=np.ones(Q_map.shape, dtype=bool))
    S_M = vibly.project_Q2S(Q_V, grids, proj_opt=np.mean)
    Q_M = vibly.map_S2Q(Q_map, S_M, s_grid, Q_V=Q_V)
    time_elapsed = tictoc.toc()
    print("time elapsed (minutes) for everything: " + str(time_elapsed/60.0))
    ###########################################################################
    # * save data as pickle
    ###########################################################################
    import pickle
    import os

    filename = 'pendulum_map_lighter20.pickle'
    # if we are in the vibly root folder:
    if os.path.exists('data'):
        path_to_file = 'data/dynamics/'
    else:  # else we assume this is being run from the /demos folder.
        path_to_file = '../../data/dynamics/'
    if not os.path.exists(path_to_file):
        os.makedirs(path_to_file)

    data2save = {"grids": grids, "Q_map": Q_map, "Q_F": Q_F, "Q_V": Q_V,
                "Q_M": Q_M, "S_M": S_M, "p": p, "x0": x0}
    outfile = open(path_to_file+filename, 'wb')
    pickle.dump(data2save, outfile)
    outfile.close()

    # * to load this data, do:
    # infile = open(filename, 'rb')
    # data = pickle.load(infile)
    # infile.close()

    ###########################################################################
    # Plot
    ###########################################################################

    plt.imshow(S_M, origin='lower')  # visualize the Q-safety measure
    plt.show()
    # plt.imshow(Q_V) # visualize the viable set
    # plt.show()