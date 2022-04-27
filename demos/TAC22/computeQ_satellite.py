import numpy as np
import matplotlib.pyplot as plt
import models.satellite as sys
from models.satellite import p_map
import viability as vibly  # algorithms for brute-force viability
from ttictoc import tictoc

# ! for detailed comments, see `computeQ_hovership.py`

if __name__ == '__main__':
    # * Planet Make Believe
    p = {'n_states': 2,
         'geocentric_constant': 10.0, # geocentric gravitational constant
         'geocentric_radius': 10.0,
         'angular_speed': 0.1,
         'mass': 1.0, # Kg
         'control_frequency': 1,  # hertz
         'thrust': 1.0, # Kg
         'radius': 1.0, # Km (radius of planet)
         'radio_range': 15.0 # before losing comms
         }
    x0 =np.array([9.5, 0.0])

    # * standard helper functions
    p_map.p = p
    p_map.x = x0
    p_map.sa2xp = sys.sa2xp
    p_map.xp2s = sys.xp2s

    # * grid used for TAC submission
    s_grid = (np.linspace(0.5, 15.5 , 301), np.linspace(-5.0, 7.0, 241))
    a_grid = (np.linspace(-1, 1, 11),)

    grids = {'states': s_grid, 'actions': a_grid}

    tictoc.tic()
    Q_map, Q_F, Q_coords = vibly.parcompute_Q_mapC(grids, p_map, verbose=1, check_grid=False, keep_coords=True)
    # Q_map, Q_F, Q_on_grid = vibly.compute_Q_map(grids, p_map, check_grid=True)

    # * compute_QV computes the viable set and viability kernel
    Q_V, S_V = vibly.compute_QV(Q_map, grids, ~Q_F,
                        Q_on_grid=np.ones(Q_map.shape, dtype=bool))
    # * compute measures (not necessary for TAC), but usedful as ground truth
    S_M = vibly.project_Q2S(Q_V, grids, proj_opt=np.mean)
    Q_M = vibly.map_S2Q(Q_map, S_M, s_grid, Q_V=Q_V)
    time_elapsed = tictoc.toc()
    print("time elapsed (minutes): "
          + str(time_elapsed/60.0))

    ###########################################################################
    # * save data as pickle
    ###########################################################################
    import pickle
    import os

    filename = 'closed_satellite11.pickle'
    # if we are in the vibly root folder:
    if os.path.exists('data'):
        path_to_file = 'data/dynamics/'
    else:  # else we assume this is being run from the /demos/TAC11 folder.
        path_to_file = '../../data/dynamics/'
    if not os.path.exists(path_to_file):
        os.makedirs(path_to_file)

    data2save = {"grids": grids, "Q_map": Q_map, "Q_F": Q_F, "Q_V": Q_V,
                "Q_M": Q_M, "S_M": S_M, "p": p, "x0": x0}
    outfile = open(path_to_file+filename, 'wb')
    pickle.dump(data2save, outfile)
    outfile.close()
    # to load this data, do:
    # infile = open(filename, 'rb')
    # data = pickle.load(infile)
    # infile.close()

    extent = [grids['states'][1][0],
              grids['states'][1][-1],
              grids['states'][0][0],
              grids['states'][0][-1]]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(S_M, origin='lower', extent=extent, aspect='auto',
              interpolation='none', cmap='viridis')
    plt.show()