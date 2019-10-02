'''
Plotting functions for damping project
'''


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import models.daslip as sys
import viability as vibly  # algorithms for brute-force viability
import seaborn as sns

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# rc('text', usetex=True)
matplotlib.rcParams['figure.figsize'] = 5.5, 7
font = {'size': 8}
matplotlib.rc('font', **font)
sns.set_style('dark')


def interp_measure(s_bin, S_M, grids):
    neighbor_idx = vibly.get_grid_indices(s_bin, grids['states'])
    measure = 0
    if len(neighbor_idx) > 0:
        for idx in neighbor_idx:
            measure += S_M[idx]
        return measure/len(neighbor_idx)
    else:
        return 0.0


def ground_perturbations(ax, trajectories, S_M, grids, p, v_threshold=0.1,
                         col_offset=0.65):
    '''
    Plot a series of trajectories, centered around level-ground as nominal
    inputs:
    trajectories: list of traj objects (sol to scipy.integrate.solve_ivp)
    v_threshold: minimum safety, otherwise don't plot it
    '''
        # * get index of nominal trajectory, max step up/down
    index0 = 0
    max_up = 0
    max_down = 0
    for idx, traj in enumerate(trajectories):
        if np.isclose(traj.y[-1, 0], 0):
            index0 = idx

        if traj.y[-1, 0] > max_up:
            max_up = traj.y[-1, 0]
        if traj.y[-1, 0] < max_down:
            max_down = traj.y[-1, 0]
    else:
        if index0 < 0:
            print("WARNING: no nominal trajectory!")
        num_up = index0-1
        num_down = len(trajectories)-index0

    # * plot step-ups
    mycmap = plt.get_cmap("Blues")
    for up_dx in range(index0):
        traj = trajectories[up_dx]
        x = traj.y[:, -1]  # ground
        s = sys.xp2s_y_xdot(x, p)
        sbin = vibly.digitize_s(s, grids['states'])
        s_m = interp_measure(sbin, S_M, grids)

        if s_m > v_threshold:
            col = mycmap(col_offset-np.abs(traj.y[-1, 0]))
            ax.plot(traj.y[0], traj.y[1], color=col)

    # * plot step-downs
    mycmap = plt.get_cmap("Reds")
    for down_dx in range(index0+1, len(trajectories)):
        traj = trajectories[down_dx]

        x = traj.y[:, -1]
        s = sys.xp2s_y_xdot(x, p)
        sbin = vibly.digitize_s(s, grids['states'])
        s_m = interp_measure(sbin, S_M, grids)

        if s_m > 0.1:
            col = mycmap(col_offset-np.abs(traj.y[-1, 0]))
            ax.plot(traj.y[0], traj.y[1], color=col)

    if index0 >= 0:
        traj = trajectories[index0]
        ax.plot(traj.y[0], traj.y[1], color='k')
    plt.title(str(np.round(p['linear_normalized_damping_coefficient'],
             decimals=2)))
    plt.xlabel('x position')
    plt.ylabel('y position')





