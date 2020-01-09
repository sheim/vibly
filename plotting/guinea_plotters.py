'''
Plotting functions for damping project
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import models.parslip as sys
import viability as vibly  # algorithms for brute-force viability
import seaborn as sns
import matplotlib.collections as collections

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


def get_perturbation_indices(trajectories):
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
    
    return index0, max_up, max_down

def plot_ground_perturbations(ax, trajectories, S_M, grids, p, v_threshold=0.1,
                         col_offset=0.65):
    '''
    Plot a series of trajectories, centered around level-ground as nominal
    inputs:
    trajectories: list of traj objects (sol to scipy.integrate.solve_ivp)
    v_threshold: minimum safety, otherwise don't plot it
    '''

    # TODO redo this with trajectories colored by measure

    # * plot step-ups
    up_cmap = plt.get_cmap("Blues")
    down_cmap = plt.get_cmap("Reds")
    idx0 = -1
    for idx in range(len(trajectories)):
        traj = trajectories[idx]
        x = traj.y[:, -1]  # ground
        s = sys.xp2s_y_xdot(x, p)
        sbin = vibly.digitize_s(s, grids['states'])
        s_m = interp_measure(sbin, S_M, grids)

        if s_m > v_threshold:
            if np.isclose(x[-1], 0):
                # print afterwards, so it's on top of other circles
                idx0 = idx
                continue
            elif x[-1] < 0:
                col = down_cmap(col_offset-np.abs(x[-1]))
            elif x[-1] > 0:
                col = up_cmap(col_offset-np.abs(x[-1]))
            ax.plot(traj.y[0], traj.y[1], color=col)
        if idx0 > 0:
            ax.plot(trajectories[idx0].y[0], trajectories[idx0].y[1],
                    color='black')
    plt.title(str(np.round(p['damping'], decimals=5)))
    # plt.title(str(np.round(p['linear_normalized_damping_coefficient'],
    #          decimals=2)))
    plt.xlabel('x position')
    plt.ylabel('y position')


# * Waterfall plot
def compute_measure_postep(data, trajecs, S_M = None):

    if S_M is None:
        S_M = data['S_M']
    # apex state after step
    XN = [traj.y[:, -1] for traj in trajecs]
    # state in low-dim state-space
    SN = [sys.xp2s_y_xdot(xn, data['p']) for xn in XN]
    # digitalize, to bin index
    SN_dig = [vibly.digitize_s(sn, data['grids']['states']) for sn in SN]
    # measure of each point
    SNM = [interp_measure(sbin, S_M,
                                 data['grids']) for sbin in SN_dig]

    return SNM


def waterfall_plot(fig, ax, X, Y, Z,
                color='viridis',
                line_width=2):
    '''
    Make a waterfall plot
    Input:
        fig,ax : matplotlib figure and axes to populate
        Z : n,m numpy array. Must be a 2d array even if only one line should be plotted
        X,Y : n,m array
    '''
    # Set normalization to the same values for all plots
    norm = plt.Normalize(Z.min().min(), Z.max().max())
    # Check sizes to loop always over the smallest dimension
    n, m = Z.shape
    if n > m:
        X = X.T; Y = Y.T; Z = Z.T
        m, n = n, m

    for j in range(n):
        # reshape the X,Z into pairs
        points = np.array([X[j, :], Z[j, :]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = collections.LineCollection(segments, cmap=color, norm=norm)
        # Set the values used for colormapping
        lc.set_array((Z[j, 1:] + Z[j, :-1])/2)
        lc.set_linewidth(line_width)  # set linewidth a little larger to see properly the colormap variation
        line = ax.add_collection3d(lc, zs=(Y[j, 1:]+Y[j, :-1])/2, zdir='y')  # add line to axes

    fig.colorbar(lc)  # add colorbar, as the normalization is the same for all, it doesent matter which of the lc objects we use


def get_max_measure(SM_list):
    v_max = 0
    for SM in SM_list:
        if np.max(SM) > v_max:
            v_max = np.max(SM)
    return v_max


def poincare_plot(fig, ax, data, vmax=1, trajectories=None, min_M = 0.0,
                  col_offset=0.65):

    grids = data['grids']
    extent = [grids['states'][1][0],
              grids['states'][1][-1],
              grids['states'][0][0],
              grids['states'][0][-1]]
    # vmax = get_max_measure((data['S_M'] for data in data_list))
    ax.imshow(data['S_M'], origin='lower', extent=extent, aspect='auto',
                interpolation='none', vmin=0, vmax=vmax, cmap='viridis')
    plt.title(str(np.round(data['p']['damping'], decimals=5)))

    if trajectories is not None:
        # X0 = [traj.y[:, 0] for traj in trajectories]
        # XN = [traj.y[:, -1] for traj in trajectories]
        # SN = [sys.xp2s_y_xdot(xn, data['p']) for xn in XN]

        # s_grid_shape = list(map(np.size, grids['states']))
        # # s_bin_shape = tuple(dim+1 for dim in s_grid_shape)
        # SN_dig = [vibly.digitize_s(sn, grids['states']) for sn in SN]
        # SNM = np.array([interp_measure(sbin, data['S_M'], grids)
        #                 for sbin in SN_dig])
        # ground_heights = [traj.y[-1, 0] for traj in trajectories]
        # indices = np.arange(SNM.size)
        # SNp = np.array(SN).T

        # * get index of nominal trajectory, max step up/down
        index0, max_up, max_down = get_perturbation_indices(trajectories)
        # index0 = -1
        # up_indices = list()
        # down_indices = list()
        # for idx, traj in enumerate(trajectories):


        # * plot each step at next step
        up_cmap = plt.get_cmap("Blues")
        down_cmap = plt.get_cmap("Reds")
        idx0 = -1
        for idx in range(len(trajectories)):
            traj = trajectories[idx]
            xn = traj.y[:, -1]  # state at next step
            sn = sys.xp2s_y_xdot(xn, data['p'])
            sbin = vibly.digitize_s(sn, grids['states'])
            s_m = interp_measure(sbin, data['S_M'], grids)
            if s_m > min_M:
                if np.isclose(xn[-1], 0):
                    # print afterwards, so it's on top of other circles
                    sn0 = sn.copy()
                    idx0 = 1
                    continue
                elif xn[-1] < 0:
                    col = down_cmap(col_offset-np.abs(xn[-1]))
                elif xn[-1] > 0:
                    col = up_cmap(col_offset-np.abs(xn[-1]))

                ax.scatter(sn[1], sn[0],
                           facecolors='none', edgecolors=col, s=20)
        if idx0 > 0:
            ax.scatter(sn0[1], sn0[0],
                       facecolors='none', edgecolors='black', s=20)
