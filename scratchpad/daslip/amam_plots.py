import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import models.daslip as sys
import viability as vibly  # algorithms for brute-force viability
import seaborn as sns

import pickle

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

sns.set_style('dark')
sns.set_context('poster')

FLAG_ALL_SM = False
FLAG_SM4 = True
FLAG_TRAJS = True
FLAG_DAMPING_2D = False
FLAG_WATERFALL = False


def interp_measure(s_bin, S_M, grids):
    neighbor_idx = vibly.get_grid_indices(s_bin, grids['states'])
    measure = 0
    if len(neighbor_idx) > 0:
        for idx in neighbor_idx:
            measure += S_M[idx]
        return measure/len(neighbor_idx)
    else:
        return 0.0


def get_step_trajectories(x0_, p_, ground_heights_=None):

    if ground_heights_ is None:
        total_leg_length = p_['spring_resting_length']
        total_leg_length += p_['actuator_resting_length']
        ground_heights_ = np.linspace(0, -0.5*total_leg_length, 10)
    x0_ = sys.reset_leg(x0_, p_)
    trajectories_ = list()
    for height in ground_heights_:
        x0_[-1] = height
        trajectories_.append(sys.step(x0_, p_))
    x0_[-1] = 0.0  # reset x0 back to 0
    return trajectories_


def run_trials(dat, p_keys, p_vals, ground_heights_=(0.0,)):
    x0 = dat['x0'].copy()
    p = dat['p'].copy()
    for key, val in zip(p_keys, p_val):
        p[key] = val
    trajectories = get_step_trajectories(x0, p,
                                         ground_heights_=ground_heights_)
    return trajectories

foldername = 'thurs_amamhu/'
filenames = ['LING_1.pickle',
             'LING_2.pickle',
             'LING_3.pickle',
             'LING_4.pickle',
             'LING_5.pickle',
             'LING_6.pickle',
             'LING_7.pickle',
             'LING_8.pickle',
             'LING_9.pickle',
             'LING_10.pickle',
             'LING_11.pickle',
             'LING_12.pickle',
             'LING_13.pickle',
             'LING_14.pickle',
             'LING_15.pickle']

data = list()
trajecs = list()
for filename in filenames:
    infile = open(foldername+filename, 'rb')
    infile2 = open(foldername+'Traj_'+filename, 'rb')
    data.append(pickle.load(infile))
    trajecs.append(pickle.load(infile2)['trajectories'])
    infile.close()
    infile2.close()

# total_leg_length = 1
# * regen trajectories with more resolution
# ground_heights = np.linspace(0.2*total_leg_length,
#                              -0.3*total_leg_length, 6)
# traj_list = []
# for idx in range(len(data)-10):
#     p = data[idx]['p'].copy()
#     x0 = data[idx]['x0'].copy()
#     x0 = sys.reset_leg(x0, p)
#     sys.compute_total_energy(x0, p)
#     sys.create_open_loop_trajectories(x0, p)
#     trajectories = get_step_trajectories(x0,
#                                          p,
#                                          ground_heights_=ground_heights)
#     # data[idx]['trajectories'] = trajectories.copy()
#     traj_list.append(trajectories.copy())

# * Plot all the things
if FLAG_ALL_SM:
    # prepare heatmap params
    extent = [data[0]['grids']['states'][1][0],
                data[0]['grids']['states'][1][-1],
                data[0]['grids']['states'][0][0],
                data[0]['grids']['states'][0][-1]]
    # plt.imshow(data[0]['S_M'], origin='lower', extent=extent)
    vmax = 0
    for idx in range(len(data)):
        if np.max(data[idx]['S_M']) > vmax:
            vmax = np.max(data[idx]['S_M'])

    # plot_these = range(len(data))
    plot_these = (5, 10, 12, 14)

    for idx in plot_these:
        extent = [data[idx]['grids']['states'][1][0],
                data[idx]['grids']['states'][1][-1],
                data[idx]['grids']['states'][0][0],
                data[idx]['grids']['states'][0][-1]]

        plt.figure(idx)
        plt.imshow(data[idx]['S_M'], origin='lower', extent=extent,
                interpolation='bessel', vmin=0, vmax=vmax, cmap='viridis')
        # sns.heatmap(data[idx]['S_M'],
        #             vmin=0, vmax=vmax, cmap='viridis')

        plt.title(str(np.round(data[idx]['p']['linear_normalized_damping_coefficient'], decimals=2)))
        X0 = [traj.y[:, 0] for traj in data[idx]['trajectories']]
        XN = [traj.y[:, -1] for traj in data[idx]['trajectories']]
        SN = [sys.xp2s_y_xdot(xn, data[idx]['p']) for xn in XN]
        s_grid_shape = list(map(np.size, data[idx]['grids']['states']))
        s_bin_shape = tuple(dim+1 for dim in s_grid_shape)
        SN_dig = [vibly.digitize_s(sn, data[idx]['grids']['states']) for sn in SN]
        SNM = np.array([interp_measure(sbin, data[idx]['S_M'], data[idx]['grids']) for sbin in SN_dig])
        # pick out only the ones inside the grid
        ground_heights = [x[-1] for x in X0]
        indices = np.arange(SNM.size)
        SNp = np.array(SN).T
        plt.scatter(SNp[1, indices[SNM>0.1]], SNp[0, indices[SNM>0.1]],
                    facecolors='none', edgecolors=[0.8, 0.3, 0.3], s=20)

    plt.show()

# * make a collage of 4

# todo: color-code with same coloring as trajectories plot
# todo: make trajectories plot
if FLAG_SM4:
    fig = plt.figure()
    plot_these = (5, 10, 12, 14)
    ncol = 2
    nrow = 2
    gs1 = gridspec.GridSpec(nrow, ncol,
                            wspace=0.0, hspace=0.0,
                            top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1),
                            left=0.5/(ncol+1), right=1-0.5/(ncol+1))

    vmax = 0
    for idx in plot_these:
        if np.max(data[idx]['S_M']) > vmax:
            vmax = np.max(data[idx]['S_M'])
    axes = []
    for gdx, idx in enumerate(plot_these):
        extent = [data[idx]['grids']['states'][1][0],
                data[idx]['grids']['states'][1][-1],
                data[idx]['grids']['states'][0][0],
                data[idx]['grids']['states'][0][-1]]
        axes.append(plt.subplot(gs1[gdx]))
        axes[gdx].imshow(data[idx]['S_M'], origin='lower', extent=extent,
                interpolation='bessel', vmin=0, vmax=vmax, cmap='viridis')
        axes[gdx].title.set_text(str(np.round(data[idx]['p']['linear_normalized_damping_coefficient'], decimals=2)))
        X0 = [traj.y[:, 0] for traj in data[idx]['trajectories']]
        XN = [traj.y[:, -1] for traj in data[idx]['trajectories']]
        SN = [sys.xp2s_y_xdot(xn, data[idx]['p']) for xn in XN]
        s_grid_shape = list(map(np.size, data[idx]['grids']['states']))
        s_bin_shape = tuple(dim+1 for dim in s_grid_shape)
        SN_dig = [vibly.digitize_s(sn, data[idx]['grids']['states']) for sn in SN]
        SNM = np.array([interp_measure(sbin, data[idx]['S_M'], data[idx]['grids']) for sbin in SN_dig])
        # pick out only the ones inside the grid
        ground_heights = [x[-1] for x in X0]
        indices = np.arange(SNM.size)
        SNp = np.array(SN).T
        axes[gdx].scatter(SNp[1, indices[SNM>0.1]], SNp[0, indices[SNM>0.1]],
                facecolors='none', edgecolors=[0.8, 0.3, 0.3], s=20)
        axes[gdx].set_xlabel("apex velocity")
        axes[gdx].set_ylabel("apex height")

    # plt.tight_layout()
    gs1.tight_layout(fig)
    plt.show()

# * Trajectories plot

if FLAG_TRAJS:
    # * get index of nominal trajectory, max step up/down
    index0 = 0
    max_up = 0
    max_down = 0
    for idx, traj in enumerate(trajecs[0]):
        if np.isclose(traj.y[-1, 0], 0):
            index0 = idx

        if traj.y[-1, 0] > max_up:
            max_up = traj.y[-1, 0]
        if traj.y[-1, 0] < max_down:
            max_down = traj.y[-1, 0]
    # else:
    #     print("WARNING: no nominal trajectory")

    num_up = index0-1
    num_down = len(trajecs[0])-index0

    # height_cmap = sns.diverging_palette(10, 220, l=90, s=99, as_cmap=True, center='dark')
    # sns.color_palette("RdBu")
    # plot_these = tuple(range(len(trajecs)))
    plot_these = (5, 10, 12, 14)
    # trajectories = data[idx]['trajectories']

    for idx in plot_these:
        fig = plt.figure(idx)
        mycmap = plt.get_cmap("Blues")
        for up_dx in range(index0):

            # col = height_cmap(0.5+traj.y[-1, 0])
            traj = trajecs[idx][up_dx]

            # X0 = [traj.y[:, 0] for traj in data[idx]['trajectories']]
            x = traj.y[:, -1]
            s = sys.xp2s_y_xdot(x, data[idx]['p'])
            sbin = vibly.digitize_s(s, data[idx]['grids']['states'])
            s_m = interp_measure(sbin, data[idx]['S_M'], data[idx]['grids'])

            if s_m > 0.1:
                col = mycmap(0.65-np.abs(traj.y[-1, 0]))
                plt.plot(traj.y[0], traj.y[1], color=col)

        mycmap = plt.get_cmap("Reds")
        for down_dx in range(index0+1, len(trajecs[idx])):
            traj = trajecs[idx][down_dx]

            x = traj.y[:, -1]
            s = sys.xp2s_y_xdot(x, data[idx]['p'])
            sbin = vibly.digitize_s(s, data[idx]['grids']['states'])
            s_m = interp_measure(sbin, data[idx]['S_M'], data[idx]['grids'])

            if s_m > 0.1:
                col = mycmap(0.65-np.abs(traj.y[-1, 0]))
                plt.plot(traj.y[0], traj.y[1], color=col)

        traj = trajecs[idx][index0]
        plt.plot(traj.y[0], traj.y[1], color='k')
        plt.title(str(np.round(data[idx]['p']['linear_normalized_damping_coefficient'], decimals=2)))
        plt.xlabel('x position')
        plt.ylabel('y position')
    plt.show()
# plot_these = (6,7,8,9,10)
# for tdx in plot_these:
#     plt.plot(trajectories[tdx].y[0], trajectories[tdx].y[1])
# plt.show()

# for traj in traj_list

# * plot damping on 2D

if FLAG_DAMPING_2D:
    plt.figure(15)
    sns.set_palette(sns.light_palette((210, 90, 60), input="husl",n_colors=13))
    for idx in plot_these:
        s_grid_shape = list(map(np.size, data[idx]['grids']['states']))
        s_bin_shape = tuple(dim+1 for dim in s_grid_shape)
        X0 = [traj.y[:, 0] for traj in data[idx]['trajectories']]
        XN = [traj.y[:, -1] for traj in data[idx]['trajectories']]
        SN = [sys.xp2s_y_xdot(xn, data[idx]['p']) for xn in XN]
        SN_dig = [vibly.digitize_s(sn, data[idx]['grids']['states']) for sn in SN]
        SNM = [interp_measure(sbin, data[idx]['S_M'], data[idx]['grids']) for sbin in SN_dig]
        ground_heights = [x[-1] for x in X0]
        plt.plot(ground_heights[0:25], SNM[0:25])

    leggenda = [str(data[idx]['p']['linear_normalized_damping_coefficient']) +
                'damping coeff' for idx in plot_these]

    plt.legend(leggenda)
    plt.xlabel('ground height')
    plt.ylabel('safety measure')
    plt.show()

# for idx in range(len(data)):
#     print("non-failing portion of Q: " + str(np.mean(~data[idx]['Q_F'])))
#     print("viable portion of Q: " + str(np.mean(data[idx]['Q_V'])))
#     print('')


#### waterfalling
if FLAG_WATERFALL:
    import matplotlib.collections as collections

    plt_from = 0
    plt_till = 25

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

    # Generate data
    # x: ground_heights
    # y: damping value
    # z: safety_measure
    x = ground_heights
    y = np.array([d['p']['linear_normalized_damping_coefficient'] for d in data])
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for idx in range(len(data)):
        XN = [traj.y[:, -1] for traj in data[idx]['trajectories']]
        SN = [sys.xp2s_y_xdot(xn, data[idx]['p']) for xn in XN]
        SN_dig = [vibly.digitize_s(sn, data[idx]['grids']['states']) for sn in SN]
        SNM = [interp_measure(sbin, data[idx]['S_M'], data[idx]['grids']) for sbin in SN_dig]
        Z[idx, :] = np.array(SNM)

    # Generate waterfall plot
    l = 4
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    waterfall_plot(fig, ax, X[:, plt_from:plt_till],
                   Y[:, plt_from:plt_till],
                   Z[:, plt_from:plt_till], line_width=l)
    ax.set_xlabel('ground height')
    ax.set_xlim3d(np.min(x[plt_from:plt_till]), np.max(x[plt_from:plt_till]))
    ax.set_ylabel('normalized damping coefficient')
    ax.set_ylim3d(np.min(y), np.max(y))
    ax.set_zlabel('safety measure')
    ax.set_zlim3d(0, 0.6)
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 12
    plt.show()
