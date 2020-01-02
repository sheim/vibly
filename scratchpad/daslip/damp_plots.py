import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import models.daslip as sys
import viability as vibly  # algorithms for brute-force viability
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 

import plotting.guinea_plotters as daplot

import pickle

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

sns.set_style('dark')
sns.set_context('poster')

# * Flags for which plots to generate

FLAG_TRAJS = False
WATERFALL_PLOT = True

# * Load and partially parse data

foldername = 'all_birds/'
set_files = ['all_birds_0.01.pickle',
             'all_birds_0.06.pickle',
             'all_birds_0.1.pickle',
             'all_birds_0.1.pickle',
             'all_birds_0.15.pickle',
             'all_birds_0.2.pickle']

traj_files = ['all_birds_0.01_trajs.pickle',
              'all_birds_0.06_trajs.pickle',
              'all_birds_0.1_trajs.pickle',
              'all_birds_0.1_trajs.pickle',
              'all_birds_0.15_trajs.pickle',
              'all_birds_0.2_trajs.pickle']

data = list()
trajecs = list()
for file_1, file_2 in zip(set_files, traj_files):
    infile_1 = open(foldername+file_1, 'rb')
    infile_2 = open(foldername+file_2, 'rb')
    data.append(pickle.load(infile_1))
    trajecs.append(pickle.load(infile_2)['trajectories'])
    infile_1.close()
    infile_2.close()

# Z[idx, :] = np.array(SNM)

# * compute measure after perturbation

# XN = [traj.y[:, -1] for traj in data[idx]['trajectories']]
# SN = [sys.xp2s_y_xdot(xn, data[idx]['p']) for xn in XN]

# * plot individual trajectories

if FLAG_TRAJS:
    # * get index of nominal trajectory, max step up/down
    index0 = -1
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
    plot_these = (1, 2)
    # trajectories = data[idx]['trajectories']

    for idx in plot_these:
        # fig = plt.figure(idx)
        fig, ax = plt.subplots()
        daplot.ground_perturbations(ax, trajecs[idx], data[idx]['S_M'],
                                    data[idx]['grids'], data[idx]['p'])
    plt.show()

# * plot waterfalls
if WATERFALL_PLOT:

    # prepare data
    damping_values = np.array([d['p']['damping'] for d in data])

    X0 = [traj.y[:, 0] for traj in trajecs[0]]
    ground_heights = [x[-1] for x in X0]

    x = ground_heights
    y = damping_values
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for idx in range(len(data)):
        SNM = daplot.compute_measure_postep(data[idx], trajecs[idx])
        Z[idx, :] = np.array(SNM)

    # plot

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt_from = 0
    plt_till = 4
    daplot.waterfall_plot(fig, ax, X[:, plt_from:plt_till],
                   Y[:, plt_from:plt_till],
                   Z[:, plt_from:plt_till], line_width=4)
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
