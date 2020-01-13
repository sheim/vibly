import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import models.parslip as model
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

# * Load and partially parse data

set_files = ['human_aoa/human_aoa_0.01.pickle',
'human_aoa/human_aoa_0.1.pickle',
'human_aoa/human_aoa_0.5.pickle',
'human_aoa/human_aoa_1.0.pickle',
'human_aoa/human_aoa_3.0.pickle'
]

traj_files = ['human_aoa/human_aoa_0.01_trajs3.pickle',
'human_aoa/human_aoa_0.1_trajs3.pickle',
'human_aoa/human_aoa_0.5_trajs3.pickle',
'human_aoa/human_aoa_1.0_trajs3.pickle',
'human_aoa/human_aoa_3.0_trajs3.pickle'
]

# set_files = ['higher/higher_0.0125.pickle',
#              'higher/higher_0.015.pickle',
#              'higher/higher_0.0175.pickle',
#              'higher/higher_0.02.pickle',
#              'higher/higher_0.0225.pickle',
#              'higher/higher_0.025.pickle',
#              'higher/higher_0.050.pickle',
#              'higher/higher_0.075.pickle',
#              'higher/higher_0.100.pickle',
#              'higher/higher_0.125.pickle',
#              'higher/higher_0.150.pickle',
#              'higher/higher_0.175.pickle',
#              'higher/higher_0.2.pickle',
#              'higher/higher_0.25.pickle',
#              'higher/higher_0.5.pickle',
#              'higher/higher_0.75.pickle',
#              'higher/higher_1.0.pickle',
#              'higher/higher_1.25.pickle',
#              'higher/higher_1.5.pickle',
#              'higher/higher_1.75.pickle',
#              'higher/higher_2.0.pickle',
#              'higher/higher_2.25.pickle',
#              'higher/higher_2.5.pickle',
#              'higher/higher_2.75.pickle',
#              'higher/higher_3.0.pickle',
#              'higher/higher_3.25.pickle',
#              'higher/higher_3.5.pickle']

# traj_files = ['higher/higher_0.0125_trajs2.pickle',
#               'higher/higher_0.015_trajs2.pickle',
#               'higher/higher_0.0175_trajs2.pickle',
#               'higher/higher_0.02_trajs2.pickle',
#               'higher/higher_0.0225_trajs2.pickle',
#               'higher/higher_0.025_trajs2.pickle',
#               'higher/higher_0.050_trajs2.pickle',
#               'higher/higher_0.075_trajs2.pickle',
#               'higher/higher_0.100_trajs2.pickle',
#               'higher/higher_0.125_trajs2.pickle',
#               'higher/higher_0.150_trajs2.pickle',
#               'higher/higher_0.175_trajs2.pickle',
#               'higher/higher_0.2_trajs2.pickle',
#               'higher/higher_0.25_trajs2.pickle',
#               'higher/higher_0.5_trajs2.pickle',
#               'higher/higher_0.75_trajs2.pickle',
#               'higher/higher_1.0_trajs2.pickle',
#               'higher/higher_1.25_trajs2.pickle',
#               'higher/higher_1.5_trajs2.pickle',
#               'higher/higher_1.75_trajs2.pickle',
#               'higher/higher_2.0_trajs2.pickle',
#               'higher/higher_2.25_trajs2.pickle',
#               'higher/higher_2.5_trajs2.pickle',
#               'higher/higher_2.75_trajs2.pickle',
#               'higher/higher_3.0_trajs2.pickle',
#               'higher/higher_3.25_trajs2.pickle',
#               'higher/higher_3.5_trajs2.pickle']

data_list = list()
trajec_list = list()
for file_1, file_2 in zip(set_files, traj_files):
    infile_1 = open(file_1, 'rb')
    infile_2 = open(file_2, 'rb')
    data_list.append(pickle.load(infile_1))
    trajec_list.append(pickle.load(infile_2)['trajectories'])
    infile_1.close()
    infile_2.close()

# * Flags for which plots to generate

FLAG_TRAJS = True
WATERFALL_PLOT = False
POINC_PLOT = False
ACTIVATION_PLOT = False

# * which trials

total_trials = len(data_list)
plot_these = range(total_trials)
# plot_these = [0, 5, 10, total_trials-1]

# * plot individual trajectories

if FLAG_TRAJS:
    # * get index of nominal trajectory, max step up/down
    index0 = -1
    max_up = 0
    max_down = 0
    tdx = 0
    for idx, traj in enumerate(trajec_list[tdx]):
        if np.isclose(traj.y[-1, 0], 0):
            index0 = idx

        if traj.y[-1, 0] > max_up:
            max_up = traj.y[-1, 0]
        if traj.y[-1, 0] < max_down:
            max_down = traj.y[-1, 0]
    # else:
    #     print("WARNING: no nominal trajectory")

    num_up = index0-1
    num_down = len(trajec_list[tdx])-index0

    # height_cmap = sns.diverging_palette(10, 220, l=90, s=99, as_cmap=True, center='dark')
    # sns.color_palette("RdBu")
    # plot_these = tuple(range(len(trajec_list)))
    # plot_these = (1, 2)
    # trajectories = data_list[idx]['trajectories']

    for idx in plot_these:
        # fig = plt.figure(idx)
        fig, ax = plt.subplots()
        daplot.plot_ground_perturbations(ax, trajec_list[idx],
                                         data_list[idx]['S_M'],
                                         data_list[idx]['grids'],
                                         data_list[idx]['p'], v_threshold=0)
    plt.show()

# * plot waterfalls

def weighted_mean(A, axes, weights):
    assert len(A.shape) == weights.size
    V = A.copy()
    for idx in axes:
        V = np.mean(V, axis=0)*weights[idx]  # at each iteration, dim(V) -= 1
    return V

if WATERFALL_PLOT:

    # prepare data_list
    damping_values = np.array([d['p']['damping'] for d in data_list])

    X0 = [traj.y[:, 0] for traj in trajec_list[0]]
    ground_heights = [x[-1] for x in X0]

    x = ground_heights
    y = damping_values
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for idx in range(len(data_list)):

        # optionally, recompute S_M
        # if True:


        SNM = daplot.compute_measure_postep(data_list[idx], trajec_list[idx])
        Z[idx, :] = np.array(SNM)

    # plot

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    daplot.waterfall_plot(fig, ax, X, Y, Z, line_width=4)
    # plt_from = 0
    # plt_till = 4
    # daplot.waterfall_plot(fig, ax, X[:, plt_from:plt_till],
    #                Y[:, plt_from:plt_till],
    #                Z[:, plt_from:plt_till], line_width=4)
    ax.set_xlabel('ground height')
    # ax.set_xlim3d(np.min(x[plt_from:plt_till]), np.max(x[plt_from:plt_till]))
    ax.set_xlim3d(np.min(x), np.max(x))
    ax.set_ylabel('normalized damping coefficient')
    ax.set_ylim3d(np.min(y), np.max(y))
    ax.set_zlabel('safety measure')
    ax.set_zlim3d(0, 0.6)
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 12
    plt.show()

if POINC_PLOT:

    # find the maximum measure value, to normalize colormap
    vmax = daplot.get_max_measure((data_list['S_M'] for data_list in data_list))

    rem_string = len('.pickle')
    for i, data in enumerate(data_list[i] for i in plot_these):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        daplot.poincare_plot(fig, ax, data, vmax=vmax,
                             trajectories=trajec_list[i], min_M=0.0)
        # plt.show()
        new_filename = set_files[i][0:len(set_files[i])-rem_string]+'_withdelay.pdf'
        plt.savefig(new_filename, format='pdf')
        plt.close()

if ACTIVATION_PLOT:
    for d in data_list:
        plt.plot(d['p']['actuator_force'][0],
                 d['p']['actuator_force'][1]/(d['p']['mass']*d['p']['gravity']))
    plt.xlabel('time [s]')
    plt.ylabel('activation [bodyweight]')
    plt.title('Open-loop activation, bird fit')
    plt.show()