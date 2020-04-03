import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import models.daslip as model
import viability as vibly  # algorithms for brute-force viability
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
from scipy.signal import savgol_filter

import plotting.guinea_plotters as daplot

import pickle

# rcParams['xtick.major.width']=0
# rcParams['ytick.major.width']=0
# rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

# sns.set_style('dark', {'axes.grid': False})
sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_context("paper", font_scale=2.5)


# * Load and partially parse data

set_files = ['falling/fall/fall_0.0010.pickle',
             'falling/fall/fall_0.0050.pickle',
             'falling/fall/fall_0.0100.pickle',
             'falling/fall/fall_0.0200.pickle',
             'falling/fall/fall_0.0300.pickle',
             'falling/fall/fall_0.0400.pickle',
             'falling/fall/fall_0.0500.pickle',
             'falling/fall/fall_0.0600.pickle',
             'falling/fall/fall_0.0700.pickle',
             'falling/fall/fall_0.0800.pickle',
             'falling/fall/fall_0.0900.pickle',
             'falling/fall/fall_0.1000.pickle',
             'falling/fall/fall_0.1100.pickle',
             'falling/fall/fall_0.1200.pickle',
             'falling/fall/fall_0.1300.pickle',
             'falling/fall/fall_0.1400.pickle',
             'falling/fall/fall_0.1500.pickle',
             'falling/fall/fall_0.1600.pickle',
             'falling/fall/fall_0.1700.pickle',
             'falling/fall/fall_0.1800.pickle',
             'falling/fall/fall_0.1900.pickle'
]

traj_files = ['falling/fall/fall_0.0010_trajs.pickle',
              'falling/fall/fall_0.0050_trajs.pickle',
              'falling/fall/fall_0.0100_trajs.pickle',
              'falling/fall/fall_0.0200_trajs.pickle',
              'falling/fall/fall_0.0300_trajs.pickle',
              'falling/fall/fall_0.0400_trajs.pickle',
              'falling/fall/fall_0.0500_trajs.pickle',
              'falling/fall/fall_0.0600_trajs.pickle',
              'falling/fall/fall_0.0700_trajs.pickle',
              'falling/fall/fall_0.0800_trajs.pickle',
              'falling/fall/fall_0.0900_trajs.pickle',
              'falling/fall/fall_0.1000_trajs.pickle',
              'falling/fall/fall_0.1100_trajs.pickle',
              'falling/fall/fall_0.1200_trajs.pickle',
              'falling/fall/fall_0.1300_trajs.pickle',
              'falling/fall/fall_0.1400_trajs.pickle',
              'falling/fall/fall_0.1500_trajs.pickle',
              'falling/fall/fall_0.1600_trajs.pickle',
              'falling/fall/fall_0.1700_trajs.pickle',
              'falling/fall/fall_0.1800_trajs.pickle',
              'falling/fall/fall_0.1900_trajs.pickle'
]

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

TRAJ_PLOTS = False
WATERFALL_PLOT = True
POINC_PLOT = False

# * which trials
# This is only relevant for TRAJS and POINC plots.
total_trials = len(data_list)
# plot_these = range(total_trials)  # plot all trials
# plot_these = [0, 8, total_trials-1]
plot_these = [0, 9, total_trials-1]
# plot_these = range(2)

# * plot waterfalls

def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)

FILTER = True  # plot raw data, or smoothen?
poly = 3  # by how much to smooth

# prepare data_list
damping_values = np.array([d['p']['constant_normalized_damping']
                            for d in data_list])
stiffness = data_list[0]['p']['stiffness']
X0 = [traj.y[:, 0] for traj in trajec_list[0]]
ground_heights = [x[-1] for x in X0]

x = ground_heights
y = damping_values  # *data_list[0]['p']['stiffness']
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for idx in range(len(data_list)):
    SNM = daplot.compute_measure_postep(data_list[idx], trajec_list[idx])
    Z[idx, :] = np.array(SNM)
norm = plt.Normalize(Z.min().min(), Z.max().max())
if WATERFALL_PLOT:
    # plot
    if FILTER:
        indices = np.arange(len(ground_heights))
        for ydx in range(len(y)):
            zfirst = np.min(indices[Z[ydx, :] != 0])
            zlast = np.max(indices[Z[ydx, :] != 0])
            length = zlast - zfirst

            window = round_up_to_odd((zlast-zfirst)/10)
            # we want the window to round off, so it doesn't smooth over the edge.
            if ydx + np.floor(window/2) > zlast:
                window = round_up_to_odd((zlast-ydx)*2)

            if window > poly:
                z = Z[ydx, zfirst:zlast]
                Z[ydx, zfirst:zlast] = savgol_filter(z, window, poly)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    daplot.waterfall_plot(fig, ax, X, Y, Z, line_width=4)

    ax.set_xlabel(r'ground height $[m]$')
    # ax.set_xlim3d(np.min(x[plt_from:plt_till]), np.max(x[plt_from:plt_till]))
    ax.set_xlim3d(np.min(ground_heights), np.max(ground_heights))
    ax.set_ylabel(r'         damping $[s]$', va='center', ha='left')
    ax.set_ylim3d(np.min(y), np.max(y))
    # ax.set_yticklabels(['{:.1f}'.format(x*stiffness) for x in damping_values], va='center', ha='left')
    ytickers = [x for x in damping_values[0:-1:4]]
    ax.set_yticks(ytickers)
    ax.set_yticklabels(['{:.2f}'.format(x) for x in ytickers], va='center', ha='left')

    ax.set_zlabel(r'   viability measure [deg]')
    ax.set_zlim3d(0, 0.5)
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 50


    # relabel z-axis, scaled by 60 degrees (range of AoAs)
    ztickers = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_zticks(ztickers)
    ax.set_zticklabels(['  {:.0f}'.format(x) for x in ztickers*60],
                        va='center')
    ax.zaxis.labelpad = 30
    # ax.tick_params(axis='both', which='major', length=0)
    ax.view_init(elev=42., azim=-83)

    plt.show()


# * plot individual trajectories

if TRAJ_PLOTS:
    plot_these = (1,)
    # do some data processing to normalize the labels of the xticks etc.
    # Note, this takes a bit of finessing
    # Get some defualt values. These are the same across all trials
    leg_length = data_list[0]['p']['resting_length']  # +data_list[0]['x0'][6]
    my_yticks = np.arange(-0.1, 0.35, 0.05)
    my_ytick_labels = np.round(my_yticks/leg_length, decimals=1) # ticks normalized by hip height

    my_xticks =  np.arange(0, 1.0, leg_length/2)
    my_xtick_labels = np.round(my_xticks/leg_length, decimals=1)

    for idx in plot_these:
        # fig = plt.figure(idx)
        fig, ax = plt.subplots()
        daplot.plot_ground_perturbations(ax, trajec_list[idx],
                                         data_list[idx]['S_M'],
                                         data_list[idx]['grids'],
                                         data_list[idx]['p'],
                                         v_threshold=0.5/60.0,
                                         col_offset=0.7,
                                         draw_LC=True,
                                         Q_M=data_list[idx]['Q_M'],
                                         norm=norm)
        # name = 'trajectories_'+str(idx)
        # plt.savefig(name, format='eps')
        plt.ylim(-0.075, 0.275)
        plt.xlabel(r'x position leg lengths $\ell_0$')
        plt.ylabel(r'y position leg lengths $\ell_0$')

        plt.xticks(my_xticks, my_xtick_labels)
        plt.yticks(my_yticks, my_ytick_labels)
        plt.gca().set_aspect('equal', adjustable='box')
        # ax.xlabels(my_xtick_labels)
        # ax.yticks(my_yticks)
        # ax.ylabels(my_ytick_labels)
        plt.show()
        plt.close()



if POINC_PLOT:
    # find the maximum measure value, to normalize colormap
    vmax = daplot.get_max_measure((data_list['S_M'] for data_list in data_list))

    rem_string = len('.pickle')
    for i, data in enumerate(data_list[i] for i in plot_these):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        daplot.poincare_plot(fig, ax, data, vmax=vmax,
                             trajectories=trajec_list[i], min_M=0.0)
        plt.show()
        # new_filename = set_files[i][0:len(set_files[i])-rem_string]+'_withdelay.pdf'
        # plt.savefig(new_filename, format='pdf')
        plt.close()