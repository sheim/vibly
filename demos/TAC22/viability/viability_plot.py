import numpy as np

import matplotlib as cm
cm.use("TkAgg")
cm.rcParams['pdf.fonttype'] = 42
cm.rcParams['ps.fonttype'] = 42
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import viability as vibly  # algorithms for brute-force viability
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import rcParams
# from scipy.signal import savgol_filter
import pickle
import os
import matplotlib.collections as collections

import plotting.value_plotters as vplot
colwide = 3.5
pagewide = 7.16
# import seaborn as sns
# sns.set_theme(style="darkgrid")

# * set up helper functions

# * Load data

filename = "./viability1.pickle"
# * Load and unpack

infile = open(filename, 'rb')
data = pickle.load(infile)
infile.close()
# * unpack invariants
# Q_value = data["Q_value"]
R_value = data["R_value"]
grids = data["grids"]
S_M = data["S_M"]
XV = S_M > 0.0

# *
RX_value = vibly.project_Q2S(R_value, grids, proj_opt=np.mean)

# * limit of reward
r_max = RX_value.max()
r_min = RX_value.min()
# * limit of values
alpha = 0.
mynorm = colors.TwoSlopeNorm(vmin=-2.5, vcenter=alpha, vmax=0.1)
# mynorm = colors.CenteredNorm()
mymap = vplot.get_vmap(2)



extent = [grids['states'][1][0],
        grids['states'][1][-1],
        grids['states'][0][0],
        grids['states'][0][-1]]

# * plot reward function
figR, axR = plt.subplots(figsize=(colwide, 3/6*colwide),
                         constrained_layout=True)
rmap = colors.ListedColormap(["red", "white"])
pcR = vplot.reward_function(axR, RX_value, grids, XV=XV, mymap=rmap)
axR.set_xlabel("$x_2$")
axR.set_ylabel("$x_1$")
figR.colorbar(pcR, ax=axR, orientation='horizontal', extend='both',
                spacing='proportional', shrink=1)
plt.savefig('viability_R.pdf', format='pdf')
plt.close()

# # * plot value function
# fig, ax = plt.subplots(1, figsize=(colwide, 3/6*colwide),
#                             constrained_layout=True)

# X_value = vibly.project_Q2S(data["Q_value"], grids, proj_opt=np.max)
# lvls = [-2., -1., 0.]
# print("max viable: ", X_value[XV].max())
# print("min viable: ", X_value[XV].min())
# print("max unviable: ", X_value[~XV].max())
# print("min unviable: ", X_value[~XV].min())
# print("-----")
# pc1 = vplot.value_function(ax,
#                             X_value,
#                             grids,
#                             XV=XV,
#                             mymap=mymap,
#                             cont_lvls=lvls,
#                             mynorm=mynorm)

# # fig.colorbar(pc1, ax=ax, orientation='horizontal', extend='both')
# fig.colorbar(pc1, ax=ax, orientation='horizontal', extend='min',
#              ticks=[-2., -1., 0.])
# ax.set_xlabel("$x_2$")
# ax.set_ylabel("$x_1$")
# # * Reward function
# # pc0 = vplot.reward_function(ax, RX_value, grids, XV=XV,
# #                             mymap="PRGn", mynorm=mynorm)  #,
# #                             viability_threshold=0)
# # fig.colorbar(pc0, ax=ax)

# # for idx in range(num_plots-1):
# #     plt.setp(ax[idx].get_xticklabels(), visible=False)

# plt.savefig(filename+'.pdf', format='pdf')
# plt.show()
# plt.close()
