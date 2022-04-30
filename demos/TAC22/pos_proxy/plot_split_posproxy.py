import numpy as np

import matplotlib as cm
cm.use("TkAgg")
cm.rcParams['pdf.fonttype'] = 42
cm.rcParams['ps.fonttype'] = 42
import matplotlib.colors as colors
import matplotlib.cbook as cbook
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

# ! p* at 111
# * Load data
alpha = 0.

filenames = ["posproxy1.pickle",
             "posproxy111.pickle"]

data_list = list()

try:
    for filename in filenames:
        infile = open(filename, 'rb')
        data_list.append(pickle.load(infile))
        infile.close()
except FileNotFoundError:
    print("ERROR: data not found.")

# * unpack invariants
R_value = data_list[0]["R_value"]
grids = data_list[0]["grids"]
S_M = data_list[0]["S_M"]

# * Ground Truth XV
XV = S_M > 0.0

mynorm = colors.TwoSlopeNorm(vmin=-100, vcenter=0.0, vmax=2.5)
mymap = vplot.get_vmap(0)

extent = [grids['states'][1][0],
        grids['states'][1][-1],
        grids['states'][0][0],
        grids['states'][0][-1]]

figR, axR = plt.subplots(figsize=(colwide, 3/6*colwide), constrained_layout=True)
# * plot reward function
R_value = data_list[0]["R_value"]
RX_value = vibly.project_Q2S(R_value, grids, proj_opt=np.mean)

rmap = colors.ListedColormap(["red", "white", "darkgrey"])
pcR = vplot.reward_function(axR, RX_value, grids, XV=XV, mymap=rmap)
axR.set_xlabel("$x_2$")
axR.set_ylabel("$x_1$")
figR.colorbar(pcR, ax=axR, orientation='horizontal', extend='both',
                spacing='proportional', shrink=1)
plt.savefig('posproxy_R.pdf', format='pdf')
plt.close()
# * plot value functions
lvls = [-100., -50., alpha, 1., 2.]
for pdx, data in enumerate(data_list):
    fig, ax = plt.subplots(figsize=(colwide, 3/6*colwide), constrained_layout=True)
    X_value = vibly.project_Q2S(data["Q_value"], grids, proj_opt=np.max)
    pc1 = vplot.value_function(ax,
                                X_value,
                                grids,
                                XV=XV, cont_lvls=lvls,
                                mymap=mymap,
                                mynorm=mynorm)
    print("max: ", X_value.max())
    print("min viable: ", X_value[XV].min())
    print("max unviable: ", X_value[~XV].max())
    fig.colorbar(pc1, ax=ax, orientation='horizontal', extend='both',
                spacing='proportional', ticks=lvls, shrink=0.8)
    ax.set_xlabel("$x_2$")
    ax.set_ylabel("$x_1$")
    plt.savefig('posproxy_Vp_'+str(pdx)+'.pdf', format='pdf')
plt.close()
