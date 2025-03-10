import numpy as np

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import viability as vibly  # algorithms for brute-force viability

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import rcParams
# from scipy.signal import savgol_filter
import pickle

import plotting.value_plotters as vplot

colwide = 3.5
pagewide = 7.16

# ! p* at 136
# * Load data


filenames = [  # "negproxy0.pickle",
    "wacky0.pickle",
    "wacky144.pickle",
]

# filenames = ["negproxy0.pickle",
#             "shaping/L2arti_penalty_10.pickle",
#             "shaping/L2arti_penalty_50.pickle"]

data_list = list()

try:
    for filename in filenames:
        infile = open(filename, "rb")
        data_list.append(pickle.load(infile))
        infile.close()
except FileNotFoundError:
    print("ERROR: data not found.")

# * unpack invariants
# Q_value = data["Q_value"]
R_value = data_list[0]["R_value"]
grids = data_list[0]["grids"]
# Q_map = data["Q_map"]
S_M = data_list[0]["S_M"]

# * Ground Truth XV
XV = S_M > 0.0
Xv2 = vibly.project_Q2S(data_list[1]["Q_value"], grids, proj_opt=np.max)
# alpha = Xv2[XV].min() # viability threshold
alpha = 0.0
# alpha = -4.535  # rounded

mynorm = colors.TwoSlopeNorm(vmin=-1, vcenter=alpha, vmax=1)
# mynorm = colors.CenteredNorm()
mymap = vplot.get_vmap(0)
extent = [
    grids["states"][1][0],
    grids["states"][1][-1],
    grids["states"][0][0],
    grids["states"][0][-1],
]

# these_values = [0, 3, 5, -3, -2, -1]
# these_values = range(len(data_list))
fig, ax = plt.subplots(figsize=(colwide, 3 / 6 * colwide), constrained_layout=True)

# * plot reward function
R_value = data_list[1]["R_value"]
RX_value = vibly.project_Q2S(R_value, grids, proj_opt=np.mean)
RXv = RX_value.copy()

RXv[RXv == -0.4] = 1.0
RXv[RXv == -1.4] = -1.0
RXv[RXv < -100.0] = 0.0
# rmap = colors.ListedColormap(["tab:blue","tab:red", "white"])
# newcolors = [mymap(mynorm(-1.0)), (0., 0., 0., 1), (1., 1., 1., 1.)]
rmap = colors.ListedColormap(["darkgrey", "red", "white"])
# rmap = colors.ListedColormap(newcolors)
pc0 = vplot.reward_function(ax, RXv, grids, XV=XV, mymap=rmap)
ax.set_xlabel("$x_2$")
ax.set_ylabel("$x_1$")
fig.colorbar(
    pc0,
    ax=ax,
    orientation="horizontal",
    extend="both",
    spacing="proportional",
    shrink=0.8,
)
plt.savefig("negproxy_R.pdf", format="pdf")
plt.close()
# * plot value functions
# lvls = [-0.2, -0.1, -0.001, 0., 0.001, 0.1, 0.2, 1.]
# lvls = [-100., -200., -50., -10., 0., 0.5, 1., 1.5, 2., 2.5, 3. ].sort()
# lvls = [-100., -200., -50., 0., 1., 2., 3.].sort()
# lvls=10
for pdx, data in enumerate(data_list):
    # data = data_list[ndx]
    fig, ax = plt.subplots(figsize=(colwide, 3 / 6 * colwide), constrained_layout=True)
    X_value = vibly.project_Q2S(data["Q_value"], grids, proj_opt=np.max)
    pc1 = vplot.value_function(
        ax,
        X_value,
        grids,
        XV=XV,
        # cont_lvls=lvls,
        mymap=mymap,
        mynorm=mynorm,
    )
    print("max viable: ", X_value[XV].max())
    print("min viable: ", X_value[XV].min())
    print("max unviable: ", X_value[~XV].max())
    print("min unviable: ", X_value[~XV].min())
    print("-----")
    fig.colorbar(
        pc1,
        ax=ax,
        orientation="horizontal",
        extend="both",
        spacing="proportional",
        shrink=0.8,
    )
    # fig.colorbar(pc1, ax=ax, orientation='horizontal', extend='min',
    #             spacing='proportional', ticks=lvls, shrink=0.8)
    ax.set_xlabel("$x_2$")
    ax.set_ylabel("$x_1$")
    plt.savefig("Tnegproxy_Vp_" + str(pdx) + ".pdf", format="pdf")
plt.close()
