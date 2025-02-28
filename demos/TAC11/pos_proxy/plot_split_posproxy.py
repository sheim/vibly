import numpy as np

import matplotlib as cm

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import viability as vibly  # algorithms for brute-force viability
import pickle

import plotting.value_plotters as vplot

cm.use("TkAgg")
cm.rcParams["pdf.fonttype"] = 42
cm.rcParams["ps.fonttype"] = 42

colwide = 3.5
pagewide = 7.16
# import seaborn as sns
# sns.set_theme(style="darkgrid")

# ! p* at 111
# * Load data
alpha = 0.0

filenames = [  # "posproxy0.pickle",
    "posproxy1.pickle",
    "posproxy111.pickle",
]

# filenames = ["posproxy0.pickle",
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

mynorm = colors.TwoSlopeNorm(vmin=-100, vcenter=0.0, vmax=2.5)
# mynorm = colors.CenteredNorm()
mymap = vplot.get_vmap(0)
# mymap = np.vstack((plt.cm.Blues_r(np.linspace(0.2, 0.7, 127)),
#                    plt.cm.Oranges(np.linspace(0.3, 0.8, 128))))
# mymap = colors.LinearSegmentedColormap.from_list('my_colormap', mymap)
extent = [
    grids["states"][1][0],
    grids["states"][1][-1],
    grids["states"][0][0],
    grids["states"][0][-1],
]

# these_values = [0, 3, 5, -3, -2, -1]
# these_values = range(len(data_list))
# num_plots = len(data_list)+1
# fig, axs = plt.subplots(num_plots, 1, figsize = (colwide, 16/10*colwide))
figR, axR = plt.subplots(figsize=(colwide, 3 / 6 * colwide), constrained_layout=True)
# * plot reward function
R_value = data_list[0]["R_value"]
RX_value = vibly.project_Q2S(R_value, grids, proj_opt=np.mean)


# colors.LinearSegmentedColormap.from_list('second_map', mymap)
# newcolors = [(0., 0., 0., 1), (1., 1., 1., 1.), mymap(mynorm(1.0))]
rmap = colors.ListedColormap(["red", "white", "darkgrey"])
# rmap = colors.ListedColormap(newcolors)
pcR = vplot.reward_function(axR, RX_value, grids, XV=XV, mymap=rmap)
axR.set_xlabel("$x_2$")
axR.set_ylabel("$x_1$")
figR.colorbar(
    pcR,
    ax=axR,
    orientation="horizontal",
    extend="both",
    spacing="proportional",
    shrink=1,
)
plt.savefig("posproxy_R.pdf", format="pdf")
plt.close()
# * plot value functions
lvls = [-100.0, -50.0, alpha, 1.0, 2.0]
for pdx, data in enumerate(data_list):
    fig, ax = plt.subplots(figsize=(colwide, 3 / 6 * colwide), constrained_layout=True)
    X_value = vibly.project_Q2S(data["Q_value"], grids, proj_opt=np.max)
    pc1 = vplot.value_function(
        ax, X_value, grids, XV=XV, cont_lvls=lvls, mymap=mymap, mynorm=mynorm
    )
    print("max: ", X_value.max())
    print("min viable: ", X_value[XV].min())
    print("max unviable: ", X_value[~XV].max())
    fig.colorbar(
        pc1,
        ax=ax,
        orientation="horizontal",
        extend="both",
        spacing="proportional",
        ticks=lvls,
        shrink=0.8,
    )
    ax.set_xlabel("$x_2$")
    ax.set_ylabel("$x_1$")
    plt.savefig("Tposproxy_Vp_" + str(pdx) + ".pdf", format="pdf")
# plt.show()
plt.close()
