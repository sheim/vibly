import numpy as np

import matplotlib as cm

cm.use("TkAgg")
cm.rcParams["pdf.fonttype"] = 42
cm.rcParams["ps.fonttype"] = 42
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
extent = [
    grids["states"][1][0],
    grids["states"][1][-1],
    grids["states"][0][0],
    grids["states"][0][-1],
]

# these_values = [0, 3, 5, -3, -2, -1]
# these_values = range(len(data_list))
num_plots = len(data_list) + 1
fig, axs = plt.subplots(num_plots, 1, figsize=(colwide, 16 / 10 * colwide))

# * plot reward function
R_value = data_list[0]["R_value"]
RX_value = vibly.project_Q2S(R_value, grids, proj_opt=np.mean)
rmap = colors.ListedColormap(["tab:red", "white", "tab:orange"])
pc0 = vplot.reward_function(axs[0], RX_value, grids, XV=XV, mymap=rmap)
XV0 = list()

# * plot value functions
lvls = [-100.0, -50.0, alpha, 1.0, 2.0]
for pdx, data in enumerate(data_list):
    # data = data_list[ndx]
    X_value = vibly.project_Q2S(data["Q_value"], grids, proj_opt=np.max)
    pc1 = vplot.value_function(
        axs[pdx + 1], X_value, grids, XV=XV, cont_lvls=lvls, mymap=mymap, mynorm=mynorm
    )
    print("max: ", X_value.max())
    print("min viable: ", X_value[XV].min())
    print("max unviable: ", X_value[~XV].max())
fig.colorbar(
    pc1,
    ax=axs,
    orientation="horizontal",
    extend="both",
    spacing="proportional",
    ticks=lvls,
    shrink=0.8,
)
axs[-1].set_xlabel("Velocity")

for pdx in range(len(data_list)):
    axs[pdx].set_xticks([])

# vplot.set_size(5,5,axs[-1])
plt.savefig("pos_proxy.pdf", format="pdf")
plt.show()
plt.close()
