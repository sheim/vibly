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
import matplotlib.collections as collections

import plotting.value_plotters as vplot

# import seaborn as sns
# sns.set_theme(style="darkgrid")

# * set up helper functions
alpha = 0.378
# ! p* at 282 for wacky
# ! p* at 22 for wacker
# * Load data

filenames = [  # wacky0.pickle",
    "wacky1.pickle",
    "wacky100.pickle",
    "wacky1000.pickle",
    "wacky10000.pickle",
    "wacky100000.pickle",
    "wacky1000000.pickle",
]

data_list = list()

for filename in filenames:
    infile = open(filename, "rb")
    data_list.append(pickle.load(infile))
    infile.close()

# * unpack invariants
# Q_value = data["Q_value"]
R_value = data_list[0]["R_value"]
grids = data_list[0]["grids"]
# Q_map = data["Q_map"]
S_M = data_list[0]["S_M"]

# * Ground Truth XV
XV = S_M > 0.0

mynorm = colors.TwoSlopeNorm(vmin=-100, vcenter=0.0, vmax=1)
# mynorm = colors.CenteredNorm()
mymap = vplot.get_vmap(0)


extent = [
    grids["states"][1][0],
    grids["states"][1][-1],
    grids["states"][0][0],
    grids["states"][0][-1],
]

# pick out value of x[0] to slice through
x0 = vibly.digitize_s([12.0, 0], grids["states"], to_bin=False)
x1_slice = grids["states"][1]
XV_slice = XV[x0[0], :]

these_values = [0, 3, 5]  # to plot entire value functions
num_plots = len(these_values) + 1
fig, axs = plt.subplots(num_plots, 1)

# * color settings for slice
p_cmap = plt.get_cmap("Dark2")
col_norm = 20.0
col_offset = 1.0
v_min = 0.0
v_max = 0.0

found = False
for pdx, ndx in enumerate(these_values):
    data = data_list[ndx]
    RX_value = vibly.project_Q2S(data["R_value"], grids, proj_opt=np.max)
    X_value = vibly.project_Q2S(data["Q_value"], grids, proj_opt=np.max)
    if v_max < X_value.max():
        v_max = X_value.max()
    if v_min > X_value.min():
        v_min = X_value.min()
    print("Min viable value: ", X_value[XV].min())
    print("Max unviable value: ", X_value[~XV].max())
    if ~found:
        if X_value[XV].min() > X_value[~XV].max():
            found = True
            print("p* crossed at ", RX_value.min())
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

print("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*")
print("min: ", v_min)
print("max: ", v_max)

mynorm = colors.TwoSlopeNorm(vmin=v_min, vcenter=0.0, vmax=v_max)
# mynorm = colors.TwoSlopeNorm(vmin=-100, vcenter=0., vmax=4)
# * plot slice
for idx, data in enumerate(data_list):
    # * pick out slice
    X_value = vibly.project_Q2S(data["Q_value"], grids, proj_opt=np.max)
    # # XV = X_value >  -1.7742
    # # fix bug:
    # # xtop = vibly.digitize_s([1, 0], grids['states'], to_bin=False)
    # # XV[xtop[0]-1, :] = False
    # if X_value.min() < v_min:
    #     v_min = X_value.min()
    # if X_value.max() > v_max:
    #     v_max = X_value.max()
    col = p_cmap(col_offset - np.abs(idx / col_norm))
    axs[0].plot(x1_slice, X_value[x0[0], :], color="k")
axs[0].set_yscale("log")
# axs[1].set_xlim((-4., 4.))
collection = collections.BrokenBarHCollection.span_where(
    x1_slice, ymin=v_min, ymax=v_max, where=~XV_slice, facecolor="red", alpha=0.2
)
axs[0].add_collection(collection)
axs[0].grid(True)

# * plot value functions
lvls = [-100.0, -75.0, -50.0, -25.0, alpha, 1.0]
for pdx, idx in enumerate(these_values):
    data = data_list[idx]
    X_value = vibly.project_Q2S(data["Q_value"], grids, proj_opt=np.max)
    pc3 = vplot.value_function(
        axs[pdx + 1], X_value, grids, XV=XV, mymap=mymap, mynorm=mynorm
    )

for idx in range(num_plots - 1):
    plt.setp(axs[idx].get_xticklabels(), visible=False)

fig.colorbar(
    pc1,
    ax=axs,
    orientation="horizontal",
    extend="both",
    spacing="proportional",
    ticks=lvls,
)
# plt.savefig("Test up to 1e9"+'.pdf', format='pdf')
plt.show()
plt.close()
