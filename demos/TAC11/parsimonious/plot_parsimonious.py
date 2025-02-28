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

# import seaborn as sns
# sns.set_theme(style="darkgrid")

# * set up helper functions

# * Load data

filename = "./parsimonious0.pickle"
# * Load and unpack

infile = open(filename, "rb")
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
alpha = 0.0
# mynorm = colors.TwoSlopeNorm(vmin=0., vcenter=0., vmax=2.5)
mynorm = colors.TwoSlopeNorm(vmin=-0.5, vcenter=alpha, vmax=2.5)
mymap = vplot.get_vmap(2)


extent = [
    grids["states"][1][0],
    grids["states"][1][-1],
    grids["states"][0][0],
    grids["states"][0][-1],
]


fig, axs = plt.subplots(1, figsize=(6, 5))

X_value = vibly.project_Q2S(data["Q_value"], grids, proj_opt=np.max)
# lvls = [0., 0.15]
print(X_value.max())
pc1 = vplot.value_function(
    axs, X_value, grids, XV=XV, cont_lvls=None, mymap=mymap, mynorm=mynorm
)

# fig.colorbar(pc1, ax=axs, orientation='horizontal', extend='both')
fig.colorbar(pc1, ax=axs, orientation="horizontal", extend="max", ticks=[0.0, 1, 2.0])
# * Reward function
# pc0 = vplot.reward_function(axs, RX_value, grids, XV=XV,
#                             mymap="PRGn", mynorm=mynorm)  #,
#                             viability_threshold=0)
# fig.colorbar(pc0, ax=axs)

# for idx in range(num_plots-1):
#     plt.setp(axs[idx].get_xticklabels(), visible=False)

plt.savefig(filename + ".pdf", format="pdf")
plt.show()
plt.close()
