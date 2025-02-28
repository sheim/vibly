import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import rcParams
# from scipy.signal import savgol_filter
import pickle

colwide = 3.5
pagewide = 7.16

# ! p* at 136
# * Load data


filename = "COMBO_gamma_pstar.pickle"

infile = open(filename, "rb")
data = pickle.load(infile)
infile.close()
gammas = data["gamma_list"]
pstars = data["penalty_list"]

fig, ax = plt.subplots(figsize=(colwide, 3 / 6 * colwide), constrained_layout=True)

ax.semilogy(gammas, pstars)
ax.set_xlim(1, 0.2)
ax.set_xlabel("$\gamma$")
ax.set_ylabel("$p^*$")
plt.savefig("gamma_pstar" + ".pdf", format="pdf")
# plt.show()
plt.close()
