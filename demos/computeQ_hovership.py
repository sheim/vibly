import numpy as np
import matplotlib.pyplot as plt
import models.hovership as sys
from models.hovership import p_map
import viability as vibly  # algorithms for brute-force viability

# * here we choose the parameters to use
# * we also put in a place-holder action (thrust)
p = {'n_states': 1,
     'base_gravity': 0.1,
     'gravity': 0.5,
     'thrust': 0,
     'max_thrust': 0.5,
     'ceiling': 2,
     'control_frequency': 1  # hertz
     }

# * Choose an initial condition
x0 = np.array([0.5, 0])

# * For convenience, helper functions, a default parameter dict and initial
# * condition are attached to the transition map.
p_map.p = p
p_map.x = x0
p_map.sa2xp = sys.sa2xp
p_map.xp2s = sys.xp2s

# * determine the bounds and resolution of your grids
# * note, the s_grid is a tuple of grids, such that each dimension can have
# * different resolution, and we do not need to initialize the entire array
s_grid = (np.linspace(-0.0, p['ceiling'], 101),)
# * same thing for the actions
a_grid = (np.linspace(0.0, 0.5, 51),)

# * for convenience, both grids are placed in a dictionary
grids = {'states': s_grid, 'actions': a_grid}

# * compute_Q_map computes a gridded transition map, `Q_map`, which is used as
# * a look-up table for computing viable sets
# * Q_F is a grid marking all failing state-action pairs
# * Q_on_grid is a helper grid, which marks if a state has not moved
# * this is used to catch corner cases, and is not important for most systems
# * setting `check_grid` to False will omit Q_on_grid
Q_map, Q_F, Q_on_grid = vibly.compute_Q_map(grids, p_map, check_grid=True)

# * compute_QV computes the viable set and viability kernel
Q_V, S_V = vibly.compute_QV(Q_map, grids, ~Q_F, Q_on_grid=Q_on_grid)

# * project_Q2S takens a projection of the viable set onto state-space
# * for the computing the measure, you can use either `np.mean` or `np.sum`
# * as the projection operator
S_M = vibly.project_Q2S(Q_V, grids, proj_opt=np.mean)
# * map_S2Q maps the measure back into state-action space using the gridded
# * transition map
Q_M = vibly.map_S2Q(Q_map, S_M, s_grid, Q_V=Q_V)

###############################################################################
# * save data as pickle
###############################################################################
import pickle
filename = '../data/hover_map.pickle'
data2save = {"grids": grids, "Q_map": Q_map, "Q_F": Q_F, "Q_V": Q_V,
             "Q_M": Q_M, "S_M": S_M, "p": p, "x0": x0}
outfile = open(filename, 'wb')
pickle.dump(data2save, outfile)
outfile.close()
# to load this data, do:
# infile = open(filename, 'rb')
# data = pickle.load(infile)
# infile.close()

plt.imshow(Q_M, origin='lower')  # visualize the Q-safety measure
plt.show()
# plt.imshow(Q_V) # visualize the viable set
# plt.show()