import numpy as np
import matplotlib.pyplot as plt
import slippy.spaceship as sys
import pickle

from slippy.spaceship import p_map
import slippy.viability as vibly

p = {'n_states': 4,
     'wind': 0.2,
     'gravity': 0.2,
     'thrust': 0,
     'max_thrust': 0.5,
     'min_thrust': 0.0,
     'x0_upper_bound': 2,
     'x0_lower_bound': 0.0,
     'x1_upper_bound': 1,
     'x1_lower_bound': -1,
     'control_frequency': 10}
x0 = np.array([0.5, 0.0, 0, 0.0])

# poincare_map.p = p
# poincare_map.x = x0
# poincare_map.sa2xp = mapSA2xp_height_angle
# poincare_map.xp2s = map2s
p_map.p = p
p_map.x = x0  # do we still need these? I don't think so...
p_map.sa2xp = sys.sa2xp
p_map.xp2s = sys.xp2s

s_grid = (np.linspace(1.1*p['x0_lower_bound'], 1.1*p['x0_upper_bound'], 10),
          np.linspace(1.1*p['x1_lower_bound'], 1.1*p['x1_upper_bound'], 10),
          np.linspace(2.2*p['x1_lower_bound'], 2.2*p['x1_upper_bound'], 10),
          np.linspace(2.2*p['x1_lower_bound'], 2.2*p['x1_upper_bound'], 10),
          )

a_grid = np.linspace(p['min_thrust'], p['max_thrust'], 11)

grids = {'states': s_grid, 'actions': (a_grid,)}

Q_map, Q_F, Q_on_grid = vibly.compute_Q_map(grids, p_map, verbose=2,
                                            check_grid=True)

Q_V, S_V = vibly.compute_QV(Q_map, grids, ~Q_F, Q_on_grid=Q_on_grid)
S_M = vibly.project_Q2S(Q_V, grids, np.mean)
Q_M = vibly.map_S2Q(Q_map, S_M, Q_V, Q_on_grid)

print("size of viability kernel: " + str(np.mean(S_M)))
print("size of Q_M: " + str(np.mean(Q_M)))

################################################################################
# save data as pickle
################################################################################

# filename = 'Q_map' + '.pickle'
# data2save = {"grids": grids, "Q_map": Q_map, "Q_F": Q_F, "Q_V": Q_V,
#              "Q_M": Q_M, "S_M": S_M, "p": p, "x0": x0, "P_map": p_map}
# outfile = open(filename, 'wb')
# pickle.dump(data2save, outfile)
# outfile.close()
# to load this data, do:
# infile = open(filename, 'rb')
# data = pickle.load(infile)
# infile.close()


# plt.imshow(S_M)
# plt.show()