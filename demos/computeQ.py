import numpy as np
import matplotlib.pyplot as plt
from slippy.slip import *
import slippy.viability as vibly

p = {'mass':80.0, 'stiffness':8200.0, 'resting_length':1.0, 'gravity':9.81,
'angle_of_attack':1/5*np.pi}
x0 = np.array([0, 0.85, 5.5, 0, 0, 0, 0])
x0 = reset_leg(x0, p)
p['x0'] = x0
p['total_energy'] = compute_total_energy(x0, p)
poincare_map.p = p
poincare_map.x = x0
poincare_map.sa2xp = mapSA2xp_height_angle
poincare_map.xp2s = map2s

s_grid = np.linspace(0.1, 1, 100)
s_grid = s_grid[:-1]
a_grid = np.linspace(-10/180*np.pi, 90/180*np.pi, 91)
grids = {'states': (s_grid,), 'actions': (a_grid,)}
Q_map, Q_F, Q_on_grid = vibly.compute_Q_map(grids, poincare_map,
                                            check_grid=True)
Q_V, S_V = vibly.compute_QV(Q_map, grids, Q_on_grid=Q_on_grid)
################################################################################
# save data as pickle
################################################################################
import pickle

filename = 'slip_map' + '.pickle'
data2save = {"grids": grids, "Q_map": Q_map, "Q_F": Q_F, "Q_V": Q_V,
            "p": p, "x0": x0, "P_map": poincare_map}
outfile = open(filename, 'wb')
pickle.dump(data2save, outfile)
outfile.close()
# to load this data, do:
# infile = open(filename, 'rb')
# data = pickle.load(infile)
# infile.close()

################################################################################
# basic visualization
################################################################################

plt.imshow(Q_map, origin='lower')
plt.show()