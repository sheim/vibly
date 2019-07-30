import numpy as np
import matplotlib.pyplot as plt
from models import slip
import viability as vibly

p = {'mass': 80.0, 'stiffness': 8200.0, 'spring_resting_length': 1.0,
     'gravity': 9.81, 'angle_of_attack': 1/5*np.pi,
     'actuator_resting_length': 0}
x0 = np.array([0, 0.85, 5.5, 0, 0, 0, 0])
x0 = slip.reset_leg(x0, p)
p['x0'] = x0
p['total_energy'] = slip.compute_total_energy(x0, p)
p_map = slip.p_map
p_map.p = p
p_map.x = x0
p_map.sa2xp = slip.sa2xp
p_map.xp2s = slip.xp2s

s_grid = np.linspace(0.1, 1, 51)
s_grid = (s_grid[:-1],)
a_grid = (np.linspace(-10/180*np.pi, 70/180*np.pi, 41),)
grids = {'states': s_grid, 'actions': a_grid}
Q_map, Q_F, Q_on_grid = vibly.compute_Q_map(grids, p_map,
                                            check_grid=True)
Q_V, S_V = vibly.compute_QV(Q_map, grids, Q_on_grid=Q_on_grid)
S_M = vibly.project_Q2S(Q_V, grids, proj_opt=np.mean)
Q_M = vibly.map_S2Q(Q_map, S_M, s_grid, Q_V=Q_V)
################################################################################
# save data as pickle
################################################################################
# import pickle

# filename = '../data/dynamics/slip_map' + '.pickle'
# data2save = {"grids": grids, "Q_map": Q_map, "Q_F": Q_F, "Q_V": Q_V,
#              "Q_M": Q_M, "S_M": S_M, "p": p, "x0": x0}
# outfile = open(filename, 'wb')
# pickle.dump(data2save, outfile)
# outfile.close()
# to load this data, do:
# infile = open(filename, 'rb')
# data = pickle.load(infile)
# infile.close()

################################################################################
# basic visualization
################################################################################

plt.imshow(Q_map, origin='lower')
plt.show()
