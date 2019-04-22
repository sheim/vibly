import numpy as np
import matplotlib.pyplot as plt
import timeit

from slip import *


p = {'mass':80.0, 'stiffness':8200.0, 'resting_length':1.0, 'gravity':9.81,
'angle_of_attack':1/5*np.pi}
x0 = [0, 0.85, 5.5, 0, 0, 0]
x0 = reset_leg(x0, p)
p['total_energy'] = compute_total_energy(x0, p)
poincare_map.p = p
poincare_map.x = x0
poincare_map.sa2xp = mapSA2xp_height_angle

s_grid = np.linspace(0.4, 1, 4)
a_grid = np.linspace(0, -np.pi/4, 5)

from viability import compute_Q_2D, compute_QV_2D
Q_map, Q_F = compute_Q_2D(s_grid, a_grid, poincare_map)
grids = {'states':s_grid, 'actions':a_grid}
Q_V, S_V = compute_QV_2D(Q_map, grids)

# save file

np.savez('test_computeQ2D.npz',s_grid, a_grid, Q_map, Q_F, Q_V, S_V)

# plt.plot(sol.t,sol.y[0])
# plt.plot(sol.t,sol.y[1], color='green')
# # plt.plot(sol.t,sol.y[0])
# plt.plot(sol.t,sol.y[1], color='orange')
# # plt.plot(sol.t,sol.y[0])
# plt.plot(sol.t,sol.y[1], color='blue')
# plt.show()

####

# from viability import computeP

# sol = computeP(1,2,p)