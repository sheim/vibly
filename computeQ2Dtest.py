import numpy as np
import matplotlib.pyplot as plt

from slip import *

p = {'mass':80, 'stiffness':8200, 'resting_length':1, 'gravity':9.81,'aoa':0}
x0 = [0,1.5,0,0]
p['total_energy'] = computeTotalEnergy(x0,p)
pMap.p = p
pMap.x = x0
pMap.sa2xp = mapSA2xp_height_angle

s_table = np.linspace(0,1,10)
a_table = np.linspace(0,-np.pi/4,5)

from viability import computeQ_2D, computeQV_2D
Q_map, Q_F = computeQ_2D(s_table, a_table, pMap)
grids = {'states':s_table, 'actions':a_table}
Q_V, S_V = computeQV_2D(Q_map, grids)
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