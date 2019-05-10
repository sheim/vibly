import numpy as np
import matplotlib.pyplot as plt

import slippy.viability as viability

data = np.load('test.npz')

s_grid = data['s_grid']
a_grid = data['a_grid']
Q_map = data['Q_map']
Q_F = data['Q_F']

grids = {'states':(s_grid,), 'actions':(a_grid,)}

print("grid resolution: " + str(s_grid.size) + " by " +str(a_grid.size))

Q_V, S_V = viability.compute_QV(Q_map, grids)

S_q = viability.project_Q2S(Q_V, grids, proj_opt = np.sum)

plt.plot(range(1, S_q.size+1), S_q)
plt.show()
plt.imshow(Q_map, origin = 'lower')
plt.show()

