import numpy as np
import matplotlib.pyplot as plt
import timeit

# Compute Q using the 2D version

from slippy.viability import compute_Q_map

# use the same grid etc.

# differently from the 2D version, you must pass s_grid and a_grid as a
# vector of np.arrays (in 2D you pass the np.arrays directly).
Q_map, Q_F = compute_Q((s_grid,), (a_grid,), poincare_map)

grids = {'states':s_grid, 'actions':a_grid}
Q_V, S_V = compute_QV_2D(Q_map, grids)

# save file
data2save = {"s_grid": s_grid, "a_grid": a_grid, "Q_map": Q_map, "Q_F": Q_F,
    "Q_V": Q_V, "S_V": S_V}
np.savez('test_computeQ2D.npz', **data2save)

plt.imshow(Q_map)
plt.show()

# Do the same thing with the ND version
s_grid = np.linspace(0.1, 1, 5)
a_grid = np.linspace(-10/180*np.pi, 90/180*np.pi, 5)

Q_map, Q_F = compute_Q_map((s_grid,), (a_grid,), poincare_map)
grids = {'states':s_grid, 'actions':a_grid}
Q_V, S_V = compute_QV_2D(Q_map, grids)

# save file
data2save = {"s_grid": s_grid, "a_grid": a_grid, "Q_map": Q_map, "Q_F": Q_F,
    "Q_V": Q_V, "S_V": S_V}
np.savez('test_computeQ.npz', **data2save)