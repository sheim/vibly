import numpy as np
import matplotlib.pyplot as plt
import time

from slippy.slip import *
import slippy.viability as vibly

data = np.load('slip_constr.npz')
Q_map = data['Q_map']
grids = {}
temp = data['grids']
grids['states'] = temp.item().get('states')
grids['actions'] = temp.item().get('actions')

Q_V, S_V = vibly.compute_QV(Q_map, grids)
S_M = vibly.project_Q2S(Q_V, grids, np.sum)
Q_M = vibly.map_S2Q(Q_map, S_M, Q_V)
plt.plot(S_M)
# plt.show()
plt.imshow(Q_M, origin='lower')
plt.show()
# save file
timestr = time.strftime("%Y_%m_%H_%M_%S")
print("Finished at " + timestr)
print(timestr)
# data2save = {"grids": grids, "Q_map": Q_map, "Q_F": Q_F, "p" : p,
#             "P_map" : poincare_map}
# np.savez('slip_constr_', **data2save)

# plt.imshow(Q_V, origin = 'lower')
# plt.show()