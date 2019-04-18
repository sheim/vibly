from slip import *
import numpy as np
import matplotlib.pyplot as plt

p = {'mass':80.0, 'stiffness':8200.0, 'resting_length':1.0, 'gravity':9.81,
'aoa':1/5*np.pi}
# x0 = [-np.sin(p['aoa'])*p['resting_length'],1.1*p['resting_length'], # body position
#     5*p['resting_length'],0.0, # velocities
#     0.0,0.0] # foot position
x0 = [0, 0.85, 5.5, 0, 0, 0]
x0 = resetLeg(x0,p)
p['total_energy'] = computeTotalEnergy(x0,p)
sol = step(x0,p)

# plt.plot(sol.t,sol.y[0])
# plt.plot(sol.t,sol.y[1], color='green')
# plt.plot(sol.t,sol.y[0])
# plt.plot(sol.y[0],sol.y[1], color='orange')
# plt.show()