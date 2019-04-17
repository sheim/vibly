from slip import *
import numpy as np
import matplotlib.pyplot as plt

p = {'mass':80.0, 'stiffness':8200.0, 'resting_length':1.0, 'gravity':9.81,
'aoa':-1.0*np.pi/8}
x0 = [0.0,1.5, # body position
    0.0,0.0, # velocities
    0.0,0.0] # foot position
p['total_energy'] = computeTotalEnergy(x0,p)
sol = step(x0,p)

# plt.plot(sol.t,sol.y[0])
plt.plot(sol.t,sol.y[1], color='green')
# plt.plot(sol.t,sol.y[0])
plt.plot(sol.y[0],sol.y[1], color='orange')
# plt.plot(sol.t,sol.y[0])
plt.plot(sol.t,sol.y[1], color='blue')
plt.show()

####

# from viability import computeP

# sol = computeP(1,2,p)