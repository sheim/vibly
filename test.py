from slip import *
import numpy as np
import matplotlib.pyplot as plt

p = {'mass':80, 'stiffness':8200, 'resting_length':1, 'gravity':9.81,'aoa':0}
x0 = [0,1.5,0,0]
p['total_energy'] = computeTotalEnergy(x0,p)
sol = pMap(x0,p)

# plt.plot(sol.t,sol.y[0])
plt.plot(sol.t,sol.y[1], color='green')
# plt.plot(sol.t,sol.y[0])
plt.plot(sol.t,sol.y[1], color='orange')
# plt.plot(sol.t,sol.y[0])
plt.plot(sol.t,sol.y[1], color='blue')
plt.show()