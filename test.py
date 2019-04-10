from slip import *
import numpy as np
import matplotlib.pyplot as plt

p = {'mass':1, 'stiffness':1, 'resting_length':1, 'gravity':1,'aoa':0}
x0 = [0,1.5,0,0]
sol = pMap(x0,p)

plt.plot(sol.t,sol.y[0])
plt.plot(sol.t,sol.y[1])
plt.show()