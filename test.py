from slip import *
import numpy as np
import matplotlib.pyplot as plt

p = {'mass':80, 'stiffness':8200, 'resting_length':1, 'gravity':9.81,'aoa':0}
x0 = [0,1.5,0,0]
sol = pMap(x0,p)

# plt.plot(sol[0].t,sol[0].y[0])
plt.plot(sol[0].t,sol[0].y[1], color='green')
# plt.plot(sol[1].t,sol[1].y[0])
plt.plot(sol[1].t,sol[1].y[1], color='orange')
# plt.plot(sol[2].t,sol[2].y[0])
plt.plot(sol[2].t,sol[2].y[1], color='blue')
plt.show()