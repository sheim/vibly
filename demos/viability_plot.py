import numpy as np
import matplotlib.pyplot as plt
from plotting.visualise_viability import visualise

data               = np.load('../data/low_res_slip.npz')
initial_conditions = np.array([0, 0.85, 5.5, 0, 0, 0]) 

visualise(data)
## The arguments you can pass to this function are:
# state_colormap     = 'viridis'  : other sequential colormaps: 'plasma', 'gray'
# change_colormap    = 'coolwarm' : other diverging colormaps: 'bwr', 'RdBu'

# initial_conditions = []: if you provide the initial conditions, then the function can plot the unfeasible set
# example:
visualise(data, initial_conditions = initial_conditions)

# include_end_state = False : if True, you see not only the change in set, but also the end state
# example:
visualise(data, initial_conditions = initial_conditions, include_end_state = True)

plt.show()