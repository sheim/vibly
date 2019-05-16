import numpy as np
import matplotlib.pyplot as plt
import plotting.visualise_viability

data               = np.load('../data/low_res_slip.npz')
initial_conditions = np.array([0, 0.85, 5.5, 0, 0, 0]) 

Q_map	= data['Q_map']
Q_V		= data['Q_V']
s_grid  = data['s_grid']
a_grid	= data['a_grid']

plotting.visualise_viability.visualise_no_dict(s_grid, a_grid, Q_map, Q_V) # previously: visualise(data)
## The arguments you can pass to this function are:
# state_colormap     = 'viridis'  : other sequential colormaps: 'plasma', 'gray'
# change_colormap    = 'coolwarm' : other diverging colormaps: 'bwr', 'RdBu'

# initial_conditions = []: if you provide the initial conditions, then the function can plot the unfeasible set
# example:
plotting.visualise_viability.visualise_no_dict(s_grid, a_grid, Q_map, Q_V, initial_conditions = initial_conditions)

# include_end_state = False : if True, you see not only the change in set, but also the end state
# example:
plotting.visualise_viability.visualise_no_dict(s_grid, a_grid, Q_map, Q_V, initial_conditions = initial_conditions, include_end_state = True)

plt.show()