import numpy as np
import matplotlib.pyplot as plt

state_colormap     = 'viridis'  # other sequential colormaps: 'plasma', 'gray'
change_colormap    = 'coolwarm' # other diverging colormaps: 'bwr', 'RdBu'

def visualise(data, initial_conditions = [], include_end_state = False,  state_colormap = state_colormap, change_colormap = change_colormap):

	s_grid	= data['s_grid']	# shape 50
	a_grid	= data['a_grid']	# shape 50 (in radian? angles?)
	Q_map	= data['Q_map']		# shape 50 x 50
	Q_V		= data['Q_V']		# shape 50 x 50: binary non-failure

	s_min = s_grid[0]
	s_max = s_grid[-1]
	a_min = a_grid[0]
	a_max = a_grid[-1]

	initial_state  = np.repeat(np.array([s_grid]), len(a_grid), axis = 0).T
	end_state      = np.zeros((len(s_grid), len(a_grid)))
	end_state[Q_V == 1] = Q_map[Q_V == 1]
	end_state[Q_V != 1] = np.nan
	change_state   = end_state - initial_state

	plt.figure()
	
	if include_end_state:	
		plt.subplot(211)
		plt.imshow(end_state, origin = 'lower', extent = [a_min, a_max, s_min, s_max], cmap = state_colormap)
		plt.colorbar()
		plt.subplot(212)
	min_change = np.min(change_state[Q_V == 1])
	max_change = np.max(change_state[Q_V == 1])
	change     = max(abs(min_change), abs(max_change))
	plt.imshow(change_state, origin = 'lower', extent = [a_min, a_max, s_min, s_max], vmin = - change, vmax = change, cmap = change_colormap)
	plt.colorbar()
	plt.show(block=False)
	
	if len(initial_conditions) > 0:
		## To determine the lower bound on state
		s_lower = np.cos(a_grid)/(initial_conditions[1]+initial_conditions[2]**2/(2*9.81)) # height at touchdown / (height + speed^2/2g)
		plt.plot(a_grid, s_lower, color = 'k')
		plt.fill_between(a_grid, s_lower, color = 'grey')	
		if include_end_state:		
			plt.subplot(2,1,1)
			plt.plot(a_grid, s_lower, color = 'k')
			plt.fill_between(a_grid, s_lower, color = 'grey')	
		
	## Labels
	if include_end_state:	
		plt.subplot(2,1,1)
		plt.xticks([])
		plt.ylabel('State')
		plt.title('End state')
		plt.subplot(2,1,2)
	plt.xlabel('Angle of attack (rad)')
	plt.ylabel('State')
	plt.title('Change in state')
	plt.show(block=False)
