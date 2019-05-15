import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.interpolate import interp1d


slow       = 10 # scales the animation relative to real time
interval   = 30 # ms between frames in the animation 
com_radius = 60 # size of the CoM dot
def animation_visualisation(sol, slow = slow, interval = interval, com_radius = com_radius, see = True, save = False, writer_name = 'html', filename = 'test.html', Ground = False):
	
	times    = sol.t 
	result   = sol.y
	t_events = sol.t_events
	x_com    = result[0]
	y_com    = result[1]
	foot_x   = result[4]
	foot_y   = result[5]

	### Timing
	## We determine the time-frames at which we want Animate to show data
	t_end         = times[-1]
	regular_times = np.arange(0, t_end , 0.001*interval/slow)
	duration      = len(regular_times)

	## Events: the leg is only visible between touchdown and liftoff
	# if touchdown does not occur, then it is visible from the start of the trial
	# if liftoff does not occur, then it is visible until the end of the trial
	if len(t_events[1]) > 0:
		touchdown = t_events[1][0]
	else: 
		touchdown = 0
	if len(t_events[3]) > 0	:
		liftoff   = t_events[3][0]
	else: 
		liftoff = t_end
		
	### Interpolation	
	## Animate shows frames at regular time intervals, whereas the simulation results are at irregular time intervals
	## Therefore we need to interpolate the data to have the values at the times of the frames given by regular_times

	# CoM
	x_interpolate = interp1d(times, x_com)
	y_interpolate = interp1d(times, y_com)
	regular_x_com = x_interpolate(regular_times)
	regular_y_com = y_interpolate(regular_times)

	# Leg
	foot_x_interpolate = interp1d(times, foot_x) 
	foot_y_interpolate = interp1d(times, foot_y) 
	regular_foot_x = foot_x_interpolate(regular_times)
	regular_foot_y = foot_y_interpolate(regular_times)
	leg_x_data = np.array([regular_x_com, regular_foot_x])
	leg_y_data = np.array([regular_y_com, regular_foot_y]) 

	### Visualisation
	fig, ax = plt.subplots()
	ax.set_xlim([np.min(x_com),np.max(x_com)])
	
	ax.set_aspect('equal')	
	if Ground:
		ground = result[-1]
		plt.plot(x_com, ground, color = 'k')
		ax.set_ylim([np.min(ground) - 0.1,np.max(y_com) + 0.1])
	else:	
		ax.axhline(y=0, color = 'k')
		ax.set_ylim([-0.1,np.max(y_com) + 0.1])
	ax.set_xlabel('Horizontal')
	ax.set_ylabel('Vertical')

	### Animation
	leg  = ax.plot([],[])[0] # the leg is a line
	CoM  = ax.scatter([], [], s = com_radius) # the CoM is a point

	def init():
		leg.set_data([], [])
		CoM.set_offsets([])
		return leg, CoM

	def animate(t):
		time = regular_times[t]
		if time > touchdown and time < liftoff - 0.001*interval/slow:
			leg.set_data(leg_x_data[:,t],leg_y_data[:,t])
		else:
			leg.set_data([], [])
		CoM.set_offsets(np.array([regular_x_com[t], regular_y_com[t]]).T)
		return leg, CoM

	anim = animation.FuncAnimation(fig, animate, init_func=init,
								   frames=duration, interval=interval, blit=True)
								   
	if save:							   
		Writer = animation.writers[writer_name]							   
		writer = Writer(fps=int(1000/interval), metadata=dict(artist='Me'), bitrate=1800)
		anim.save(filename, writer = writer)							   
	if see:							   
		# plt.show(block = False) # bizarrely with this you can't see the animation
		plt.show() # bizarrely without this you can't see the animation