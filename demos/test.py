from slippy.slip import *
import numpy as np
import matplotlib.pyplot as plt

p = {'mass':80.0, 'stiffness':8200.0, 'resting_length':1.0, 'gravity':9.81,
    'angle_of_attack':1/5*np.pi}
x0 = np.array([10,		# x position: forwards
				0.85,	# y position: upwards
				5.5,	# x velocity
				0,		# y velocity
				0,		# foot position x
				0])		# foot position y
x0 = reset_leg(x0, p)
p['total_energy'] = compute_total_energy(x0, p)
sol = step(x0, p)

# plt.plot(sol.y[0], sol.y[1], color='orange')
# plt.show()

### The attributes of sol are:
## sol.t 		: series of time-points at which the solution was calculated
## sol.y 		: simulation results, size 6 x times
## sol.t_events	: list of the times of 7 events: 
# - fall during flight
# - touchdown
# - fall during stance
# - lift-off
# - reversal during stance
# - apex during flight 
# - fall during flight
### If the event did not occur than the array is empty.

def full_visualisation(sol):
	times    = sol.t 
	result   = sol.y
	t_events = sol.t_events

	# If the trial was not a failure:	
	if len(t_events[1]) > 0 and len(t_events[3]) > 0 and len(t_events[5]) > 0:

		events = [t_events[1][0],t_events[3][0],t_events[5][0]]
		colors = ['k','b','g']
		labels = ['touchdown','liftoff','apex']

		## Foot trajectory
		foot_x = result[4]
		foot_y = result[5]
		plt.figure()
		plt.subplot(221)
		plt.plot(times, foot_x)
		plt.subplot(223)
		plt.plot(times, foot_y)
		plt.subplot(122)
		plt.plot(foot_x, foot_y)
		## Indicate the events
		for e in range(3):
			for i in [3,1]:
				plt.subplot(2,2,i)
				plt.axvline(x = events[e], color = colors[e], label = labels[e])
			plt.subplot(122)
			index = np.argmax(times >= events[e])
			plt.scatter(foot_x[index], foot_y[index], color = colors[e])
		## Legends and labels	
		plt.subplot(221)	
		plt.legend(loc = 2)	
		plt.xticks([])
		plt.ylabel('Forwards position')
		plt.subplot(223)	
		plt.ylabel('Upwards position')
		plt.xlabel('Time')
		plt.subplot(122)
		plt.xlabel('Forwards position')
		plt.ylabel('Upwards position')
		plt.title('Foot trajectory')
		plt.show()

		## CoM position 
		plt.figure()
		for i in range(2):
			for j in range(2):
				plt.subplot(2,3,1+i+3*j)
				plt.plot(times, result[i+2*j])
		plt.subplot(133)
		plt.plot(result[0],result[1])		
		## Indicate the events
		for e in range(3):
			for i in range(2):
				for j in range(2):
					plt.subplot(2,3,1+i+3*j)
					plt.axvline(x = events[e], color = colors[e], label = labels[e])
			plt.subplot(133)
			index = np.argmax(times >= events[e])
			plt.scatter(result[0,index], result[1,index], color = colors[e])
		## Legends and labels	
		plt.subplot(231)
		plt.xticks([])
		plt.ylabel('Forwards position')
		plt.subplot(232)
		plt.xticks([])
		plt.ylabel('Upwards position')
		plt.subplot(234)
		plt.xlabel('Time')
		plt.ylabel('Forwards speed')
		plt.subplot(235)
		plt.xlabel('Time')
		plt.ylabel('Upwards speed')
		plt.subplot(133)
		plt.xlabel('Forwards position')
		plt.ylabel('Upwards position')
		plt.title('CoM trajectory')
		plt.show()		
	else:
		print('The trial was a failure')
full_visualisation(sol)		