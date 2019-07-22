import numpy as np
import matplotlib.pyplot as plt

# colors corresponding to initial flight, stance, second flight
colors = ['k', 'b', 'g']

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

def com_visualisation(sol, leg_visibility=0.5, colors=colors, size=100, Ground=False):
	'''
	 This function plots failure events in red.
	 '''

	times    = sol.t
	result   = sol.y
	t_events = sol.t_events
	x_com    = result[0]
	y_com    = result[1]

	# plt.figure()

	### Initial position
	plt.scatter(x_com[0], y_com[0], color = colors[0], s = size)
	foot_x = result[4,0]
	foot_y = result[5,0]
	plt.plot([foot_x,x_com[0]],[foot_y,y_com[0]], color = colors[0], 
			alpha = leg_visibility)

	### First flight phase
	if len(t_events[1]) == 0: # no touch-down
		## Time of failure
		if len(t_events[0]) == 0: # no fall during initial flight
			print('No touch-down but no fall during flight')
		else:
			failure = t_events[0][0]
			fail_index = np.argmax(times > failure)
			plt.plot(x_com[:fail_index],y_com[:fail_index], color = colors[0])
			plt.scatter(x_com[fail_index -1],y_com[fail_index-1],
					color = 'r', s = size)
	else:
		touchdown = t_events[1][0]
		index     = np.argmax(times > touchdown)
		foot_x    = result[4,index]
		plt.plot(x_com[:index],y_com[:index], color = colors[0])
		plt.scatter(x_com[index-1],y_com[index-1], color = colors[1], s = size)
		plt.plot([foot_x,x_com[index-1]],[0,y_com[index-1]], color = colors[1],
				alpha = leg_visibility)

		### Stance phase
		if len(t_events[3]) == 0: # no lift-off
			## Time of failure
			failure = False
			if len(t_events[2]) == 0: # no fall during initial flight
				if len(t_events[4]) == 0: # no reversal during initial flight
					print('No lift-off but no failure during stance')
				else:
					failure = t_events[4][0] # time of reversal
			else:
				failure = t_events[2][0] # time of fall
			if failure:
				fail_index = np.argmax(times > failure)
				plt.plot(x_com[index:fail_index],y_com[index:fail_index],
						color = colors[1])
				plt.scatter(x_com[fail_index -1],y_com[fail_index-1],
						color = 'r', s = size)
		else:
			liftoff = t_events[3][0]
			lift_index = np.argmax(times > liftoff)
			plt.plot(x_com[index-1:lift_index],y_com[index-1:lift_index],
					color = colors[1])
			plt.scatter(x_com[lift_index-1],y_com[lift_index-1],
					color = colors[2], s = size)
			plt.plot([foot_x,x_com[lift_index-1]],[0,y_com[lift_index-1]],
					color = colors[2], alpha = leg_visibility)

			### Flight phase
			if len(t_events[5]) == 0: # no apex
				## Time of failure
				if len(t_events[6]) == 0: # no fall
					print('No apex but no fall during flight')
				else:
					failure = t_events[6][0]
					fail_index = np.argmax(times > failure)
					plt.plot(x_com[lift_index-1:fail_index],y_com[lift_index-1:fail_index], color = colors[2])
					plt.scatter(x_com[fail_index -1],y_com[fail_index-1], color = 'r', s = size)
			else:
				apex = t_events[5][0]
				if times[-1] > apex:
					apex_index = np.argmax(times > apex)
				else:
					apex_index = len(times)
				plt.plot(x_com[lift_index-1:apex_index],
						y_com[lift_index-1:apex_index], color = colors[2])
				plt.scatter(x_com[apex_index-1],y_com[apex_index-1],
						color = colors[0], s = size)
				plt.plot([result[4,apex_index-1],x_com[apex_index-1]],
						[result[5,apex_index-1],y_com[apex_index-1]],
						color = colors[0], alpha = leg_visibility)
						
	if Ground:
		ground = result[-1]
		plt.plot(x_com, ground, color = 'k')
	else:					
		plt.axhline(y=0, color = 'k')
	plt.xlabel('Horizontal position')
	plt.ylabel('Vertical position')


def full_visualisation(sol, colors = colors, foot = False):
	'''
	 This function only plots if there was no failure in the trial
	 '''
	times    = sol.t
	result   = sol.y
	t_events = sol.t_events
	labels = ['touchdown','liftoff','apex']
	# If the trial was not a failure:
	if len(t_events[1]) > 0 and len(t_events[3]) > 0 and len(t_events[5]) > 0:

		events 	= [t_events[1][0],t_events[3][0],t_events[5][0]]
		indices = [0]
		for e in range(3):
			indices.append(np.argmax(times >= events[e]))

		if foot:
			## Foot trajectory
			foot_x = result[4]
			foot_y = result[5]
			plt.figure()
			for e in range(3):
				plt.subplot(221)
				plt.plot(times[indices[e]:indices[e+1]], foot_x[indices[e]:indices[e+1]], color = colors[e])
				plt.subplot(223)
				plt.plot(times[indices[e]:indices[e+1]], foot_y[indices[e]:indices[e+1]], color = colors[e])
				plt.subplot(122)
				plt.plot(foot_x[indices[e]:indices[e+1]], foot_y[indices[e]:indices[e+1]], color = colors[e])
				plt.scatter(foot_x[indices[e]], foot_y[indices[e]], color = colors[e])
				## Indicate the events
				for i in [3,1]:
					plt.subplot(2,2,i)
					plt.axvline(x = events[e], color = colors[e], label = labels[e])
			## Legends and labels
			plt.subplot(221)
			plt.xticks([])
			plt.ylabel('Horizontal position')
			plt.subplot(223)
			plt.ylabel('Vertical position')
			plt.xlabel('Time')
			plt.subplot(122)
			plt.xlabel('Horizontal position')
			plt.ylabel('Vertical position')
			plt.title('Foot trajectory')

		## CoM position
		plt.figure()
		for e in range(3):
			for i in range(2):
				for j in range(2):
					plt.subplot(2,3,1+i+3*j)
					plt.plot(times[indices[e]:indices[e+1]+1],
							result[i+2*j,indices[e]:indices[e+1]+1],
							color = colors[e])
			plt.subplot(133)
			plt.plot(result[0,indices[e]:indices[e+1]+1],
					result[1,indices[e]:indices[e+1]+1], color = colors[e])
			## Indicate the events
			for i in range(2):
				for j in range(2):
					plt.subplot(2,3,1+i+3*j)
					plt.axvline(x = events[e], color = colors[e],
							label = labels[e])
			plt.subplot(133)
			index = np.argmax(times >= events[e])
			plt.scatter(result[0,index], result[1,index], color = colors[e])
		## Legends and labels
		plt.subplot(231)
		plt.legend(loc = 2)
		plt.xticks([])
		plt.ylabel('Horizontal position')
		plt.subplot(232)
		plt.xticks([])
		plt.ylabel('Vertical position')
		plt.subplot(234)
		plt.xlabel('Time')
		plt.ylabel('Horizontal speed')
		plt.subplot(235)
		plt.xlabel('Time')
		plt.ylabel('Vertical speed')
		plt.subplot(133)
		plt.xlabel('Horizontal position')
		plt.ylabel('Vertical position')
		plt.title('CoM trajectory')
	else:
		print('The trial was a failure')