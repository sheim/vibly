from models import nslip
import numpy as np
import matplotlib.pyplot as plt

p = {'mass': 80.0, 'stiffness': 705.0, 'resting_angle': 17/18*np.pi, 'gravity': 9.81,
    'angle_of_attack': 1/5*np.pi, 'upper_leg': 0.5, 'lower_leg': 0.5}
x0 = np.array([10,		# x position: forwards
               0.85,  # y position: upwards
               5.5,	 # x velocity
               0,  # y velocity
               0,  # foot position x
               0,  # foot position y
               0])  # ground position y
x0 = nslip.reset_leg(x0, p)
p['x0'] = x0
p['total_energy'] = nslip.compute_total_energy(x0, p)
sol = nslip.step(x0, p)

# import plotting.animation
# plotting.animation.animation_visualisation(sol)
## The arguments you can pass to the function are:
# slow = 10       		: scales the animation relative to real time, default 10
# interval = 30   		: time (in ms) between frames in the animation, default 30ms
# com_radius = 60 		: size of the CoM dot
# see = True      		: calls up show() within the animation_visualisation function; I don't know why but if you call show here, you don't see the animation 
# save = False    		: saves the animation, for this you need to have a matplotlib animation writer installed 
# writer_name = 'html'	: this default writer can only write html files, 'pillow' writes gif, ffmpeg writes mp4
# filename = 'test.html': name of the animation file that will be saved

import plotting.single_trials
plotting.single_trials.com_visualisation(sol)
## The arguments you can pass to the function are:
# leg_visibility = 0.5	: set to 0 if you don't want to see the leg 
# colors = ['k','b','g']: colors of the [flight, stance, flight] phases
# size = 100			: size of the CoM dot
plotting.single_trials.full_visualisation(sol)	
## The arguments you can pass to the function are:
# colors = ['k','b','g']: colors of the [flight, stance, flight] phases
# foot = False			: set to True if you want to see the foot trajectory
plt.show()
