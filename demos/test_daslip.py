import slippy.daslip
import numpy as np
import matplotlib.pyplot as plt
	
slip_model = 0	#0 (slip), 1 (daslip)
	
p_daslip = { 'model_type': slip_model,    #0 (slip), 1 (daslip)
      'mass':80,                          #kg
      'stiffness':8200.0,                 #K : N/m
      'spring_resting_length':0.9,        #m
      'gravity':9.81,                     #N/kg
      'angle_of_attack':1/5*np.pi,        #rad
      'actuator_resting_length':0.1,      # m
      'actuator_force':[],                # * 2 x M matrix of time and force
      'actuator_force_period':10,         # * s
      'damping_type':0,                   # * 0 (constant), 1 (linear-with-force)
      'constant_normalized_damping':0.75,          # *    s   : D/K : [N/m/s]/[N/m]
      'linear_normalized_damping_coefficient':3.5, # * A: s/m : D/F : [N/m/s]/N : 0.0035 N/mm/s -> 3.5 1/m/s from Kirch et al. Fig 12
      'linear_minimum_normalized_damping':0.05,    # *   1/A*(kg*N/kg) :
      'swing_type':0,                    # 0 (constant angle of attack), 1 (linearly varying angle of attack)
      'swing_leg_norm_angular_velocity': 1.1,  # [1/s]/[m/s] (omega/(vx/lr))
      'swing_leg_angular_velocity':0,   # rad/s (set by calculation)
      'angle_of_attack_offset':0}        # rad   (set by calculation)
p_daslip['swing_type'] = 0
x0 = np.array([10,		# x position: forwards
				0.85,	# y position: upwards
				5.5,	# x velocity
				0,		# y velocity
				0,		# foot position x
				0,		# foot position y
				0])		# height of the ground

x0 = slippy.daslip.reset_leg(x0, p_daslip)
p_daslip['total_energy'] = slippy.daslip.compute_total_energy(x0, p_daslip)
sol = slippy.daslip.step(x0, p_daslip)

import plotting.animation
plotting.animation.animation_visualisation(sol, Ground = True)
## The arguments you can pass to the function are:
# slow = 10       		: scales the animation relative to real time, default 10
# interval = 30   		: time (in ms) between frames in the animation, default 30ms
# com_radius = 60 		: size of the CoM dot
# see = True      		: calls up show() within the animation_visualisation function; I don't know why but if you call show here, you don't see the animation 
# save = False    		: saves the animation, for this you need to have a matplotlib animation writer installed 
# writer_name = 'html'	: this default writer can only write html files, 'pillow' writes gif, ffmpeg writes mp4
# filename = 'test.html': name of the animation file that will be saved
# Ground = False    	: if False, plots the Ground at height 0, otherwise finds the height of the ground in sol.y[-1]

import plotting.single_trials
plotting.single_trials.com_visualisation(sol, Ground = True)
## The arguments you can pass to the function are:
# leg_visibility = 0.5	: set to 0 if you don't want to see the leg 
# colors = ['k','b','g']: colors of the [flight, stance, flight] phases
# size = 100			: size of the CoM dot
# Ground = False    	: if False, plots the Ground at height 0, otherwise finds the height of the ground in sol.y[-1]
plt.show()