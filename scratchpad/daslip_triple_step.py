from slippy.daslip import *
import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import sys
from matplotlib import gridspec

slip_model  = 0 #spring-loaded-inverted-pendulum
daslip_model= 1 #damper-actuator-spring-loaded inverted pendulum

#Model parameters for both slip/daslip. Parameters only used by daslip are *
p = { 'model_type':slip_model,            #0 (slip), 1 (daslip)
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

# set up a helper triple-step integrator

def triple_step(x0, p, step, height_perturbation = 0):
    '''
    just setting up shorthand for doing 3 steps, with a possible height perturbation
    in between
    '''
    # first step
    sol = step(x0, p)
    #  one step with perturbation
    x0 = sol.y[:,-1]
    x0[-1] = x0[-1] + height_perturbation
    sol = step(x0, p, sol)
    # final step without perturbation
    x0 = sol.y[:,-1]
    x0[-1] = x0[-1] - height_perturbation
    sol = step(x0, p, sol)
    return sol

#===============================================================================
#Initialization: Slip, Daslip and perturbations
#===============================================================================

x0   = np.array([0, 0.85, 5.5, 0, 0, 0, 0])

x0 = reset_leg(x0, p)
# * Get limit-cycle (LC)
search_width = np.pi*0.25
swing_type = p['swing_type']
# * To compute the limit cycle make sure the swing type is fixed to constant.
p['swing_type'] = 0
# p['angle_of_attack'] = 0.65
p, success = limit_cycle(x0,p, 'angle_of_attack', search_width)
sol_LC = step(x0, p)

# * SLIP initial conditions
p_slip = p.copy()
x0_slip = reset_leg(x0, p_slip)

# * DASLIP initial conditions
x0_daslip = np.concatenate((x0[:-1],
                            np.array([p['actuator_resting_length'], 0, 0, 0])))
# #Use the limit cycle to form the actuator open-loop force time series
actuator_time_force = create_actuator_open_loop_time_series(sol_LC, p)
p_daslip = p.copy() # python doesn't do deep copies by default
p_daslip['model_type'] = daslip_model
p_daslip['actuator_force'] = actuator_time_force
p_daslip['actuator_force_period'] = np.max(actuator_time_force[0, :])

x0_daslip = reset_leg(x0_daslip, p_daslip)

# perturbations

perturbations = np.arange(-0.15, 0.151, 0.05)

#===============================================================================
# SLIP
#===============================================================================

# one step on the LC
slip_list = list()

for idx, perturbation in np.ndenumerate(perturbations):
    slip_list.append(triple_step(x0_slip, p_slip, step, perturbation))

##### repeat with damping
daslip_list = list()

for idx, perturbation in np.ndenumerate(perturbations):
    daslip_list.append(triple_step(x0_daslip, p_daslip, step, perturbation))

# plotting
# plt.figure(figsize=(9,6))
# gsBasic= gridspec.GridSpec(2, 2, width_ratios=[2, 1])

# ax=plt.subplot(gsBasic[0])
#Creates two subplots and unpacks the output array immediately

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.set_title('SLIP and DASLIP')

for sol in slip_list:
    ax1.plot(sol.y[0], sol.y[1])

# plt=plt.subplot(gsBasic[1])
for sol in daslip_list:
    ax2.plot(sol.y[0], sol.y[1])

plt.show()