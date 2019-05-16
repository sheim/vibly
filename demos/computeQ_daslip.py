import numpy as np
import matplotlib.pyplot as plt

from slippy.daslip import *
import slippy.viability as vibly

# * First, solve for the operating point to get an open-loop force traj
# Model parameters for both slip/daslip. Parameters only used by daslip are *
p_slip = { 'model_type':0,            #0 (slip), 1 (daslip)
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

x0_slip   = np.array([0, 0.85, 5.5, 0, 0, 0, 0])
x0_slip = reset_leg(x0_slip, p_slip)

p_slip['total_energy'] = compute_total_energy(x0_slip, p_slip)

search_width = np.pi*0.25
# find p for LC
p_lc, success = limit_cycle(x0_slip, p_slip, 'angle_of_attack', search_width)
x0_slip = reset_leg(x0_slip, p_lc)
# get high-res SLIP traj
sol_slip = step(x0_slip, p_lc)
# create the open_loop_force
actuator_time_force = create_actuator_open_loop_time_series(sol_slip, p_lc)

p_daslip = p_lc.copy()
p_daslip['model_type'] = 1
p_daslip['actuator_force'] = actuator_time_force
p_daslip['actuator_force_period'] = np.max(actuator_time_force[0, :])

# initialize default x0_daslip
x0_daslip = np.array([0, 0.85, 5.5, 0, 0, 0,
        p_daslip['actuator_resting_length'], 0, 0, 0])
for idx, val in enumerate(x0_slip): # copy over default values from SLIP
    x0_daslip[idx] = val
x0_daslip = reset_leg(x0_daslip, p_daslip)
poincare_map.p = p_daslip
poincare_map.x = x0_daslip
poincare_map.sa2xp = mapSA2xp_energy_normalizedheight_aoa
poincare_map.xp2s = map2s_energy_normalizedheight_aoa

s_grid_energy = np.linspace(1500, 2500, 10)
s_grid_normalizedheight = np.linspace(0.4, 1, 10)
s_grid = (s_grid_energy, s_grid_normalizedheight)
a_grid = (np.linspace(0/180*np.pi, 70/180*np.pi, 15), )

grids = {'states':s_grid, 'actions':a_grid}
Q_map, Q_F = vibly.compute_Q_map(grids, poincare_map, verbose = 2)
Q_V, S_V = vibly.compute_QV(Q_map, grids)
print("non-failing portion of Q: " + str(np.sum(Q_F)/Q_F.size))
print("viable portion of Q: " + str(np.sum(Q_V)/Q_V.size))

################################################################################
# save data as pickle
################################################################################
import pickle
import time
filename = 'Q_map_daslip' + time.strftime("%Y_%m_%H_%M_%S") + '.pickle'
data2save = {"grids": grids, "Q_map": Q_map, "Q_F": Q_F, "Q_V":Q_V,
            "p" : p_daslip, "x0":x0_daslip, "P_map" : poincare_map}
outfile = open(filename, 'wb')
pickle.dump(data2save, outfile)
outfile.close()
# to load this data, do:
# infile = open(filename, 'rb')
# data = pickle.load(infile)
# infile.close()