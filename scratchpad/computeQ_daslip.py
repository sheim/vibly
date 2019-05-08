import numpy as np
import matplotlib.pyplot as plt
from slippy.daslip import *
from slippy.viability import compute_Q_map

p = { 'model_type':1,
      'mass':80,
      'stiffness':8200.0,
      'spring_resting_length':0.9,
      'gravity':9.81,
      'angle_of_attack':1/5*np.pi,
      'actuator_resting_length':0.1,
      'actuator_normalized_damping':0.75,
      'actuator_force':[],
      'actuator_force_period':10}

x0 = np.array([0, 0.85, 5.5, 0, 0, 0, p['actuator_resting_length'], 0])

x0 = reset_leg(x0, p)
p['total_energy'] = compute_total_energy(x0, p)

def p_map(x, p):
    '''
    Define the poincare section as finding first the open-loop action,
    and then playing it out. This will be slower, and should be replaced by
    first doincomputing a look-up table that can be interpolated over.
    '''
    # check if infeasible:
    if x[5] < 0: # foot starts underground
        return x, True

    # x_lc = np.copy(x)
    p_lc = p.copy()
    p_lc['model_type'] = 0
    search_width = np.pi*0.25
    x0_slip = np.copy(x[0:6])
    #Get a high-resolution state trajectory of the limit cycle
    x0_slip = reset_leg(x0_slip, p_lc)
    (p_lc, success) = limit_cycle(x0_slip, p_lc, 'angle_of_attack', search_width)
    #Get a high-resolution state trajectory of the limit cycle
    x0_slip = reset_leg(x0_slip, p_lc)
    # p['total_energy'] = compute_total_energy(x0_slip, p_lc)

    sol_slip = step(x0_slip, p_lc)
    actuator_time_force = create_actuator_open_loop_time_series(sol_slip, p_lc)
    p_daslip = p_lc # TODO should not be necessary? check
    p_daslip['model_type'] = 1
    p_daslip['actuator_force'] = actuator_time_force
    p_daslip['actuator_force_period']=np.max(actuator_time_force[0,:])
    sol_daslip = step(x, p_daslip)
    # check for failures
    xk = sol_daslip.y[:, -1]
    if xk[1] <= 0:
        failed = True
    elif xk[2] <= 0:
        failed = True
    else:
        failed = False
    return xk, failed


p_map.p = p
p_map.x = x0
p_map.sa2xp = mapSA2xp_aoa
p_map.xp2s = map2s

s_grid_y = np.linspace(0.85, 1, 3)
s_grid_xdot = np.linspace(5, 6, 4)
s_grid = (s_grid_y, s_grid_xdot)
a_grid = (np.linspace(30/180*np.pi, 70/180*np.pi, 5), )

Q_map, Q_F = compute_Q_map(s_grid, a_grid, p_map)
grids = {'states':s_grid, 'actions':a_grid}

# save file
data2save = {"grids": grids, "Q_map": Q_map, "Q_F": Q_F}
np.savez('test.npz', **data2save)