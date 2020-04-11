import models.daslip as model
import numpy as np
import matplotlib.pyplot as plt
import pickle
from ttictoc import TicToc
import viability as vibly
import os
import pickle

def compute_viability(x0, p, name, visualise=False):

    # * Solve for nominal open-loop limit-cycle

    # legStiffnessSearchWidth = p['stiffness']*0.5

    # limit_cycle_options = {'search_initial_state': False,
    #                        'state_index': 0,
    #                        'state_search_width': 0,
    #                        'search_parameter': True,
    #                        'parameter_name': 'stiffness',
    #                        'parameter_search_width': legStiffnessSearchWidth}

    # # print(p['stiffness'],' N/m :Leg stiffness prior to fitting')
    # x0, p = model.create_open_loop_trajectories(x0, p, limit_cycle_options)
    # # print(p['stiffness'],' N/m :Leg stiffness prior after fitting')
    # p['x0'] = x0.copy()

    # * Set-up P maps for comutations
    p_map = model.poincare_map
    p_map.p = p
    p_map.x = x0.copy()

    # * choose high-level represenation
    # p_map.sa2xp = model.sa2xp_amp
    p_map.sa2xp = model.sa2xp_y_xdot_timedaoa
    p_map.xp2s = model.xp2s_y_xdot

    # * set up grids
    s_grid_height = np.linspace(0.25, 1.25, 101)  # 21)
    s_grid_velocity = np.linspace(2, 12, 111)  # 51)
    s_grid = (s_grid_height, s_grid_velocity)
    a_grid_aoa = np.linspace(10/180*np.pi, 70/180*np.pi, 61)
    a_grid = (a_grid_aoa, )
    # a_grid_amp = np.linspace(0.75, 1.25, 11)
    # a_grid = (a_grid_aoa, a_grid_amp)

    grids = {'states': s_grid, 'actions': a_grid}

    # * compute

    t = TicToc()
    t.tic()
    # * compute transition matrix and boolean matrix of failures
    Q_map, Q_F = vibly.parcompute_Q_map(grids, p_map, verbose=1)
    t.toc()
    print("time elapsed " + str(t.elapsed/60))
    # * compute viable sets
    Q_V, S_V = vibly.compute_QV(Q_map, grids)
    # * compute the measure in state-space
    S_M = vibly.project_Q2S(Q_V, grids, proj_opt=np.mean)
    # * map the measure to Q-space
    Q_M = vibly.map_S2Q(Q_map, S_M, s_grid, Q_V=Q_V)

    print("non-failing portion of Q: " + str(np.sum(~Q_F)/Q_F.size))
    print("viable portion of Q: " + str(np.sum(Q_V)/Q_V.size))

    # * save data
    if not os.path.exists(name):
        os.makedirs(name)
    filename = name+'/'+name+'_'+'{:.4f}'.format(damping)

    data2save = {"grids": grids, "Q_map": Q_map, "Q_F": Q_F, "Q_V": Q_V,
                 "Q_M": Q_M, "S_M": S_M, "p": p, "x0": x0}
    outfile = open(filename+'.pickle', 'wb')
    pickle.dump(data2save, outfile)
    outfile.close()

    if visualise:
        print("SAVING FIGURE")
        print(" ")
        plt.figure()
        plt.imshow(S_M, origin='lower', vmin=0, vmax=1, cmap='viridis')
        plt.title('bird ' + name)
        plt.savefig(filename+'.pdf', format='pdf')
        # plt.show()
        plt.close()


# * Set up parameters for average of all birds

p = {'mass': 80,                          # kg
     'stiffness': 7800.0,                 # K : N/m
     'resting_length': 0.9,        # m
     'gravity': 9.81,                     # N/kg
     'angle_of_attack': 1/5*np.pi,        # rad
     'actuator_resting_length': 0.1,      # m
     'actuator_force': [],                # * 2 x M matrix of time and force
     'actuator_force_period': 10,         # * s
     'activation_delay': 0.0,  # * a delay for when to start activation
     'activation_amplification': 1.0,
     'constant_normalized_damping': 0.0,          # * s : D/K : [N/m/s]/[N/m]
     'linear_normalized_damping': 0.0,  # * A: s/m : D/F : [N/m/s]/N : 0.0035 N/mm/s -> 3.5 1/m/s from Kirch et al. Fig 12
     'linear_minimum_normalized_damping': 0.05,    # *   1/A*(kg*N/kg) :
     'swing_velocity': 0,   # rad/s (set by calculation)
     'angle_of_attack_offset': 0,        # rad   (set by calculation)
     'swing_extension_velocity': 0,    # m/s
     'swing_leg_length_offset' : 0}                 # m (set by calculation) 

x0 = np.array([0, 0.95,    # x_com , y_com
               3.3, 0,     # vx_com, vy_com
               0, 0,         # x_f   , y_f
	      p['actuator_resting_length'],  # actuator initial length
		0, 0,  # work actuator, work damper
               0])           # h
x0 = model.reset_leg(x0, p)
p['total_energy'] = model.compute_total_energy(x0, p)

# * Set up experiment parameters
damping_vals = np.around(np.arange(0.005, 0.3, 0.005), decimals=3)
# damping_vals = np.array([0.5, 0.05])

# * start
t_total = TicToc()
t_total.tic()
name = 'const_human_daslip'
# legStiffnessSearchWidth = p['stiffness']*0.5
limit_cycle_options = {'search_initial_state': True,
                        'state_index': 2,
                        'state_search_width': 2.0,
                        'search_parameter': False,
                        'parameter_name': 'stiffness',
                        'parameter_search_width': 0}

x0, p = model.create_open_loop_trajectories(x0, p, limit_cycle_options)
print("forward velocity with LC: "+str(x0[2]))

for damping in damping_vals:
    p['constant_normalized_damping'] = damping
    print("now computing: "+str(damping))
    # print(p['stiffness'],' N/m :Leg stiffness prior to fitting')
    # x0, p = model.create_open_loop_trajectories(x0, p, limit_cycle_options)
    # print(p['stiffness'],' N/m :Leg stiffness prior after fitting')
    p['x0'] = x0.copy()
    compute_viability(x0, p, name, visualise=True)

t_total.toc()
print("time elapsed for one set of damping values: " + str(t_total.elapsed/60))

# # * Average of single birds

# for name in [2, ]:
#     aTD = data['a_bird'][name] - np.pi/2
#     yApex = data['y_bird'][name]*resting_length
#     vApex = data['v_bird'][name]*np.sqrt(resting_length*gravity)
#     stiffness = data['k_bird'][name]*m*gravity/resting_length
#     x0[1] = yApex
#     x0[2] = vApex
#     p['mass'] = m
#     p['stiffness'] = stiffness
#     p['angle_of_attack'] = aTD
#     for damping in damping_vals:
#         compute_viability(x0, p, damping, 'bird%i' % name, visualise=True)
#         trajectories = get_step_trajectories(x0, p, perturbation_vals)
#         filename = name+'/'+name+'_'+str(damping)+'_trajs.pickle'
#         data2save = {"trajectories": trajectories}
#         outfile = open(filename, 'wb')
#         pickle.dump(data2save, outfile)
#         outfile.close()

t_total.toc()
print("total time elapsed: " + str(t_total.elapsed/60))
