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
    s_grid_height = np.linspace(0.1, 1.4, 131)
    s_grid_velocity = np.linspace(0.5, 7.5, 141)
    s_grid = (s_grid_height, s_grid_velocity)
    a_grid_aoa = np.linspace(10/180*np.pi, 70/180*np.pi, 31)  # 91)
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
bird_idx = 1
gravity = 9.81
data = np.load('stiffness.npz')
mass_list = np.array([ 1.3667, 1.31, 1.23, 1.4198, 1.3])  # missing data
mass = mass_list[bird_idx]
resting_length = data['l_bird'][bird_idx]
aTD = data['a_bird'][bird_idx] - np.pi/2
yApex = data['y_bird'][bird_idx]*resting_length
vApex = data['v_bird'][bird_idx]*np.sqrt(resting_length*gravity)
stiffness = data['k_bird'][bird_idx]*mass*gravity/resting_length

p = {'mass': mass,                          # kg
     'stiffness': stiffness,                 # K : N/m
     'resting_length': 0.9*resting_length,        # m
     'gravity': gravity,                     # N/kg
     'angle_of_attack': 1/5*np.pi,        # rad
     'actuator_resting_length': 0.1*resting_length,      # m
     'actuator_force': [],                # * 2 x M matrix of time and force
     'actuator_force_period': 10,         # * s
     'activation_delay': 0.0,  # * a delay for when to start activation
     'activation_amplification': 1.0,
     'constant_normalized_damping': 0.0,          # * s : D/K : [N/m/s]/[N/m]
     'linear_normalized_damping': 0.0,  # * A: s/m : D/F : [N/m/s]/N : 0.0035 N/mm/s -> 3.5 1/m/s from Kirch et al. Fig 12
     'linear_minimum_normalized_damping': 0.015,    # *   1/A*(kg*N/kg) :
     'swing_velocity': 0,   # rad/s (set by calculation)
     'angle_of_attack_offset': 0,        # rad   (set by calculation)
     'swing_extension_velocity': 0,    # m/s
     'swing_leg_length_offset' : 0}                 # m (set by calculation) 

x0 = np.array([0, yApex,    # x_com , y_com
               vApex, 0,     # vx_com, vy_com
               0, 0,         # x_f   , y_f
	      p['actuator_resting_length'],  # actuator initial length
		0, 0,  # work actuator, work damper
               0])           # h
x0 = model.reset_leg(x0, p)
p['total_energy'] = model.compute_total_energy(x0, p)

# * Set up experiment parameters
# damping_vals = np.around(np.arange(0.01, 0.00000001, -0.0025), decimals=4)
damping_vals = np.concatenate((np.array([0.001, 0.005]),
                               np.around(np.arange(0.01, 0.2, 0.01),
                                         decimals=4)))

# * start
t_total = TicToc()
t_total.tic()
name = 'fall'
legStiffnessSearchWidth = p['stiffness']*0.5
limit_cycle_options = {'search_initial_state': False,
                        'state_index': 2,
                        'state_search_width': 2.0,
                        'search_parameter': True,
                        'parameter_name': 'stiffness',
                        'parameter_search_width': legStiffnessSearchWidth}

x0, p = model.create_open_loop_trajectories(x0, p, limit_cycle_options)
print(p['stiffness'],' N/m :Leg stiffness prior after fitting')

# save parameter fit
if not os.path.exists(name):
    os.makedirs(name)
filename = name+'/'+name+'_parameter_fit'

data2save = {"p": p, "x0": x0,
             "limit_cycle_options":limit_cycle_options,
             "damping_vals": damping_vals}
outfile = open(filename+'.pickle', 'wb')
pickle.dump(data2save, outfile)
outfile.close()

for damping in damping_vals:
    p['constant_normalized_damping'] = damping
    # p['linear_normalized_damping'] = damping
    p['x0'] = x0.copy()
    compute_viability(x0, p, name, visualise=False)

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
