import models.parslip as model
import numpy as np
import matplotlib.pyplot as plt
import pickle
from ttictoc import TicToc
import viability as vibly
import os
import pickle


def get_step_trajectories(x0, p, ground_heights=None):
    '''
    helper function to apply a battery of ground-height perturbations.
    returns a list of trajectories.
    '''

    if ground_heights is None:
        total_leg_length = p['resting_length']
        total_leg_length += p['actuator_resting_length']
        ground_heights = np.linspace(0, -0.5*total_leg_length, 10)
    x0 = model.reset_leg(x0, p)
    trajectories = list()
    for height in ground_heights:
        x0[-1] = height
        trajectories.append(model.step(x0, p))
    x0[-1] = 0.0  # reset x0 back to 0
    return trajectories


def compute_viability(x0, p, bird_name, visualise=False):

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
    p_map.sa2xp = model.sa2xp_amp
    # p_map.sa2xp = model.sa2xp_y_xdot_timedaoa
    p_map.xp2s = model.xp2s_y_xdot

    # * set up grids
<<<<<<< Updated upstream
    s_grid_height = np.linspace(0.15, 0.45, 61)  # 21)
    s_grid_velocity = np.linspace(1.5, 5.5, 81)  # 51)
||||||| constructed merge base
    s_grid_height = np.linspace(0.1, 0.3, 81)  # 21)
    s_grid_velocity = np.linspace(1.5, 3.5, 51)  # 51)
=======
    s_grid_height = np.linspace(0.15, 0.45, 61)  # 21)
    s_grid_velocity = np.linspace(1.5, 5.5, 101)  # 51)
>>>>>>> Stashed changes
    s_grid = (s_grid_height, s_grid_velocity)
    a_grid_aoa = np.linspace(10/180*np.pi, 70/180*np.pi, 61)
<<<<<<< Updated upstream
    a_grid = (a_grid_aoa, )
    # a_grid_amp = np.linspace(0.75, 1.25, 11)
    # a_grid = (a_grid_aoa, a_grid_amp)
||||||| constructed merge base
    a_grid = (a_grid_aoa, )
    # a_grid_amp = np.linspace(0.8, 1.2, 20)
    # a_grid = (a_grid_aoa, a_grid_amp)
=======
    # a_grid = (a_grid_aoa, )
    a_grid_amp = np.linspace(0.75, 1.25, 11)
    a_grid = (a_grid_aoa, a_grid_amp)
>>>>>>> Stashed changes

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
    if not os.path.exists(bird_name):
        os.makedirs(bird_name)
    filename = bird_name+'/'+bird_name+'_'+str(damping)+'.pickle'

    data2save = {"grids": grids, "Q_map": Q_map, "Q_F": Q_F, "Q_V": Q_V,
                 "Q_M": Q_M, "S_M": S_M, "p": p, "x0": x0}
    outfile = open(filename, 'wb')
    pickle.dump(data2save, outfile)
    outfile.close()

    if visualise:
        print("SAVING FIGURE")
        print(" ")
        plt.figure()
        plt.imshow(S_M, origin='lower', vmin=0, vmax=1, cmap='viridis')
        plt.title('bird ' + bird_name)
        plt.savefig(filename+'.pdf', format='pdf')
        # plt.show()


# * Set up parameters for average of all birds
gravity = 9.81
data = np.load('stiffness.npz')
mass = 1.31  # [ 1.3667, 1.31, 1.23, 1.4198, 1.3]
# resting_length = data['l_bird']
# aTD = data['a_all'] - np.pi/2
# yApex = data['y_all']*resting_length
# vApex = data['v_all']*np.sqrt(resting_length*gravity)
# stiffness = data['k_all']*m*gravity/resting_length
resting_length = data['l_bird'][1]
aTD = data['a_bird'][1] - np.pi/2
yApex = data['y_bird'][1]*resting_length
vApex = data['v_bird'][1]*np.sqrt(resting_length*gravity)
stiffness = data['k_bird'][1]*mass*gravity/resting_length

p = {'mass': mass,                             # kg
     'stiffness': stiffness,     # K : N/m
     'resting_length': resting_length,          # m
     'gravity': gravity,                    # N/kg
     'angle_of_attack': aTD,                # rad
     'actuator_resting_length': 0.,                 # m
     'actuator_force': [],         # * 2 x M matrix of time and force
     'actuator_force_period': 10,  # * s
     'activation_delay': 0.0,      # * a delay for when to start activation
     'activation_amplification': 1.0,
     'damping': 0}

x0 = np.array([0, yApex,    # x_com , y_com
               vApex, 0,     # vx_com, vy_com
               0, 0,         # x_f   , y_f
               0])           # h
x0 = model.reset_leg(x0, p)
p['total_energy'] = model.compute_total_energy(x0, p)

# * Set up experiment parameters
<<<<<<< Updated upstream
# damping_vals = np.around(np.arange(0.0225, 0.00001, -0.0025), decimals=4)
damping_vals = np.array([0.5, 0.01])
# step_down_max = -0.4*p['resting_length']
# step_up_max = 0.4*p['resting_length']
# perturbation_vals = np.around(np.linspace(step_down_max, step_up_max, 21),
#                               decimals=2)
||||||| constructed merge base
damping_vals = np.around(np.linspace(0.1, 1.0, 5), decimals=2)
# damping_vals = np.array([0.5, 0.2, 0.1, 0.01, 0.0])
step_down_max = -0.4*p['resting_length']
step_up_max = 0.4*p['resting_length']
perturbation_vals = np.around(np.linspace(step_down_max, step_up_max, 5),
                              decimals=2)
=======
damping_vals = np.around(np.arange(0.0225, 0.00001, -0.0025), decimals=4)
# damping_vals = np.array([0.5, 0.2, 0.1, 0.01, 0.0])
# step_down_max = -0.4*p['resting_length']
# step_up_max = 0.4*p['resting_length']
# perturbation_vals = np.around(np.linspace(step_down_max, step_up_max, 21),
#                               decimals=2)
>>>>>>> Stashed changes

# * start
t_total = TicToc()
t_total.tic()
bird_name = 'higher'
legStiffnessSearchWidth = p['stiffness']*0.5
limit_cycle_options = {'search_initial_state': False,
                        'state_index': 0,
                        'state_search_width': 0,
                        'search_parameter': True,
                        'parameter_name': 'stiffness',
                        'parameter_search_width': legStiffnessSearchWidth}
for damping in damping_vals:
    p['damping'] = damping
    # print(p['stiffness'],' N/m :Leg stiffness prior to fitting')
    x0, p = model.create_open_loop_trajectories(x0, p, limit_cycle_options)
    print(p['stiffness'],' N/m :Leg stiffness prior after fitting')
    p['x0'] = x0.copy()
    compute_viability(x0, p, bird_name, visualise=True)
    # trajectories = get_step_trajectories(x0, p, perturbation_vals)
    # filename = bird_name+'/'+bird_name+'_'+str(damping)+'_trajs.pickle'
    # data2save = {"trajectories": trajectories}
    # outfile = open(filename, 'wb')
    # pickle.dump(data2save, outfile)
    # outfile.close()

t_total.toc()
print("time elapsed for one set of damping values: " + str(t_total.elapsed/60))

# # * Average of single birds

# for bird_name in [2, ]:
#     aTD = data['a_bird'][bird_name] - np.pi/2
#     yApex = data['y_bird'][bird_name]*resting_length
#     vApex = data['v_bird'][bird_name]*np.sqrt(resting_length*gravity)
#     stiffness = data['k_bird'][bird_name]*m*gravity/resting_length
#     x0[1] = yApex
#     x0[2] = vApex
#     p['mass'] = m
#     p['stiffness'] = stiffness
#     p['angle_of_attack'] = aTD
#     for damping in damping_vals:
#         compute_viability(x0, p, damping, 'bird%i' % bird_name, visualise=True)
#         trajectories = get_step_trajectories(x0, p, perturbation_vals)
#         filename = bird_name+'/'+bird_name+'_'+str(damping)+'_trajs.pickle'
#         data2save = {"trajectories": trajectories}
#         outfile = open(filename, 'wb')
#         pickle.dump(data2save, outfile)
#         outfile.close()

t_total.toc()
print("total time elapsed: " + str(t_total.elapsed/60))
