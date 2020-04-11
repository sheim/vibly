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

    # * Set-up P maps for comutations
    p_map = model.poincare_map
    p_map.p = p
    p_map.x = x0.copy()

    # * choose high-level represenation

    p_map.xp2s = model.xp2s_y_xdot
    # this maps the full simulated state to the high-level representation,
    # in this case the relevant states at apex: (y, xdot) (height an velocity)

    p_map.sa2xp = model.sa2xp_y_xdot_timedaoa
    # this maps the high-level representation of state and actions back to
    # the full state and parameters used for the simulation

    # p_map.sa2xp = model.sa2xp_amp
    # this representation includes an amplification coefficient `a' for the
    # muscle activation a*f(t). It adds a dimension to the grids, which
    # substantially increases computation time, and also makes visualization
    # much less straightforward (due to the 4-dimensional state-action space)
    # We have tried this out, and a from a preliminary look, there does not
    # seem to be much qualitative difference in the results.

    # * set up grids for computing the viable set and measure
    # a denser grid will yield more precision, but also require more compute
    s_grid_height = np.linspace(0.1, 1.4, 131)
    s_grid_velocity = np.linspace(0.5, 7.5, 141)
    s_grid = (s_grid_height, s_grid_velocity)
    a_grid_aoa = np.linspace(10/180*np.pi, 70/180*np.pi, 91)
    a_grid = (a_grid_aoa, )

    # if you use the representation `sa2xp_amp` (see above), the action grid
    # also includes an extra dimension
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
        # plt.show()  # to just see it on the fly
        plt.close()


# * Set up parameters for average of a single bird
p = {'mass': 1.31,  # kg
     'stiffness': 748.7648416462149,  # K : N/m
     'resting_length': 0.25211325904476056,  # m
     'gravity': 9.81,  # N/kg
     'angle_of_attack': 0.5307319798714443,  # rad
     'actuator_resting_length': 0.02801258433830673,  # m
     'actuator_force': [],  # * 2 x M matrix of time and force
     'actuator_force_period': 10,  # * s
     'activation_delay': 0.0,  # * a delay for when to start activation
     'activation_amplification': 1.0,  # 1 for no amplification
     'constant_normalized_damping': 0.0,  # * s : D/K : [N/m/s]/[N/m]
     'linear_normalized_damping': 0.0,  # * A: s/m : D/F : [N/m/s]/N
     'linear_minimum_normalized_damping': 0.015,  # for numerical stability)
     'swing_velocity': 0,  # rad/s (set by calculation)
     'angle_of_attack_offset': 0,  # rad   (set by calculation)
     'swing_extension_velocity': 0,  # m/s
     'swing_leg_length_offset': 0}  # m (set by calculation) 

x0 = np.array([0, 0.249369,  # x_com , y_com
               3.30643233, 0,  # vx_com, vy_com
               0, 0,  # foot x, foot y (set below)
               p['actuator_resting_length'],  # actuator initial length
               0, 0,  # work actuator, work damper
               0])  # ground height h
x0 = model.reset_leg(x0, p)
p['total_energy'] = model.compute_total_energy(x0, p)

# * Set up experiment parameters
# damping_vals = np.concatenate((np.array([0.001, 0.005]),
#                                np.around(np.arange(0.01, 0.2, 0.01),
#                                          decimals=4)))
damping_vals = [0.1, ]

# * start computation
t_total = TicToc()
t_total.tic()
name = 'TEST'  # name folder and files to save data

# * find spring stiffness that results in limit cycle motion
legStiffnessSearchWidth = p['stiffness']*0.5
limit_cycle_options = {'search_initial_state': False,
                       'state_index': 2,  # not used
                       'state_search_width': 2.0,  # not used
                       'search_parameter': True,
                       'parameter_name': 'stiffness',
                       'parameter_search_width': legStiffnessSearchWidth}

print(p['stiffness'], ' N/m :Leg stiffness prior to fitting')
x0, p = model.create_open_loop_trajectories(x0, p, limit_cycle_options)
print(p['stiffness'], ' N/m :Leg stiffness after fitting')

# * Save parameter fit
if not os.path.exists(name):
    os.makedirs(name)
filename = name+'/'+name+'_parameter_fit'

data2save = {"p": p, "x0": x0,
             "limit_cycle_options":limit_cycle_options,
             "damping_vals": damping_vals}
outfile = open(filename+'.pickle', 'wb')
pickle.dump(data2save, outfile)
outfile.close()

# * For each damping coefficient, compute the viability measure
for damping in damping_vals:
    p['constant_normalized_damping'] = damping
    # p['linear_normalized_damping'] = damping
    # use the commented line if you want to try a damping force that scales
    # linearly with muscle-activation (f(t)). Not discussed in the paper.
    p['x0'] = x0.copy()
    compute_viability(x0, p, name, visualise=True)

t_total.toc()
print("time elapsed for one set of damping values: " + str(t_total.elapsed/60))