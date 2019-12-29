import models.daslip as model
import numpy as np
import matplotlib.pyplot as plt
import pickle
from ttictoc import TicToc
import viability as vibly
import os

## To keep the normalisation consistent (for damping)
gravity = 9.81
params 	= np.load('ref_params.npz')
m 		= params['mass']
LTD 	= params['spring_resting_length']
adotTD 	  = 0
LdotTD 	  = 0

def compute_viability(aTD, yApex, vApex, stiffness, damping, bird_name,
                      visualise = False):
    p = {'mass': m,                             # kg
            'stiffness': stiffness,     # K : N/m
            'spring_resting_length': LTD,          # m
            'gravity': gravity,                    # N/kg
            'angle_of_attack': aTD,                # rad
            'actuator_resting_length': 0.,                 # m
            'actuator_force': [],                   # * 2 x M matrix of time and force
            'actuator_force_period': 10,                   # * s
            'activation_delay': 0.0,         # * a delay for when to start activation
            'activation_amplification': 1.0,
            'constant_normalized_damping': 0.0,     # * s : D/K : [N/m/s]/[N/m]
            'linear_normalized_damping_coefficient': damping,  # * A: s/m : D/F : [N/m/s]/N : 0.0035 N/mm/s -> 3.5 1/m/s from Kirch et al. Fig 12
            'linear_minimum_normalized_damping': 0.01,     # *   1/A*(kg*N/kg) :
            'swing_velocity': adotTD,                      # rad/s (set by calculation)
            'angle_of_attack_offset': 0,                   # rad   (set by calculation)
            'swing_extension_velocity': LdotTD,           # m/s
            'swing_leg_length_offset' : 0}                # m (set by calculation)
    ##
    # * Initialization: Slip & Daslip
    ##

    # State vector of the Damped-Actuated-Slip (daslip)
    #
    # Index  Name   Description                       Units
    #     0  x       horizontal position of the CoM   m
    #     1  y       vertical   position "        "   m
    #     2  dx/dt   horizontal velocity "        "   m/s
    #     3  dy/dt   vertical   velocity "        "   m/s
    #     4  xf      horizontal foot velocity         m
    #     5  yf      vertical foot velocity           m
    #     6  la      actuator length                  m
    #     7  wa      actuator-force-element work      J
    #     8  wd      actuator-damper-element work     J
    #     9  h       floor height (normally fixed)    m


    x0 = np.array([0, yApex,    # x_com , y_com
                    vApex, 0,             # vx_com, vy_com
                    0,              0,              # x_f   , y_f
                    p['actuator_resting_length'],    # l_a
                    0, 0,                           # wa, wd
                    0])                             # h

    x0 = model.reset_leg(x0, p)
    p['total_energy'] = model.compute_total_energy(x0, p)
    # print('x0',x0)

    # * Solve for nominal open-loop trajectories

    legStiffnessSearchWidth = p['stiffness']*0.5

    limit_cycle_options = {'search_initial_state' : False,
                            'state_index'          : 0,
                            'state_search_width'   : 0,
                            'search_parameter'     : True,
                            'parameter_name'       : 'stiffness',
                            'parameter_search_width': legStiffnessSearchWidth}

    print(p['stiffness'],' N/m :Leg stiffness prior to fitting')
    x0, p = model.create_open_loop_trajectories(x0, p, limit_cycle_options) ## even if you don't want to fit stiffness,
    ## this changes the values of the 5, 6 and 7th coordinates of x0: x_f, y_f, l_a
    print(p['stiffness'],' N/m :Leg stiffness prior after fitting')
    # print('x0',x0)
    p['x0'] = x0.copy()

    # * Set-up P maps for comutations
    p_map = model.poincare_map
    p_map.p = p
    p_map.x = x0.copy()

    # * choose high-level represenation
    # p_map.sa2xp = model.sa2xp_amp
    p_map.sa2xp = model.sa2xp_y_xdot_timedaoa
    p_map.xp2s = model.xp2s_y_xdot

    # * set up grids
    s_grid_height = np.linspace(0.1, 0.3, 81)  # 21)
    s_grid_velocity = np.linspace(1.5, 3.5, 51)  # 51)
    s_grid = (s_grid_height, s_grid_velocity)
    a_grid_aoa = np.linspace(10/180*np.pi, 70/180*np.pi, 61)
    a_grid = (a_grid_aoa, )
    # a_grid_amp = np.linspace(0.8, 1.2, 5)
    # a_grid = (a_grid_aoa, a_grid_amp)

    grids = {'states': s_grid, 'actions': a_grid}

    # * turn off swing dynamics
    # for second step, we assume perfect control, such that the chosen aoa is the
    # desired one.
    p['swing_velocity'] = 0
    p['swing_extension_velocity'] = 0
    p['swing_leg_length_offset'] = 0
    p['angle_of_attack_offset'] = 0
    model.reset_leg(x0, p)

    # * compute

    # Q_map, Q_F = vibly.parcompute_Q_map(grids, p_map, verbose=5)
    t = TicToc()
    t.tic()
    Q_map, Q_F = vibly.parcompute_Q_map(grids, p_map, verbose=1)
    t.toc()
    print("time elapsed " + str(t.elapsed/60))
    Q_V, S_V = vibly.compute_QV(Q_map, grids) # compute the viable sets
    S_M = vibly.project_Q2S(Q_V, grids, proj_opt=np.mean) # compute the measure in S-space
    Q_M = vibly.map_S2Q(Q_map, S_M, s_grid, Q_V=Q_V) # map the measure to Q-space
    # plt.scatter(Q_map[1], Q_map[0])
    print("non-failing portion of Q: " + str(np.sum(~Q_F)/Q_F.size))
    print("viable portion of Q: " + str(np.sum(Q_V)/Q_V.size))

    if not os.path.exists(bird_name):
        os.makedirs(bird_name)
    filename = bird_name+'/'+bird_name+'_'+str(damping)
    np.savez(filename, grids = grids, Q_map = Q_map, Q_F = Q_F, Q_V = Q_V,
             Q_M = Q_M, S_M = S_M, p = p, x0 = x0)

    if visualise:
        plt.figure()
        plt.imshow(S_M, origin='lower')
        plt.title('Bird ' + bird_name)
        plt.savefig(filename+'.pdf', format='pdf')
		# plt.show()

## Average of the birds
t_total = TicToc()
t_total.tic()
data      = np.load('stiffness.npz')
aTD       = data['a_all'] - np.pi/2
yApex     = data['y_all']*LTD
vApex  	  = data['v_all']*np.sqrt(LTD*gravity)
stiffness = data['k_all']*m*gravity/LTD
for damping in np.around(np.linspace(1.0, 0.6, 5), decimals=1):
    compute_viability(aTD, yApex, vApex, stiffness, damping, 'all_birds',
                      visualise=True)
t_total.toc()
print("time elapsed for one set of damping values: " + str(t_total.elapsed/60))
## Average of single birds

for bird_name in range(5):
    aTD       = data['a_bird'][bird_name] - np.pi/2
    yApex     = data['y_bird'][bird_name]*LTD
    vApex  	  = data['v_bird'][bird_name]*np.sqrt(LTD*gravity)
    stiffness = data['k_bird'][bird_name]*m*gravity/LTD
    for damping in np.around(np.linspace(1.0, 0.6, 5), decimals=1):
        compute_viability(aTD, yApex, vApex, stiffness, damping,
                          'bird%i'%bird_name, visualise=True)

t_total.toc()
print("total time elapsed: " + str(t_total.elapsed/60))
