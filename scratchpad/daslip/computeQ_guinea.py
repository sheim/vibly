import models.daslip as model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from enum import Enum
from ttictoc import TicToc

import viability as vibly
# * helper functions


def get_step_trajectories(x0, p, ground_heights=None):
    '''
    helper function to apply a battery of ground-height perturbations.
    returns a list of trajectories.
    '''

    if ground_heights is None:
        total_leg_length = p['spring_resting_length']
        total_leg_length += p['actuator_resting_length']
        ground_heights = np.linspace(0, -0.5*total_leg_length, 10)
    x0 = model.reset_leg(x0, p)
    trajectories = list()
    for height in ground_heights:
        x0[-1] = height
        trajectories.append(model.step(x0, p))
    x0[-1] = 0.0  # reset x0 back to 0
    return trajectories


# Parameters from Blum, Vejdani, Birn-Jeffery, Hubicki, Hurst, & Daley 2014
#
# As a start the parameters have been set to the average of the 5 birds. Why?
# Almost all of the parameters needed to simulate the 367 steps recorded,
# except for: the height at apex, and the forward velocity at apex. The average
# forward velocity is reported, the height at apex is not. For now we can
# solve for the height at apex that leads to a limit cycle for the 'averaged'
# bird.
# GFData_normParams
folder_Blum14 = "data/BlumVejdaniBirnJefferyHubickiHurstDaley2014"
data_norm_Blum14 = np.loadtxt(folder_Blum14+"/GFData_normParams.csv",
                              delimiter=",", skiprows=1)
#    individual=0
#    L0_m =1
#    m_kg =2

dataStepBlum2014 = np.loadtxt(folder_Blum14+"/GFData_stepVariables_upd.csv",
                              delimiter=",", skiprows=1)
#    stepType  =0
#    dropHeight  =1
#    individual  =2
#    aTD  =3
#    adotTD =4
#    LTD  =5
#    LdotTD =6
#    LdotTD_sc =7
#    kLeg =8
#    FmaxLeg   =9
#    FmaxLeg_sc  =10
#    Ix =11
#    ILeg =12
#    dECoM  =13

##
# Simulation Configuration
#  Comment overall:
#     There is a slight error between the start of the actuator
#     function and the contact time in simulation due to integration
#     error of the state and foot position (which now has swing leg retration
#     and extension following Blum et al. 2014). That means that even the
#     nominal trial has a big damper force at time zero. MM is suspects that
#     this can be improved but at the moment doesn't have the time to look into
#     this in detail.
#
#     Blum et al. and the data from Monica almost contain enough information
#     for the simulations. We have everything on a step-by-step basis except
#     for the forward velocity and the height. The average forward velocity
#     is reported in the paper so I'm taking that for the initial forward
#     velocity. This is a bit undesiredable as the forward velocity varied a
#     lot and so not every birds average angle of attack and stiffness is
#     compatible with the average velocity
#
#     The apex height is not included at all and so here we solve for the
#     height that yields a limit cycle. This is not always possible. See
#     comments below for details. I'm going to ask Monica if this trajectory
#     level information is available. It would be nice to have it, though we
#     can make due without
#     it.
#
#     Brief observations thus far:
#       -Swing leg retraction & extension make a HUGE improvement during
#        the step down trials.
#       -The results are also quite sensitive to damping - too much or too
#        little results in a fall. Thus far the nicest values for
#        linear_normalized_damping_coefficient that I see are around 0.1.
#
#  Bird id: Step Type:  Comment
#  1      : -10         MM cannot find a stable limit cycle.
#  2      : -10         works, 3/4 trials end in success
#  3      : -10         works, 2/4 trials succeed
#  4      : -10         works, 4/4 trials succeed
#  5      : -10         works, 3/4 trials succeed
#
# Step Type:
#  -10: level running
#  -1 : pre-drop
#   0 : drop step
#
# Trial Type:
#   Right now there are trials run over [0., -0.02, -0.04, -0.06] which is
#   almost the same as used in the experiment [0., -0.04, -0.06]. Note that
#   the script keeps the parameters: if you use a flat running step type (-10)
#   only the data from these step types will be used to compute the parameters
#   for the model, but all of the step-down experiments will be done. This is
#   in contrast to the actual bird where these parameters vary.
#
#   What is being done in this script (as its configured) is closer to an
#   experiment where the drop is a surprise.
##
id = 4
stepType = -10  # -10 Flat running, -1 Before drop, 0 Drop, 1 After Drop
dropHeight = 0.  # options [0, 4, 6] in cm

forwardVelocity = 2.84  # Average reported in paper. Trial data not available


# Go through all of the recorded steps and grab the average step parameters
# for the specified individual and step type
idFound = False
mass = 0
gravity = 9.81
leg_length = 0
for i in range(0, np.shape(data_norm_Blum14)[0]):
    if(data_norm_Blum14[i, 0] == id):
        mass = data_norm_Blum14[i, 2]
        leg_length = data_norm_Blum14[i, 1]
        idFound = True
assert(idFound)

normBW = mass*gravity
normT = np.sqrt(leg_length/gravity)

angleOfAttackTDSum = 0.
angleDotOfAttackNormTDSum = 0.
legLengthNormTDSum = 0.
legLengthDotNormTDSum = 0.
legStiffnessNormTDSum = 0.

nSteps = 0.

for i in range(0, np.shape(dataStepBlum2014)[0]):
    dropErr = np.abs(dataStepBlum2014[i, 1]-dropHeight)
    if(dataStepBlum2014[i, 0] == stepType
       and dataStepBlum2014[i, 2] == id and dropErr < 0.001):
        nSteps += 1.
        angleOfAttackTDSum += dataStepBlum2014[i, 3]
        angleDotOfAttackNormTDSum += dataStepBlum2014[i, 4]
        legLengthNormTDSum += dataStepBlum2014[i, 5]
        legLengthDotNormTDSum += dataStepBlum2014[i, 6]
        legStiffnessNormTDSum += dataStepBlum2014[i, 8]

assert(nSteps > 0.)

angleOfAttackDegreesTD = (angleOfAttackTDSum/nSteps)
angleDotOfAttackDegreesTD = (1./(normT))*(angleDotOfAttackNormTDSum/nSteps)
legLengthTD = leg_length*(legLengthNormTDSum/nSteps)
legLengthDotTD = (leg_length/normT)*(legLengthDotNormTDSum/nSteps)
legStiffnessTD = (normBW/leg_length)*(legStiffnessNormTDSum/nSteps)

angleOfAttackTD = (angleOfAttackDegreesTD - 90)*(np.pi/180)
angleDotOfAttackTD = (angleDotOfAttackDegreesTD)*(np.pi/180)

# Lin damping: 0.0035 N/mm/s -> 3.5 1/m/s from Kirch et al. Fig 12
p = {'mass': mass,                                  # kg
     'stiffness': legStiffnessTD,                   # K: N/m
     'spring_resting_length': legLengthTD,          # m
     'gravity': gravity,                            # N/kg
     'angle_of_attack': angleOfAttackTD,            # rad
     'actuator_resting_length': 0.,                 # m
     'actuator_force': [],                   # * 2 x M matrix of time and force
     'actuator_force_period': 10,                   # * s
     'activation_delay': 0.0,         # * a delay for when to start activation
     'activation_amplification': 1.0,
     'constant_normalized_damping': 0.75,           # * s: D/K: [N/m/s]/[N/m]
     'linear_normalized_damping_coefficient': 0.1,  # * A: s/m: D/F: [N/m/s]/N:
     'linear_minimum_normalized_damping': 0.01,     # *   1/A*(kg*N/kg):
     'swing_velocity': angleDotOfAttackTD,       # rad/s (set by calculation)
     'angle_of_attack_offset': 0,                # rad   (set by calculation)
     'swing_extension_velocity': legLengthDotTD,    # m/s
     'swing_leg_length_offset': 0}                 # m (set by calculation)
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

heightTD = leg_length*np.cos(angleOfAttackTD)

x0 = np.array([0, heightTD + 0.05,    # x_com , y_com
               forwardVelocity, 0,             # vx_com, vy_com
               0,              0,              # x_f   , y_f
               p['actuator_resting_length'],    # l_a
               0, 0,                           # wa, wd
               0])                             # h
x0 = model.reset_leg(x0, p)
p['total_energy'] = model.compute_total_energy(x0, p)

# * Solve for nominal open-loop trajectories

heightSearchWidth = 0.025

limit_cycle_options = {'search_initial_state': True,
                       'state_index': 1,
                       'state_search_width': heightSearchWidth,
                       'search_parameter': False,
                       'parameter_name': 'angle_of_attack',
                       'parameter_search_width': np.pi*0.25}

x0, p = model.create_open_loop_trajectories(x0, p, limit_cycle_options)
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
s_grid_height = np.linspace(0.1, 0.3, 61)  # 21)
s_grid_velocity = np.linspace(0.5, 2.5, 61)  # 51)
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


t = TicToc()
t.tic()
Q_map, Q_F = vibly.parcompute_Q_map(grids, p_map, verbose=5)
t.toc()


print("time elapsed: " + str(t.elapsed/60))
Q_V, S_V = vibly.compute_QV(Q_map, grids)
S_M = vibly.project_Q2S(Q_V, grids, proj_opt=np.mean)
Q_M = vibly.map_S2Q(Q_map, S_M, s_grid, Q_V=Q_V)
# plt.scatter(Q_map[1], Q_map[0])
print("non-failing portion of Q: " + str(np.sum(~Q_F)/Q_F.size))
print("viable portion of Q: " + str(np.sum(Q_V)/Q_V.size))

# plt.imshow(S_M, origin='lower')

# S_N = vibly.project_Q2S(~Q_F, grids, proj_opt=np.mean)
# plt.imshow(S_N, origin='lower')
# plt.show()

import pickle

filename = 'guinea' + '.pickle'
data2save = {"grids": grids, "Q_map": Q_map, "Q_F": Q_F, "Q_V": Q_V,
             "Q_M": Q_M, "S_M": S_M, "p": p, "x0": x0}
outfile = open(filename, 'wb')
pickle.dump(data2save, outfile)
outfile.close()