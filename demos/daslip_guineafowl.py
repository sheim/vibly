import models.daslip as model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from enum import Enum

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
p_map.sa2xp = model.sa2xp_amp
# p_map.sa2xp = model.sa2xp_y_xdot_timedaoa
p_map.xp2s = model.xp2s_y_xdot


# * set up range of heights for first step
# at the moment, just doing 1 (0)
total_leg_length = p['spring_resting_length'] + p['actuator_resting_length']
ground_heights = np.linspace(0.0, -0.06, 4)

# * MM For now I'm not iterating over damping as the solution is
#   quite sensitive the values used: 0.1 works, 0.2 starts to see error
#
# * Set up range of damping values to compute
# at the moment, just doing 1
# damping_values = tuple(np.round(np.linspace(0.3, 0.02, 1), 2))
# for lin_d in damping_values:
#    p['linear_normalized_damping_coefficient'] = lin_d
x0, p = model.create_open_loop_trajectories(x0, p, limit_cycle_options)
trajectories = get_step_trajectories(x0, p, ground_heights=ground_heights)

energetics = list()
for idx, traj in enumerate(trajectories):
    energy = model.compute_potential_kinetic_work_total(traj.y, p)
    energetics.append(energy)

springDamperActuatorForces = list()
for idx, traj in enumerate(trajectories):
    sda = model.compute_spring_damper_actuator_force(traj.t, traj.y, p)
    # This function is only valid during stance, so zero out all the
    # entries during the flight phase
    for j in range(sda.shape[1]):
        if(traj.t[j] <= traj.t_events[1] or traj.t[j] >= traj.t_events[3]):
            sda[0, j] = 0.
            sda[1, j] = 0.
            sda[2, j] = 0.
    springDamperActuatorForces.append(sda)

leg_length = list()
for ids, traj in enumerate(trajectories):
    legLen = model.compute_leg_length(traj.y)
    leg_length.append(legLen)

# * basic plot
# Tex rendering slows the plots down, but good for final pub quality plots
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')


figBasic = plt.figure(figsize=(16, 9))
gsBasic = gridspec.GridSpec(2, 3)

maxHeight = 0
for idx, traj in enumerate(trajectories):
    if(max(traj.y[1]) > maxHeight):
        maxHeight = max(traj.y[1])

maxKineticEnergy = 0
maxPotentialEnergy = 0

for idx, pkwt in enumerate(energetics):
    if(max(pkwt[0]) > maxKineticEnergy):
        maxKineticEnergy = max(pkwt[0])
    if(max(pkwt[1]) > maxPotentialEnergy):
        maxPotentialEnergy = max(pkwt[1])

color0 = np.array([0, 0, 0])  # Nominal
color1 = np.array([1, 0, 0])  # Largest height perturbation
colorPlot = np.array([0, 0, 0])

axCoM = plt.subplot(gsBasic[0])
for idx, traj in enumerate(trajectories):
    n01 = float(max(idx, 0))/max(float(len(trajectories)-1), 1)
    colorPlot = color0*(1-n01) + color1*(n01)
    axCoM.plot(traj.y[0],  traj.y[1],
               color=(colorPlot[0],  colorPlot[1],  colorPlot[2]),
               label=ground_heights[idx])
    plt.ylim((0, maxHeight))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('CoM Trajectory')
axCoM.spines['top'].set_visible(False)
axCoM.spines['right'].set_visible(False)
plt.legend(loc='upper left')


axAF = plt.subplot(gsBasic[1])
for idx, sda in enumerate(springDamperActuatorForces):
    n01 = float(max(idx, 0))/max(float(len(trajectories)-1), 1)
    colorPlot = color0*(1-n01) + color1*(n01)
    traj = trajectories[idx]
    axAF.plot(traj.t, sda[2],
              color=(colorPlot[0],  colorPlot[1],  colorPlot[2]),
              label=ground_heights[idx])
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Actuator Forces')
axAF.spines['top'].set_visible(False)
axAF.spines['right'].set_visible(False)

axDF = plt.subplot(gsBasic[2])
for idx, sda in enumerate(springDamperActuatorForces):
    n01 = float(max(idx, 0))/max(float(len(trajectories)-1), 1)
    colorPlot = color0*(1-n01) + color1*(n01)
    traj = trajectories[idx]
    axDF.plot(traj.t, sda[1],
              color=(colorPlot[0], colorPlot[1], colorPlot[2]),
              label=ground_heights[idx])
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Damping Forces')
axDF.spines['top'].set_visible(False)
axDF.spines['right'].set_visible(False)

axLL = plt.subplot(gsBasic[3])
for idx, ll in enumerate(leg_length):
    n01 = float(max(idx, 0))/max(float(len(trajectories)-1), 1)
    colorPlot = color0*(1-n01) + color1*(n01)
    traj = trajectories[idx]
    axLL.plot(traj.t, ll,
              color=(colorPlot[0], colorPlot[1], colorPlot[2]),
              label=ground_heights[idx])
    plt.xlabel('Time (s)')
    plt.ylabel('Length (m)')
    plt.title('Leg Length')
axLL.spines['top'].set_visible(False)
axLL.spines['right'].set_visible(False)
plt.legend(loc='upper left')

axLF = plt.subplot(gsBasic[4])
for idx, sda in enumerate(springDamperActuatorForces):
    n01 = float(max(idx, 0))/max(float(len(trajectories)-1), 1)
    colorPlot = color0*(1-n01) + color1*(n01)
    traj = trajectories[idx]
    axLF.plot(traj.t, sda[0],
              color=(colorPlot[0], colorPlot[1], colorPlot[2]),
              label=ground_heights[idx])
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Leg Forces')
axLF.spines['top'].set_visible(False)
axLF.spines['right'].set_visible(False)
plt.legend(loc='upper left')

axLW = plt.subplot(gsBasic[5])
for idx, pkwt in enumerate(energetics):
    traj = trajectories[idx]
    n01 = float(max(idx, 0))/max(float(len(energetics)-1), 1)
    colorPlot = color0*(1-n01) + color1*(n01)
    axLW.plot(traj.t, pkwt[2]+pkwt[3], color=(colorPlot[0], colorPlot[1],
              colorPlot[2]))
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title('Leg Work: Actuator+Damper')
axLW.spines['top'].set_visible(False)
axLW.spines['right'].set_visible(False)

plt.show()

# TODO (STEVE) update this intro-documentation
# Model types
#
#  Daslip (damper-actuator-slip)
#  (c)   -center-of-mass-  (c)
#   -                     -----            -     -
#   |                     |   |            |     |
#   |        actuator-    f   d -damper    | la  | lr
#   |                     |   |            |     |
#   -                     -----            -     |
#   /                       /                    |
#   \ k      -spring-       \ k                  |
#   /                       /                    |
#   \                       \                    |
#    +p   -contact point-    +p                  -
#
#
#    Damping that varies linearly with the force of the actuator. This specific
#    damping model has been chosen because it emulates the intrinsic damping of
#    active muscle Since we cannot allow the damping to go to zero for
#    numerical reasons we use
#
#    d = max( d_min, A*f )
#
#    Where A takes a value of 3.5 (N/N)/(m/s) which comes from Kirch et al.'s
#    experimental data which were done on cat soleus. Fig. 12. Note in Fig.
#    12 a line of best fit has a slope of approximately 0.0035 [N/(mm/s)]/[N],
#    which in units of N/(m/s) becomes 3.5 [N/(mm/s)]/[N]. For d_min we choose
#    a value that is some small value, eta, that will scale with the mass of
#    the body (M) (and thus the expected peak value of F): d_min = A*(M*g)*eta.
#
# Kirsch RF, Boskov D, Rymer WZ. Muscle stiffness during transient and
# continuous movements of cat muscle: perturbation characteristics and
# physiological relevance. IEEE Transactions on Biomedical Engineering.
# 1994 Aug;41(8):758-70.
#
# The swing leg for the daslip now has a few options as well:
#
# 0. Constant angle of attack
#
# 1. Linearly varying angle of attack
# The leg of attack varies with time. So that this parameter does not
# have to be recomputed for each new forward velocity, compute the angular
# velocity of the leg, omega, assuming that it scales with the forward
# velocity vx of the body and a scaling factor W
# ('swing_foot_norm_velocity')

# omega = -W(vx/lr)

# thus for an W of -1 omega will be set so that when the leg is straight
# velocity of the foot exactly counters the forward velocity of the body.
# If W is set to -1.1 then the foot will be travelling backwards 10%
# faster than the foward velocity of the body.

# At the apex angle of the leg is reset to the angle of attack with an
# offset (angle_of_attack_offset) so that during the nominal model.step the
# leg lands exactly with the desired angle of attack.

# It would be ideal to set W so that it corresponded to a value that
# fits Monica's guinea fowl, or perhaps people. I don't have this data
# on hand so for now I'm just setting this to -1.1
#
# Model parameters for both slip/daslip. Parameters only used by daslip are *
