import models.daslip as model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

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


# * initialize parameters. Compared to the original version...
# * removed:
# model_type: the type is always a daslip
# damping_type: both parameters for both damping types need to be set. The resulting forces are summed. To use only one, just set the parameter of the other to 0
# swing_type: to choose "no swing', set the swing velocity to 0
# * added
# activation_delay: this will add an extra delay (can be negative) to the open-loop trajectory. This is used to adjust the timing to coincide with touch-down
# activation_amplification: a coefficient to modulate the open-loop activation
# * renamed
# swing_leg_angular_velcoity --> swing_velocity

p = {'mass': 80,                          # kg
     'stiffness': 8200.0,                 # K : N/m
     'spring_resting_length': 0.9,        # m
     'gravity': 9.81,                     # N/kg
     'angle_of_attack': 1/5*np.pi,        # rad
     'actuator_resting_length': 0.1,      # m
     'actuator_force': [],                # * 2 x M matrix of time and force
     'actuator_force_period': 10,         # * s
     'activation_delay': 0.0,  # * a delay for when to start activation
     'activation_amplification': 1.0,
     'constant_normalized_damping': 0.75,          # * s : D/K : [N/m/s]/[N/m]
     'linear_normalized_damping_coefficient': 3.5,  # * A: s/m : D/F : [N/m/s]/N : 0.0035 N/mm/s -> 3.5 1/m/s from Kirch et al. Fig 12
     'linear_minimum_normalized_damping': 0.05,    # *   1/A*(kg*N/kg) :
     'swing_leg_norm_angular_velocity':  0,  # [1/s]/[m/s] (omega/(vx/lr))
     'swing_velocity': 0,   # rad/s (set by calculation)
     'angle_of_attack_offset': 0}        # rad   (set by calculation)

# * Initialization: Slip & Daslip

x0 = np.array([0, 1.00, 5.5, 0, 0, 0, p['actuator_resting_length'], 0, 0, 0])
x0 = model.reset_leg(x0, p)
p['total_energy'] = model.compute_total_energy(x0, p)
# * Solve for nominal open-loop trajectories
x0, p = model.create_open_loop_trajectories(x0, p)
p['x0'] = x0.copy()

# * Set-up P maps for comutations
p_map = model.poincare_map
p_map.p = p
p_map.x = x0.copy()

# * choose high-level represenation
p_map.sa2xp = model.sa2xp_amam
# p_map.sa2xp = model.sa2xp_y_xdot_timedaoa
p_map.xp2s = model.xp2s_y_xdot

# * Set up range of damping values to compute
# at the moment, just doing 1
damping_values = tuple(np.round(np.linspace(0.3, 0.02, 1), 2))

# * set up range of heights for first step
# at the moment, just doing 1 (0)
total_leg_length = p['spring_resting_length'] + p['actuator_resting_length']
ground_heights = np.linspace(0.0*total_leg_length,
                             -0.2*total_leg_length, 2)

for lin_d in damping_values:
    p['linear_normalized_damping_coefficient'] = lin_d
    model.create_open_loop_trajectories(x0, p)
    trajectories = get_step_trajectories(x0, p, ground_heights=ground_heights)

energetics = list()
for idx, traj in enumerate(trajectories):    
    energy = model.compute_potential_kinetic_work_total(traj.y, p)
    energetics.append(energy)

springDamperActuatorForces = list()
for idx,traj in enumerate(trajectories):    
    sda = model.compute_spring_damper_actuator_force(traj.t,traj.y, p)
    #This function is only valid during stance, so zero out all the 
    #entries during the flight phase
    for j in range(sda.shape[1]):
        if(traj.t[j] <= traj.t_events[1] or traj.t[j] >= traj.t_events[3]):
            sda[0,j] = 0.
            sda[1,j] = 0.
            sda[2,j] = 0.
    springDamperActuatorForces.append(sda)

legLength = list()
for ids,traj in enumerate(trajectories):
    legLen = model.compute_leg_length(traj.y)
    legLength.append(legLen)

# * basic plot
# Tex rendering slows the plots down, but good for final pub quality plots
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')


figBasic = plt.figure(figsize=(16,9))
gsBasic = gridspec.GridSpec(2,3)

maxHeight=0
for idx, traj in enumerate(trajectories):
    if(max(traj.y[1])>maxHeight):
        maxHeight = max(traj.y[1])

maxKineticEnergy=0
maxPotentialEnergy=0

for idx, pkwt in enumerate(energetics):
    if(max(pkwt[0])>maxKineticEnergy):
        maxKineticEnergy = max(pkwt[0])
    if(max(pkwt[1])>maxPotentialEnergy):
        maxPotentialEnergy = max(pkwt[1])



color0 = np.array([   0,   0,   0]) #Nominal
color1 = np.array([   1,   0,   0]) #Largest height perturbation
colorPlot = np.array([0,0,0])

axCoM = plt.subplot(gsBasic[0])
for idx, traj in enumerate(trajectories):
    n01 = float(max(idx,0))/max(float(len(trajectories)-1),1)
    colorPlot = color0*(1-n01) + color1*(n01)
    axCoM.plot(traj.y[0], traj.y[1],
            color=(colorPlot[0],colorPlot[1],colorPlot[2]),
            label=ground_heights[idx])
    plt.ylim((0,maxHeight))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('CoM Trajectory')
axCoM.spines['top'].set_visible(False)
axCoM.spines['right'].set_visible(False)
plt.legend(loc='upper left')


axAF = plt.subplot(gsBasic[1])
for idx, sda in enumerate(springDamperActuatorForces):
    n01 = float(max(idx,0))/max(float(len(trajectories)-1),1)
    colorPlot = color0*(1-n01) + color1*(n01)
    traj = trajectories[idx]
    axAF.plot(traj.t, sda[2],
            color=(colorPlot[0],colorPlot[1],colorPlot[2]),
            label=ground_heights[idx])
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Actuator Forces')
axAF.spines['top'].set_visible(False)
axAF.spines['right'].set_visible(False)

axDF = plt.subplot(gsBasic[2])
for idx, sda in enumerate(springDamperActuatorForces):
    n01 = float(max(idx,0))/max(float(len(trajectories)-1),1)
    colorPlot = color0*(1-n01) + color1*(n01)
    traj = trajectories[idx]
    axDF.plot(traj.t, sda[1],
            color=(colorPlot[0],colorPlot[1],colorPlot[2]),
            label=ground_heights[idx])
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Damping Forces')
axDF.spines['top'].set_visible(False)
axDF.spines['right'].set_visible(False)

axLL = plt.subplot(gsBasic[3])
for idx, ll in enumerate(legLength):
    n01 = float(max(idx,0))/max(float(len(trajectories)-1),1)
    colorPlot = color0*(1-n01) + color1*(n01)
    traj = trajectories[idx]
    axLL.plot(traj.t, ll,
            color=(colorPlot[0],colorPlot[1],colorPlot[2]),
            label=ground_heights[idx])
    plt.xlabel('Time (s)')
    plt.ylabel('Length (m)')
    plt.title('Leg Length')
axLL.spines['top'].set_visible(False)
axLL.spines['right'].set_visible(False)
plt.legend(loc='upper left')

axLF = plt.subplot(gsBasic[4])
for idx, sda in enumerate(springDamperActuatorForces):
    n01 = float(max(idx,0))/max(float(len(trajectories)-1),1)
    colorPlot = color0*(1-n01) + color1*(n01)
    traj = trajectories[idx]
    axLF.plot(traj.t, sda[0],
            color=(colorPlot[0],colorPlot[1],colorPlot[2]),
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
    n01 = float(max(idx,0))/max(float(len(energetics)-1),1)
    colorPlot = color0*(1-n01) + color1*(n01)
    axLW.plot(traj.t, pkwt[2]+pkwt[3],color=(colorPlot[0],colorPlot[1],colorPlot[2]))
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
