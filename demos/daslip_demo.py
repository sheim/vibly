import models.daslip as model
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import gridspec

FLAG_SAVE_PLOTS = False
height_perturbation = 0.1
# Human readable state index labels and enums.
state_labels = ['x_c', ' y_c', '\dot{x}_c', '\dot{y}_c', 'x_f', 'y_f', 'l_a',
                'w_a', 'w_d', 'y_g']

# slip & daslip: shared states
x_c = 0  # x_c  : horizontal coordinate of the center-of-mass (c) (+ right)
y_c = 1  # y_c  : vertical '...' (+ up)
vx_c = 2  # vx_c : horizontal velocity of the center-of-mass (+ right)
vy_c = 3  # vy_c : vertical velocity '...' (+ up)
x_f = 4  # x_f : horizontal coordinate from c to the foot (f)
y_f = 5  # y_f : vertical coordinate '...'
# daslip: appended extra states
la = 6  # la    : length of the actuator
wa = 7  # wa    : work of the actuator
wd = 8  # wd
y_g = -1  # ground height

slip_model = 0  # spring-loaded-inverted-pendulum
daslip_model = 1  # damper-actuator-spring-loaded inverted pendulum

# Model types
#
#
# Slip:                   Daslip (damper-actuator-slip)
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
# The damping model for the daslip has two options:
#
# 0. Constant fixed damping.
#
# 1. Damping that varies linearly with the force of the actuator. This specific
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
p = {'model_type': slip_model,            # 0 (slip), 1 (daslip)
     'mass': 80,                          # kg
     'stiffness': 8200.0,                 # K : N/m
     'spring_resting_length': 0.9,        # m
     'gravity': 9.81,                     # N/kg
     'angle_of_attack': 1/5*np.pi,        # rad
     'actuator_resting_length': 0.1,      # m
     'actuator_force': [],                # * 2 x M matrix of time and force
     'actuator_force_period': 10,         # * s
     'damping_type': 0,                   # * 0 (constant), 1 (linear-with-force)
     'constant_normalized_damping': 0.75,          # *    s   : D/K : [N/m/s]/[N/m]
     'linear_normalized_damping_coefficient': 3.5,  # * A: s/m : D/F : [N/m/s]/N : 0.0035 N/mm/s -> 3.5 1/m/s from Kirch et al. Fig 12
     'linear_minimum_normalized_damping': 0.05,    # *   1/A*(kg*N/kg) :
     'swing_type': 0,                    # 0 (constant angle of attack), 1 (linearly varying angle of attack)
     'swing_leg_norm_angular_velocity':  1.1,  # [1/s]/[m/s] (omega/(vx/lr))
     'swing_leg_angular_velocity': 0,   # rad/s (set by calculation)
     'angle_of_attack_offset': 0}        # rad   (set by calculation)

# * Initialization: Slip & Daslip

x0_slip = np.array([0, 0.85, 5.5, 0, 0, 0, 0])
x0_daslip = np.array([0, 0.85, 5.5, 0, 0, 0,
                     p['actuator_resting_length'], 0, 0, 0])

x0_slip = model.reset_leg(x0_slip, p)
p['total_energy'] = model.compute_total_energy(x0_slip, p)


# * Limit Cycle: nominal Slip

search_width = np.pi*0.25
swing_type = p['swing_type']

# To compute the limit cycle make sure the swing type is fixed to constant.
p['swing_type'] = 0
p_lc, success = model.find_limit_cycle(x0_slip, p, 'angle_of_attack',
                                       search_width)

# Get a high-resolution state trajectory of the limit cycle
x0_slip = model.reset_leg(x0_slip, p_lc)
p_lc['total_energy'] = model.compute_total_energy(x0_slip, p_lc)
sol_slip = model.step(x0_slip, p_lc)

# If swing_type is in retraction mode, update the angle_of_attack_offset, and
# the angular velocity of the swing leg
if swing_type == 1:
    t_contact = sol_slip.t_events[1][0]
    p_lc['swing_type']=swing_type
    p_lc['swing_leg_angular_velocity'] = (
            -(p_lc['swing_leg_norm_angular_velocity']*x0_slip[vx_c])/
            (p_lc['spring_resting_length']+p_lc['actuator_resting_length']))
    p_lc['angle_of_attack_offset'] = -t_contact*p_lc['swing_leg_angular_velocity']
    # Update the model.step solution
    x0_slip = model.reset_leg(x0_slip, p_lc)
    p_lc['total_energy'] = model.compute_total_energy(x0_slip, p_lc)
    sol_slip = model.step(x0_slip, p_lc)

n = len(sol_slip.t)
slip_spring_deflection = np.zeros((1,n))
slip_leg_force = np.zeros((1,n))
for i in range(0, n):
    slip_spring_deflection[0,i] = (model.compute_spring_length(sol_slip.y[:,i], p_lc)
    -p_lc['spring_resting_length'])
    slip_leg_force[0,i] = model.compute_leg_force(sol_slip.y[:,i],p_lc)


#Evaluate the energetics of the solution. Note
#   t: kinetic energy
#   v: potential energy
#   w: work
tvw_slip = model.compute_potential_kinetic_work_total(sol_slip.y, p_lc)
tvw_slip_error = np.zeros(np.shape(tvw_slip)[1])
for i in range(0, len(tvw_slip_error)):
    tvw_slip_error[i] = tvw_slip[3, i] - tvw_slip[3, 0]

#Print the difference between the starting and ending state
print('Limit Cycle State Error: x(start) - x(end)')
print('  Note: state erros in x_com and x_foot will not go to zero')
for i in range(0, 4):
  print('  '+state_labels[i]+'  '+str(sol_slip.y[i,0]-sol_slip.y[i,-1]))

#===============================================================================
# Perturbed SLIP
#===============================================================================

#Get a high-resolution state trajectory of the limit cycle
x0_slip[y_c] = x0_slip[y_c] + height_perturbation
x0_slip = model.reset_leg(x0_slip, p_lc)
p['total_energy'] = model.compute_total_energy(x0_slip, p_lc)
sol_slip_pert = model.step(x0_slip, p_lc)

n = len(sol_slip_pert.t)
slip_spring_deflection_pert = np.zeros((1, n))
for i in range(0, n): #TODO: pythonize
    slip_spring_deflection_pert[0,i] = (model.compute_spring_length(
            sol_slip_pert.y[:,i], p_lc) - p_lc['spring_resting_length'])

#Evaluate the energetics of the solution. Note
#   t: kinetic energy
#   v: potential energy
#   w: work
tvw_slip_pert = model.compute_potential_kinetic_work_total(sol_slip_pert.y, p_lc)
tvw_slip_error_pert = np.zeros(np.shape(tvw_slip_pert)[1])
for i in range(0, len(tvw_slip_error_pert)):
    tvw_slip_error_pert[i] = tvw_slip_pert[3, i] - tvw_slip_pert[3, 0]

#Print the difference between the starting and ending state
print('Perturbed SLIP State Error: x(start) - x(end)')
# print('  Note: state errors in x_com and x_foot will not go to zero')
for i in range(1, 4):
  print('  ' + state_labels[i] + '  ' + str(sol_slip_pert.y[i,0]
        - sol_slip_pert.y[i,-1]))

#===============================================================================
# Daslip
#===============================================================================

#Use the limit cycle to form the actuator open-loop force time series
actuator_time_force = model.create_force_trajectory(sol_slip, p_lc)
p_daslip = p_lc.copy() # python doesn't do deep copies by default
p_daslip['model_type'] = daslip_model
p_daslip['actuator_force'] = actuator_time_force
p_daslip['actuator_force_period'] = np.max(actuator_time_force[0, :])

# Solve for a nominal model.step of the daslip model. The output trajectory
# should perfectly match the limit cycle model.step of the slip model
x0_daslip[y_c] = x0_daslip[y_c] + height_perturbation
x0_daslip = model.reset_leg(x0_daslip, p_daslip)
p_daslip['total_energy'] = model.compute_total_energy(x0_daslip, p_daslip)
sol_daslip = model.step(x0_daslip, p_daslip)

#Evaluate the energetics of the solution
tvw_daslip = model.compute_potential_kinetic_work_total(sol_daslip.y,p_daslip)
tvw_daslip_error =np.zeros(np.shape(tvw_daslip)[1])
for i in range(0, len(tvw_daslip_error)):
    tvw_daslip_error[i]=tvw_daslip[3,i]-tvw_daslip[3,0]

n = len(sol_daslip.t)
daslip_spring_deflection = np.zeros((1,n))
daslip_leg_force = np.zeros((1,n))
for i in range(0, n):
    daslip_spring_deflection[0,i] = (model.compute_spring_length(sol_daslip.y[:,i],
        p_daslip) - p_daslip['spring_resting_length'])
    daslip_leg_force[0,i] = model.compute_leg_force(sol_daslip.y[:,i], p_daslip)


#===============================================================================
#Kinematic Error: Slip vs. Daslip
#===============================================================================

sol_daslip_err = np.zeros((len(sol_daslip.t)))
pos_slip  = np.zeros(2)
pos_daslip= np.zeros(2)
pos_err   = np.zeros(2)

for i in range(0, len(sol_daslip.t)):
    for j in range(0,2):
        pos_daslip[j] = sol_daslip.y[j,i]
        pos_slip[j]   = np.interp(sol_daslip.t[i],sol_slip_pert.t,
                sol_slip_pert.y[j])
        pos_err[j]    = pos_daslip[j]-pos_slip[j]

    sol_daslip_err[i] = np.linalg.norm(pos_err)


#===============================================================================
#Plotting
#===============================================================================

plt.rc('font',family='serif')
color_slip   = [0.5, 0.5, 0.5]
color_slip_pert = [1, 0.4, 0.4]
color_daslip = [0., 0., 0.]
color_ground = [0., 0., 0.]

color_t  = [1., 0., 0.]
color_v  = [0., 0., 1.]
color_w  = [1., 0., 1.]
color_tvw= [0., 0., 0. ]

linewidth_thick = 2.0
linewidth_thin=1.0

plotWidth  = 3
plotHeight = 3

#========================================
#Basic Plots
#========================================
plt.figure(figsize=(plotWidth*3,plotHeight*2))

gsBasic= gridspec.GridSpec(2, 2, width_ratios=[2, 1])

ax=plt.subplot(gsBasic[0])
ax.plot(sol_slip.y[x_c], sol_slip.y[y_c],
        color=color_slip, linewidth=linewidth_thick, label ='SLIP')
ax.plot(sol_slip_pert.y[x_c], sol_slip_pert.y[y_c],
        color=color_slip_pert, linewidth=linewidth_thick, label ='SLIP perturbed')

#Plot the leg when events are triggered
contact_event = False
toe_off_event = False
for i in range(0, len(sol_slip.t_events)):
    if len(sol_slip.t_events[i]) > 0:
        idx = (np.abs( sol_slip.t - sol_slip.t_events[i][0])).argmin()
        if(contact_event == True and toe_off_event == False):
            idx = idx-1
            toe_off_event = True
        ax.plot([sol_slip.y[x_c,idx], sol_slip.y[x_f,idx]],
                [sol_slip.y[y_c,idx], sol_slip.y[y_f,idx]],
                color=color_slip, linewidth=linewidth_thin)
        contact_event = True
ax.plot(sol_daslip.y[x_c],sol_daslip.y[y_c],
        color=color_daslip, linewidth=linewidth_thin,
        linestyle='--', label ='DASLIP')

#Plot the leg when events are triggered
contact_event = False
toe_off_event = False
for i in range(0, len(sol_daslip.t_events)):
    if len(sol_daslip.t_events[i]) > 0:
        idx = (np.abs( sol_daslip.t - sol_daslip.t_events[i][0])).argmin()
        if(contact_event == True and toe_off_event == False):
            idx = idx-1
            toe_off_event = True
        ax.plot([sol_daslip.y[x_c,idx], sol_daslip.y[x_f,idx]],
                [sol_daslip.y[y_c,idx], sol_daslip.y[y_f,idx]], 
                color=color_daslip, linewidth=linewidth_thin,
                linestyle='--')
        contact_event = True

plt.legend()
plt.legend(frameon=True)
plt.plot([np.min(sol_slip.y[x_c]),np.max(sol_slip.y[x_c])],
         [0.,0.], color=color_ground,linestyle='-',linewidth=0.5)

plt.xlabel('$'+state_labels[0]+'$')
plt.ylabel('$'+state_labels[1]+'$')
plt.title('Limit Cycle Trajectory')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.axis('equal')
plt.tight_layout()

#Open-Loop Function
ax=plt.subplot(gsBasic[1])
ax.plot(actuator_time_force[0],
            actuator_time_force[1], color=[0.,0.,0.])

plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('DASLIP: Actuator Open Loop Force')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.tight_layout()

if(FLAG_SAVE_PLOTS==True):
    plt.savefig('fig_slip_daslip_basic.pdf')
#Ground forces of the two models
ax=plt.subplot(gsBasic[2])
ax.plot(sol_slip.t, slip_leg_force[0],
        color=color_slip, linewidth=linewidth_thick, label ='SLIP')
ax.plot(sol_daslip.t, daslip_leg_force[0],
        color=color_daslip, linewidth=linewidth_thin,
        linestyle='--', label ='DASLIP-Total')


plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Leg Force')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.tight_layout()

#Actuator Length
ax=plt.subplot(gsBasic[3])
ax.plot([np.min(sol_slip.t),np.max(sol_slip.t)],
        [p['actuator_resting_length'],p['actuator_resting_length']],
        color=color_slip, linewidth=linewidth_thick, label ='SLIP')
ax.plot(sol_daslip.t, sol_daslip.y[la],
        color=color_daslip, linewidth=linewidth_thin,
        linestyle='--', label ='DASLIP')

plt.xlabel('Time (s)')
plt.ylabel('Force (m)')
plt.title('Actuator Length')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.tight_layout()

if(FLAG_SAVE_PLOTS==True):
    plt.savefig('fig_slip_daslip_basic.pdf')


#========================================
#Debugging Plots
#========================================

plt.figure(figsize=(plotWidth*3,plotHeight*2))
gsDebug= gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1])

#Kinematic Error Between Slip-Daslip
ax=plt.subplot(gsDebug[0])
plt.plot(sol_daslip.t, sol_daslip_err,
         color=color_daslip, linewidth=linewidth_thin,label ='DASLIP')

plt.xlabel('Time (s)')
plt.ylabel('Distance')
plt.title('SLIP-DASLIP Error')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.tight_layout()

#Energy Balance
ax=plt.subplot(gsDebug[1])
ax.plot(sol_slip.t, tvw_slip_error,
        color=color_slip, linewidth=linewidth_thick, label ='SLIP')
ax.plot(sol_slip_pert.t, tvw_slip_error_pert, color=color_slip_pert,
        linewidth=linewidth_thick, label ='SLIP perturbed')
ax.plot(sol_daslip.t, tvw_daslip_error, color=color_daslip,
        linewidth=linewidth_thin, linestyle='--', label ='DASLIP')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('$T+V-W$')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.tight_layout()

# Spring Lengths
ax=plt.subplot(gsDebug[2])
ax.plot(sol_slip.t, slip_spring_deflection[0],
        color=color_slip, linewidth=linewidth_thick, label ='SLIP')
ax.plot(sol_slip_pert.t, slip_spring_deflection_pert[0], color=color_slip_pert,
        linewidth=linewidth_thick, label ='SLIP perturbed')
ax.plot(sol_daslip.t, daslip_spring_deflection[0], color=color_daslip,
        linewidth=linewidth_thin, linestyle='--', label ='DASLIP')
plt.xlabel('Time (s)')
plt.ylabel('Length (m)')
plt.title('Spring Deflection')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.tight_layout()

#Kinetic Energy
ax=plt.subplot(gsDebug[3])
ax.plot(sol_slip.t, tvw_slip[0],
        color=color_slip, linewidth=linewidth_thick, label ='SLIP')
ax.plot(sol_slip_pert.t, tvw_slip_pert[0], color=color_slip_pert,
        linewidth=linewidth_thick, label ='SLIP perturbed')
ax.plot(sol_daslip.t, tvw_daslip[0], color=color_daslip,
        linewidth=linewidth_thin, linestyle='--', label ='DASLIP')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Kinetic Energy')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.tight_layout()

#Potential Energy
ax=plt.subplot(gsDebug[4])
ax.plot(sol_slip.t, tvw_slip[1],
        color=color_slip, linewidth=linewidth_thick, label ='SLIP')
ax.plot(sol_slip_pert.t, tvw_slip_pert[1], color=color_slip_pert, 
        linewidth=linewidth_thick, label ='SLIP perturbed')
ax.plot(sol_daslip.t,tvw_daslip[1], color=color_daslip,
        linewidth=linewidth_thin, linestyle='--', label ='DASLIP')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Potential Energy')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.tight_layout()

#Work
ax=plt.subplot(gsDebug[5])
ax.plot(sol_slip.t, tvw_slip[2],
        color=color_slip, linewidth=linewidth_thick, label ='SLIP')
ax.plot(sol_daslip.t,tvw_daslip[2], color=color_daslip,
        linewidth=linewidth_thin, linestyle='--', label ='DASLIP actuator work')
ax.plot(sol_daslip.t,tvw_daslip[3], color=color_daslip,
        linewidth=linewidth_thin, linestyle='-.', label ='DASLIP damper work')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Work')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.tight_layout()

if(FLAG_SAVE_PLOTS==True):
    plt.savefig('fig_slip_daslip_detail.pdf')

plt.show()