from slippy.slip import *
import numpy as np
import matplotlib.pyplot as plt
import sys 
# Human readable state index labels and enums. 

daslip_height_purturbation= 0.0

state_labels = ['x_c',' y_c','\dot{x}_c','\dot{y}_c','x_f','y_f','l_a','w_a']
x_c  = 0  # x_c  : horizontal coordinate of the center-of-mass (c) (+ right)
y_c  = 1  # y_c  : vertical '...' (+ up)
vx_c = 2  # vx_c : horizontal velocity of the center-of-mass (+ right)
vy_c = 3  # vy_c : vertical velocity '...' (+ up)
x_f   = 4 # x_f : horizontal coordinate from c to the foot (f) 
y_f   = 5 # y_f : vertical coordinate '...' 
la    = 6 # l    : length of the actuator 
wa    = 7 # w    : work of the actuator

slip_model  = 0 #spring-loaded-inverted-pendulum
daslip_model= 1 #damper-actuator-spring-loaded inverted pendulum

# Model types
#
#
# Slip:                   Daslip (damper-actuator-slip)
#  (c)   -center-of-mass-  (c)         
#   -                     -----            -     -
#   |                     |   |            |     |
#   |        actuator-    a   d -damper    | la  | lr
#   |                     |   |            |     | 
#   -                     -----            -     |  
#   /                       /                    |
#   \ k      -spring-       \ k                  | 
#   /                       /                    | 
#   \                       \                    |          
#    + f     -foot-          + f                 -

#Model parameters
p = { 'model_type':slip_model,
      'mass':80, 
      'stiffness':8200.0, 
      'spring_resting_length':0.9, 
      'gravity':9.81,
      'angle_of_attack':1/5*np.pi, 
      'actuator_resting_length':0.1,
      'actuator_normalized_damping':0.1, 
      'actuator_force':[],
      'actuator_force_period':10}


#===============================================================================
#Initialization: Slip & Daslip
#===============================================================================

x0_slip   = np.array([0, 0.85, 5.5, 0, 0, 0])
x0_daslip = np.array([0, 0.85, 5.5, 0, 0, 0, p['actuator_resting_length'], 0])

x0_slip = reset_leg(x0_slip, p)
p['total_energy'] = compute_total_energy(x0_slip, p)

#===============================================================================
#Limit Cycle: Slip
#===============================================================================
search_width = np.pi*0.25
(p_lc, success) = limit_cycle(x0_slip,p,'angle_of_attack',search_width)

#Get a high-resolution state trajectory of the limit cycle
x0_slip = reset_leg(x0_slip, p_lc)
p['total_energy'] = compute_total_energy(x0_slip, p_lc)
sol_slip = step(x0_slip, p_lc)

#Evaluate the energetics of the solution. Note
#   t: kinetic energy
#   v: potential energy
#   w: work
tvw_slip = compute_potential_kinetic_work_total(sol_slip.y,p_lc)
twv_slip_error =np.zeros(np.shape(tvw_slip)[1])
for i in range(0, len(twv_slip_error)):
    twv_slip_error[i]=tvw_slip[3,i]-tvw_slip[3,0]

#Print the difference between the starting and ending state
print('Limit Cycle State Error: x(start) - x(end)')
print('  Note: state erros in x_com and x_foot will not go to zero')
for i in range(0,6):
  print('  '+state_labels[i]+'  '+str(sol_slip.y[i,0]-sol_slip.y[i,-1]))

#===============================================================================
# Daslip
#===============================================================================

#Use the limit cycle to form the actuator open-loop force time series
actuator_time_force = create_actuator_open_loop_time_series(sol_slip, p_lc)
p_daslip = p_lc
p_daslip['model_type'] = daslip_model
p_daslip['actuator_force'] = actuator_time_force
p_daslip['actuator_force_period']=np.max(actuator_time_force[0,:])

#Solve for a nominal step of the daslip model. The output trajectory
#should perfectly match the limit cycle step of the slip model
x0_daslip[y_c] = x0_daslip[y_c] + daslip_height_purturbation
x0_daslip = reset_leg(x0_daslip, p_daslip)
p_daslip['total_energy'] = compute_total_energy(x0_daslip, p_daslip)
sol_daslip = step(x0_daslip, p_daslip)

#Evaluate the energetics of the solution
tvw_daslip = compute_potential_kinetic_work_total(sol_daslip.y,p_daslip)
twv_daslip_error =np.zeros(np.shape(tvw_daslip)[1])
for i in range(0, len(twv_daslip_error)):
    twv_daslip_error[i]=tvw_daslip[3,i]-tvw_daslip[3,0]


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
        pos_slip[j]   = np.interp(sol_daslip.t[i],sol_slip.t, sol_slip.y[j])
        pos_err[j]    = pos_daslip[j]-pos_slip[j]

    sol_daslip_err[i] = np.linalg.norm(pos_err)


#===============================================================================
#Plotting
#===============================================================================

plt.rc('font',family='serif')
color_slip   = [0.5,0.5,0.5]
color_daslip = [0.,0.,0.]
color_ground = [0.,0.,0.]

color_t  = [1.,0.,0.,]
color_v  = [0.,0.,1.,]
color_w  = [1.,0.,1.,]
color_tvw= [0.,0.,0. ]

linewidth_thick = 2.0
linewidth_thin=1.0

plotWidth  = 8
plotHeight = 6

#Trajectory
plt.figure(figsize=(plotWidth,plotHeight))
ax=plt.subplot(2,3,1)

ax.plot(sol_slip.y[x_c], sol_slip.y[y_c], 
        color=color_slip, linewidth=linewidth_thick,label='slip')
ax.plot(sol_daslip.y[x_c],sol_daslip.y[y_c], 
        color=color_daslip, linewidth=linewidth_thin,label='daslip')

plt.legend()
plt.legend(frameon=False)
plt.plot([np.min(sol_slip.y[x_c]),np.max(sol_slip.y[x_c])], 
         [0.,0.], color=color_ground,linestyle='-',linewidth=0.5)

plt.xlabel('$'+state_labels[0]+'$')
plt.ylabel('$'+state_labels[1]+'$')
plt.title('Limit Cycle Trajectory')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')   

#plt.tight_layout()

#Kinematic Error Between Slip-Daslip
ax=plt.subplot(2,3,2)
plt.plot(sol_daslip.t, sol_daslip_err, 
         color=color_daslip, linewidth=linewidth_thin,label='daslip')

plt.xlabel('Time (s)')
plt.ylabel('Distance')
plt.title('SLIP-DASLIP Error')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')   

#plt.tight_layout()


#Open-Loop Function
ax=plt.subplot(2,3,3)
plt.plot(actuator_time_force[0], 
            actuator_time_force[1], color=[0.,0.,0.])

plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('DASLIP: Actuator Open Loop Force')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')   

#plt.tight_layout()

#Slip System Energy
ax=plt.subplot(2,3,4)
ax.plot(sol_slip.t, tvw_slip[0],  
        color=color_t, linewidth=linewidth_thin,label='T')
ax.plot(sol_slip.t, tvw_slip[1],  
        color=color_v, linewidth=linewidth_thin,label='V')
ax.plot(sol_slip.t, tvw_slip[2],  
        color=color_w, linewidth=linewidth_thin,label='W')
ax.plot(sol_slip.t, tvw_slip[3],  
        color=color_tvw, linewidth=linewidth_thick,label='T+V-W')
#ax.plot(sol_slip.t_events,np.zeros(np.shape(sol_slip.t_events)),mark='*' )

plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('SLIP Energics')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')   

#Daslip System Energy
ax=plt.subplot(2,3,5)
ax.plot(sol_daslip.t, tvw_daslip[0],  
        color=color_t, linewidth=linewidth_thin,label='T')
ax.plot(sol_daslip.t, tvw_daslip[1],  
        color=color_v, linewidth=linewidth_thin,label='V')
ax.plot(sol_daslip.t, tvw_daslip[2],  
        color=color_w, linewidth=linewidth_thin,label='W')
ax.plot(sol_daslip.t, tvw_daslip[3],  
        color=color_tvw, linewidth=linewidth_thick,label='T+V-W')
#ax.plot(sol_daslip.t_events,np.zeros(np.shape(sol_daslip.t_events)),mark='*' )

plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('DASLIP Energics')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  

#Energy Losses
ax=plt.subplot(2,3,6)

ax.plot(sol_slip.t, twv_slip_error, 
        color=color_slip, linewidth=linewidth_thick,label='slip')
ax.plot(sol_daslip.t,twv_daslip_error, 
        color=color_daslip, linewidth=linewidth_thin,label='daslip')
plt.legend()
plt.legend(frameon=False)

plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Energy Balance Error: T+V-W')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')   

#plt.tight_layout()



plt.show()
