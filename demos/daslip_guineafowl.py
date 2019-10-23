import models.daslip as model
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import pickle
from matplotlib import gridspec

# Functions to read in a MAT file and then convert it to an easily accessible 
# dictionary. These functions come thanks to the post:
# https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

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

#===============================================================================
#
# README
#   
#  The fields below match the experimental data (not public) from the MAT
#  file GF_Drop_AllSteps_SIUnits.mat that contains raw data from Blum et al.
#  Note that this MAT file contains almost everything needed to simulate a
#  step: the leg stiffness is not contained in this MAT file. To match Blum
#  et al. as closely as possible we solve for leg stiffness by adjusting the
#  leg stiffness until a limit cycle for the slip model is found. 
# 
#  The iteration begins with a leg stiffness estimate which is evaluated using 
#  maximum change in leg force divided by the mean leg compression. The leg 
#  length is not equal between touch-down and toe-off, here we take the average 
#  of these two long lengths to define the maximum leg length. To see see this
#  in more detail set flag_plotLegCompressionCycle to true. From this plot it
#  is clear that the SLIP model is a pretty rough approximation of the the 
#  guinea fowl. M.Millard would be very interested to see the center-of-pressure
#  recordings, if there are any, to see how this compares to the SLIP model.
#
#  Coming back to the struct, all of the step parameters and timeseries data is 
#  contained in the fields
#  
#  'Step'
#       'Bird1',...,'Bird5'
#           m     : mass of the bird in kg
#           L0    : leg length of the bird in meters
#           ObsH0 : data from flat running trials
#           ObsH4 : data from 4 cm drop-step trials
#           ObsH6 : data from 6 cm drop-step trials
#
#   Within ObsH0 .. 4 are the following fields
#       STm3 : 3 steps before the drop (empty for ObsH0)
#       STm2 : 2 steps " ... "
#       STm1 : 1 step " ... "
#       STze : drop step (all data for flat running trials here)
#       STp1 : 1 step after the drop (empty for ObsH0)
#       STp2 : 2 steps after " ... "
#       STp3 : 3 steps after " ... "
#  
#   Within STm3 ... STp3 are many fields related to step parameters as well as 
#   time series data recorded from each step. In this script I will adopt the
#   variable naming convention to match Blum et al. to minimize confusion for 
#   those working with this script and the data from Blum et al. For details 
#   on each variable name please see GFData_stepVariables_READEME.rtf and the 
#   paper.
#
#   To simulate a specific trial you need to choose:
# 
#   birdNo      : select one of 'Bird1' ... 'Bird5'
#   observation : select one of 'ObsH0' ... 'ObsH6'
#   stepType    : select one of 'STm3' ... 'STp3'
#   stepNo      : select one of the step indices between 0 and maxSteps
#
#   Note:   maxSteps varies for each combination of birdNo, observation, and 
#           stepType. The value of maxSteps for the selected combination is 
#           printed to the terminal prior to simulation.
#
# Blum Y, Vejdani HR, Birn-Jeffery AV, Hubicki CM, Hurst JW, Daley MA. 
# Swing-leg trajectory of running guinea fowl suggests task-level priority 
# of force regulation rather than disturbance rejection. PloS one. 2014 Jun 
# 30;9(6):e100399.
#  

# =============================================================================
#  Set configuration here
birdNo      = 'Bird1' #'Bird1','Bird2','Bird3''Bird4''Bird5'
observation = 'ObsH0' #'ObsH0, 'ObsH4', 'ObsH6'
stepType    = 'STze'  #'STm3','STm2','STm1','STze','STp1','STp2','STp3'
stepNo      = 0       #The trial number which varies for each combination of 
                      #bird, observation, and step type

flag_plotLegCompressionCycle = True

#===============================================================================
# Rarely should these variables be touched
folderBlum2014 = "data/BlumVejdaniBirnJefferyHubickiHurstDaley2014"
fileName       = "/GF_Drop_AllSteps_SIUnits"

flag_readMATFile = False #This is slow, so the MAT file contents are pickled.
                        #Once the pickle file exists use it instead

#===============================================================================
# WARNING: You should not have to adjust anything below. Experts only.
if(flag_readMATFile):
    dataMat = loadmat(folderBlum2014+fileName+".mat")
    pklFileName = open(folderBlum2014+fileName+".pkl",'wb')
    pickle.dump(dataMat, pklFileName)
    pklFileName.close()

pklFileName = open(folderBlum2014+fileName+".pkl",'rb')
dataBlum2014SIUnits = pickle.load(pklFileName)
pklFileName.close()


print('Selected:'+birdNo+' '+observation+' '+stepType)
totalTrials  = np.shape(dataBlum2014SIUnits['Step'][birdNo][observation][stepType]['aTD'])[0]

print('Total recorded trails:',totalTrials)
assert(stepNo < totalTrials and stepNo >= 0)

#Go get all of the parameters necessary for the simulation:

#Physical Parameters
gravity = 9.81
m      = dataBlum2014SIUnits['Step'][birdNo]['m']
L0     = dataBlum2014SIUnits['Step'][birdNo]['L0']

#Step Parameters
yApex  = dataBlum2014SIUnits['Step'][birdNo][observation][stepType]['yApex'][stepNo]
vApex  = dataBlum2014SIUnits['Step'][birdNo][observation][stepType]['vApex'][stepNo]
LTD    = dataBlum2014SIUnits['Step'][birdNo][observation][stepType]['LTD'][stepNo]
LdotTD = dataBlum2014SIUnits['Step'][birdNo][observation][stepType]['LdotTD'][stepNo]
aTDDegrees  = dataBlum2014SIUnits['Step'][birdNo][observation][stepType]['aTD'][stepNo]
adotTDDegrees = dataBlum2014SIUnits['Step'][birdNo][observation][stepType]['adotTD'][stepNo]

aTD         = (aTDDegrees  - 90)*(np.pi/180)
adotTD      = (   adotTDDegrees)*(np.pi/180)

#Time series data: used to extract out an initial leg stiffness. The final 
#leg stiffness used in simulation is solved by iteration as in Blum et al.
#This data is stored as a list-of-lists. There is no easy way that I can find
#to grab a single column, so I'm just copying it over element by element.
ele = np.shape(dataBlum2014SIUnits['Step'][birdNo][observation][stepType]['t'])[0]

timeSeries = np.zeros((ele,1))
LSeries    = np.zeros((ele,1))
fLegSeries = np.zeros((ele,1))

for i in range(0,ele):
    timeSeries[i,0] = dataBlum2014SIUnits['Step'][birdNo][observation][stepType]['t'][i][stepNo]
    LSeries[i,0]    = dataBlum2014SIUnits['Step'][birdNo][observation][stepType]['LStance'][i][stepNo]
    fLegSeries[i,0] = dataBlum2014SIUnits['Step'][birdNo][observation][stepType]['FLeg'][i][stepNo] 

#Go get the index of toe-off
fLow = np.max(fLegSeries)*0.01
idxTD = 0
idxT0 = 0

for i in range(np.shape(fLegSeries)[0]-1,1,-1):
    dfL = fLegSeries[i-1,0]-fLow
    dfR = fLegSeries[i,0]-fLow
    if( dfL*dfR <= 0. and idxT0 == 0):
        if(idxT0 == 0):
            idxT0 = i
        
#The leg is shorter at toe-off than touch down. To get the leg compression we 
# subtract the average leg length at 0 force from the minimum leg length during 
# stance        
meanLTD              = 0.5*(LSeries[idxTD,0]+LSeries[idxT0,0])
legCompression       = np.min(LSeries) - meanLTD 
maxLegForce          = np.max(fLegSeries)
legStiffnessEstimate = -maxLegForce/legCompression 



p = {'mass': m,                             # kg
     'stiffness': legStiffnessEstimate,     # K : N/m
     'spring_resting_length': LTD,          # m
     'gravity': gravity,                    # N/kg
     'angle_of_attack': aTD,                # rad
     'actuator_resting_length': 0.,                 # m
     'actuator_force': [],                   # * 2 x M matrix of time and force
     'actuator_force_period': 10,                   # * s
     'activation_delay': 0.0,         # * a delay for when to start activation
     'activation_amplification': 1.0,
     'constant_normalized_damping': 0.75,           # * s : D/K : [N/m/s]/[N/m]
     'linear_normalized_damping_coefficient': 0.1,  # * A: s/m : D/F : [N/m/s]/N : 0.0035 N/mm/s -> 3.5 1/m/s from Kirch et al. Fig 12
     'linear_minimum_normalized_damping': 0.01,     # *   1/A*(kg*N/kg) :
     'swing_velocity': adotTD,                      # rad/s (set by calculation)
     'angle_of_attack_offset': 0,                   # rad   (set by calculation)
     'swing_extension_velocity': LdotTD,            # m/s
     'swing_leg_length_offset' : 0}                 # m (set by calculation) 
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

# * Solve for nominal open-loop trajectories

legStiffnessSearchWidth = p['stiffness']*0.5

limit_cycle_options = {'search_initial_state' : False,
                       'state_index'          : 0,
                       'state_search_width'   : 0,
                       'search_parameter'     : True,
                       'parameter_name'       : 'stiffness',
                       'parameter_search_width': legStiffnessSearchWidth}

print(p['stiffness'],' N/m :Leg stiffness prior to fitting')
x0, p = model.create_open_loop_trajectories(x0, p, limit_cycle_options)
print(p['stiffness'],' N/m :Leg stiffness prior after fitting')
p['x0'] = x0.copy()
if(flag_plotLegCompressionCycle == True):
    springForce = np.zeros((ele,1))
    springForceLimitCycle = np.zeros((ele,1))
    slipL = (p['spring_resting_length']+p['actuator_resting_length'])
    for i in range(0,ele):
        if( LSeries[i,0] <=  meanLTD):
            springForce[i,0] = - legStiffnessEstimate*(LSeries[i,0]-meanLTD)
        if( LSeries[i,0] <= slipL):
            springForceLimitCycle[i,0] = - p['stiffness']*(LSeries[i,0]-slipL)
    figBlum2014 = plt.figure(figsize=(3,3))
    plt.rc('font', family='serif')
    plt.plot(LSeries,fLegSeries,color=(0.,0.,0.),label='Blum 2014')
    plt.plot(LSeries,springForce,color=(0.,0.,1.),label='Stiffness (Init.)')
    plt.plot(LSeries,springForceLimitCycle,color=(1.,0.,0.),
    label='Stiffness (Limit Cycle)')    
    plt.xlabel('Leg Length (m)')
    plt.ylabel('Leg Force (N)')
    plt.title('Leg Compression Cycle')

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
    sda = model.compute_leg_forces(traj.t, traj.y, p)
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
