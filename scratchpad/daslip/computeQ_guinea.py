import models.daslip as model
import numpy as np
import matplotlib.pyplot as plt
import pickle
from ttictoc import TicToc

import viability as vibly
# * helper functions
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
        A recursive function which constructs from matobjects nested
        dictionaries
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


#  data wrangling. See demos/daslip_guineafowl.py for more details
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
folderBlum2014 = "../../data/BlumVejdaniBirnJefferyHubickiHurstDaley2014"
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

# * Set-up P maps for comutations
p_map = model.poincare_map
p_map.p = p
p_map.x = x0.copy()

# * choose high-level represenation
# p_map.sa2xp = model.sa2xp_amp
p_map.sa2xp = model.sa2xp_y_xdot_timedaoa
p_map.xp2s = model.xp2s_y_xdot

# * set up grids
s_grid_height = np.linspace(0.1, 0.3, 11)  # 21)
s_grid_velocity = np.linspace(1.5, 3.5, 11)  # 51)
s_grid = (s_grid_height, s_grid_velocity)
a_grid_aoa = np.linspace(10/180*np.pi, 70/180*np.pi, 21)
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