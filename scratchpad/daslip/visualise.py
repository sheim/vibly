import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

path     = '..\\..\\data\\damping\\'

# filename    = 'all_birds'
filename    = 'bird4'

Show = True
Save = False

### Normalisation
gravity = 9.81
params 	= np.load('ref_params.npz')
m 		= params['mass']
LTD 	= params['spring_resting_length']

### Reference limit cycle
reference = np.load('stiffness.npz')
## Average of the birds
if filename == 'all_birds':
	yApex     = reference['y_all']*LTD
	vApex  	  = reference['v_all']*np.sqrt(LTD*gravity)
## Single bird	
elif filename[:4] == 'bird':
	yApex     = reference['y_bird'][int(filename[4])]*LTD
	vApex  	  = reference['v_bird'][int(filename[4])]*np.sqrt(LTD*gravity)

### Space grids from one of the files
damping     = 0.1
simulations = np.load(path + filename + '\\' + filename + '_' + str(damping) + '.npz')
grids       = simulations['grids']
s_grids     = grids[()]['states'] # this notation is a trick for when numpy has been used to save dictionnaries
a_grids     = grids[()]['actions']
s_grid_height   = s_grids[0]
s_grid_velocity = s_grids[1]
v_min = np.min(s_grid_velocity)
v_max = np.max(s_grid_velocity)
h_min = np.min(s_grid_height)
h_max = np.max(s_grid_height)

### Looping over different values of samping
dampings = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.figure()	
for d, damping in enumerate(dampings):	
    file = path + filename + '\\' + filename + '_' + str(damping) + '.npz'
    if os.path.exists(file):
        simulations = np.load(path + filename + '\\' + filename + '_' + str(damping) + '.npz')
		
        ## Viable 
        S_M = simulations['S_M']
        plt.subplot(2,len(dampings)+1,1+d)
        plt.imshow(S_M, interpolation = 'nearest', origin = 'lower', aspect = 'auto', extent = (v_min,v_max,h_min,h_max), cmap = 'binary', norm = norm)
        plt.title('Damping %0.3g' % damping)
        plt.scatter(vApex, yApex, color = 'b')
		
        ## Immediate failure
        Q_F = simulations['Q_F']
        S_F = np.mean(Q_F, axis = 2)
        plt.subplot(2,len(dampings)+1,len(dampings)+2+d)
        plt.imshow(1-S_F, interpolation = 'nearest', origin = 'lower', aspect = 'auto', extent = (v_min,v_max,h_min,h_max), cmap = 'binary', norm = norm)
        plt.scatter(vApex, yApex, color = 'b')
		
## Reference colorbar with scale from 0 to 1		
plt.subplot(2,len(dampings)+1,len(dampings)+1)
plt.imshow(np.array([np.linspace(0,1,100)]).T, norm = norm, cmap = 'binary')
plt.colorbar()
for s in range(9):
    plt.subplot(2,len(dampings)+1,2+s)
    plt.yticks([])
    plt.subplot(2,len(dampings)+1,len(dampings)+3+s)
    plt.yticks([])
for s in range(10):
    plt.subplot(2,len(dampings)+1,1+s)
    plt.xticks([])
    plt.subplot(2,len(dampings)+1,len(dampings)+2+s)
    plt.xlabel('Apex speed (m/s)')
for s in range(2):
    plt.subplot(2,len(dampings)+1,1+(len(dampings)+1)*s)
    plt.ylabel('Apex height (m)')
	
if Show:	
	plt.show()
if Save:	
	plt.savefig(filename+'dampings.png')

