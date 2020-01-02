import numpy as np
import models.parslip as model
import pickle


def get_step_trajectories(x0, p, ground_heights=None):
    '''
    helper function to apply a battery of ground-height perturbations.
    returns a list of trajectories.
    '''

    if ground_heights is None:
        total_leg_length = p['resting_length']
        total_leg_length += p['actuator_resting_length']
        ground_heights = np.linspace(0, -0.5*total_leg_length, 10)
    x0 = model.reset_leg(x0, p)
    trajectories = list()
    for height in ground_heights:
        x0[-1] = height
        trajectories.append(model.step(x0, p))
    x0[-1] = 0.0  # reset x0 back to 0
    return trajectories


foldername = 'all_birds/'
set_files = ['all_birds_0.01.pickle',
             'all_birds_0.06.pickle',
             'all_birds_0.1.pickle',
             'all_birds_0.1.pickle',
             'all_birds_0.15.pickle',
             'all_birds_0.2.pickle']

perturbation_vals = np.arange(-0.1, 0.1,0.005)

rem_string = len('.pickle')
data = list()
for filename in set_files:
    infile_1 = open(foldername+filename, 'rb')
    data = pickle.load(infile_1)
    p = data['p']
    x0 = data['x0']
    # x0 = pickle.load(infile_1)
    trajectories = get_step_trajectories(x0, p, perturbation_vals)
    infile_1.close()

    new_filename = filename[0:len(filename)-rem_string]+'_trajs2.pickle'
    data2save = {"trajectories": trajectories}
    outfile = open(foldername+new_filename, 'wb')
    pickle.dump(data2save, outfile)
    outfile.close()