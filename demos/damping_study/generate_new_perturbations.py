import numpy as np
import models.daslip as model
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

    # start_idx = np.argwhere(~np.isclose(p['actuator_force'][1], 0))[0]
    # time_to_activation = p['actuator_force'][0, start_idx]

    trajectories = list()
    for height in ground_heights:
        x0[-1] = height
        # time_to_touchdown = np.sqrt(2*(x0[5] - x0[-1])/p['gravity'])
        # p['activation_delay'] = time_to_touchdown - time_to_activation
        trajectories.append(model.step(x0, p))
    x0[-1] = 0.0  # reset x0 back to 0
    return trajectories

set_files = ['../../data/guineafowl/guineafowl_0.0010.pickle',
             '../../data/guineafowl/guineafowl_0.0050.pickle',
             '../../data/guineafowl/guineafowl_0.0100.pickle',
             '../../data/guineafowl/guineafowl_0.0200.pickle',
             '../../data/guineafowl/guineafowl_0.0300.pickle',
             '../../data/guineafowl/guineafowl_0.0400.pickle',
             '../../data/guineafowl/guineafowl_0.0500.pickle',
             '../../data/guineafowl/guineafowl_0.0600.pickle',
             '../../data/guineafowl/guineafowl_0.0700.pickle',
             '../../data/guineafowl/guineafowl_0.0800.pickle',
             '../../data/guineafowl/guineafowl_0.0900.pickle',
             '../../data/guineafowl/guineafowl_0.1000.pickle',
             '../../data/guineafowl/guineafowl_0.1100.pickle',
             '../../data/guineafowl/guineafowl_0.1200.pickle',
             '../../data/guineafowl/guineafowl_0.1300.pickle',
             '../../data/guineafowl/guineafowl_0.1400.pickle',
             '../../data/guineafowl/guineafowl_0.1500.pickle',
             '../../data/guineafowl/guineafowl_0.1600.pickle',
             '../../data/guineafowl/guineafowl_0.1700.pickle',
             '../../data/guineafowl/guineafowl_0.1800.pickle',
             '../../data/guineafowl/guineafowl_0.1900.pickle']


perturbation_vals = np.arange(-0.07, 0.03, 0.0025)

rem_string = len('.pickle')
data = list()
for filename in set_files:
    infile_1 = open(filename, 'rb')
    data = pickle.load(infile_1)
    p = data['p']
    x0 = data['x0']
    # x0 = pickle.load(infile_1)
    trajectories = get_step_trajectories(x0, p, perturbation_vals)
    infile_1.close()

    new_filename = filename[0:len(filename)-rem_string]+'_trajs.pickle'
    data2save = {"trajectories": trajectories}
    outfile = open(new_filename, 'wb')
    pickle.dump(data2save, outfile)
    outfile.close()