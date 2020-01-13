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

    start_idx = np.argwhere(~np.isclose(p['actuator_force'][1], 0))[0]
    time_to_activation = p['actuator_force'][0, start_idx]

    trajectories = list()
    for height in ground_heights:
        x0[-1] = height
        time_to_touchdown = np.sqrt(2*(x0[5] - x0[-1])/p['gravity'])
        p['activation_delay'] = time_to_touchdown - time_to_activation
        trajectories.append(model.step(x0, p))
    x0[-1] = 0.0  # reset x0 back to 0
    return trajectories

# set_files = ['higher/higher_0.025.pickle',
#              'higher/higher_0.050.pickle',
#              'higher/higher_0.075.pickle',
#              'higher/higher_0.100.pickle',
#              'higher/higher_0.125.pickle',
#              'higher/higher_0.150.pickle',
#              'higher/higher_0.175.pickle',
#              'higher/higher_0.2.pickle',
#              'higher/higher_0.25.pickle',
#              'higher/higher_0.5.pickle',
#              'higher/higher_0.75.pickle',
#              'higher/higher_1.0.pickle',
#              'higher/higher_1.25.pickle',
#              'higher/higher_1.5.pickle',
#              'higher/higher_1.75.pickle',
#              'higher/higher_2.0.pickle',
#              'higher/higher_2.25.pickle',
#              'higher/higher_2.5.pickle',
#              'higher/higher_2.75.pickle',
#              'higher/higher_3.0.pickle',
#              'higher/higher_3.25.pickle',
#              'higher/higher_3.5.pickle']

set_files = ['human_daslip/human_daslip_0.025.pickle',
             'human_daslip/human_daslip_0.175.pickle',
             'human_daslip/human_daslip_0.05.pickle',
             'human_daslip/human_daslip_0.2.pickle',
             'human_daslip/human_daslip_0.075.pickle',
             'human_daslip/human_daslip_0.225.pickle',
             'human_daslip/human_daslip_0.1.pickle',
             'human_daslip/human_daslip_0.25.pickle',
             'human_daslip/human_daslip_0.125.pickle',
             'human_daslip/human_daslip_0.275.pickle',
             'human_daslip/human_daslip_0.15.pickle',
             'human_daslip/human_daslip_0.3.pickle']

perturbation_vals = np.arange(-0.2, 0.2, 0.005)

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

    new_filename = filename[0:len(filename)-rem_string]+'_trajs2.pickle'
    data2save = {"trajectories": trajectories}
    outfile = open(new_filename, 'wb')
    pickle.dump(data2save, outfile)
    outfile.close()