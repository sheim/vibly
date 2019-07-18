import models.slip as true_model
import numpy as np
import pickle

import plotting.corl_plotters as cplot

import measure.active_sampling as sampling


def run_demo(dynamics_model_path='./data/dynamics/', gp_model_path='./data/gp_model/', results_path='./results/'):

    ###########################################################################
    # Load model data
    ###########################################################################

    dynamics_file = dynamics_model_path + 'slip_map.pickle'
    # use 'slip_prior_proxy.npy' for incorrect prior
    # use 'slip_prior_true.npy' for prior regressed over ground truth
    gp_model_file = gp_model_path + 'slip_prior_true.npy'

    infile = open(dynamics_file, 'rb')
    data = pickle.load(infile)
    infile.close()

    # The ground truth measure

    X = np.meshgrid(*(data['grids']['actions']), *(data['grids']['states']))
    Y = data['Q_M']

    X = np.vstack(map(np.ravel, X)).T
    Y = np.atleast_2d(Y.ravel()).T

    Q_V = data['Q_V']
    idx_safe = np.argwhere(Q_V.ravel()).ravel()

    idx = np.random.choice(idx_safe, size=np.min([100, len(idx_safe)]), replace=False)

    X = X[idx, :]
    Y = Y[idx].reshape(-1, 1)

    seed_data = {'X': X, 'y': Y}

    sampler = sampling.MeasureLearner(model=true_model, model_data=data)
    sampler.init_estimation(seed_data=seed_data,
                            prior_model_path=gp_model_file,
                            learn_hyperparameters=False)

    sampler.exploration_confidence_s = 0.85
    sampler.exploration_confidence_e = 0.85
    sampler.measure_confidence_s = 0.85
    sampler.measure_confidence_e = 0.85
    sampler.safety_threshold_s = 0.01
    sampler.safety_threshold_e = 0.01

    # randomize, but keep track of it in case you want to reproduce
    sampler.seed = np.random.randint(1, 100)
    print('Seed: ' + str(sampler.seed))

    n_samples = 200

    random_string = str(np.random.randint(1, 10000))

    plot_callback = cplot.create_plot_callback(n_samples,
                                               experiment_name='slip_prior',
                                               random_string=random_string,
                                               save_path=results_path)

    s0 = .45

    sampler.run(n_samples=n_samples, s0=s0, callback=plot_callback)


