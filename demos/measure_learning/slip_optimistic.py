import slippy.slip as true_model
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

    # A prior state action pair that is considered safe (from system knowledge)
    X_seed = np.atleast_2d(np.array([38 / (180) * np.pi, .45]))
    y_seed = np.array([[.2]])

    seed_data = {'X': X_seed, 'y': y_seed}

    sampler = sampling.MeasureLearner(model=true_model, model_data=data)
    sampler.init_estimation(seed_data=seed_data,
                            prior_model_path=gp_model_file,
                            learn_hyperparameters=False)

    sampler.exploration_confidence_s = 0.80
    sampler.exploration_confidence_e = 0.90
    sampler.measure_confidence_s = 0.50
    sampler.measure_confidence_e = 0.70
    sampler.safety_threshold_s = 0.0
    sampler.safety_threshold_e = 0.0

    # randomize, but keep track of it in case you want to reproduce
    sampler.seed = np.random.randint(1, 100)
    print('Seed: ' + str(sampler.seed))

    n_samples = 200

    random_string = str(np.random.randint(1, 10000))

    plot_callback = cplot.create_plot_callback(n_samples,
                                               experiment_name='slip_optimistic',
                                               random_string=random_string,
                                               save_path=results_path)

    s0 = .45

    sampler.run(n_samples=n_samples, s0=s0, callback=plot_callback)


