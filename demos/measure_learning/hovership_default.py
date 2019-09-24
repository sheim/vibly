import models.hovership as true_model
import numpy as np
import pickle

import plotting.corl_plotters as cplot

import measure.active_sampling as sampling


def run_demo(dynamics_model_path='./data/dynamics/', gp_model_path='./data/gp_model/', results_path='./results/'):

    ################################################################################
    # Load model data
    ################################################################################

    dynamics_file = dynamics_model_path + 'hover_map.pickle'
    gp_model_file = gp_model_path + 'hover_prior.npy'

    infile = open(dynamics_file, 'rb')
    data = pickle.load(infile)
    infile.close()

    # A prior state action pair that is considered safe (from system knowledge)
    X_seed = np.atleast_2d(np.array([1.8, .6]))
    y_seed = np.array([[.5]])

    seed_data = {'X': X_seed, 'y': y_seed}

    sampler = sampling.MeasureLearner(model=true_model, model_data=data)
    sampler.init_estimation(seed_data=seed_data,
                            prior_model_path=gp_model_file,
                            learn_hyperparameters=False)

    # No adjustment, just learning
    sampler.exploration_confidence_s = 0.8
    sampler.exploration_confidence_e = 0.8
    sampler.measure_confidence_s = 0.6
    sampler.measure_confidence_e = 0.8

    n_samples = 500

    random_string = str(np.random.randint(1, 10000))

    plot_callback = cplot.create_plot_callback(n_samples,
                                               experiment_name='hovership',
                                               random_string=random_string,
                                               every=50,
                                               save_path=results_path)

    s0 = 2
    sampler.run(n_samples=n_samples, s0=s0, callback=plot_callback)

if __name__ == "__main__":
    dynamics_model_path = '../../data/dynamics/'
    gp_model_path = '../../data/gp_model/'
    results_path = '../../results/'

    run_demo(dynamics_model_path=dynamics_model_path,
                        gp_model_path=gp_model_path,
                        results_path=results_path)
