import models.spaceship4 as true_model
import numpy as np
import pickle

import plotting.corl_plotters as cplot
import measure.active_sampling as sampling

def run_demo(dynamics_model_path = './data/dynamics/', gp_model_path='./data/gp_model/', results_path='./results/'):

    ################################################################################
    # Load model data
    ################################################################################

    dynamics_file = dynamics_model_path + 'spaceship4_lowres.pickle'
    # use 'slip_prior_proxy.npy' for incorrect prior
    # use 'slip_prior_true.npy' for prior regressed over ground truth
    gp_model_file = gp_model_path + 'spaceship4_lowres.npy'

    infile = open(dynamics_file, 'rb')
    data = pickle.load(infile)
    infile.close()

    # A prior state action pair that is considered safe (from system knowledge)
    X_seed = np.atleast_2d(np.array([.3, 1, 0, 0, 0]))
    y_seed = np.array([[1]])

    seed_data = {'X': X_seed, 'y': y_seed}

    sampler = sampling.MeasureLearner(model=true_model, model_data=data)
    sampler.init_estimation(seed_data=seed_data, prior_model_path=gp_model_file, learn_hyperparameters=False)

    sampler.exploration_confidence_s = 0.70
    sampler.exploration_confidence_e = 0.70
    sampler.measure_confidence_s = 0.70
    sampler.measure_confidence_e = 0.70
    sampler.safety_threshold_s = 0.0
    sampler.safety_threshold_e = 0.0

    # randomize, but keep track of it in case you want to reproduce
    sampler.seed = np.random.randint(1, 100)
    print('Seed: ' + str(sampler.seed))

    n_samples = 200

    s0 = np.array([1, 0, 0, 0])

    def plot_callback(sampler, ndx, thresholds):
        # Plot every n-th iteration
        if ndx % 5 == 0 or ndx + 1 == n_samples or ndx == -1:

            S_M_true = sampler.model_data['S_M']

            Q_V = sampler.current_estimation.safe_level_set(safety_threshold=0,
                                                            confidence_threshold=thresholds['measure_confidence'])
            S_M_0 = sampler.current_estimation.project_Q2S(Q_V)


            if sampler.y is not None:
                print(str(ndx) + " ACCUMULATED ERROR: "
                      + str(np.sum(np.abs(S_M_0 - S_M_true)))
                      + " Failure rate: " + str(np.mean(sampler.y < 0)))


    sampler.run(n_samples=n_samples, s0=s0, callback=plot_callback)


