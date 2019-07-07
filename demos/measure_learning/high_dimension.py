import slippy.spaceship4 as true_model
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
    y_seed = np.array([[.2]])

    seed_data = {'X': X_seed, 'y': y_seed}

    sampler = sampling.MeasureLearner(model=true_model, model_data=data)
    sampler.init_estimation(seed_data=seed_data, prior_model_path=gp_model_file, learn_hyperparameters=False)

    sampler.exploration_confidence_s = 0.90
    sampler.exploration_confidence_e = 0.999
    sampler.measure_confidence_s = 0.70
    sampler.measure_confidence_e = 0.999
    sampler.safety_threshold_s = 0.1
    sampler.safety_threshold_e = 0.0

    # randomize, but keep track of it in case you want to reproduce
    sampler.seed = np.random.randint(1, 100)
    print('Seed: ' + str(sampler.seed))

    n_samples = 500

    s0 = np.array([1, 0, 0, 0])

    sampler.run(n_samples=n_samples, s0=s0, callback=None)



## TODO: Start from good prior

# idx_safe = np.argwhere(Q_V_proxy.ravel()).ravel()
# idx_unsafe = np.argwhere(~Q_V_proxy.ravel()).ravel()
#
# idx_sample_safe = np.random.choice(idx_safe, size=np.min([200, len(idx_safe)]), replace=False)
# idx_sample_unsafe = np.random.choice(idx_unsafe, size=np.min([100, len(idx_unsafe)]), replace=False)
#
# idx = np.concatenate((idx_sample_safe, idx_sample_unsafe))
#
# X_prior = X_grid[idx, :]
# y_prior = Q_M_proxy.ravel()
# y_prior = y_prior[idx].reshape(-1, 1)