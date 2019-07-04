import slippy.slip as true_model
import numpy as np
import pickle

import matplotlib.pyplot as plt
import plotting.corl_plotters as cplot

import scratchpad.active_sampling as sampling


# TODO Steve
true_model.mapSA2xp = true_model.mapSA2xp_height_angle
true_model.p_map = true_model.poincare_map

################################################################################
# Load model data
################################################################################
infile = open('../data/slip_map.pickle', 'rb')
data = pickle.load(infile)
infile.close()

## Start from bad prior

# This comes from knowledge of the system

X_seed = np.atleast_2d(np.array([38 / (180) * np.pi, .45]))
y_seed = np.array([[.2]])

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

seed_data = {'X': X_seed, 'y': y_seed}

sampler = sampling.MeasureLearner(model=true_model, model_data=data)
sampler.init_estimation(seed_data=seed_data, prior_model_path='./model/prior.npy', learn_hyperparameters=False)


n_samples = 250

def plot_callback(sampler, ndx, thresholds):
    # Plot every n-th iteration
    if ndx % 50 == 0 or ndx + 1 == n_samples:

        Q_map_true = sampler.model_data['Q_map']
        grids = sampler.grids

        Q_M_true = sampler.model_data['Q_M']
        Q_V_true = sampler.model_data['Q_V']
        S_M_true = sampler.model_data['S_M']


        Q_V = sampler.current_estimation.safe_level_set(safety_threshold=thresholds['safety_threshold'],
                                                     confidence_threshold=thresholds['measure_confidence'])
        S_M_0 = sampler.current_estimation.project_Q2S(Q_V)

        Q_V_exp = sampler.current_estimation.safe_level_set(safety_threshold=0,
                                                         confidence_threshold=thresholds['exploration_confidence'])

        fig = cplot.plot_Q_S(Q_V_true, Q_V_exp, Q_V, S_M_0, S_M_true, grids,
                             samples=(sampler.X, sampler.y),
                             failed_samples=sampler.failed_samples,
                             S_labels=("safe estimate", "ground truth"),
                             action_space_label='angle of attack [rad]',
                             state_space_label='normalized height at apex')

        # plt.savefig('./sample'+str(ndx))
        # plt.close('all')
        plt.show()

        if sampler.y is not None:
            print(str(ndx) + " ACCUMULATED ERROR: "
                  + str(np.sum(np.abs(S_M_0 - S_M_true)))
                  + " Failure rate: " + str(np.mean(sampler.y < 0)))


s0 = .45

sampler.run(n_samples=n_samples, s0=s0, callback=plot_callback)
