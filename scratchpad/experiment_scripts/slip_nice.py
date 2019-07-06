import slippy.slip as true_model
import numpy as np
import pickle
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import plotting.corl_plotters as cplot

import scratchpad.active_sampling as sampling


# TODO Steve
true_model.mapSA2xp = true_model.mapSA2xp_height_angle
true_model.p_map = true_model.poincare_map

################################################################################
# Load model data
################################################################################
infile = open('../../data/slip_map.pickle', 'rb')
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

sampler.exploration_confidence_s = 0.95
sampler.exploration_confidence_e = 0.98
sampler.measure_confidence_s = 0.75
sampler.measure_confidence_e = 0.98

sampler.seed = 66

n_samples = 200

random_string = str(np.random.randint(1, 10000))

plot_callback = cplot.create_plot_callback(n_samples, 'slip_nice', random_string=random_string)

s0 = .45

sampler.run(n_samples=n_samples, s0=s0, callback=plot_callback)
