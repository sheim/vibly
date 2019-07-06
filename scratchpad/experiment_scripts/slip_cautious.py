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

# Start from bad prior

# This comes from knowledge of the system

X_seed = np.atleast_2d(np.array([38 / (180) * np.pi, .45]))
y_seed = np.array([[.2]])

seed_data = {'X': X_seed, 'y': y_seed}

sampler = sampling.MeasureLearner(model=true_model, model_data=data)
# use '../model/slip_prior_proxy.npy' for incorrect prior
# use '../model/slip_prior_true.npy' for prior regressed over ground truth
sampler.init_estimation(seed_data=seed_data, prior_model_path='../model/slip_prior_proxy.npy', learn_hyperparameters=False)

sampler.measure_confidence_s = 0.75
sampler.measure_confidence_e = 0.99

sampler.exploration_confidence_s = 0.95
sampler.exploration_confidence_e = 0.99
sampler.safety_threshold_s = 0.01
sampler.safety_threshold_e = 0.05

# randomize, but keep track of it in case you want to reproduce
sampler.seed = np.random.randint(1, 100)
print(sampler.seed)

n_samples = 200

random_string = str(np.random.randint(1, 10000))

plot_callback = cplot.create_plot_callback(n_samples, 'slip_nice', random_string=random_string)

s0 = .45

sampler.run(n_samples=n_samples, s0=s0, callback=plot_callback)
