import slippy.hovership as true_model
import numpy as np
import pickle
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import plotting.corl_plotters as cplot

import scratchpad.active_sampling as sampling

################################################################################
# Load model data
################################################################################
infile = open('../../data/hover_map.pickle', 'rb')
data = pickle.load(infile)
infile.close()

# TODO Steve
true_model.mapSA2xp = true_model.sa2xp
true_model.map2s = true_model.xp2s

# This comes from knowledge of the system
X_seed = np.atleast_2d(np.array([.3, 2]))
y_seed = np.array([[.5]])

seed_data = {'X': X_seed, 'y': y_seed}

sampler = sampling.MeasureLearner(model=true_model, model_data=data)
sampler.init_estimation(seed_data=seed_data, prior_model_path='../model/hover_prior.npy', learn_hyperparameters=False)

sampler.seed = 69

sampler.exploration_confidence_s = 0.98
sampler.exploration_confidence_e = 0.9
sampler.measure_confidence_s = 0.7
sampler.measure_confidence_e = 0.9

n_samples = 500

random_string = str(np.random.randint(1, 10000))


plot_callback = cplot.create_plot_callback(n_samples, 'hovership_nice', random_string=random_string)

s0 = 2

sampler.run(n_samples=n_samples, s0=s0, callback=plot_callback)
