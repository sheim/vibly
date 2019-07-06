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
X_seed = np.atleast_2d(np.array([.4, 1.5]))
y_seed = np.array([[.5]])

seed_data = {'X': X_seed, 'y': y_seed}

sampler = sampling.MeasureLearner(model=true_model, model_data=data)
sampler.init_estimation(seed_data=seed_data,
                        prior_model_path='../model/hover_prior.npy',
                        learn_hyperparameters=False)

sampler.seed = 69

sampler.exploration_confidence_s = 0.94
sampler.exploration_confidence_e = 0.999
sampler.measure_confidence_s = 0.55
sampler.measure_confidence_e = 0.999
sampler.safety_threshold_s = 0.0
sampler.safety_threshold_e = 0.15

n_samples = 250

random_string = str(np.random.randint(1, 1000))

plot_callback = cplot.create_plot_callback(n_samples, 'hovership_bad_start', random_string=random_string)


# TODO get initial state from estimator
s0 = 1.5
# Q_V_explore = estimation.safe_level_set(safety_threshold=0,
#                                         confidence_threshold=exploration_confidence)
# S_M_safe = estimation.project_Q2S(Q_V_explore)
# s_0_idx = np.random.choice(np.where(S_M_safe > 0)[0])
# s_0 = self.grids['states'][0][s_next_idx]

sampler.run(n_samples=n_samples, s0=s0, callback=plot_callback)
