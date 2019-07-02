import slippy.hovership as true_model
import numpy as np
import pickle

import plotting.corl_plotters as cplot


import scratchpad.active_sampling as sampling

################################################################################
# Load model data
################################################################################
infile = open('../data/hover_map.pickle', 'rb')
data = pickle.load(infile)
infile.close()

# TODO Steve
true_model.mapSA2xp = true_model.sa2xp
true_model.map2s = true_model.xp2s

# This comes from knowledge of the system
X_seed = np.atleast_2d(np.array([.3, 0]))
y_seed = np.array([[.5]])

seed_data = {'X': X_seed, 'y': y_seed}

sampler = sampling.MeasureLearner(model=true_model, model_data=data)
sampler.init_estimation(seed_data=seed_data, prior_model_path='./model/hover_prior.npy')

s0 = .45

sampler.run(n_samples=1000, s0=s0)
