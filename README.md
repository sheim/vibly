# Implementation accompanying the paper "A Learnable Safety Measure"


## Prerequisites

Tested with Python 3.7, using the Anaconda distribution on Linux and MacOS.
The needed packaged are:
- Numpy (https://www.numpy.org/)
- SciPy (https://www.scipy.org)
- GPy (https://sheffieldml.github.io/GPy/)

TODO Anaconda environment

If you are not using the anaconda environment make sure you have the required packages installed in your system.

## Reproduce the experiments presented in the paper

To reproduce the presented results you can run any of the experiments located in 'demos/measure_learning'.

For your convenience we have prepared a script 'run_learning_examples.py' from which these experiments can be run.
Just uncomment the script you want to excecute. The file should look something like this:

```python
dynamics_model_path = './data/dynamics/'
gp_model_path = './data/gp_model/'
results_path = './results/'


import demos.measure_learning.hovership as experiment

experiment.run_demo(dynamics_model_path=dynamics_model_path,
                              gp_model_path=gp_model_path,
                              results_path=results_path)

```
The figures will be shown and saved on the results path as PDFs.
Make sure that the paths are configured correctly.
Especially in Windows path naming is different.


## Run your own experiment
Defining an experiment is simply done by choosing a supplied model or implementing your own.
See the next section on how to implement your own dynamics.

After you chose the desired confidence threshold and level set as well as an initial safe state action pair and run the script.
For example:

```python

import slippy.slip as true_model
import numpy as np
import pickle

import plotting.corl_plotters as cplot
import measure.active_sampling as sampling

dynamics_model_path = './data/dynamics/'
gp_model_path = './data/gp_model/'
results_path = './results/'

################################################################################
# Load model data
################################################################################

dynamics_file = dynamics_model_path + 'hover_map.pickle'
gp_model_file = gp_model_path + 'hover_prior.npy'

infile = open(dynamics_file, 'rb')
data = pickle.load(infile)
infile.close()

################################################################################
# Setup estimate
################################################################################

# A prior state action pair (first action, than state) that is considered safe (from system knowledge)
X_seed = np.atleast_2d(np.array([.3, 2])) # action, state
y_seed = np.array([[.5]])

seed_data = {'X': X_seed, 'y': y_seed}

sampler = sampling.MeasureLearner(model=true_model, model_data=data)
sampler.init_estimation(seed_data=seed_data,
                        prior_model_path=gp_model_file,
                        learn_hyperparameters=True)

################################################################################
# Tuning parameter
################################################################################

# The start and end of the thresholds are linearly interpolated between iterations

# The start of the exploration confidence threshold
sampler.exploration_confidence_s = 0.98
# The end of the exploration confidence threshold
sampler.exploration_confidence_e = 0.9
# The start of the measure confidence threshold
sampler.measure_confidence_s = 0.7
# The start of the measure confidence threshold
sampler.measure_confidence_e = 0.9
# Number of total samples
n_samples = 500


################################################################################
# Run the experiment
################################################################################

# To avoid accidental overwriting of data when saving the results
random_string = str(np.random.randint(1, 10000))

# The callback is called every e.g. 50 iteration and plots the results as well as saving the figures and data
plot_callback = cplot.create_plot_callback(n_samples,
                                           experiment_name='hovership',
                                           random_string=random_string,
                                           every=50,
                                           save_path=results_path)
# Starting state (should estimated to be safe)
s0 = 2

# Run everything :)
sampler.run(n_samples=n_samples, s0=s0, callback=plot_callback)
```


## Implement your own dynamics

### Implement your simulation

Since our algorithms treat the system as an oracle, dynamics can be implemented
quite arbitrarily. We recommend starting with `hovership.py`, which is simple and has been well commented, as an example to modify.
Specifically, our algorithms require:
- a transition map, `p_map`
- a mapping from high-level state-action to low-level state, `sa2xp`
- a mapping from low-level state to high-level state, `xp2s`
Note, we refer to a high level state _s_, used in the viability algorithms, and a low-level state _x_, used by the simulation. These can be the same (in which case `xp2s` would, for example, simply return the input `x` as `s`). This has been done since the black-box oracle may live in a higher-dimensional state-space. This is a common case if hierarchical control is used. For an example, see the `slip.py` example. Additionally, the simulation does not consider a separate control input. Instead, each action is stored in the parameter dict `p`. The helper function `sa2xp` also maps the action to the parameter dict.

### Create ground truth data

For small systems (the high-level state-action space `(S,A)` should be up to 4-5 dimensions, since the ground truth is computed via brute force. The low-levle state-action space can be higher-dimensional), you can compute a grid of the ground-truth using brute-force to compute a discretized viable set.
This uses the algorithms in `viability.py`, which uses the same objects described above. The main functions are listed below. For an example script, please see `computeQ_hovership.py`.

- `compute_Q_map` computes a gridded transition map, as well as a grid of failing state-action pairs
- `compute_QV`, using the previously computed gridded transition map, compute the viable set and viability kernel
- `project_Q2S` can be used with `np.mean` to obtain the safety measure. Note, you can also use `np.sum`, however `np.mean` will normalize your measure by the total volume of your state-action space, which is convenient
- `map_S2Q` uses the gridded transition map, map the measure from state-space back into Q-space

And voila!