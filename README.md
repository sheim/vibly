# Implementation accompanying the paper "A Learnable safety measure"


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


import demos.measure_learning.hovership as hovership_experiment

hovership_experiment.run_demo(dynamics_model_path=dynamics_model_path,
                              gp_model_path=gp_model_path,
                              results_path=results_path)

```
The figures will be shown and saved on the results path as PDFs.
Make sure that the paths are configured correctly.
Especially in Windows path naming is different.


## Run your own experiment
Defining an experiment is simply done by choosing a supplied model (TODO LIST?) or implementing your own.
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

true_model.mapSA2xp = true_model.sa2xp
true_model.map2s = true_model.xp2s

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

### Implement model
(TODO) what about high dimension

### Create ground truth data