# Vibly: viability in state-action space

A Python framework for working with viable sets in state-action space, learning safety constraints, and for evaluating the effect of morphology on robustness and stability. This code accompanies the following papers:

1. [**A Learnable Safety Measure**](https://arxiv.org/abs/1910.02835) Steve Heim, Alexander von Rohr, Sebastian Trimpe, Alexander Badri-Spröwitz, _Conference on Robot Learning_ 2019. Paper-specific examples in `demos/measure_learning/`. [Code examples](#learning)

2. [**Beyond Basins of Attraction: Quantifying Robustness of Natural Dynamics**](https://arxiv.org/abs/1806.08081) Steve Heim and Alexander Badri-Spröwitz _IEEE Transaction on Robotics_ 2019. Note: This code implements the core algorithm for the paper, but does not reproduce all results and figures. The Arxiv version contains fewer typos and is recommended.

3. [**A Little Damping Goes a Long Way: a simulation study of how damping influences task-level stability in running**](https://royalsocietypublishing.org/doi/10.1098/rsbl.2020.0467) Steve Heim, Matthew Millard, Charlotte Le Mouel, Alexander Badri-Spröwitz _Royal Society Biology Letters_ 2020. Paper-specific examples in `demos/damping_study/`. [Code examples](#damping)

<!-- ## What is viability in state-action space?

A dynamical system is in a _viable_ state if there exist control inputs that allow it to avoid a set of _failure states_ forever. The _viability kernel_ is the set of all viable states. We extend this notion into state-action space, and define sets of viability-maintaining state-action pairs, or _viable sets_, which allows certain insights. -->

## Installation
We recommend using a virtual environment. Note, `GPy` is only required for the safe learning examples, and can be safely removed from the requirements. Install from your terminal with

PyEnv + PipEnv (recommended):  
`pipenv install -r requirements.txt`  
`pipenv install -e .`


pip:  
`pip install -r requirements.txt`  
`pip install -e .`


conda:  
`conda create -n vibly python=3.9`
`conda activate vibly`
then follow the same instructions as for pip:
`pip install -r requirements.txt && pip install -e .`

You should now be able to run all examples in `demos/`, and import packages as following:
```python 
# import brute-force viability tools
import viability as vibly
# import a model of a dynamical systems
from models import slip
# import classes for active sampling and learning
from measure import active_sampling
```

## Use for viability

Examples are shown in the `demos` folder. We recommend starting with `slip_demo.py`, and then `computeQ_slip.py`.  
The `viability` package contains:
- `compute_Q_map`: a utility to compute a gridded transition map for N-dimensional systems. Note, this can be computationally intensives (it is essentially brute-forcing an N-dimensional problem). It typically works reasonably well for up to ~4 dimensions.
- `parcompute_Q_map`: same as above, but parallelized. You typically want to use this, unless running a debugger.
- `compute_QV`: computes the viability kernel and viable set to within conservative discrete approximation, using the grid generated by `compute_Q_map`.
- `get_feasibility_mask`: this can be used to exclude parts of the grid which are infeasible (i.e. are not physically meaningful)
- `project_Q2S`: Apply an operator (default is an orthogonal projection) from state-action space to state space. Used to compute measures.
- `map_S2Q`: maps values of each state to state-action space. Used for mapping measures from state space to state-action space.

## Reproduce CoRL safe learning study <a name="learning"/>

You will need to first regenerate the ground-truth data used for comparison, by running `demos/computeQ_hovership.py` and `demos/computeQ_slip.py`.  
Then, select and run an experiment by running `run_learning_examples.py` in `demos/measure_learning/`. The experiment details, including algorithm hyper-parameters and initialization, are defined in the experiment file, e.g. `demos/measure_learning/hovership_default.py`.  
The learning algorithm is split between `measure/active_sampling`, which includes sampling strategy, and `measure/estimate_measure`, which handles everything dealing with the measure in different spaces. If you're looking to use the measure for a different learning approach, you probably want to look into the `active_sampling.py` file.

## Reproduce RSBL damping study <a name="damping"/>
The code to reproduce the results from the paper are in `/demos/damping_study/`. You can run `compute_measure_damping.py`, which will generate all the data needed; however, this can take a _long_ time (~20 hours on a 24-core desktop). If you just want to inspect the results, all the pre-computed data (and code) can be downloaded from [Dryad](https://doi.org/10.5061/dryad.44j0zpcbj). We encourage you to use this code, which may have improvements/bugfixes, and simply copy/paste the dataset from `data/guineafowl` into the `data` folder.

## Create your own dynamics

You can easily add your own models in the `models` package. The `viability` code does expect a few helper functions which need to be implemented as well. We recommend using the `hovership.py` example as a template, as it is simple and heavily commented. For examples of simulating a more complex system, see the `slip.py` model.

## Contact

Feel free to open an issue, or get in touch via heim.steve@gmail.com for any questions, comments, etc.