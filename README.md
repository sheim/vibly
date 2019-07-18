# Viability in state-action space

This code implements the algorithms for computing viable sets in state-action space, as discussed in the paper [Beyond Basins of Attraction: Quantifying Robustness of Natural Dynamics](https://arxiv.org/abs/1806.08081). Viable sets in state-action space relate control to failures (and thus, to avoiding failures) in dynamical systems, and is relevant to robustness and learning.  

It also includes code in the follow-up paper _A Learnable Safety Measure_, submitted to [CoRL](https://arxiv.org/abs/1806.08081).

## Prerequisites

Tested with Python 3.7, using the Anaconda distribution on Linux and MacOS.
The needed packaged are:
- [Numpy](https://www.numpy.org/)
- [SciPy](https://www.scipy.org)
- [Matplotlib](https://matplotlib.org/) (only for plotting)
- **optional** [GPy](https://sheffieldml.github.io/GPy/) is needed for the `measure` code (only needed for learning using Gaussian processes)

If you are not using the anaconda environment make sure you have the required packages installed in your system.

## Installation

Once you have set up your environment, install this package locally by running `pip install -e .` in the root directory. You should now be able to import packages as following:

```python
# import brute-force viability tools
import viability as vibly
# import a model of a dynamical systems
from models import slip
# import classes for active sampling and learning
from measure import active_sampling
```

## Usage for viability

Examples are shown in the `demos` folder. If you're interested in SLIP models and legged locomotion, we recommend starting with `slip_demo.py`, and then `computeQ_slip.py`.  
The `viability` package contains:
- `compute_Q_map`: a utility to compute a gridded transition map for N-dimensional systems. Note, this can be computationally intensives (it is essentially brute-forcing an N-dimensional problem). It typically works reasonably well for up to ~5 dimensions.
- `compute_QV`: computes the viability kernel and viable set to within conservative discrete approximation, using the grid computed by `compute_Q_map`.
- `get_feasibility_mask`: this can be used to exclude parts of the grid which are infeasible (i.e. are not physically meaningful)
- `project_Q2S`: Apply an operator (usually just an orthogonal projection) from state-action space to state space. Used to compute measures.
- `map_S2Q`: maps values of each state to state-action space. Used for mapping measures from state space to state-action space.


## Usage for learning measures and viable sets

Coming soon. For now, see the readme in the `submission` branch. To dive right in, see `run_learning_examples.py`. **NOTE** this requires the `GPy` package.

## create your own dynamics

You can easily add your own models in the `models` package. The `viability` code does expect a few helper functions which need to be implemented as well. We recommend using the `hovership.py` example as a template, as it is simple and heavily commented. For examples of a more complex system, see the `slip.py` model.

## Contact

Feel free to open an issue, or get in touch via heim.steve@gmail.com for any questions, comments, etc.