# Safe Value Functions

Code to reproduce figures from _Safe Value Functions_, and other reward-shaping cases.

## Installation

General installation instructions are on the [main page of this repo](https://github.com/sheim/vibly/tree/TAC22).
We provide here a summary with conda:
```
conda create -n vibly python=3.9
conda activate vibly 
pip install -r requirements.txt && pip install -e .
```

Before running these commands, you may remove the GPy dependency of the file `requirements.txt`; it is not needed for these results.


### Continuous-time Example

To reproduce Figure 2 from the article, run:
```
conda activate vibly
cd continuous_time && python main.py && cd ..
```
This will create three PDF files in the folder `continuous_time` with names following the pattern `CONTINUOUS_TIME_[TAU]_[Tf].pdf`. 
Here, `[TAU]` and `[Tf]` correspond respectively to the values of $\tau$ and $T_f$ used to plot V_p (top plot of Figure 2).
You can change these parameters freely by editing the last lines of the file `continuous_time/main.py`.

### Discretized Numerical Examples
We illustrate the effects of reward shaping on a simple 2-dimensional system, loosely modeled as an orbiting satellite.

1. First run `computeQ_satellite.py` in this folder. This will set up the transition map and ground-truth viability kernel (computed via brute-force).
2. In one of the specific folders (listed below), run `VIS_example.py` to compute the given value function. Where relevant, an additional plotting script is also included.

In each example, the penalties used in the paper are hard-coded, and can be changed. Reward functions can also be modified (see code in `miscellaneous/`).

- `parsimonious/` for figure 3: receive a reward only at the equilibrium point at $\left[x_1, x_2\right] = \left[10, 0\right].
- `viability/` for figure 4: receive no positive reward, only a penalty.
- `pos_proxy/` for figure 5: receive positive reward any time $|x_1-10| \leq 1$
- `neg_proxy/` for figure 6: receive a negative reward when $|x_2| > 2$
- `miscellaneous/` shows a case where positive rewards are only attainable outside the viability kernel, encouraging the controller to fail. This case is not included in the paper due to limited space. Several additional reward functions are included here.
- `gamma/` produces a study that evaluate how $p^*$ changes as a function of the discount factor $\gamma$

---
#### Exploring additional examples
- To change reward functions, define them as functions and add tehm to the reward list, e.g. `reward_functions = (L2_diverge, actuator_cost, penalty)`.
- To change the penalties, add them to the penalty list `failure_penalties = [0, 10., 100, 500.]`
---
#### System description

The model used is loosely inspired as an orbiting satellite, and has the following continuous-time dynamics, which are discretized into a grid-world through numerical integration.
$$
\dot{x}_1 = x_2, \\
\, \\
\dot{x}_2 = - \frac{g}{x_1^2} +\omega^2x_1 + u
$$
with state $x = \left [x_1, x_2 \right] \in \left[0, 16\right]\times\left[-5, 7\right]$, control input $u \in \left[-1, 1\right]$, and system parameters $g = 10$ and $\omega=0.1$.
The model is implemented in `vibly/models/satellite.py`.
Other models are also included (mostly idealized models of legged locomotion) and can also be used (will need custom plotting); to implement your own, we recommend modifying the `vibly/models/hovership.py` model.