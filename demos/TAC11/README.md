# Safe Value Functions

Code to reproduce figures from _Safe Value Functions_, and other reward-shaping cases.

## Continuous Space Example

Please fill this in :)

## Discretized Numerical Examples

We illustrate the effects of reward shaping on a simple 2-dimensional system, loosely modeled as an orbiting satellite.
The system dynamics are

$$
\dot{x}_1 = x_2, \\
\, \\
\dot{x}_2 = - \frac{g}{x_1^2} +\omega^2x_1 + u
$$
with state $x = \left [x_1, x_2 \right] \in \left[0, 16\right]\times\left[-5, 7\right]$, control input $u \in \left[-1, 1\right]$, and system parameters $g = 10$ and $\omega=0.1$.
The model is implemented in `vibly/models/satellite.py`.
A discretized transition map is first computed using `computeQ_satellite.py`, which also computes the viability kernel, and contains the system parameters.  
The value function of each case study is computed in the various folders:

- `parsimonious/` for figure 3: receive a reward only at the equilibrium point at $\left[x_1, x_2\right] = \left[10, 0\right].
- `viability/` for figure 4: receive no positive reward, only a penalty.
- `pos_proxy/` for figure 5: receive positive reward any time $|x_1-10| \leq 1$
- `neg_proxy/` for figure 6: receive a negative reward when $|x_2| > 2$
- `progressive/` shows a case where positive rewards are only attainable outside the viability kernel, encouraging the controller to fail. This case is not included in the paper due to limited space. Additional rewards can be easily added (see code).
- `gamma/` produces a study that evaluate how $p^*$ changes as a function of the discount factor $\gamma$