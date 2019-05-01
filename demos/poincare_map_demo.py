import numpy as np

from slippy.slip import *

p = {'mass':80.0, 'stiffness':8200.0, 'resting_length':1.0, 'gravity':9.81,
'angle_of_attack':1/5*np.pi}
x0 = np.array([0, 0.85, 5.5, 0, 0, 0]) # ensure you are starting a vector!
x0 = reset_leg(x0, p)
p['total_energy'] = compute_total_energy(x0, p)

# compute individually

x0 = x0
x_delta = np.array([0, 0.01, 0, 0, 0, 0])

xk, x0_failed = poincare_map(x0, p)
xk_delta, x_delta_failed  = poincare_map(x0 + x_delta, p)

# compute a vector of initial states and parameters
# to use this, you need to provide
# - an m by n array of initial conditions [x0_1, x0_2, ..., x0_n]
# - a tuple of parameter dicts of length n

x0_traj = np.column_stack((x0, x0 + x_delta))
p_traj = (p, p)

x0_traj, x_failed_traj = poincare_map(x0_traj, p_traj)

#### check results are the same

if (not np.all(np.isclose(np.column_stack((xk,xk_delta)), x0_traj)) and
    not np.all(np.column_stack((x0_failed, x_delta_failed)) == x_failed_traj)):
    print("WARNING: difference in results between individual and stacked")