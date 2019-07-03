import numpy as np
import scipy.integrate as integrate
'''
space attempting to reconnoitre the surface of a planet.
Must ensure not to go to the dark side of the planet.
x_{k+1} = map(x_k, p)
x: (x1, x2)
x1: altitude
p: dict of parameters. For convenience, actions are also stored here.
'''

# map: x_k+1, failed = map


def p_map(x, p):
    '''
    Dynamics function of your system
    Note that the control input is included in the parameter,
    and needs to be unpacked.
    '''
    if check_failure(x, p):
        return x, True

    # unpack
    THRUST = np.min([p['max_thrust'], p['thrust']])
    BASE_GRAVITY = p['base_gravity']
    GRAVITY = p['gravity']
    MAX_TIME = 1.0/p['control_frequency']

    def continuous_dynamics(t, x):
        x[0] += BASE_GRAVITY + np.max([0, np.tanh(0.75*x[0])])*GRAVITY - THRUST
        x[0] = np.max([0, x[0]])  # saturate at ceiling (x=0)
        return x

    sol = integrate.solve_ivp(continuous_dynamics, t_span=[0, MAX_TIME], y0=x)

    return sol.y[:, -1], check_failure(sol.y[:, -1], p)


def check_failure(x, p):
    '''
    Check if a state is in the failure set.
    '''
    if x[0] > p['ground_height']:
        return True
    else:
        return False


# Viability functions
def sa2xp(state_action, p):
    x = np.atleast_1d(state_action[:p['n_states']])
    p['thrust'] = np.atleast_1d(state_action[p['n_states']:])
    return x.flatten(), p


def xp2s(x, p):
    return x
