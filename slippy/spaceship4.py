import numpy as np
import scipy.integrate as integrate
'''
space attempting to reconnoitre the surface of a planet.
Must ensure not to go to the dark side of the planet.
x_{k+1} = map(x_k, p)
x: (x1, x2)
x1: altitude
x2: longitude
p: dict of parameters. For convenience, actions are also stored here.
'''

# map: x_k+1, failed = map


def wind(x, p):
    # return -0.5*x[1]
    # stretch = (p['x0_upper_bound']-p['x0_lower_bound'])*2*np.pi
    return p['wind']*np.sin(x[0]*np.pi)


def gravity(x, p):
    return np.max([0, x[0]])*p['gravity']


def p_map(x, p):
    '''
    Dynamics function of your system
    Note that the control input is included in the parameter,
    and needs to be unpacked.
    '''
    if check_failure(x, p):
        return x, True

    THRUST = p['thrust']
    WIND = p['wind']
    GRAVITY = p['gravity']
    MAX_TIME = 1.0/p['control_frequency']

    def continuous_dynamics(t, x):
        x[0] += x[2]
        x[1] += x[3]
        x[2] += np.max([0, x[0]])*GRAVITY - THRUST
        x[3] += WIND*np.sin(x[0]*np.pi)
        return x

    sol = integrate.solve_ivp(continuous_dynamics, t_span=[0, MAX_TIME], y0=x)

    return sol.y[:, -1], check_failure(sol.y[:, -1], p)


def check_failure(x, p):
    '''
    Check if a state is in the failure set.
    '''
    if x[0] >= p['x0_upper_bound']:
        return True
    elif x[0] < p['x0_lower_bound']:
        return True
    elif x[1] > p['x1_upper_bound']:
        return True
    elif x[1] < p['x1_lower_bound']:
        return True
    else:
        return False


# Viability functions
def sa2xp(state_action, x, p):
    x = np.atleast_1d(state_action[:p['n_states']])
    p['thrust'] = np.atleast_1d(state_action[p['n_states']:])
    return x, p


def xp2s(x, p):
    return x
