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


def p_map(x, p):
    '''
    Dynamics function of your system
    Note that the control input is included in the parameter,
    and needs to be unpacked.
    '''
    if check_failure(x, p):
        return x, True

    OMEGA = p['angular_speed']
    GM = p['geocentric_constant']
    THRUST = p['thrust']
    MAX_TIME = 1.0/p['control_frequency']
    MASS = p['mass']

    def continuous_dynamics(t, x):

        f = np.zeros_like(x)
        # np.max([0, np.tanh(0.75*(CEILING - x[0]))])*GRAVITY
        f[0] = x[1]
        # x[0] = np.min([CEILING, x[0]])  # saturate at ceiling (x=0)
        f[1] = (-GM/(x[0]**2) + OMEGA**2*x[0] + THRUST/MASS)
        return f

    # do Euler integration manually, probably faster and just as good
    timestep = 0.01
    for t in np.arange(0.0, MAX_TIME, timestep):
        x += timestep*continuous_dynamics(t, x)

    return x, check_failure(x, p)

    # sol = integrate.solve_ivp(continuous_dynamics, t_span=[0, MAX_TIME], y0=x)
    # return sol.y[:, -1], check_failure(sol.y[:, -1], p)


def check_failure(x, p):
    '''
    Check if a state is in the failure set.
    '''
    if x[0] >= p['radio_range']:  # out of comms range
        # print("satellite lost")
        return True
    elif x[0] <= p['radius']: # earth radius  # hits the ground
        # print("satellite crashed")
        return True
    return False


# Viability functions
def sa2xp(state_action, p):
    p_new = p.copy()
    x = np.atleast_1d(state_action[:p['n_states']])
    actions = np.atleast_1d(state_action[p['n_states']:])
    p_new['thrust'] = actions[0]
    # p_new['thrust_horizontal'] = actions[1]
    return x, p_new


def xp2s(x, p):
    return x
