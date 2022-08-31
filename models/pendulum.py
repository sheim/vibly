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
    # * no failures for the pendulum

    MASS = p['mass']
    GRAVITY = p['gravity']
    LENGTH = p['length']
    TORQUE = p['torque']
    CTRL_DT = 1.0/p['control_frequency']
    SIM_DT = 1.0/p['simulation_frequency']

    def continuous_dynamics(t, x):

        f = np.zeros_like(x)
        f[0] = x[1]
        # x[0] = np.min([CEILING, x[0]])  # saturate at ceiling (x=0)
        f[1] = -GRAVITY/LENGTH*np.sin(x[0]) + TORQUE/(MASS*LENGTH**2)
        return f

    # do Euler integration manually, probably faster and just as good
    for t in np.arange(0.0, CTRL_DT, SIM_DT):
        x += SIM_DT*continuous_dynamics(t, x)

    return x, False


# * Viability functions


def sa2xp(state_action, p):
    p_new = p.copy()
    x = np.atleast_1d(state_action[:p['n_states']])
    p_new['torque'] = np.atleast_1d(state_action[p['n_states']:])
    return x, p_new


def xp2s(x, p):
    return x
