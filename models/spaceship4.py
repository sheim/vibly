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

    THRUST_V = p['thrust_vertical']
    THRUST_H = p['thrust_horizontal']
    WIND = p['wind']
    GRAVITY = p['gravity']
    BASE_GRAVITY = p['base_gravity']
    CEILING = p['ceiling']
    MAX_TIME = 1.0/p['control_frequency']

    def continuous_dynamics(t, x):
        grav_field = np.max([0, np.tanh(0.75*(CEILING - x[0]))])*GRAVITY
        f = np.zeros_like(x)
        # np.max([0, np.tanh(0.75*(CEILING - x[0]))])*GRAVITY
        f[0] = - BASE_GRAVITY - grav_field + THRUST_V
        # x[0] = np.min([CEILING, x[0]])  # saturate at ceiling (x=0)
        f[1] = WIND*np.sin(x[0]*np.pi) + THRUST_H
        return f

    def ceiling_event(t, x):
        '''
        Event function to detect the body hitting the floor (failure)
        '''
        return x[0]-CEILING
    ceiling_event.terminal = True
    ceiling_event.direction = 1

    sol = integrate.solve_ivp(continuous_dynamics, t_span=[0, MAX_TIME], y0=x,
                              events=ceiling_event)

    return sol.y[:, -1], check_failure(sol.y[:, -1], p)


def check_failure(x, p):
    '''
    Check if a state is in the failure set.
    '''
    # * For this example, the system has failed if it ever hits the ground
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
def sa2xp(state_action, p):
    p_new = p.copy()
    x = np.atleast_1d(state_action[:p['n_states']])
    actions = np.atleast_1d(state_action[p['n_states']:])
    p_new['thrust_vertical'] = actions[0]
    p_new['thrust_horizontal'] = actions[1]
    return x, p_new


def xp2s(x, p):
    return x
