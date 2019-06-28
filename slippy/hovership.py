import numpy as np

'''
space attempting to reconnoitre the surface of a planet.
Must ensure not to go to the dark side of the planet.
x_{k+1} = map(x_k, p)
x: (x1, x2)
x1: altitude
p: dict of parameters. For convenience, actions are also stored here.
'''

# map: x_k+1, failed = map


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

    x[0] += gravity(x, p) - p['thrust']
    x[0] = np.max([0, x[0]])  # saturate at ceiling (x=0)
    return x, check_failure(x, p)


def check_failure(x, p):
    '''
    Check if a state is in the failure set.
    '''
    if x[0] > p['ground_height']:
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
