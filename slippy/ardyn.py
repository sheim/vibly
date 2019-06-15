import numpy as np

'''
Arbitrary dynamics
x_{k+1} = map(x_k, p)
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

    x += (np.minimum(1, np.linalg.norm(x))*(p['nonlinear'](x, p)) +
          p['actions'])

    return x, check_failure(x, p)

def check_failure(x, p):
    '''
    Check if a state is in the failure set. Pass in a tuple of indices for which
    failure conditions to check. Currently: 0 for falling, 1 for direction rev.
    '''
    if np.linalg.norm(x) <= p['fail_bound']:
        return False
    else:
        return True

# Viability functions

def sa2xp(state_action, x, p):
    x = np.atleast_2d(state_action[:p['n_states']])
    p['actions'] = np.atleast_2d(state_action[p['n_states']:])
    return x, p

def xp2s(x, p):
    return x