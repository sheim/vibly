import numpy as np

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

def mass_matrix(x, p):
    # unpack
    m1 = p['m1']
    m2 = p['m2']
    l1 = p['l1']
    l2 = p['l2']

    a = m1*l1**2 + m2*l1**2
    b = m2*l2**2
    c = m2*l1*l2

    return np.array([[a + b + 2*c*np.cos(x[1]), b + c*np.cos(x[1])],
                     [b + c*np.cos(x[1]), b]])


def coriolis(x, p):
    m2 = p['m2']
    l1 = p['l1']
    l2 = p['l2']

    c = m2*l1*l2

    return np.array([[-c*np.sin(x[1])*x[3], -c*np.sin(x[0]+x[1])],
                     [c*np.sin(x[1])*x[2], 0]])


def gravitational(x, p):
    m1 = p['m1']
    m2 = p['m2']
    l1 = p['l1']
    l2 = p['l2']
    g = p['g']

    d = g*m1*l1 + g*m2*l1
    e = g*m2*l2

    return np.array([-d*np.sin(x[0]) - e*np.sin(x[0]+x[1]),
                    e*np.sin(x[0]+x[1])])


def p_map(x, p):
    '''
    Dynamics function of your system
    Note that the control input is included in the parameter,
    and needs to be unpacked.
    '''
    if check_failure(x, p):
        return x, True

    M = mass_matrix(x, p)
    C = coriolis(x, p)
    G = gravitational(x, p)

    tau = np.array([0, p['torque']])

    # TODO check dimensions
    x += p['t_step']*np.stack(x[0:2], np.linalg.solve(M, tau - C*x[2:] - G))

    return x, check_failure(x, p)


def check_failure(x, p):
    '''
    Check if a state is in the failure set.
    '''
    elbow_height = p['l1']*np.cos(x[0])
    end_eff_height = p['l1']*np.cos(x[0]) + p['l2']*np.cos(x[1]-x[0])
    if elbow_height <= 0.0:
        return True
    elif end_eff_height <= 0.0:
        return True
    else:
        return False


# Viability functions
def sa2xp(state_action, x, p):
    x = np.atleast_1d(state_action[:p['n_states']])
    p['torque'] = np.atleast_1d(state_action[p['n_states']:])
    p['torque'] = np.min([p['u_upper_bound'], p['torque']])  # bound torque
    p['torque'] = np.max([p['u_lower_bound'], p['torque']])
    return x, p


def xp2s(x, p):
    return x
