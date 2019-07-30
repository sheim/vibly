import numpy as np
import scipy.integrate as integrate
'''
A spaceship attempting to reconnoitre the surface of a planet.
However, the planet has an unusual gravitational field... getting too close to
the surface may result in getting sucked in, with no escape!
'''

# * Transition Map. This is your oracle.
def p_map(x, p):
    '''
    The transition map of your system.
    inputs:
    x: ndarray of the current state
    p: dict containing all parameters and action (control input)
    outputs:
    x: ndarray state at next iteration
    failed: boolean indicating if the system has failed
    '''

    # * we first check if the state is already in the failure state
    # * this can happen if the initial state of a new sequence is chosen poorly
    # * this can also happen if the failure state depends on the action
    if check_failure(x, p):
        return x, True

    # unpack the parameter dict
    THRUST = np.min([p['max_thrust'], p['thrust']])
    BASE_GRAVITY = p['base_gravity']
    GRAVITY = p['gravity']
    MAX_TIME = 1.0/p['control_frequency']
    CEILING = p['ceiling']

    # * for convenience, we define the continuous-time dynamics, and use
    # * scipy.integrate to solve this over one control time-step (MAX_TIME)
    # * what you put in here can be as complicated as you like.
    def continuous_dynamics(t, x):
        grav_field = np.max([0, np.tanh(0.75*(CEILING - x[0]))])*GRAVITY
        f = - BASE_GRAVITY - grav_field + THRUST
        return f

    def ceiling_event(t, x):
        '''
        Event function to detect the body hitting the floor (failure)
        '''
        return x-CEILING
    ceiling_event.terminal = True
    ceiling_event.direction = 1

    sol = integrate.solve_ivp(continuous_dynamics, t_span=[0, MAX_TIME], y0=x,
                              events=ceiling_event)

    # * we return the final
    return sol.y[:, -1], check_failure(sol.y[:, -1], p)


def check_failure(x, p):
    '''
    Check if a state-action pair is in the failure set.
    inputs:
    x: ndarray of the state
    p: dict of parameters and actions
    outputs:
    failed: bool indicating true if the system failed, false otehrwise

    '''

    # * For this example, the system has failed if it ever hits the ground
    if x[0] < 0:
        return True
    else:
        return False


# Viability functions

# * since the simulation may live in a higher-dimensional state-action space,
# * (this is the case when using a hierarchical control structure),
# * we need to map a state-action pair to the x (state) and p (parameters)
# * we also properly assign the action in the parameter dict
def sa2xp(state_action, p):
    '''
    maps a state-action pair to continuous-time state x and parameter dict p
    inputs:
    state_action: ndarray of state-action pair (s, a)
    p: dict of parameters (and actions)
    outputs:
    x: ndarray of states for simulation
    p: dict of parameters updated with new action
    '''
    x = np.atleast_1d(state_action[:p['n_states']])
    p['thrust'] = np.atleast_1d(state_action[p['n_states']:])
    return x.flatten(), p

# * we also need to provide a function to go back from the continuous-time
# * state x to the high-level state s
def xp2s(x, p):
    '''
    maps a state x to the high-level state s
    inputs:
    x: ndarray of state
    p: dict of parameters and actions
    outputs:
    s: high-level state used in the viability algorithms
    '''
    return x
