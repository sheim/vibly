import numpy as np
import scipy.integrate as integrate
'''
The linear-inverted pendulum in sagittal plane, from Zaytsev & Ruina.
Poincare section at mid-stance (no x coordinate for high-level).
state: \dot{x}
actions: step-length, step-time
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

    v0 = x[0]

    Tst = p['step_timing']
    Xst = p['step_location']
    # GRAVITY = p['gravity']
    # HEIGHT = p['height']

    # * analytical solution

    # v1_squared = v0**2 - Xst**2 + 2*v0*Xst + 2*v0*Xst*np.sinh(Tst)
    # if v1_squared<0:  # moved backwards
    #     return x, True
    # elif v0*np.sinh(Tst) >= p['max_step']:  # stance leg hits limit
    #     print("stretching stance leg")
    #     return x, True
    # elif v0*np.sinh(Tst) <= Xst - p['max_step']:  # next step too far
    #     print("next step too far")
    #     return x, True
    # elif v0*np.sinh(Tst) >= Xst:  # always step ahead of CoG
    #     print("stepping backwards")
    #     return x, True
    # else:
    #     return np.sqrt(v1_squared), False
    # TODO constraint on velocity at step transition... to make sure you actually reach the next mid-stance...?


    # * Numerical implementation
    def continuous_dynamics(t, x):
        return np.array(x[1], x[0])

    # simulate till touch-down
    x0 = x.copy()
    dt = 0.01
    for t in np.arange(start=0, stop=Tst, step=dt):
        x0 += dt*x0[::-1]  # x0[0] += x0[1], x0[1] += x0[0]

    # check constraints
    if x0[0] >= p['max_step']: # stance leg over-stretched
        return x, True
    elif x0[0] >= Xst:  # stepping backwards
        return x, True
    elif Xst - x0[0] >= p['max_step']:  # trying to step too far
        return x, True

    # coordinate-switch
    x0[0] = x0[0] - Xst

    # simulate till hip reaches stance-foot
    dt = 0.0001 # HACK do proper crossing event
    for t in np.arange(start=0, stop=5.0, step=dt):
        x0 += dt*x0[::-1]  # x0[0] += x0[1], x0[1] += x0[0]
        if x0[0]>=0.0:
            # print("REACHED")
            break
    else:  # did not reach mid-stance for toooo long
        return x, True
    
    return x, False


# def check_failure(x, p):
#     '''
#     Check if a state-action pair is in the failure set.
#     inputs:
#     x: ndarray of the state
#     p: dict of parameters and actions
#     outputs:
#     failed: bool indicating true if the system failed, false otehrwise

#     '''

#     # * If the CoG reaches is too far from the commanded step-location
#     if p['step_location'] - x[0] > p['leg_reach']:
#         return True
#     elif np.abs(x[0]) > p['leg_reach']:  # if we stretch stance too far
#         return True
#     else:
#         return False


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
    p_new = p.copy()
    p_new['step_timing'] = np.atleast_1d(state_action[1])
    p_new['step_location'] = np.atleast_1d(state_action[2])
    return x.flatten(), p_new

def sa2xp_num(state_action, p):
    '''
    maps a state-action pair to continuous-time state x and parameter dict p
    inputs:
    state_action: ndarray of state-action pair (s, a)
    p: dict of parameters (and actions)
    outputs:
    x: ndarray of states for simulation
    p: dict of parameters updated with new action
    '''
    x = np.array([0, state_action[0]])  # always start with x0 = 0 (Poinc)
    p_new = p.copy()
    p_new['step_timing'] = np.atleast_1d(state_action[1])
    p_new['step_location'] = np.atleast_1d(state_action[2])
    return x.flatten(), p_new

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

def xp2s_num(x, p):
    '''
    maps a state x to the high-level state s
    inputs:
    x: ndarray of state
    p: dict of parameters and actions
    outputs:
    s: high-level state used in the viability algorithms
    '''
    return x[1]
