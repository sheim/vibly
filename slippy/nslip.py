import numpy as np
import scipy.integrate as integrate
# from numba import jit

# Parameter set:
# p.g = 9.81;
# p.k10 = 30;
# p.alpha = pi/9;
# p.m = 80;
# p.beta0 = 170/180*pi; % resting angle. Cannot be n*pi
# p.l1 = sqrt(1/(2*(1-cos(p.beta0))));
# p.l2 = p.l1;
# p.l0 = sqrt(p.l1^2 + p.l2^2 - 2*p.l1*p.l2*np.cos(p.beta0));
# p.beta10 = acos( (p.l1^2 + p.l2^2 - (0.9*p.l0)^2)/(2*p.l1*p.l2));
# p.c = p.k10*p.m*p.g/p.l0 * (p.l1*p.l2*0.1)/(0.9) * sin(p.beta10) / (p.beta0 - p.beta10);

def feasible(x, p):
    '''
    check if state is at all feasible (body/foot underground)
    returns a boolean
    '''
    if x[5] < 0 or x[1] < 0:
        return False
    return True

def poincare_map(x, p):
    '''
    Wrapper function for step function, returning only x_next, and -1 if failed
    Essentially, the Poincare map.
    '''
    if type(p) is dict:
        if not feasible(x, p):
            return x, True # return failed if foot starts underground
        sol = step(x, p)
        # if len(sol.t_events) < 7:
        #     # print(len(sol.t_events))
        #     return sol.y[:, -1], True
        return sol.y[:, -1], check_failure(sol.y[:, -1])
    elif type(p) is tuple:
        vector_of_x = np.zeros(x.shape) # initialize result array
        vector_of_fail = np.zeros(x.shape[1])
        # TODO: for shorthand, allow just a single tuple to be passed in
        # this can be done easily with itertools
        for idx, p0 in enumerate(p):
            if not feasible(x, p):
                vector_of_x[:, idx] = x[:, idx]
                vector_of_fail[idx] = True
            else:
                sol = step(x[:, idx], p0) # p0 = p[idx]
                vector_of_x[:, idx] = sol.y[:, -1]
                vector_of_fail[idx] = check_failure(sol.y[:, -1])
        return vector_of_x, vector_of_fail
    else:
        print("WARNING: I got a parameter type that I don't understand.")
        return (x, True)


def step(x0, p, prev_sol = None):
    '''
    Take one step from apex to apex/failure.
    returns a sol object from integrate.solve_ivp, with all phases
    '''

    # * nested functions - scroll down to step code * #

    # unpacking constants for faster lookup
    AOA = p['angle_of_attack'] #
    GRAVITY = p['gravity'] #
    MASS = p['mass'] #
    RESTING_ANGLE = p['resting_angle']
    UPPER_LEG = p['upper_leg']
    LOWER_LEG = p['lower_leg']
    RESTING_LENGTH = (np.sqrt(p['upper_leg']**2 + p['lower_leg']**2
                    - 2*p['upper_leg']*p['lower_leg']*np.cos(p['resting_angle']) ))
    STIFFNESS = p['stiffness']
    TOTAL_ENERGY = p['total_energy']
    # SPECIFIC_STIFFNESS = p['stiffness'] / p['mass']
    MAX_TIME = 5

    # @jit(nopython=True)
    def flight_dynamics(t, x):
        # code in flight dynamics, xdot_ = f()
        return np.array([x[2], x[3], 0, -GRAVITY, x[2], x[3], 0])

    # @jit(nopython=True)
    def stance_dynamics2(t, x):
        # stance dynamics
        alpha = np.arctan2(x[1] - x[5], x[0] - x[4]) - np.pi/2.0
        leg_length = np.hypot(x[0]-x[4], x[1]-x[5])

        # leg_angle = np.arccos( (UPPER_LEG**2 + LOWER_LEG**2 - leg_length**2) /
        #                     (2*UPPER_LEG*LOWER_LEG))
        leg_angle = np.arccos( 1 - 2*leg_length**2/RESTING_LENGTH**2)
        leg_force = 4*leg_length*STIFFNESS*(leg_angle - RESTING_ANGLE) / (
                    RESTING_LENGTH**2*np.sin(leg_angle))
        xdotdot = leg_force/MASS*np.sin(alpha)
        ydotdot =  leg_force/MASS*np.cos(alpha) - GRAVITY
        return np.array([x[2], x[3], xdotdot, ydotdot, 0, 0, 0])

    def stance_dynamics(t, x):
        # since legs are massless, the orientation of the knee doesn't matter.
        alpha = np.arctan2(x[1] - x[5], x[0] - x[4]) - np.pi/2.0
        leg_length = np.hypot(x[0]-x[4], x[1]-x[5])
        # if np.greater_equal((UPPER_LEG**2+LOWER_LEG**2 - leg_length**2)
        #                 /(2*UPPER_LEG*LOWER_LEG), 1):
        #     print("warning")
        beta = np.arccos((UPPER_LEG**2+LOWER_LEG**2 - leg_length**2)
                        /(2*UPPER_LEG*LOWER_LEG))
        # if np.isnan(beta): #TODO test for minimum value...
        #     print("HELLO!")
        # sinbeta = max(np.sin(beta), 1e-5)
        tau = STIFFNESS*(RESTING_ANGLE - beta)
        leg_force = leg_length/(UPPER_LEG*LOWER_LEG) * tau / np.sin(beta)
        xdotdot = -leg_force/MASS*np.sin(alpha)
        ydotdot =  leg_force/MASS*np.cos(alpha) - GRAVITY
        return np.array([x[2], x[3], xdotdot, ydotdot, 0, 0, 0])

    # @jit(nopython=True)
    def fall_event(t, x):
        '''
        Event function to detect the body hitting the floor (failure)
        '''
        return x[1]
    fall_event.terminal = True
    fall_event.terminal = -1

    # @jit(nopython=True)
    def touchdown_event(t, x):
        '''
        Event function for foot touchdown (transition to stance)
        '''
            # x[1]- np.cos(p['angle_of_attack'])*RESTING_LENGTH
            # (which is = x[5])
        return x[5]
    touchdown_event.terminal = True # no longer actually necessary...
    touchdown_event.direction = -1

    # @jit(nopython=True)
    def liftoff_event(t, x):
        '''
        Event function to reach maximum spring extension (transition to flight)
        '''
        return np.hypot(x[0]-x[4], x[1]-x[5]) - RESTING_LENGTH**2
    liftoff_event.terminal = True
    liftoff_event.direction = 1

    # @jit(nopython=True)
    def apex_event(t, x):
        '''
        Event function to reach apex
        '''
        return x[3]
    apex_event.terminal = True


    # @jit(nopython=True)
    def reversal_event(t, x):
        '''
        Event function for direction reversal
        '''
        return x[2] + 1e-5 # for numerics, allow for "straight up"
    reversal_event.terminal = True
    reversal_event.direction = -1

    # * Start of step code * #

    # TODO: properly update sol object with all info, not just the trajectories

    if prev_sol is not None:
        t0 = prev_sol.t[-1]
    else:
        t0 = 0 # starting time

    # * FLIGHT: simulate till touchdown
    events = [fall_event, touchdown_event]
    sol = integrate.solve_ivp(flight_dynamics,
        t_span = [t0, t0 + MAX_TIME], y0 = x0, events = events, max_step = 0.01)

    # TODO Put each part of the step into a list, so you can concat them
    # TODO programmatically, and reduce code length.
        # if you fell, stop now
    if sol.t_events[0].size != 0: # if empty
        if prev_sol is not None:
            sol.t = np.concatenate((prev_sol.t, sol.t))
            sol.y = np.concatenate((prev_sol.y, sol.y), axis =1)
            sol.t_events = prev_sol.t_events + sol.t_events
        return sol

    # * STANCE: simulate till liftoff
    events = [fall_event, liftoff_event, reversal_event]
    x0 = sol.y[:, -1]
    sol2 = integrate.solve_ivp(stance_dynamics,
        t_span = [sol.t[-1], sol.t[-1] + MAX_TIME], y0 = x0,
        events=events, max_step=0.0005)

    # if you fell, stop now
    if sol2.t_events[0].size != 0 or sol2.t_events[2].size != 0: # if empty
        # concatenate all solutions
        sol.t = np.concatenate((sol.t, sol2.t))
        sol.y = np.concatenate((sol.y, sol2.y), axis = 1)
        sol.t_events += sol2.t_events
        if prev_sol is not None: # concatenate to previous solution
            sol.t = np.concatenate((prev_sol.t, sol.t))
            sol.y = np.concatenate((prev_sol.y, sol.y), axis =1)
            sol.t_events = prev_sol.t_events + sol.t_events
        return sol

    # * FLIGHT: simulate till apex
    events = [fall_event, apex_event]

    x0 = reset_leg(sol2.y[:, -1], p)
    sol3 = integrate.solve_ivp(flight_dynamics,
            t_span = [sol2.t[-1], sol2.t[-1] + MAX_TIME], y0 = x0,
            events=events, max_step=0.01)

    # concatenate all solutions
    sol.t = np.concatenate((sol.t, sol2.t, sol3.t))
    sol.y = np.concatenate((sol.y, sol2.y, sol3.y), axis = 1)
    sol.t_events += sol2.t_events + sol3.t_events

    if prev_sol is not None:
        sol.t = np.concatenate((prev_sol.t, sol.t))
        sol.y = np.concatenate((prev_sol.y, sol.y), axis =1)
        sol.t_events = prev_sol.t_events + sol.t_events

    return sol

def check_failure(x, fail_idx = (0, 1, 2)):
    '''
    Check if a state is in the failure set. Pass in a tuple of indices for which
    failure conditions to check. Currently: 0 for falling, 1 for direction rev.
    '''
    for idx in fail_idx:
        if idx is 0: # check for falling
            if np.less_equal(x[1], 0.0) or np.isclose(x[1], 0.0):
                return True
        elif idx is 1:
            if np.less_equal(x[2], 0.0): # check for direction reversal
                return True
        # elif idx is 2: # check if you're still on the ground
        #     if np.less_equal(x[5], 0.0):
        #         return True
        # else:
        #     print("WARNING: checking for a non-existing failure id.")
    else: # loop completes, no fail conditions triggered
        return False

def reset_leg(x, p):
    # TODO: check if sqrt can be replaced with np.hypot
    resting_length = (np.sqrt(p['upper_leg']**2 + p['lower_leg']**2
                    - 2*p['upper_leg']*p['lower_leg']*np.cos(p['resting_angle']) ))
    x[4] = x[0] + np.sin(p['angle_of_attack'])*resting_length
    x[5] = x[1] - np.cos(p['angle_of_attack'])*resting_length
    return x

def compute_total_energy(x, p):
    # TODO: make this accept a trajectory, and output parts as well
    # resting_length = (np.sqrt(p['upper_leg']**2 + p['lower_leg']**2
    #                 - 2*p['upper_leg']*p['lower_leg']*np.cos(p['resting_angle']) ))
    leg_length = np.hypot(x[0]-x[4], x[1]-x[5])
    beta = np.arccos( (p['upper_leg']**2 + p['lower_leg']**2 - leg_length**2)
                /(2*p['upper_leg']*p['lower_leg']))
    return (p['mass']/2*(x[2]**2+x[3]**2) +
            p['gravity']*p['mass']*(x[1]) +
            p['stiffness']/2*(beta - p['resting_angle'])**2)

### Functions for Viability
def map2s(x, p):
    '''
    map an apex state to its dimensionless normalized height
    TODO: make this accept trajectories
    '''
    # assert np.isclose(x[3], 0), "state x: " + str(x)
    potential_energy = p['mass']*p['gravity']*x[1]
    kinetic_energy = p['mass']/2*x[2]**2
    return potential_energy/(potential_energy + kinetic_energy)

# TODO: Steve cleans this
def map2x(x, p, s):
    '''
    map a desired dimensionless height `s` to it's state-vector
    '''
    if 'total_energy' not in p:
        print('WARNING: you did not initialize your parameters with '
        'total energy. You really should do this...')
    # check that we are at apex
    assert np.isclose(x[3], 0), "state x: " + str(x) + " and e: " + str(s)

    x_new = x
    x_new[1] = p['total_energy']*s/p['mass']/p['gravity']
    x_new[2] = np.sqrt(p['total_energy']*(1-s)/p['mass']*2)
    x_new[3] = 0.0 # shouldn't be necessary, but avoids errors accumulating
    x = reset_leg(x, p)
    return x_new

def mapSA2xp_height_angle(state_action, p):
    '''
    Specifically map state_actions to x and p
    '''
    p['angle_of_attack'] = state_action[1]
    x = map2x(p['x0'], p, state_action[0])
    return x, p
