import numpy as np
import scipy.integrate as integrate
# from numba import jit


def feasible(x, p):
    '''
    check if state is at all feasible (body/foot underground)
    returns a boolean
    '''
    if x[5] < 0 or x[1] < 0:
        return False
    return True


def p_map(x, p):
    '''
    Wrapper function for step function, returning only x_next, and -1 if failed
    Essentially, the Poincare map.
    '''
    if type(p) is dict:
        if not feasible(x, p):
            return x, True  # return failed if foot starts underground
        sol = step(x, p)
        # if len(sol.t_events) < 7:
        #     # print(len(sol.t_events))
        #     return sol.y[:, -1], True
        return sol.y[:, -1], check_failure(sol.y[:, -1])
    elif type(p) is tuple:
        vector_of_x = np.zeros(x.shape)  # initialize result array
        vector_of_fail = np.zeros(x.shape[1])
        # TODO: for shorthand, allow just a single tuple to be passed in
        # this can be done easily with itertools
        for idx, p0 in enumerate(p):
            if not feasible(x, p):
                vector_of_x[:, idx] = x[:, idx]
                vector_of_fail[idx] = True
            else:
                sol = step(x[:, idx], p0)  # p0 = p[idx]
                vector_of_x[:, idx] = sol.y[:, -1]
                vector_of_fail[idx] = check_failure(sol.y[:, -1])
        return vector_of_x, vector_of_fail
    else:
        print("WARNING: I got a parameter type that I don't understand.")
        return (x, True)


def step(x0, p, prev_sol=None):
    '''
    Take one step from apex to apex/failure.
    returns a sol object from integrate.solve_ivp, with all phases
    '''

    # * nested functions - scroll down to step code * #

    # unpacking constants for faster lookup
    GRAVITY = p['gravity']
    MASS = p['mass']
    RESTING_LENGTH = p['resting_length']
    STIFFNESS = p['stiffness']
    MAX_TIME = 5
    LEG_LENGTH_OFFSET = p['actuator_resting_length']

    # @jit(nopython=True)
    def flight_dynamics(t, x):
        # code in flight dynamics, xdot_ = f()
        return np.array([x[2], x[3], 0, -GRAVITY, x[2], x[3], 0])

    # @jit(nopython=True)
    def stance_dynamics(t, x):
        # stance dynamics
        alpha = np.arctan2(x[1] - x[5], x[0] - x[4]) - np.pi/2.0
        spring_length = np.hypot(x[0]-x[4], x[1]-x[5]) - LEG_LENGTH_OFFSET
        leg_force = STIFFNESS/MASS*(RESTING_LENGTH - spring_length)
        xdotdot = -leg_force*np.sin(alpha)
        ydotdot = leg_force*np.cos(alpha) - GRAVITY
        return np.array([x[2], x[3], xdotdot, ydotdot, 0, 0, 0])

    # @jit(nopython=True)
    def fall_event(t, x):
        '''
        Event function to detect the body hitting the floor (failure)
        '''
        return x[1]
    fall_event.terminal = True
    fall_event.direction = -1

    # @jit(nopython=True)
    def touchdown_event(t, x):
        '''
        Event function for foot touchdown (transition to stance)
        '''
        # x[1]- np.cos(p['angle_of_attack'])*RESTING_LENGTH
        # (which is = x[5])
        return x[5]
    touchdown_event.terminal = True  # no longer actually necessary...
    touchdown_event.direction = -1

    # @jit(nopython=True)
    def liftoff_event(t, x):
        '''
        Event function to reach maximum spring extension (transition to flight)
        '''
        spring_length = (np.hypot(x[0]-x[4], x[1]-x[5])
                         - p['actuator_resting_length'])
        return spring_length - RESTING_LENGTH
        # ((x[0]-x[4])**2 + (x[1]-x[5])**2) - RESTING_LENGTH**2
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
        return x[2] + 1e-5  # for numerics, allow for "straight up"
    reversal_event.terminal = True
    reversal_event.direction = -1

    # * Start of step code * #

    # TODO: properly update sol object with all info, not just the trajectories

    if prev_sol is not None:
        t0 = prev_sol.t[-1]
    else:
        t0 = 0  # starting time

    # * FLIGHT: simulate till touchdown
    events = [fall_event, touchdown_event]
    sol = integrate.solve_ivp(flight_dynamics, t_span=[t0, t0 + MAX_TIME],
                              y0=x0, events=events, max_step=0.01)

    # TODO Put each part of the step into a list, so you can concat them
    # TODO programmatically, and reduce code length.
    # if you fell, stop now
    if sol.t_events[0].size != 0:  # if empty
        if prev_sol is not None:
            sol.t = np.concatenate((prev_sol.t, sol.t))
            sol.y = np.concatenate((prev_sol.y, sol.y), axis=1)
            sol.t_events = prev_sol.t_events + sol.t_events
        return sol

    # * STANCE: simulate till liftoff
    events = [fall_event, liftoff_event, reversal_event]
    x0 = sol.y[:, -1]
    sol2 = integrate.solve_ivp(stance_dynamics,
                               t_span=[sol.t[-1], sol.t[-1] + MAX_TIME],
                               y0=x0, events=events, max_step=0.001)

    # if you fell, stop now
    if sol2.t_events[0].size != 0 or sol2.t_events[2].size != 0:  # if empty
        # concatenate all solutions
        sol.t = np.concatenate((sol.t, sol2.t))
        sol.y = np.concatenate((sol.y, sol2.y), axis=1)
        sol.t_events += sol2.t_events
        if prev_sol is not None:  # concatenate to previous solution
            sol.t = np.concatenate((prev_sol.t, sol.t))
            sol.y = np.concatenate((prev_sol.y, sol.y), axis=1)
            sol.t_events = prev_sol.t_events + sol.t_events
        return sol

    # * FLIGHT: simulate till apex
    events = [fall_event, apex_event]

    x0 = reset_leg(sol2.y[:, -1], p)
    sol3 = integrate.solve_ivp(flight_dynamics,
                               t_span=[sol2.t[-1], sol2.t[-1] + MAX_TIME],
                               y0=x0, events=events, max_step=0.01)

    # concatenate all solutions
    sol.t = np.concatenate((sol.t, sol2.t, sol3.t))
    sol.y = np.concatenate((sol.y, sol2.y, sol3.y), axis=1)
    sol.t_events += sol2.t_events + sol3.t_events

    if prev_sol is not None:
        sol.t = np.concatenate((prev_sol.t, sol.t))
        sol.y = np.concatenate((prev_sol.y, sol.y), axis=1)
        sol.t_events = prev_sol.t_events + sol.t_events

    return sol


# TODO (Steve): refactor without fail_idx for consistency
def check_failure(x, fail_idx=(0, 1, 2)):
    '''
    Check if a state is in the failure set. Pass in a tuple of indices, which
    failure conditions to check. Currently: 0 for falling, 1 for direction rev.
    '''
    for idx in fail_idx:
        if idx == 0:  # check for falling
            if np.less_equal(x[1], 0.0) or np.isclose(x[1], 0.0):
                return True
        elif idx == 1:
            if np.less_equal(x[2], 0.0):  # check for direction reversal
                return True
        # elif idx is 2: # check if you're still on the ground
        #     if np.less_equal(x[5], 0.0):
        #         return True
        # else:
        #     print("WARNING: checking for a non-existing failure id.")
    else:  # loop completes, no fail conditions triggered
        return False


def reset_leg(x, p):
    leg_length = p['resting_length'] + p['actuator_resting_length']
    x[4] = x[0] + np.sin(p['angle_of_attack'])*leg_length
    x[5] = x[1] - np.cos(p['angle_of_attack'])*leg_length
    return x


def compute_spring_length(x, p):
    return np.hypot(x[0]-x[4], x[1]-x[5]) - p['actuator_resting_length']


def compute_total_energy(x, p):
    # TODO: make this accept a trajectory, and output parts as well
    spring_length = (np.hypot(x[0]-x[4], x[1]-x[5])
                     - p['actuator_resting_length'])
    return (p['mass']/2*(x[2]**2+x[3]**2) +
            p['gravity']*p['mass']*(x[1]) +
            p['stiffness']/2 *
            (p['resting_length']-spring_length)**2)


def find_limit_cycle(x, p, options):
    '''
    Iterates over the value of key-name until limit cycle is reached
    '''
    # Settings for the root finding methods

    max_iter_bisection = 20
    max_iter_newton = 20
    tol_newton = 1e-12

    limit_cycle_found = False

    if type(p) is not dict:
        print("WARNING: p is not a dict and should be.")
        return (p, False)

    if type(options) is not dict:
        print("WARNING: p is not a dict and should be.")
        return (p, False)

    if(options['search_initial_state'] is True and
       options['search_parameter'] is True):
        print("WARNING: options search_initial_state and search_parameter "
              "cannot both be True - choose one.")
        return(p, False)

    if(options['search_initial_state'] is False
       and options['search_parameter'] is False):
        print("WARNING: options search_initial_state and search_parameter "
              "cannot both be False - choose one.")
        return(p, False)

    if(options['search_initial_state'] is True
       and (options['state_index'] < 0 or options['state_index'] > 5)):
        print("WARNING: Only one of the first 6 states can be varied to "
              "find a limit cycle")

    searchP = options['search_parameter']

    # Use the bisection method to get a good initial guess for key

    # Initial solution
    reset_leg(x, p)
    # * check for feasibility
    # Somewhat hacky, very specific to AoA
    if not feasible(x, p):
        # if we're searching for AOAs, start from a feasible one
        if p_key_name is 'angle_of_attack':
            # starting infeasible
            # assuming we're looking for an aoa
            for aoa in np.linspace(p['angle_of_attack'], np.pi/2, 9):
                p['angle_of_attack'] = aoa
                reset_leg(x, p)
                if feasible(x, p):
                    break
            else:
                return aoa, limit_cycle_found
        else:
            # if not, out of luck
            print("WARNING: requested LC at infeasible point")
            return key_delta, limit_cycle_found

    p['total_energy'] = compute_total_energy(x, p)
    (pm, step_failed) = p_map(x, p)
    err = np.abs(pm[1] - x[1])

    # Memory for the left and right solutions
    pm_left = pm
    pm_right = pm
    err_left = 0
    err_right = 0

    val = 0
    val_delta = 0
    if(searchP):
        val = p[options['parameter_name']]
        val_delta = options['parameter_search_width']
    else:
        val = x[options['state_index']]
        val_delta = options['state_search_width']

    val_left = 0
    val_right = 0
    # After this for loop returns the angle of attack will be known to
    # a tolerance of pi/4 / 2^(max_iter_bisection)
    for i in range(0, max_iter_bisection):
        # Evaluate the solution to the left of the current best solution val
        val_left = val - val_delta
        if(searchP):
            p[options['parameter_name']] = val_left
        else:
            x[options['state_index']] = val_left

        x = reset_leg(x, p)
        p['total_energy'] = compute_total_energy(x, p)
        (pm_left, step_failed_left) = p_map(x, p)
        err_left = np.abs(pm_left[1] - x[1])

        # Evaluate the solution to the right of the current best solution val
        val_right = val + val_delta
        if(searchP):
            p[options['parameter_name']] = val_right
        else:
            x[options['state_index']] = val_right

        x = reset_leg(x, p)
        p['total_energy'] = compute_total_energy(x, p)
        (pm_right, step_failed_right) = p_map(x, p)
        err_right = np.abs(pm_right[1] - x[1])

        if((err_left < err and step_failed_left is False) and
           (err_left <= err_right or step_failed_right is True)):
            err = err_left
            val = val_left

        if((err_right < err and step_failed_right is False) and
           (err_right < err_left or step_failed_left is True)):
            err = err_right
            val = val_right

        val_delta = 0.5*val_delta

    # polish the root using Newton's method

    idx = 0
    h = np.sqrt(np.finfo('float64').eps)
    while np.abs(err) > tol_newton and idx < max_iter_newton:

        # Compute the error
        if(searchP):
            p[options['parameter_name']] = val
        else:
            x[options['state_index']] = val
        x = reset_leg(x, p)
        p['total_energy'] = compute_total_energy(x, p)
        (pm, step_failed) = p_map(x, p)

        if step_failed:
            break
        err = pm[1]-x[1]

        # Compute D(error)/D(val) using a numerical derivative
        if(searchP):
            p[options['parameter_name']] = val-h
        else:
            x[options['state_index']] = val-h

        x = reset_leg(x, p)
        p['total_energy'] = compute_total_energy(x, p)
        (pm, step_failed) = p_map(x, p)
        errL = pm[1]-x[1]

        if(searchP):
            p[options['parameter_name']] = val+h
        else:
            x[options['state_index']] = val+h

        x = reset_leg(x, p)
        p['total_energy'] = compute_total_energy(x, p)
        (pm, step_failed) = p_map(x, p)
        errR = pm[1]-x[1]

        # Compute a Newton step and take it
        DerrDval = (errR-errL)/(2*h)
        val = val - err/DerrDval
        idx = idx+1

    if np.abs(err) > tol_newton:
        print("WARNING: Newton method failed to converge")
        limit_cycle_found = False
    else:
        limit_cycle_found = True

    # p[p_key_name] = val
    return val, limit_cycle_found

# * Functions for Viability


def xp2s(x, p):
    '''
    map an apex state to its dimensionless normalized height
    TODO: make this accept trajectories
    '''
    # assert np.isclose(x[3], 0), "state x: " + str(x)
    potential_energy = p['mass']*p['gravity']*x[1]
    kinetic_energy = p['mass']/2*x[2]**2
    return potential_energy/(potential_energy + kinetic_energy)


def s2x(x, p, s):
    '''
    map a desired dimensionless height `s` to it's state-vector
    '''
    if 'total_energy' not in p:
        print('WARNING: you did not initialize your parameters with '
              'total energy. You really should do this...')
    # check that we are at apex
    assert np.isclose(x[3], 0), "state x: " + str(x) + " and e: " + str(s)

    x_new = p['x0']
    x_new[1] = p['total_energy']*s/p['mass']/p['gravity']
    x_new[2] = np.sqrt(p['total_energy']*(1-s)/p['mass']*2)
    x_new[3] = 0.0  # shouldn't be necessary, but avoids errors accumulating
    x = reset_leg(x, p)
    return x_new


def sa2xp(state_action, p):
    '''
    Specifically map state_actions to x and p
    '''
    p_new = p.copy()
    p_new['angle_of_attack'] = state_action[1]
    x = s2x(p_new['x0'], p_new, state_action[0]).copy()
    return x, p_new
