import numpy as np
import scipy.integrate as integrate
from numba import jit

def limit_cycle(x, p, p_key_name, p_key_width):
    '''
    Iterates over the angle of attack of the leg until a limit cycle is reached.
    '''
    #Settings for the root finding methods
    max_iter_bisection  = 10
    max_iter_newton     = 10
    tol_newton          = 1e-12

    limit_cycle_found = False

    if type(p) is dict:
        if p['model_type'] != 0:
            print( "limit_cycle: can only be called with a slip model "
                  +"(model_type 0)")
            limit_cycle_found = False
            return (p, limit_cycle_found)

        #Use the bisection method to get a good initial guess for key
        key_delta = p_key_width

        #Initial solution
        x                 = reset_leg(x, p)
        (pm, step_failed) = poincare_map(x, p)
        err               = np.abs(pm[1] - x[1])

        #Memory for the left and right solutions
        pm_left     = pm
        pm_right    = pm
        err_left    = 0
        err_right   = 0

        key = p[p_key_name]

        #After this for loop returns the angle of attack will be known to
        # a tolerance of pi/4 / 2^(max_iter_bisection)
        for i in range(0, max_iter_bisection):
            p[p_key_name] = key - key_delta
            x = reset_leg(x, p)
            (pm_left, step_failed_left) = poincare_map(x, p)
            err_left = np.abs(pm_left[1] - x[1])

            p[p_key_name] = key + key_delta
            x = reset_leg(x, p)
            (pm_right, step_failed_right) = poincare_map(x, p)
            err_right= np.abs(pm_right[1] - x[1])

            if( (err_left < err and step_failed_left==False) and
                (err_left <= err_right or step_failed_right==True)):
                err = err_left
                key = key - key_delta

            if( (err_right < err     and step_failed_right==False) and
                (err_right < err_left or step_failed_left==True)):
                err = err_right
                key = key + key_delta

            key_delta = 0.5*key_delta

        #Polish the root using Newton's method

        iter=0
        h = np.sqrt(np.finfo('float64').eps)
        while np.abs(err) > tol_newton and iter < max_iter_newton:

            #Compute the error
            p[p_key_name] = key
            x = reset_leg(x, p)
            (pm, step_failed) = poincare_map(x, p)
            err = pm[1]-x[1]

            #Compute D(error)/D(key) using a numerical derivative
            p[p_key_name] = key-h
            x = reset_leg(x, p)
            (pm, step_failed) = poincare_map(x, p)
            errL = pm[1]-x[1]

            p[p_key_name] = key+h
            x = reset_leg(x, p)
            (pm, step_failed) = poincare_map(x, p)
            errR = pm[1]-x[1]

            #Compute a Newton step and take it
            DerrDkey = (errR-errL)/(2*h)
            key = key -err/DerrDkey
            iter=iter+1

        if np.abs(err) > tol_newton:
            print("WARNING: Newton method failed to converge")
            limit_cycle_found = False
        else:
            limit_cycle_found = True

        p[p_key_name] = key
        return (p,limit_cycle_found)

    else:
        print("WARNING: p is not a dict and should be.")
        return p



def poincare_map(x, p):
    '''
    Wrapper function for step function, returning only x_next, and -1 if failed
    Essentially, the Poincare map.
    '''
    if type(p) is dict:
        if x[5] < 0:
            return x, True # return failed if foot starts underground
        sol = step(x, p)
        return sol.y[:, -1], sol.failed
    elif type(p) is tuple:
        vector_of_x = np.zeros(x.shape) # initialize result array
        vector_of_fail = np.zeros(x.shape[1])
        # TODO: for shorthand, allow just a single tuple to be passed in
        # this can be done easily with itertools
        for idx, p0 in enumerate(p):
            if x[5] < 0:
                vector_of_x[:, idx] = x[:, idx]
                vector_of_fail[idx] = True
            else:
                sol = step(x[:, idx], p0) # p0 = p[idx]
                vector_of_x[:, idx] = sol.y[:, -1]
                vector_of_fail[idx] = sol.failed
        return (vector_of_x, vector_of_fail)
    else:
        print("WARNING: I got a parameter type that I don't understand.")
        return x


def step(x, p):
    '''
    Take one step from apex to apex/failure.
    returns a sol object from integrate.solve_ivp, with all phases
    '''

    # * nested functions - scroll down to step code * #

    # unpacking constants
    MAX_TIME = 5

    MODEL_TYPE = p['model_type']
    assert( MODEL_TYPE == 0 or MODEL_TYPE == 1)
    if MODEL_TYPE == 0 :
        assert(len(x) == 6)
    elif MODEL_TYPE == 1:
        assert(len(x) == 8)
    else:
        raise Exception('model_type is not set correctly')

    AOA = p['angle_of_attack']
    GRAVITY = p['gravity']

    MASS = p['mass']
    SPRING_RESTING_LENGTH = p['spring_resting_length']
    STIFFNESS = p['stiffness']
    TOTAL_ENERGY = p['total_energy']
    SPECIFIC_STIFFNESS = p['stiffness'] / p['mass']

    ACTUATOR_RESTING_LENGTH = p['actuator_resting_length']
    ACTUATOR_DAMPING = p['actuator_normalized_damping']*STIFFNESS
    ACTUATOR_FORCE = 0

    ACTUATOR_SPECIFIC_DAMPING = ACTUATOR_DAMPING/MASS
    ACTUATOR_SPECIFIC_FORCE   = ACTUATOR_FORCE/MASS

#    @jit(nopython=True)
    def flight_dynamics(t, x):
        # code in flight dynamics, xdot_ = f()
        if(MODEL_TYPE == 0):
            return np.array([x[2], x[3], 0, -GRAVITY, x[2], x[3]])
        elif(MODEL_TYPE == 1):
            #The actuator length does not change, and no work is done.
            return np.array([x[2], x[3], 0, -GRAVITY, x[2], x[3], 0, 0])
        else:
            raise Exception('model_type is not set correctly')

#    @jit(nopython=True)
    def stance_dynamics(t, x):
        # stance dynamics
        alpha = np.arctan2(x[1] - x[5], x[0] - x[4]) - np.pi/2.0
        leg_length = np.hypot(x[0]-x[4], x[1]-x[5])
        output = x

        spring_length = compute_spring_length(x, p)
        spring_force  = -STIFFNESS*(spring_length-SPRING_RESTING_LENGTH)

        ldotdot = spring_force/MASS
        xdotdot = -ldotdot*np.sin(alpha)
        ydotdot =  ldotdot*np.cos(alpha) - GRAVITY

        if MODEL_TYPE == 0:
            output = np.array([x[2], x[3], xdotdot, ydotdot, 0, 0])
        elif MODEL_TYPE == 1:
            actuator_open_loop_force = 0
            if np.shape(p['actuator_force'])[0] > 0:
                actuator_open_loop_force = np.interp(t,
                    p['actuator_force'][0,:],
                    p['actuator_force'][1,:],
                    period=p['actuator_force_period'])

            actuator_damping_force = spring_force-actuator_open_loop_force

            ladot = -actuator_damping_force/ACTUATOR_DAMPING
            wadot = (actuator_damping_force+actuator_open_loop_force)*ladot

            #These forces are identical to the slip: there's no (other) mass
            # between the spring and the point mass
            #ldotdot = (actuator_open_loop_force+actuator_damping_force)/MASS
            #xdotdot = -ldotdot*np.sin(alpha)
            #ydotdot =  ldotdot*np.cos(alpha) - GRAVITY
            output = np.array([x[2], x[3], xdotdot, ydotdot, 0, 0, ladot, wadot])
        else:
            raise Exception('model_type is not set correctly')

        return output

#    @jit(nopython=True)
    def fall_event(t, x):
        '''
        Event function to detect the body hitting the floor (failure)
        '''
        return x[1]
    fall_event.terminal = True
    fall_event.terminal = -1

#    @jit(nopython=True)
    def touchdown_event(t, x):
        '''
        Event function for foot touchdown (transition to stance)
        '''
            # x[1]- np.cos(p['angle_of_attack'])*SPRING_RESTING_LENGTH
            # (which is = x[5])
        return x[5]
    touchdown_event.terminal = True # no longer actually necessary...
    touchdown_event.direction = -1

#    @jit(nopython=True)
    def liftoff_event(t, x):
        '''
        Event function to reach maximum spring extension (transition to flight)
        '''
        spring_length = compute_spring_length(x, p)
        event_val = spring_length - SPRING_RESTING_LENGTH
        return event_val
    liftoff_event.terminal = True
    liftoff_event.direction = 1


#    @jit(nopython=True)
    def apex_event(t, x):
        '''
        Event function to reach apex
        '''
        return x[3]
    apex_event.terminal = True

    # * Start of step code * #

    # TODO: properly update sol object with all info, not just the trajectories

    # take one step (apex to apex)
    # the "step" function in MATLAB
    # x is the state vector, a list or np.array
    # p is a dict with all the parameters

    # set integration options

    x0 = x
    t0 = 0 # starting time

    # FLIGHT: simulate till touchdown
    # events = [lambda t, x: fall_event(t, x),
    #     lambda t, x: touchdown_event(t, x)]
    # for ev in events:
    #     ev.terminal = True
    events = [fall_event, touchdown_event]
    sol = integrate.solve_ivp(flight_dynamics,
        t_span = [t0, t0 + MAX_TIME], y0 = x0, events = events, max_step = 0.01)

    # STANCE: simulate till liftoff
    # events = [lambda t, x: fall_event(t, x),
    #     lambda t, x: liftoff_event(t, x)]
    events = [fall_event, liftoff_event]
    # for ev in events:
    #     ev.terminal = True
    # events[1].direction = 1 # only trigger when spring expands
    x0 = sol.y[:, -1]
    sol2 = integrate.solve_ivp(stance_dynamics,
        t_span = [sol.t[-1], sol.t[-1] + MAX_TIME], y0 = x0,
        events=events, max_step=0.001)

    # FLIGHT: simulate till apex
    # events = [lambda t, x: fall_event(t, x),
    #     lambda t, x: apex_event(t, x)]
    events = [fall_event, apex_event]
    # for ev in events:
    #     ev.terminal = True

    x0 = reset_leg(sol2.y[:, -1], p)
    sol3 = integrate.solve_ivp(flight_dynamics,
        t_span = [sol2.t[-1], sol2.t[-1] + MAX_TIME], y0 = x0,
        events=events, max_step=0.01)

    # concatenate all solutions
    sol.t = np.concatenate((sol.t, sol2.t, sol3.t))
    sol.y = np.concatenate((sol.y, sol2.y, sol3.y), axis = 1)
    sol.t_events += sol2.t_events + sol3.t_events

    # TODO: mark different phases
    for fail_idx in (0, 2, 4):
        if sol.t_events[fail_idx].size != 0: # if empty
            sol.failed = True
            break
    else:
        sol.failed = False
        # TODO: clean up the list

    return sol

def create_actuator_open_loop_time_series(step_sol, p):
    MODEL_TYPE = p['model_type']
    assert( MODEL_TYPE == 0 or MODEL_TYPE == 1)

    actuator_time_force = np.zeros(shape=(2,len(step_sol.t)))
    for i in range(0, len(step_sol.t)):
        spring_length = compute_spring_length(step_sol.y[:,i],p)
        spring_force  = -p['stiffness']*(spring_length-p['spring_resting_length'])
        actuator_time_force[0,i] = step_sol.t[i]
        actuator_time_force[1,i] = spring_force

    return actuator_time_force

def compute_spring_length(x, p):
    spring_length = 0
    leg_length = np.sqrt( (x[0]-x[4])**2
                        + (x[1]-x[5])**2)
    if p['model_type'] == 0:
        spring_length = leg_length - p['actuator_resting_length']
    elif p['model_type'] == 1:
        spring_length = leg_length - x[6]
    else:
        raise Exception('model_type is not set correctly')

    return spring_length

def reset_leg(x, p):
    x[4] = x[0] + np.sin(p['angle_of_attack'])*(
                    p['spring_resting_length']+p['actuator_resting_length'])
    x[5] = x[1] - np.cos(p['angle_of_attack'])*(
                    p['spring_resting_length']+p['actuator_resting_length'])
    if p['model_type'] == 1:
        x[6] = p['actuator_resting_length']
    return x

def compute_total_energy(x, p):
    # TODO: make this accept a trajectory, and output parts as well
    energy = 0
    spring_length = compute_spring_length(x, p)
    energy=(p['mass']/2*(x[2]**2+x[3]**2)
    + p['gravity']*p['mass']*(x[1]) +
    p['stiffness']/2*(spring_length-p['spring_resting_length'])**2)

    return energy

def compute_potential_kinetic_work_total(sol,p):

    cols = np.shape(sol)[1]
    t_v_w = np.zeros((4,cols))
    for i in range(0, cols):
        spring_length = compute_spring_length(sol[:,i],p)
        work = 0

        if p['model_type'] == 0:
            work = 0.
        elif p['model_type'] == 1:
            work = sol[7,i]
        else:
            raise Exception('model_type is not set correctly')

        spring_energy  = 0.5*p['stiffness']*(
                spring_length-p['spring_resting_length'])**2

        t_v_w[0,i] = p['mass']/2*(sol[2,i]**2+sol[3,i]**2)
        t_v_w[1,i] = p['gravity']*p['mass']*(sol[1,i]) + spring_energy
        t_v_w[2,i] = work
        t_v_w[3,i] = t_v_w[0,i]+t_v_w[1,i]-t_v_w[2,i]

    return t_v_w

### Functions for Viability
def map2e(x, p):
    '''
    map an apex state to its dimensionless normalized height
    TODO: make this accept trajectories
    '''
    assert(np.isclose(x[3], 0))
    potential_energy = p['mass']*p['gravity']*x[1]
    kinetic_energy = p['mass']/2*x[2]**2
    return potential_energy/(potential_energy + kinetic_energy)

def map2x(x, p, e):
    '''
    map a desired dimensionless height `e` to it's state-vector
    '''
    if 'total_energy' not in p:
        print('WARNING: you did not initialize your parameters with '
        'total energy. You really should do this...')

    assert(np.isclose(x[3], 0)) # check that we are at apex

    x_new = x
    x_new[1] = p['total_energy']*e/p['mass']/p['gravity']
    x_new[2] = np.sqrt(p['total_energy']*(1-e)/p['mass']*2)
    x_new[3] = 0.0 # shouldn't be necessary, but avoids errors accumulating
    return x_new

def mapSA2xp_height_angle(state_action, x, p):
    '''
    Specifically map state_actions to x and p
    '''
    p['angle_of_attack'] = state_action[1]
    x = map2x(x, p, state_action[0])
    x = reset_leg(x, p)
    return x, p
