import numpy as np
import scipy.integrate as integrate
from numba import jit
import sys

def limit_cycle(x,p):
    '''
    Iterates over the angle of attack of the leg until a limit cycle is reached. 
    '''
    #Settings for the root finding methods
    max_iter_bisection  = 10
    max_iter_newton     = 10
    tol_newton          = 1e-12   

    limit_cycle_found = False

    if type(p) is dict:
      if p['damping'] > 0:
        print("limit_cycle: damping is ignored")

      #This will only affect the local copy of p according to
      #  https://www.python-course.eu/passing_arguments.php
      damping = p['damping']
      p['damping'] = 0.
      
      #Use the bisection method to get a good initial guess for angle_of_attack
      angle_delta = np.pi*0.25

      #Initial solution      
      x                 = reset_leg(x,p)
      (pm, step_failed) = poincare_map(x,p)
      err               = np.abs(pm[1]-x[1])         

      #Memory for the left and right solutions
      pm_left     = pm
      pm_right    = pm
      err_left    = 0
      err_right   = 0

      angle_of_attack = p['angle_of_attack']

      #After this for loop returns the angle of attack will be known to
      # a tolerance of pi/4 / 2^(max_iter_bisection)      
      for i in range(0,max_iter_bisection):
        p['angle_of_attack'] = angle_of_attack-angle_delta
        x = reset_leg(x,p)
        (pm_left, left_step_failed) = poincare_map(x,p)      
        err_left = np.abs(pm_left[1]-x[1])

        p['angle_of_attack'] = angle_of_attack+angle_delta
        x = reset_leg(x,p)
        (pm_right, right_step_failed) = poincare_map(x,p)
        err_right= np.abs(pm_right[1]-x[1])

        if( (err_left < err       and left_step_failed==False) and 
            (err_left <= err_right or right_step_failed==True)):
          err = err_left
          angle_of_attack=angle_of_attack-angle_delta

        if( (err_right < err     and right_step_failed==False) and 
            (err_right < err_left or left_step_failed==True)):
          err = err_right
          angle_of_attack = angle_of_attack+angle_delta

        angle_delta = 0.5*angle_delta
      
      #Polish the root using Newton's method
   
      iter=0
      h = np.sqrt(sys.float_info.epsilon)
      while np.abs(err) > tol_newton and iter < max_iter_newton:

        #Compute the error
        p['angle_of_attack'] = angle_of_attack
        x = reset_leg(x,p)
        (pm, step_failed) = poincare_map(x,p)      
        err = pm[1]-x[1]

        #Compute D(error)/D(angle_of_attack) using a numerical derivative
        p['angle_of_attack'] = angle_of_attack-h
        x = reset_leg(x,p)
        (pm, step_failed) = poincare_map(x,p)      
        errL = pm[1]-x[1]

        p['angle_of_attack'] = angle_of_attack+h
        x = reset_leg(x,p)
        (pm, step_failed) = poincare_map(x,p)      
        errR = pm[1]-x[1]

        #Compute a Newton step and take it
        DerrDangle = (errR-errL)/(2*h)
        angle_of_attack = angle_of_attack -err/DerrDangle
        iter=iter+1

      if np.abs(err) > tol_newton:
        print("WARNING: Newton method failed to converge")
        limit_cycle_found = False
      else:
        limit_cycle_found = True

      p['angle_of_attack'] = angle_of_attack
      p['damping']=damping
      return (p,limit_cycle_found)


    else:
        print("WARNING: I got a parameter type that I don't understand.")
        return p



def poincare_map(x, p):
    '''
    Wrapper function for step function, returning only x_next, and -1 if failed
    Essentially, the Poincare map.
    '''
    if type(p) is dict:
        sol = step(x, p)
        return sol.y[:, -1], sol.failed
    elif type(p) is tuple:
        vector_of_x = np.zeros(x.shape) # initialize result array
        vector_of_fail = np.zeros(x.shape[1])
        # TODO: for shorthand, allow just a single tuple to be passed in
        # this can be done easily with itertools
        for idx, p0 in enumerate(p):
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
    AOA = p['angle_of_attack']
    GRAVITY = p['gravity']
    MASS = p['mass']
    RESTING_LENGTH = p['resting_length']
    STIFFNESS = p['stiffness']
    TOTAL_ENERGY = p['total_energy']
    SPECIFIC_STIFFNESS = p['stiffness'] / p['mass']
    MAX_TIME = 5

#    @jit(nopython=True)
    def flight_dynamics(t, x):
        # code in flight dynamics, xdot_ = f()
        return np.array([x[2], x[3], 0, -GRAVITY, x[2], x[3]])

#    @jit(nopython=True)
    def stance_dynamics(t, x):
        # stance dynamics
        alpha = np.arctan2(x[1] - x[5], x[0] - x[4]) - np.pi/2.0
        leg_length = np.hypot(x[0]-x[4], x[1]-x[5])
        xdotdot = -STIFFNESS/MASS*(RESTING_LENGTH -
                    leg_length)*np.sin(alpha)
        ydotdot =  STIFFNESS/MASS*(RESTING_LENGTH -
                    leg_length)*np.cos(alpha) - GRAVITY
        return np.array([x[2], x[3], xdotdot, ydotdot, 0, 0])

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
            # x[1]- np.cos(p['angle_of_attack'])*RESTING_LENGTH
            # (which is = x[5])
        return x[5]
    touchdown_event.terminal = True # no longer actually necessary...
    touchdown_event.direction = -1

#    @jit(nopython=True)
    def liftoff_event(t, x):
        '''
        Event function to reach maximum spring extension (transition to flight)
        '''
        return ((x[0]-x[4])**2 + (x[1]-x[5])**2) - RESTING_LENGTH**2
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

def reset_leg(x, p):
    x[4] = x[0] + np.sin(p['angle_of_attack'])*p['resting_length']
    x[5] = x[1] - np.cos(p['angle_of_attack'])*p['resting_length']
    return x

def compute_total_energy(x, p):
    # TODO: make this accept a trajectory, and output parts as well
    return (p['mass']/2*(x[2]**2+x[3]**2) +
    p['gravity']*p['mass']*(x[1]) +
    p['stiffness']/2*
    (p['resting_length']-np.sqrt((x[0]-x[4])**2 + (x[1]-x[5])**2))**2)

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
