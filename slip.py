import numpy as np
import scipy.integrate as integrate

def poincare_map(x, p):
    '''
    Wrapper function for step function, returning only x_next, and -1 if failed
    Essentially, the Poincare map.
    '''
    sol = step(x, p)
    return sol.y[:, -1], sol.failed

def step(x, p):
    '''
    Take one step from apex to apex/failure.
    returns a sol object from integrate.solve_ivp, with all phases
    '''

    # TODO: properly update sol object with all info, not just the trajectories

    # take one step (apex to apex)
    # the "step" function in MATLAB
    # x is the state vector, a list or np.array
    # p is a dict with all the parameters

    # set integration options

    x0 = x
    max_time = 5
    t0 = 0 # starting time

    # FLIGHT: simulate till touchdown
    events = [lambda t, x: fall_event(t, x, p),
        lambda t, x: touchdown_event(t, x, p)]
    for ev in events:
        ev.terminal = True
    sol = integrate.solve_ivp(fun = lambda t, x: flight_dynamics(t, x, p),
        t_span = [t0, t0 + max_time], y0 = x0, events = events, max_step = 0.01)

    # STANCE: simulate till liftoff
    events = [lambda t, x: fall_event(t, x, p),
        lambda t, x: liftoff_event(t, x, p)]
    for ev in events:
        ev.terminal = True
    events[1].direction = 1 # only trigger when spring expands
    x0 = sol.y[:, -1]
    sol2 = integrate.solve_ivp(fun = lambda t, x: stance_dynamics(t, x, p),
        t_span = [sol.t[-1], sol.t[-1] + max_time], y0 = x0,
        events=events, max_step=0.001)

    # FLIGHT: simulate till apex
    events = [lambda t, x: fall_event(t, x, p),
        lambda t, x: apex_event(t, x, p)]
    for ev in events:
        ev.terminal = True

    x0 = reset_leg(sol2.y[:, -1], p)
    sol3 = integrate.solve_ivp(fun = lambda t, x: flight_dynamics(t, x, p),
        t_span = [sol2.t[-1], sol2.t[-1] + max_time], y0 = x0,
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

def flight_dynamics(t, x, p):
    # code in flight dynamics, xdot_ = f()
    return np.array([x[2], x[3], 0, -p['gravity'], x[2], x[3]])

def stance_dynamics(t, x, p):
    # stance dynamics
    # energy = compute_total_energy(x, p)
    # print(energy)
    alpha = np.arctan2(x[1] - x[5], x[0] - x[4]) - np.pi/2.0
    leg_length = np.sqrt((x[0] - x[4])**2 + (x[1]-x[5])**2)
    xdotdot = -p["stiffness"]/p["mass"]*(p["resting_length"] -
                leg_length)*np.sin(alpha)
    ydotdot =  p["stiffness"]/p["mass"]*(p["resting_length"] -
                leg_length)*np.cos(alpha) - p["gravity"]
    return np.array([x[2], x[3], xdotdot, ydotdot, 0, 0])

def fall_event(t, x, p):
    '''
    Event function to detect the body hitting the floor (failure)
    '''
    return x[1]
fall_event.terminal = True
# TODO: direction

def touchdown_event(t, x, p):
    '''
    Event function for foot touchdown (transition to stance)
    '''
        # x[1]- np.cos(p["angle_of_attack"])*p["resting_length"] 
        # (which is = x[5])
    return x[5]
touchdown_event.terminal = True # no longer actually necessary...
# direction

def liftoff_event(t, x, p):
    '''
    Event function to reach maximum spring extension (transition to flight)
    '''
    return ((x[0]-x[4])**2 + (x[1]-x[5])**2) - p["resting_length"]**2
liftoff_event.terminal = True
liftoff_event.direction = 1

def apex_event(t, x, p):
    '''
    Event function to reach apex
    '''
    return x[3]
apex_event.terminal = True

def compute_total_energy(x, p):
    # TODO: make this accept a trajectory, and output parts as well
    return (p["mass"]/2*(x[2]**2+x[3]**2) +
    p["gravity"]*p["mass"]*(x[1]) +
    p["stiffness"]/2*
    (p["resting_length"]-np.sqrt((x[0]-x[4])**2 + (x[1]-x[5])**2))**2)

### Functions for Viability
def map2e(x, p):
    '''
    map an apex state to its dimensionless normalized height
    TODO: make this accept trajectories
    '''
    assert(np.isclose(x[3], 0))
    potential_energy = p['mass']*p['gravity']*x[1]
    kinetic_energy = p['mass']/2*x[3]**2
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
    return x, p
