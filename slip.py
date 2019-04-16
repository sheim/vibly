import numpy as np
import scipy.integrate as integrate

def pMap(x,p):
    # take one step (apex to apex)
    # the "step" function in MATLAB
    # x is the state vector, a list or np.array
    # p is a dict with all the parameters

    # set integration options

    x0 = x
    # TODO: dict unpacker

    max_time = 5
    t0 = 0 # starting time

    events = [lambda t,x: fallEvent(t,x,p), lambda t,x: touchdownEvent(t,x,p)]
    for ev in events:
        ev.terminal = True
    
    sol = integrate.solve_ivp(fun=lambda t, x: flightDynamics(t, x, p), 
    t_span = [t0, t0+max_time], y0 = x0, events=events, max_step=0.01, 
    dense_output=True)

    
    events = [lambda t,x: fallEvent(t,x,p), lambda t,x: liftoffEvent(t,x,p)]
    for ev in events:
        ev.terminal = True
    events[1].direction = 1 # only trigger when spring expands
    sol2 = integrate.solve_ivp(fun=lambda t, x: stanceDynamics(t, x, p), 
    t_span = [sol.t[-1], sol.t[-1]+max_time], y0 = sol.y[:,-1], 
    events=events, max_step=0.01, dense_output=True)

    events = [lambda t,x: fallEvent(t,x,p), lambda t,x: apexEvent(t,x,p)]
    for ev in events:
        ev.terminal = True
    sol3 = integrate.solve_ivp(fun=lambda t, x: flightDynamics(t, x, p), 
    t_span = [sol2.t[-1], sol2.t[-1]+max_time], y0 = sol2.y[:,-1], 
    events=events, max_step=0.01, dense_output=True)

    # concatenate all solutions
    sol.t = np.concatenate((sol.t,sol2.t,sol3.t))
    sol.y = np.concatenate((sol.y,sol2.y,sol3.y),axis=1)
    sol.t_events.append(sol2.t_events.append(sol3.t_events))
    # TODO: mark different phases

    return sol    

def flightDynamics(t,x,p):
    # code in flight dynamics, xdot_ = f()
    return np.array([x[2], x[3], 0, -p["gravity"]])

def stanceDynamics(t, x,p):
    # stance dynamics
    alpha = np.arctan2(x[1],x[0]) - np.pi/2.0
    leg_length = np.sqrt(x[0]**2+x[1]**2)
    xdotdot = - p["stiffness"]/p["mass"]*(p["resting_length"] - 
                leg_length)*np.sin(alpha)
    ydotdot = p["stiffness"]/p["mass"]*(p["resting_length"] - 
                leg_length)*np.cos(alpha) - p["gravity"]
    return np.array([x[2], x[3], xdotdot, ydotdot])

def fallEvent(t,x,p):
    return x[1]
fallEvent.terminal = True
# TODO: direction

def touchdownEvent(t,x,p):
    return x[1]- np.cos(p["aoa"])*p["resting_length"]
touchdownEvent.terminal = True
# direction

def liftoffEvent(t,x,p):
    return (x[0]**2 + x[1]**2) - p["resting_length"]**2
liftoffEvent.terminal = True

def apexEvent(t,x,p):
    return x[3]
apexEvent.terminal = True