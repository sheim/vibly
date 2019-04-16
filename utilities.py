import numpy as np

def map2e(x, p):
    '''
    map an apex state to its dimensionaless normalized height
    TODO: make this accept trajectories
    '''
    assert(np.isclose(x[3],0))
    potential_energy = p['mass']*p['gravity']*x[1]
    kinetic_energy = p['mass']/2*x[3]**2
    return potential_energy/(potential_energy+kinetic_energy)

def map2x(x,p,e):
    '''
    map a desired dimensionless energy `e` to it's state-vector
    '''
    if 'total_energy' not in p:
        p['total_energy'] = computeTotalEnergy(x,p)
        print('WARNING: you did not initialize your parameters with '
        'total energy. You really should do this...')
    
    assert(np.isclose(x[3],0))

    potential_energy = p['mass']*p['gravity']*x[1]
    kinetic_energy = p['mass']/2*x[3]**2
    x_new = x
    x_new[1] = potential_energy/p['mass']/p['gravity']
    x_new[2] = np.sqrt(total_energy/p['mass']*2)
    x_new[3] = 0.0 # shouldn't be necessary, but avoids errors accumulating
    return x_new


