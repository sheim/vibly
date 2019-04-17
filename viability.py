# compute viability of a system

import itertools as it
import numpy as np
# from utilities import map2e, map2x
from slip import map2x, map2e

def computeQ_2D(s_grid, a_grid, pMap):
    ''' Compute the transition map of a system with 1D state and 1D action

    NOTES
    - s_grid and a_grid have to be iterable lists of lists
    e.g. if they have only 1 dimension, they should be `s_grid = ([1,2],)`
    - use pMap to carry parameters
    '''

    # create iterators s_grid, a_grid
    # TODO: pass in iterators/generators instead
    # compute each combination, store result in a huge matrix
    
    # initialize 1D, reshape later
    Q_map = np.zeros((s_grid.size*a_grid.size,1))
    Q_F = np.zeros((s_grid.size*a_grid.size,1))

    # TODO: also compute transitions diff maps etc.
    # QTransition = Q_map

    for idx, state_action in enumerate(it.product(s_grid,a_grid)):
        x,p = pMap.sa2xp(state_action,pMap.x,pMap.p)
        
        x_next, failed = pMap(x,p)

        if not failed:
            s_next = map2e(x_next,p)
            # note: Q_map is implicitly already excluding transitions that
            # move straight to a failure. While this is not equivalent to the
            # algorithm in the paper, for our systems it is a bit more efficient
            Q_map[idx] = s_next
        else:
            Q_F[idx] = 1

    
    return ( Q_map.reshape((s_grid.size,a_grid.size)), # only 2D
    Q_F.reshape((s_grid.size,a_grid.size)) )

def projectQ2S_2D(Q):
    S = np.zeros((Q.shape[0],1))
    for sdx, val in enumerate(S):
        if sum(Q[sdx,:]) > 0:
            S[sdx] = 1
    return S

def outside_2D(s, s_grid, S_levelset):
    ''' 
    given a level set S, check if s is inside S or not
    '''
    if sum(S_leveset) <= 1:
        return True

    s_min,s_max = s_grid[S_levelset>0][[0,-1]]
    if s>s_max or s<s_min:
        return True
    else:
        return False

def computeQV_2D(Q_map, grids, Q_V = None):
    ''' Starting from the transition map and set of non-failing state-action
    pairs, compute the viable sets. The input Q_V is referred to as Q_N in the
    paper when passing it in, but since it is immediately copied to Q_V, we
    directly use this naming.
    '''

    # Take Q_map as the non-failing set if Q_N is omitted
    if Q_V is None:
        Q_V=Q_map
        Q_V[Q_V>0] = 1
        
    S_old = np.zeros((Q_V.shape[0],1))
    S_V = projectQ2S_2D(Q_V)
    while np.array_equal(S_V, S_old):
        for qdx, is_viable in enumerate(np.nditer(Q_V)): # compare with np.enum
            if is_viable: # only check previously viable (s,a)
                if outside_2D(Q_map[qdx],S_V,grids['states']):
                    Q_V[qdx] = 0 # remove
        S_old = S_V
        S_V = projectQ2S_2D(Q_V)
    
    return Q_V, S_V
