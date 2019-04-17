# compute viability of a system

import itertools as it
import numpy as np
# from utilities import map2e, map2x
from slip import map2x, map2e

def computeQ_2D(s_table, a_table, pMap):
    ''' Compute the transition map of a system with 1D state and 1D action

    NOTES
    - s_table and a_table have to be iterable lists of lists
    e.g. if they have only 1 dimension, they should be `s_table = ([1,2],)`
    - use pMap to carry parameters
    '''

    # create iterators s_table, a_table
    # TODO: pass in iterators/generators instead
    # compute each combination, store result in a huge matrix
    
    # initialize 1D, reshape later
    Q_map = np.zeros((s_table.size*a_table.size,1))
    Q_F = Q_map
    # TODO: also compute transitions diff maps etc.
    # QTransition = Q_map

    for idx, state_action in enumerate(it.product(s_table,a_table)):
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

    
    return ( Q_map.reshape((s_table.size,a_table.size)), # only 2D
    Q_F.reshape((s_table.size,a_table.size)) )

def projectQ2S_2D(Q):
    S = np.zeros((Q.shape[0],1))
    for sdx, val in enumerate(S):
        if sum(Q[sdx,:]) > 0:
            S[sdx] = 1
    return S
