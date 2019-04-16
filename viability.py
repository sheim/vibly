# compute viability of a system

import itertools as it
import numpy as np
# from utilities import map2e, map2x
from slip import map2x, map2e

def computeQ2D(s_table, a_table, pMap,mapSA2xp):
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
    QMap = np.zeros((s_table.size*a_table.size,1))
    # TODO: also compute transitions diff maps etc.
    # QTransition = QMap

    for idx, state_action in enumerate(it.product(s_table,a_table)):
        x,p = mapSA2xp(state_action,pMap.x,pMap.p)

        x_next, failed = pMap(x,p)

        if not failed:
            s_next = map2e(x_next,p)
            QMap[idx] = s_next
    
    return QMap.reshape((s_table.size,a_table.size))