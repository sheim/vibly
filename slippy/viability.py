# compute viability of a system

import itertools as it
import numpy as np

# def save(filename):
#     '''
#     Utility function to save all the default files generated
#     '''
#     save_list = ('Q_map', 'Q_F', 'Q_V', 'grids', 'S_V', 'P_map', 'p')
#     data2save = {}
#     for var in save_list:
#         if var in locals(): # check if it is instantiated
#             data2save[var] = eval(var)

#     if len(data2save) > 0:
#         np.savez(filename, **data2save)
#     else:
#         print('Warning: nothing to save')

# def load():
#     # load has to pass in references to be overwritten, or something like that

def compute_Q_2D(s_grid, a_grid, poincare_map):
    ''' Compute the transition map of a system with 1D state and 1D action
    NOTES
    - s_grid and a_grid have to be iterable lists of lists
    e.g. if they have only 1 dimension, they should be `s_grid = ([1, 2], )`
    - use poincare_map to carry parameters
    '''

    # create iterators s_grid, a_grid
    # TODO: pass in iterators/generators instead
    # compute each combination, store result in a huge matrix

    # initialize 1D, reshape later
    Q_map = np.zeros((s_grid.size*a_grid.size, 1))
    Q_F = np.zeros((s_grid.size*a_grid.size, 1))

    # TODO: also compute transitions diff maps etc.
    # QTransition = Q_map
    n = len(s_grid)*len(a_grid)
    for idx, state_action in enumerate(it.product(s_grid, a_grid)):
        if idx%(n/10)==0:
            print('.', end=' ')
        x, p = poincare_map.sa2xp(state_action, poincare_map.x, poincare_map.p)

        x_next, failed = poincare_map(x, p)

        if not failed:
            s_next = poincare_map.xp2s(x_next, p)
            # note: Q_map is implicitly already excluding transitions that
            # move straight to a failure. While this is not equivalent to the
            # algorithm in the paper, for our systems it is a bit more efficient
            Q_map[idx] = s_next
        else:
            Q_F[idx] = 1


    return ( Q_map.reshape((s_grid.size, a_grid.size)), # only 2D
    Q_F.reshape((s_grid.size, a_grid.size)) )

def project_Q2S_2D(Q):
    S = np.zeros((Q.shape[0], 1))
    for sdx, val in enumerate(S):
        if sum(Q[sdx, :]) > 0:
            S[sdx] = 1
    return S

def is_outside_2D(s, S_V, s_grid):
    '''
    given a level set S, check if s is inside S or not
    '''
    if sum(S_V) <= 1:
        return True

    s_min, s_max = s_grid[S_V>0][[0, -1]]
    if s>s_max or s<s_min:
        return True
    else:
        return False

def compute_QV_2D(Q_map, grids, Q_V = None):
    ''' Starting from the transition map and set of non-failing state-action
    pairs, compute the viable sets. The input Q_V is referred to as Q_N in the
    paper when passing it in, but since it is immediately copied to Q_V, we
    directly use this naming.
    '''

    # Take Q_map as the non-failing set if Q_N is omitted
    if Q_V is None:
        Q_V=np.copy(Q_map)
        Q_V[Q_V>0] = 1

    S_old = np.zeros((Q_V.shape[0], 1))
    S_V = project_Q2S_2D(Q_V)
    while np.array_equal(S_V, S_old):
        for qdx, is_viable in enumerate(np.nditer(Q_V)): # compare with np.enum
            if is_viable: # only check previously viable (s, a)
                if is_outside_2D(Q_map[qdx], S_V, grids['states']):
                    Q_V[qdx] = 0 # remove
        S_old = S_V
        S_V = project_Q2S_2D(Q_V)

    return Q_V, S_V

###### Reimplement everything as N-D

def get_state_from_ravel(bin_idx, s_grid):
    '''
    Get state from bin id. Ideally, interpolate
    For now, just returning a grid point
    '''
    print("TO DO: interpolate properly") # TODO
    bin_idx = np.atleast_1d(bin_idx)
    grid_idx = np.zeros(len(s_grid), dtype='int')
    s = np.zeros(len(s_grid))
    for dim, grid in enumerate(s_grid):
        if bin_idx[dim] >= grid.size:
            grid_idx[dim] = grid.size-1 # upper-est entry
            s[dim] = grid[grid_idx[dim]] #TODO: Super ugly
        else:
            grid_idx[dim] = bin_idx[dim] # just put the right-closest grid
            s[dim] = grid[grid_idx[dim]]
    return s

def digitize_s(s, s_grid, s_bin_shape = None):
    '''
    s_grid is a tuple/list of 1-D grids. digitize_s finds the corresponding
    index of the bin overlaid on the N-dimensional grid S.

    output:
    - either an array of indices of N dimensions
    - a raveled index
    '''
    # assert type(s_grid) is tuple or type(s_grid) is list
    s = np.atleast_1d(s)
    s_bin = np.zeros((len(s_grid)), dtype = int)
    for dim_idx, grid in enumerate(s_grid): # TODO: can zip this with s
        s_bin[dim_idx] = np.digitize(s[dim_idx], grid)
        # TODO: fix this hack nicely.
        # deprecated!
        # if s_bin[dim_idx] >= grid.size:
        #     print("WARNING: exited grid in " + str(dim_idx) + " dimension.")
        #     s_bin[dim_idx] = grid.size-1 # saturating at end of grid
    if s_bin_shape is None:
        return s_bin
    else:
        return np.ravel_multi_index(s_bin, s_bin_shape)


def compute_Q_map(grids, poincare_map, verbose = 0):
    ''' Compute the transition map of a system with 1D state and 1D action
    NOTES
    - s_grid and a_grid have to be iterable lists of lists
    e.g. if they have only 1 dimension, they should be `s_grid = ([1, 2], )`
    - use poincare_map to carry parameters
    '''

    # initialize 1D, reshape later
    s_grid_shape = list(map(np.size, grids['states'])) # shape of state-space grid
    s_bin_shape = tuple(dim+1 for dim in s_grid_shape)
    a_grid_shape = list(map(np.size, grids['actions']))
    a_bin_shape = tuple(dim+1 for dim in a_grid_shape)
    total_bins = np.prod(s_bin_shape)*np.prod(a_bin_shape)
    total_gridpoints = np.prod(s_grid_shape)*np.prod(a_grid_shape)

    if verbose > 0:
        print('computing a total of ' + str(total_bins) + ' points.')

    Q_map = np.zeros((total_gridpoints, 1), dtype = int)
    Q_F = np.zeros((total_gridpoints, 1), dtype = bool)

    # TODO: also compute transitions diff maps etc.
    # TODO: generate purely with numpy (meshgrids?).
    # TODO ... since converting lists to np.arrays is slow
    # QTransition = Q_map
    for idx, state_action in enumerate(np.array(list(
            it.product(*grids['states'], *grids['actions'])))):

        if verbose > 1:
            # NOTE: requires running python unbuffered (python -u)
            if idx%(total_bins/10)==0:
                print('.', end=' ')

        x, p = poincare_map.sa2xp(state_action, poincare_map.x, poincare_map.p)

        x_next, failed = poincare_map(x, p)

        if not failed:
            s_next = poincare_map.xp2s(x_next, p)
            # note: Q_map is implicitly already excluding transitions that
            # move straight to a failure. While this is not equivalent to the
            # algorithm in the paper, for our systems it is a bit more efficient
            # bin_idx = np.digitize(state_val, s_grid[state_dx])
            # sbin = np.digitize(s_next, s)
            Q_map[idx] = digitize_s(s_next, grids['states'], s_bin_shape)
        else:
            Q_F[idx] = 1

    Q_map = Q_map.reshape(s_grid_shape + a_grid_shape)
    Q_F = Q_F.reshape(s_grid_shape + a_grid_shape)
    return ( Q_map, Q_F)

def project_Q2S(Q, grids, proj_opt = None):
    if proj_opt is None:
        proj_opt = np.any
    a_axes = tuple(range(Q.ndim - len(grids['actions']), Q.ndim))
    return proj_opt(Q, a_axes)

def compute_QV(Q_map, grids, Q_V = None):
    '''
    Starting from the transition map and set of non-failing state-action
    pairs, compute the viable sets. The input Q_V is referred to as Q_N in the
    paper when passing it in, but since it is immediately copied to Q_V, we
    directly use this naming.
    '''

    # initialize estimate of Q_V
    if Q_V is None:
        Q_V = np.copy(Q_map) # TODO: is this okay in general?
        Q_V = Q_V.astype(bool)
    # initialize empty of S_old
    # initialize estimate of S_V
    S_V = project_Q2S(Q_V, grids)
    S_old = np.zeros_like(S_V)

    # while S_V != S_old
    while not np.array_equal(S_V, S_old):
        # iterate over all s in S_V
        for qdx, is_viable in np.ndenumerate(Q_V):
            # iterate over all a
            if is_viable:
                # if s_k isOutside S_V:
                if is_outside(Q_map[qdx], grids['states'], S_V):
                    Q_V[qdx] = False
                    # remove (s,a) from Q_V
        S_old = S_V
        S_V = project_Q2S(Q_V, grids)

    return Q_V, S_V

def is_outside(s, s_grid, S_V, already_binned = True):
    '''
    given a level set S, check if s lands in a bin inside of S or not
    '''
    # only checking 1 state vector
    if not already_binned: #TODO Check this
        bin_idx = digitize_s(s, s_grid) # get unraveled indices
    else:
        bin_idx = np.unravel_index(s, tuple(x+1 for x in map(np.size, s_grid)))

    for dim_idx, grid in enumerate(s_grid):
        # if outside the left-most or right-most side of grid, mark as outside
        # * NOTE: this can result in disastrous underestimations if the grid is
        # * not larger than the viable set!
        if bin_idx[dim_idx] == 0:
            return True
        elif bin_idx[dim_idx] >= grid.size:
            return True
        else:
            # check if enclosing grid points are viable or not
            index_vec = np.zeros(len(s_grid), dtype = int)
            index_vec[dim_idx] = 1
            if (not S_V[tuple(bin_idx)] or
                not S_V[tuple(bin_idx - index_vec)]):
                return True

        return False

def map_S2Q(Q_map, S_M, Q_V = None):
    '''
    map the measure of robustness of S to the state action space Q, via
    inverse dynamics (using the lookup table).
    '''
    if Q_V is None:
        Q_V = np.copy(Q_map)

    Q_M = np.zeros_like(Q_map)
    # iterate over viable state-action pairs in Q_V
    for qdx, is_viable in np.ndenumerate(Q_V): # compare with np.enum
            if is_viable: # only check states-action pairs that are viable
                sdx = np.unravel_index(Q_map[qdx], S_M.shape)
                Q_M[qdx] = S_M[sdx]

    return Q_M

def get_feasibility_mask(feasible, sa2xp, grids, x0, p0):
    '''
    cycle through the state and action grids, and check if that state-action
    pair is feasible or nay. Returns an ND array of booleans.

    feasible: function to check if a state and parameter are feasible
    sa2xp: mapping to go from state-action to x and p
    grids: grids of states and actions
    x0: a default x0, used to fill out x0 (used by sa2xp)
    p: a default parameter dict, used to fill out p
    '''
    # initialize 1D, reshape later
    s_shape = list(map(np.size, grids['states'])) # shape of state-space grid
    a_shape = list(map(np.size, grids['actions']))
    Q_feasible = np.zeros(np.prod(s_shape)*np.prod(a_shape), dtype=bool)

    # TODO: can probably simplify this
    for idx, state_action in enumerate(np.array(list(
            it.product(*grids['states'], *grids['actions'])))):
        x, p = sa2xp(state_action, x0, p0)
        Q_feasible[idx] = feasible(x, p)

    return Q_feasible.reshape(s_shape + a_shape)