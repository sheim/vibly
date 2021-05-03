import numpy as np
import itertools as it

def get_grid_indices(bin_idx, s_grid):
    '''
    from a bin index (unraveled), get surrounding grid indices, and also check
    if you're on the grid edge. Returns a list of tuples
    '''

    # if outside the left-most or right-most side of grid, mark as outside
    # for dim_idx, grid in enumerate(s_grid):
    #     # TODO handle this nicely, and still return neighboring grid indices
    #     if bin_idx[dim_idx] == 0:
    #         return list()
    #     elif bin_idx[dim_idx] >= grid.size:
    #         return list()

    # get all neighboring grid coordinates
    # grid_coords = list(it.product(*[(x-1, x) for x in bin_idx]))
    # keepers = list()
    # for coord in grid_coords:
    #     for idx, x in np.ndenumerate(coord):
    #         if x<0 or x>s_grid[idx].size:
    #             keepers.append(coord)
    grid_coords = list()
    for neighbor in it.product(*[(x-1, x) for x in bin_idx]):
        # check if neighbor is out of bounds
        for idx, x in enumerate(neighbor):
            if x<0 or x>=s_grid[idx].size:
                break
        else:  # if for loop completes
            grid_coords.append(neighbor)

    return grid_coords

def Q_value_iteration(Q_map, grids, reward_functions, gamma,
                      stopping_threshold=1e-5, max_iter=1000):
    """
    Standard value iteration.
    Inputs:
    Q_map: transition map
    grids: list of grids for states and actions
    reward_functions: list of rewards
    gamma: discount factor
    convergence_threshold: threshold on improvement to stop iterating
    max_iter: maximum number of iterations
    """


    Q_values = np.zeros_like(Q_map, dtype=float)
    # S_Value = np.zeros([g.size for g in grids['states']])

    s_grid = grids['states']
    a_grid = grids['actions']
    n_states = len(s_grid)

    for iteration in range(max_iter):

        # iterate over each q
        max_change = 0.0  # for stopping

        for qdx, next_s in np.ndenumerate(Q_map):
            bin_idx = np.unravel_index(next_s, tuple(x+1 for x in map(np.size, s_grid)))
            # pass transition through each reward function
            reward = 0.0
            s = [grid[qdx[i]] for i, grid in enumerate(s_grid)]
            a = [grid[qdx[n_states + i]] for i, grid in enumerate(a_grid)]
            for rfunc in reward_functions:
                reward += rfunc(s, a)
            # update values
            # if reward >= 0:
                # print("hello")
            grid_indices = get_grid_indices(bin_idx, s_grid)
            # average bin value by neighboring q-values from grid
            bin_value = 0.0
            for g in grid_indices:
                bin_value += Q_values[g].max()/len(grid_indices)

            # keep track fo changes, for stopping condit    ion
            diff = np.abs(Q_values[qdx] - reward - gamma*bin_value)
            if (diff > max_change):
                max_change = diff
            Q_values[qdx] = reward + gamma*bin_value
        if max_change < stopping_threshold:
            print("Stopped early after ", iteration, " iterations.")
            break
    print("max change in value: ", max_change)
    return Q_values