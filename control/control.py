import numpy as np
from viability import get_grid_indices, project_Q2S

def Q_value_iteration(Q_map, grids, reward_functions, gamma, Q_on_grid=None,
                      stopping_threshold=1e-5, max_iter=1000, output_R=False,
                      neighbor_option=np.mean, 
                      Q_values = None):
    """
    Standard value iteration.
    Inputs:
    Q_map: transition map
    grids: list of grids for states and actions
    reward_functions: list of rewards
    gamma: discount factor
    convergence_threshold: threshold on improvement to stop iterating
    max_iter: maximum number of iterations
    output_R: toggle true to also return array of reward for each (s, a)
    neighbor_option: how to interpolate grid-borders of a bin
    Q_values: initial guess for Q_values, can help speed things up
    """

    if Q_values is None:
        Q_values = np.zeros_like(Q_map, dtype=float)
    else:
        assert (Q_values.shape == Q_map.shape), "initial_guess is bad"

    # can be pre-computed one time
    R_values = np.zeros_like(Q_map, dtype=float)

    s_grid = grids['states']
    a_grid = grids['actions']
    n_states = len(s_grid)

    if Q_on_grid is None:
        for qdx, next_s in np.ndenumerate(Q_map):
            bin_idx = np.unravel_index(next_s, tuple(x+1 for x in map(np.size, s_grid)))
            # pass transition through each reward function
            reward = 0.0
            s = [grid[qdx[i]] for i, grid in enumerate(s_grid)]
            a = [grid[qdx[n_states + i]] for i, grid in enumerate(a_grid)]
            for rfunc in reward_functions:
                reward += rfunc(s, a)
            R_values[qdx] = reward

        # iterate over each q
        for iteration in range(max_iter):
            max_change = 0.0  # for stopping
            for qdx, next_s in np.ndenumerate(Q_map):
                bin_idx = np.unravel_index(next_s, tuple(x+1 for x in map(np.size, s_grid)))
                grid_indices = get_grid_indices(bin_idx, s_grid)
                # average bin value by neighboring q-values from grid
                # bin_value = 0.0
                # for g in grid_indices:
                #     bin_value += Q_values[g].max()/len(grid_indices)
                bin_value = neighbor_option([Q_values[g].max()
                                            for g in grid_indices])

                # keep track of changes, for stopping condit    ion
                diff = np.abs(Q_values[qdx] - R_values[qdx] - gamma*bin_value)
                if (diff > max_change):
                    max_change = diff
                Q_values[qdx] = R_values[qdx] + gamma*bin_value
            if max_change < stopping_threshold:
                print("Stopped early after ", iteration, " iterations.")
                break
        print("max change in value: ", max_change)

    else:  # * Q_on_grid given
        for qdx, next_s in np.ndenumerate(Q_map):
            # * Pre-compute R
            # pass transition through each reward function
            reward = 0.0
            s = [grid[qdx[i]] for i, grid in enumerate(s_grid)]
            a = [grid[qdx[n_states + i]] for i, grid in enumerate(a_grid)]
            for rfunc in reward_functions:
                reward += rfunc(s, a)
            R_values[qdx] = reward

        # iterate over each q
        for iteration in range(max_iter):
            max_change = 0.0  # for stopping
            X_val = project_Q2S(Q_values, grids, proj_opt=np.max)
            X_val = X_val.flatten()
            for qdx, next_s in np.ndenumerate(Q_map):
                # average bin value by neighboring q-values from grid
                # bin_value = 0.0
                # for g in grid_indices:
                #     bin_value += Q_values[g].max()/len(grid_indices)
                # bin_value = neighbor_option([Q_values[g].max()
                #                             for g in grid_indices])
                q_value = X_val[next_s]
                # keep track fo changes, for stopping condit    ion
                diff = np.abs(Q_values[qdx] - R_values[qdx] - gamma*q_value)
                if (diff > max_change):
                    max_change = diff
                Q_values[qdx] = R_values[qdx] + gamma*q_value
            if max_change < stopping_threshold:
                print("Stopped early after ", iteration, " iterations.")
                break
        print("max change in value: ", max_change)

    if output_R:
        return Q_values, R_values
    else:
        return Q_values


def QValue_Iteration(Q_map, grids, reward_functions, gamma,
                      stopping_threshold=1e-5, max_iter=1000, output_R=False,
                      Q_values=None):
    """
    A better (faster!) implementation, with fewer for-loops.
    Less options for now (hence keeping both for a while).
    Assumes reward function take three arguments: (s, a, s').
    """

    if Q_values is None:
        Q_values = np.zeros_like(Q_map, dtype=float)
    else:
        assert (Q_values.shape == Q_map.shape), "initial_guess is bad"

    s_grid = grids['states']
    a_grid = grids['actions']
    n_states = len(s_grid)
    s_grid_shape = list(map(np.size, grids['states']))

    # * build the initial state array Q0 and reward array R

    # can be pre-computed one time
    R_values = np.zeros_like(Q_map, dtype=float)
    for qdx, next_sdx in np.ndenumerate(Q_map):
        # pass transition through each reward function
        reward = 0.0
        s = [grid[qdx[i]] for i, grid in enumerate(s_grid)]
        a = [grid[qdx[n_states + i]] for i, grid in enumerate(a_grid)]
        next_sdx = np.unravel_index(next_sdx, s_grid_shape)
        next_s = [grid[next_sdx[i]] for i, grid in enumerate(s_grid)]
        for rfunc in reward_functions:
            reward += rfunc(s, a, next_s)
        R_values[qdx] = reward
    # iterate over each q
    for iteration in range(max_iter):
        max_change = 0.0  # for stopping
        X_val = project_Q2S(Q_values, grids, proj_opt=np.max)
        # X_val = X_val.flatten()

        # * value V(s_next)
        Values_next = X_val.flatten()[Q_map.flatten()].reshape(Q_map.shape)
        Q_values_new = R_values + gamma*Values_next
        if np.max(np.abs(Q_values-Q_values_new)) < stopping_threshold:
            print("Stopped early after ", iteration, " iterations.")
            Q_values = Q_values_new  # might as well still update
            break
        Q_values = Q_values_new

    if output_R:
        return Q_values, R_values
    else:
        return Q_values
