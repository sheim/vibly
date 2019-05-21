# Run `initialize_active_sampling.py` first to load everything and the prior

################################################################################
# initialize "true" model
################################################################################
import numpy as np

import slippy.viability as vibly
import slippy.nslip as true_model

# This if you use nslip instead of slip
# * already loaded in initialization
# p_true = {'mass':80.0, 'stiffness':705.0, 'resting_angle':17/18*np.pi,
#         'gravity':9.81, 'angle_of_attack':1/5*np.pi, 'upper_leg':0.5,
#         'lower_leg':0.5}
# this for slip
# p_true = {'mass':80.0, 'stiffness':8200.0, 'resting_length':1.0, 'gravity':9.81,
# 'angle_of_attack':1/5*np.pi}
x0 = np.array([0, 0.85, 5.5, 0, 0, 0, 0])
p_true['total_energy'] = true_model.compute_total_energy(x0, p_true)

s_grid_shape = list(map(np.size, grids['states']))
s_bin_shape = tuple(dim+1 for dim in s_grid_shape)
#### from GP approximation, choose parts of Q to sample
n_samples = 100
active_threshold = np.array([0.5, 0.7])
# pick initial state
s0 = np.random.uniform(0.4, 0.7)
s0_idx = vibly.digitize_s(s0, grids['states'], s_bin_shape)
X_observe = np.zeros([n_samples, 2])
Y_observe = np.zeros(n_samples)

verbose = True

for ndx in range(n_samples):
        if verbose:
                print('iteration '+str(ndx))
        # slice actions available for those states
        A_slice = Q_M_est[s0_idx, slice(None)]
        thresh_idx = np.where(np.greater_equal(A_slice, active_threshold[0]),
                        [True], [False])
        # TODO: explore or don't more smartly
        # choose exploration based on uncertainty. Plot this uncertainty first
        if not thresh_idx.any(): # empty, pick the safest
                if verbose:
                        print('taking safest')
                a_idx = np.argmax(A_slice)
                expected_measure = A_slice[a_idx]
        else: # not empty, pick one of these
                if verbose:
                        print('explore!')
                A_slice[~thresh_idx] = np.nan
                a_idx = np.nanargmin(A_slice)
                expected_measure = A_slice[a_idx]
        a = grids['actions'][0][a_idx]
        # apply action, get to the next state
        x0, p_true = true_model.mapSA2xp_height_angle((s0, a), x0, p_true)
        x_next, failed = true_model.poincare_map(x0, p_true)
        if failed:
                if verbose:
                        print("FAILED!")
                #break
                a = a
                q_new = np.array([[s0, a]])
                X = np.concatenate((X, q_new), axis=0)
                y_new = np.array(0).reshape(-1, 1)
                y = np.concatenate((y, y_new))
                # TODO: restart from expected good results
                # Currently, just restart from some magic numbers
                s_next = np.random.uniform(0.4, 0.8)
                # s0_idx = vibly.digitize_s(s0, grids['states'], s_bin_shape)
                s_next_idx = vibly.digitize_s(s_next, grids['states'], s_bin_shape)
                s_next_idx = np.min([s_next_idx, S_M.size - 1])
                measure = 0
        else:
                s_next = true_model.map2s(x_next, p_true)
        # compare expected measure with true measure
                s_next_idx = vibly.digitize_s(s_next, grids['states'], s_bin_shape)
                # HACK: digitize_s returns the index of the BIN, not the grid
                s_next_idx = np.min([s_next_idx, S_M.size - 1])
                measure = S_M_est[s_next_idx]
                # TODO: once we have a proper gp-projection for S_M, predict this
        if verbose:
                print('measure mismatch: ' + str(measure - expected_measure))
                print("s: "+str(s0) + " a: " +str(a/np.pi*180))

        #Add state action pair to dataset
        q_new = np.array([[s0, a]])
        X = np.concatenate((X, q_new), axis=0)

        y_new = np.array(measure).reshape(-1,1)
        y = np.concatenate((y, y_new))
        gp.set_XY(X=X, Y=y)
        Q_M_est, S_M_est = estimate_sets(gp, X_grid)
        # take another step
        s0 = s_next
        s0_idx = s_next_idx

# batch update
# gp_prior.set_XY()


# repeat