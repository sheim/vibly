import numpy as np
import matplotlib.pyplot as plt
import pickle

import slippy.viability as vibly
import slippy.slip as proxy_model

import slippy.estimate_measure as estimation
from scipy.stats import norm

################################################################################
# Load and unpack data
################################################################################
infile = open('../data/slip_map.pickle', 'rb')
data = pickle.load(infile)
infile.close()

Q_map_proxy = data['Q_map']
x0 = data['x0']
p_proxy = data['p']
grids = data['grids']

################################################################################
# Compute measure from grid for warm-start
################################################################################
Q_V_proxy, S_V_proxy = vibly.compute_QV(Q_map_proxy, grids)

Q_feas = vibly.get_feasibility_mask(proxy_model.feasible, proxy_model.mapSA2xp_height_angle,
            grids=grids, x0=x0, p0=p_proxy)
S_M_proxy = vibly.project_Q2S(Q_V_proxy, grids, np.mean)
#S_M_proxy = S_M_proxy / grids['actions'][0].size
Q_M_proxy = vibly.map_S2Q(Q_map_proxy, S_M_proxy, Q_V_proxy)

estimator = estimation.MeasureEstimation(state_dim=1, action_dim=1, seed=1)

AS_grid = np.meshgrid(grids['actions'][0], grids['states'][0])
estimator.prepare_data(AS_grid=AS_grid, Q_M=Q_M_proxy, Q_V=Q_V_proxy, Q_feas=Q_feas)
estimator.create_prior(load='./model/prior.npy', save='./model/prior.npy')


################################################################################
# Stuff
################################################################################

X_grid_1, X_grid_2 = np.meshgrid(grids['actions'], grids['states'])
X_grid = np.column_stack((X_grid_1.flatten(), X_grid_2.flatten()))

viable_threshold = 0.01 # TODO: tune this! and perhaps make it adaptive...
# TODO: I have a feeling using this more is quite key

# TODO: using the GP with no data doesn't work put in a data point far away?
estimator.set_data(X=np.array([[-100,-100]]), Y=np.array([[0]]))

Q_M_est, Q_M_est_s2, S_M_est = estimator.estimate_sets(X_grid=X_grid, grids=grids, Q_feas=Q_feas,
                                                       viable_threshold=viable_threshold)
Q_M_prior = np.copy(Q_M_est) # make a copy to compare later
Q_M_prior_s2 = np.copy(Q_M_est_s2)
S_M_prior = np.copy(S_M_est)

################################################################################
# load ground truth
################################################################################
import slippy.nslip as true_model

infile = open('../data/nslip_map.pickle', 'rb')
data = pickle.load(infile)
infile.close()

Q_map_true = data['Q_map']
x0 = data['x0']
p_true = data['p']
x0 = true_model.reset_leg(x0, p_true)
grids = data['grids']

################################################################################
# Compute measure from grid for warm-start
################################################################################
Q_V_true, S_V_true = vibly.compute_QV(Q_map_true, grids)

Q_feas = vibly.get_feasibility_mask(true_model.feasible, true_model.mapSA2xp_height_angle,
            grids=grids, x0=x0, p0=p_true)
S_M_true = vibly.project_Q2S(Q_V_true, grids, np.mean)
#S_M_true = S_M_true / grids['actions'][0].size
Q_M_true = vibly.map_S2Q(Q_map_true, S_M_true, Q_V_true)
# plt.imshow(Q_M_true - Q_M_proxy, origin='lower')
# plt.show()

# plt.imshow(Q_M_true - Q_M_est, origin='lower')
# plt.show()
# np.max(Q_M_est-Q_M_true)

################################################################################
# Active Sampling part
################################################################################

s_grid_shape = list(map(np.size, grids['states']))
s_bin_shape = tuple(dim+1 for dim in s_grid_shape)
a_grid_shape = list(map(np.size, grids['actions']))
a_bin_shape = tuple(dim+1 for dim in a_grid_shape)
#### from GP approximation, choose parts of Q to sample
active_threshold = 0.2
# pick initial state
s0 = np.random.uniform(0.4, 0.7)
s0_idx = vibly.digitize_s(s0, grids['states'], s_grid_shape, to_bin = False)

verbose = 1
np.set_printoptions(precision=4)

def learn(estimator, x0, p_true, n_samples = 100, verbose = 0, X = None, y = None):

    # X_observe = np.zeros([n_samples, 2])
    # Y_observe = np.zeros(n_samples)
    if X is None or y is None:
        X = np.empty((0,2))
        y = np.empty((0,1))

    Q_M_est, Q_M_est_s2, S_M_est = estimator.estimate_sets(X_grid=X_grid, grids=grids, Q_feas=Q_feas,
                                                           viable_threshold=viable_threshold)
    s0 = np.random.uniform(0.4, 0.7)
    s0_idx = vibly.digitize_s(s0, grids['states'],
                            s_grid_shape, to_bin = False)

    failed_samples = [False]*n_samples

    for ndx in range(n_samples):
        if verbose:
            print('iteration '+str(ndx+1))
        # slice actions available for those states
        A_slice = Q_M_est[s0_idx, slice(None)]
        A_slice_s2 = Q_M_est_s2[s0_idx, slice(None)]

        # Calculate probability of failure for current actions
        failure_threshold = 0
        prop_fail = 1-norm.cdf((A_slice - failure_threshold) / np.sqrt(A_slice_s2))

        probability_threshold = 0.1
        thresh_idx = np.where(np.less(prop_fail, probability_threshold),
                        [True], [False])

        # TODO: explore or don't more smartly
        # choose exploration based on uncertainty. Plot this uncertainty first
        if not thresh_idx.any(): # empty, pick the safest
            if verbose > 1:
                print('taking safest')
            a_idx = np.argmax(A_slice)
            expected_measure = A_slice[a_idx]
        else: # not empty, pick one of these
            if verbose > 1:
                print('explore!')
            A_slice[~thresh_idx] = np.nan
            nan_idxs = np.argwhere(np.isnan(A_slice))
            # TODO: There seems to be a bug variance should be all equal when there is no data
            A_slice_s2[nan_idxs] = np.nan
            a_idx = np.nanargmax(A_slice_s2) # use this for variance
            # a_idx = np.nanargmin(A_slice)
            expected_measure = A_slice[a_idx]

        a = grids['actions'][0][a_idx]
        # apply action, get to the next state
        x0, p_true = true_model.mapSA2xp_height_angle((s0, a), x0, p_true)
        x_next, failed = true_model.poincare_map(x0, p_true)
        if failed:
            failed_samples[ndx] = True
            if verbose:
                print("FAILED!")
            #break
            q_new = np.array([[s0, a]])
            y_new = np.array(0).reshape(-1, 1)
            # TODO: restart from expected good results
            # Currently, just restart from some magic numbers
            s_next = np.random.uniform(0.3, 0.8)
            # s0_idx = vibly.digitize_s(s0, grids['states'], s_bin_shape)
            s_next_idx = vibly.digitize_s(s_next, grids['states'],
                                    s_grid_shape, to_bin = False)
            # TODO: weight failures more than successes
            measure = 0
        else:
            s_next = true_model.map2s(x_next, p_true)
    # compare expected measure with true measure
            s_next_idx = vibly.digitize_s(s_next, grids['states'],
                                    s_grid_shape, to_bin = False)
            measure = S_M_est[s_next_idx]
            # TODO: once we have a proper gp-projection for S_M, predict this
        if verbose > 1:
            print('measure mismatch: ' + str(measure - expected_measure))
            print("s: "+str(s0) + " a: " +str(a/np.pi*180))

        #Add state action pair to dataset
        q_new = np.array([[a, s0]])
        X = np.concatenate((X, q_new), axis=0)

        y_new = np.array(measure).reshape(-1,1)
        y = np.concatenate((y, y_new))
        estimator.set_data(X=X, Y=y)
        if verbose > 1:
            print("mapped measure (Q_map)" +
                " est: "+ str(S_M_est[Q_map_proxy[s0_idx, a_idx]]) +
                " prox: " + str(S_M_proxy[Q_map_proxy[s0_idx, a_idx]]))
            print("mapped measure (dyn)" +
                " est: " + str(S_M_est[s_next_idx]) +
                " prox: " + str(S_M_proxy[s_next_idx]))
            print("predicted measure (Q_M) " +
                " est: " + str(Q_M_est[s0_idx, a_idx]) +
                " prox: " + str(Q_M_proxy[s0_idx, a_idx]))
        # elif verbose > 0:
        #         print("error: " + str(np.sum(np.abs(Q_M_est-Q_M_true))))

        Q_M_est, Q_M_est_s2, S_M_est = estimator.estimate_sets(X_grid=X_grid, grids=grids, Q_feas=Q_feas,
                                                               viable_threshold=viable_threshold)
        # take another step
        s0 = s_next
        s0_idx = s_next_idx
    estimator.failed_samples.extend(failed_samples)
    return estimator

print("INITIAL ACCUMULATED ERROR: " + str(np.sum(np.abs(Q_M_prior-Q_M_true))))
# plt.imshow(np.abs(Q_M_est-Q_M_true), origin='lower')
# plt.show()

steps = 10
# gp = learn(gp, x0, p_true, steps=1, verbose = 1, tabula_rasa=True)
# Q_M_est, Q_M_est_s2, S_M_est = estimate_sets(gp, X_grid)
# print(" ACCUMULATED ERROR: " + str(np.sum(np.abs(Q_M_est-Q_M_true))))
estimator.failed_samples = list()

import plotting.corl_plotters as cplot
tabula_rasa = True

np.random.seed(1)
X = None
Y = None
for ndx in range(steps): # in case you want to do small increments

    # TODO: I don't understand what x0 is. What are those 7 values?
    estimator = learn(estimator, x0, p_true, n_samples=20, verbose=1, X=X, y=Y)

    Q_M_est, Q_M_est_s2, S_M_est = estimator.estimate_sets(X_grid=X_grid, grids=grids, Q_feas=Q_feas,
                                                           viable_threshold=viable_threshold)
    Q_M_safe = np.copy(Q_M_est)
    Q_M_safe[np.less(Q_M_safe, active_threshold)] = 0
    Q_M_safe[np.greater_equal(Q_M_safe, active_threshold)] = 1
    S_M_safe = np.mean(Q_M_safe, axis=1)
    fig = cplot.plot_Q_S(Q_M_est, (S_M_safe, S_M_est, S_M_true), grids,
                         samples = (estimator.gp.X, estimator.gp.Y),
                         failed_samples = estimator.failed_samples,
                         S_labels=("safe estimate", "viable estimate", "ground truth"))
    #plt.savefig('./sample'+str(ndx))
    #plt.close('all')
    plt.show()
    print(str(ndx) + " ACCUMULATED ERROR: " + str(np.sum(np.abs(Q_M_est-Q_M_true))))
    tabula_rasa = False

    # Recompute prior, and restart (naive forgetting)

    #Keep the failures:
    if np.any(estimator.failed_samples):
        X = estimator.gp.X[estimator.failed_samples,:]
        Y = estimator.gp.Y[estimator.failed_samples,:]
        estimator.failed_samples = [x for x in estimator.failed_samples if x]

    else:
        # TODO still a hack
        X = np.array([[-100, -100]])
        Y = np.array([[0]])
        estimator.failed_samples = [True]

    AS_grid = np.meshgrid(grids['actions'][0], grids['states'][0])

    Q_V = np.where(np.greater_equal(Q_M_est, viable_threshold), [True], [False])

    estimator.prepare_data(AS_grid=AS_grid, Q_M=Q_M_est, Q_V=Q_V, Q_feas=Q_feas)
    estimator.create_prior(save=False)

    estimator.set_data(X=X, Y=Y)


# Good things to plot
# plt.imshow(np.abs(Q_M_est-Q_M_true), origin='lower')
# Q_M_trimmed = np.copy(Q_M_est) # see estimate_sets()
# Q_M_trimmed[np.less(Q_M_trimmed, viable_threshold)] = 0 # level set
# Q_M_trimmed[np.greater_equal(Q_M_trimmed, viable_threshold)] = 1
# plt.imshow(Q_M_trimmed, origin='lower')
# plt.imshow(np.abs(Q_M_trimmed-Q_V_true), origin='lower')

# TODO plot sampled points and their true values
# TODO check how many sampled points are outside the true viable set, and check what the state of the gp was at that point, to see why it sampled there.

# TODO batch update