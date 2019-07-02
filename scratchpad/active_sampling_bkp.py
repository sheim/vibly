import numpy as np
import matplotlib.pyplot as plt
import pickle

import slippy.viability as vibly
import slippy.slip as proxy_model

import slippy.estimate_measure as estimation


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

# ** Compute measure from grid for warm-start
Q_V_proxy, S_V_proxy = vibly.compute_QV(Q_map_proxy, grids)

S_M_proxy = vibly.project_Q2S(Q_V_proxy, grids, np.mean)
Q_M_proxy = vibly.map_S2Q(Q_map_proxy, S_M_proxy, Q_V_proxy)

seed_n = np.random.randint(1, 100)

print("Seed: " + str(seed_n))
np.random.seed(seed_n)
estimator = estimation.MeasureEstimation(state_dim=1, action_dim=1, seed=seed_n)

learn_hyperparameters = False
if learn_hyperparameters:
    AS_grid = np.meshgrid(grids['actions'][0], grids['states'][0])
    estimator.learn_hyperparameter(AS_grid=AS_grid, Q_M=Q_M_proxy, save='./model/prior.npy')




X_grid_1, X_grid_2 = np.meshgrid(grids['actions'], grids['states'])
X_grid = np.column_stack((X_grid_1.flatten(), X_grid_2.flatten()))

estimator.set_grid_shape(X_grid, Q_map_proxy.shape)

estimator.set_data_empty()

## Start from bad prior

# This comes from knowledge of the system
initial_measure = .2
X_seed = np.atleast_2d(np.array([38 / (180) * np.pi, .45]))
y_seed = np.array([[initial_measure]])

estimator.init_estimator(X_seed, y_seed, load='./model/prior.npy')


## Start from good prior
# idx_safe = np.argwhere(Q_V_proxy.ravel()).ravel()
# idx_unsafe = np.argwhere(~Q_V_proxy.ravel()).ravel()
#
# idx_sample_safe = np.random.choice(idx_safe, size=np.min([200, len(idx_safe)]), replace=False)
# idx_sample_unsafe = np.random.choice(idx_unsafe, size=np.min([100, len(idx_unsafe)]), replace=False)
#
# idx = np.concatenate((idx_sample_safe, idx_sample_unsafe))
#
# X_prior = X_grid[idx, :]
# y_prior = Q_M_proxy.ravel()
# y_prior = y_prior[idx].reshape(-1, 1)
#
# estimator.init_estimator(X_prior, y_prior, load='./model/prior.npy')


################################################################################
# Marching Thresholds
################################################################################

def interpo(a, b, n):
    # assert n > 0 and n <=1
    return a + n*(b-a)


exploration_confidence_s = 0.95
exploration_confidence_e = 0.95
measure_confidence_s = 0.7
measure_confidence_e = 0.7

safety_threshold_s = 0.0
safety_threshold_e = 0.0

################################################################################
# Safe prior for later comparisons
################################################################################


Q_V_prior = estimator.safe_level_set(safety_threshold = 0, confidence_threshold = measure_confidence_s)
Q_M_prior, Q_M_s2_prior = estimator.Q_M()
S_M_prior = estimator.project_Q2S(Q_V_prior)

################################################################################
# load ground truth model
################################################################################

# ** load ground truth
import slippy.slip as true_model
# import slippy.nslip as true_model

infile = open('../data/slip_map.pickle', 'rb')
data = pickle.load(infile)
infile.close()

Q_map_true = data['Q_map']
x0 = data['x0']
p_true = data['p']
p_true['x0'] = data['x0']
x0 = true_model.reset_leg(x0, p_true)
grids = data['grids']

# x0 is just a placeholder needed for the simulation
true_model.mapSA2xp = true_model.mapSA2xp_height_angle

# ** Compute measure from grid for warm-start
Q_V_true, S_V_true = vibly.compute_QV(Q_map_true, grids)
Q_feas = vibly.get_feasibility_mask(true_model.feasible,
                                    true_model.mapSA2xp_height_angle,
                                    grids=grids, x0=x0, p0=p_true)
S_M_true = vibly.project_Q2S(Q_V_true, grids, np.mean)
Q_M_true = vibly.map_S2Q(Q_map_true, S_M_true, Q_V_true)

s_grid_shape = list(map(np.size, grids['states']))
s_bin_shape = tuple(dim+1 for dim in s_grid_shape)
a_grid_shape = list(map(np.size, grids['actions']))
a_bin_shape = tuple(dim+1 for dim in a_grid_shape)

verbose = 2
np.set_printoptions(precision=4)


def learn(estimator, s0, p_true, n_samples=100, X=None, y=None):

    # Init empty Dataset
    if X is None or y is None:
        X = np.empty((0, estimator.input_dim))
        y = np.empty((0, 1))

    s0_idx = vibly.digitize_s(s0, grids['states'],
                              s_grid_shape, to_bin=False)

    failed_samples = [False]*n_samples

    for ndx in range(n_samples):

        Q_V = estimator.safe_level_set(safety_threshold=0,
                                       confidence_threshold=measure_confidence)
        S_M_0 = estimator.project_Q2S(Q_V)

        Q_M, Q_M_s2 = estimator.Q_M()

        Q_V_explore = estimator.safe_level_set(safety_threshold=0,
                                       confidence_threshold=exploration_confidence)

        # TODO Alex: dont grid states, dont evalutate the whole sets
        # slice actions available for those states
        A_slice = np.copy(Q_V_explore[s0_idx, slice(None)])

        A_slice_s2 = np.copy(Q_M_s2[s0_idx, slice(None)])

        thresh_idx = np.array(A_slice, dtype=bool)

        if not thresh_idx.any():  # empty, pick the safest
            if verbose > 1:
                print('taking safest on iteration '+str(ndx+1))

            Q_V_prop = estimator.safe_level_set(safety_threshold=0,
                                           confidence_threshold=None)
            Q_prop_slice = np.copy(Q_V_prop[s0_idx, slice(None)])

            a_idx = np.argmax(Q_prop_slice
                              + np.random.randn(A_slice.shape[0])*0.01)
        else:  # not empty, pick one of these

            A_slice[~thresh_idx] = np.nan
            A_slice_s2[~thresh_idx] = np.nan

            exploration_heuristic = np.sqrt(A_slice_s2)
            a_idx = np.nanargmax(exploration_heuristic)

        a = grids['actions'][0][a_idx]  #+ (np.random.rand()-0.5)*np.pi/36
        # apply action, get to the next state
        x0, p_true = true_model.mapSA2xp((s0, a), p_true)
        x_next, failed = true_model.poincare_map(x0, p_true)
        if failed:
            failed_samples[ndx] = True
            if verbose:
                print('FAILED on iteration '+str(ndx+1))

            S_M_safe = estimator.project_Q2S(Q_V_explore)

            # TODO make dimensions work
            s_next_idx = np.random.choice(np.where(S_M_safe > 0)[0])
            s_next = grids['states'][0][s_next_idx]

            measure = estimator.failure_value
        else:
            s_next = true_model.map2s(x_next, p_true)
            s_next_idx = vibly.digitize_s(s_next, grids['states'],
                                          s_grid_shape, to_bin=False)

            measure = S_M_0[s_next_idx]

        #Add action state pair to dataset
        q_new = np.array([[a, s0]])
        X = np.concatenate((X, q_new), axis=0)

        y_new = np.array(measure).reshape(-1,1)
        y = np.concatenate((y, y_new))
        estimator.set_data(X=X, Y=y)

        # take another step
        s0 = s_next
        s0_idx = s_next_idx

    estimator.failed_samples.extend(failed_samples)

    return estimator, s_next


S_M_safe_prior = np.copy(S_M_prior)
print("INITIAL ACCUMULATED ERROR: " + str(np.sum(np.abs(S_M_safe_prior-S_M_true))))


steps = 20
estimator.failed_samples = list([False])

import plotting.corl_plotters as cplot

np.random.seed(seed_n)


X = np.atleast_2d(np.array([38/(180)*np.pi, .45]))
Y = np.array([[initial_measure]])

s0 = .45

for ndx in range(steps):  # in case you want to do small increments
    # active_threshold = active_threshold_f(ndx, steps, adapt_type='linear')
    exploration_confidence = interpo(exploration_confidence_s,
                                     exploration_confidence_e, ndx/steps)
    measure_confidence = interpo(measure_confidence_s,
                                 measure_confidence_e, ndx/steps)
    safety_threshold = interpo(safety_threshold_s, safety_threshold_e,
                               ndx/steps)

    estimator, s0 = learn(estimator, s0, p_true, n_samples=5, X=X, y=Y)

    Q_V = estimator.safe_level_set(safety_threshold=safety_threshold,
                                   confidence_threshold=measure_confidence)
    Q_M, Q_M_s2 = estimator.Q_M()
    S_M_0 = estimator.project_Q2S(Q_V)

    Q_V_exp = estimator.safe_level_set(safety_threshold=0,
                                       confidence_threshold=
                                       exploration_confidence)

    fig = cplot.plot_Q_S(Q_V+Q_V_exp+Q_V_true*3, (S_M_0, S_M_true), grids,
                         samples=(estimator.gp.X, estimator.gp.Y),
                         failed_samples=estimator.failed_samples,
                         S_labels=("safe estimate", "ground truth"))
    # plt.savefig('./sample'+str(ndx))
    # plt.close('all')
    plt.show()

    X = estimator.gp.X
    Y = estimator.gp.Y

    print(str(ndx) + " ACCUMULATED ERROR: "
          + str(np.sum(np.abs(S_M_0-S_M_true)))
          + " Failure rate: " + str(np.mean(Y < 0)))



Q_V = estimator.safe_level_set(safety_threshold=0.05, confidence_threshold=exploration_confidence)
Q_M, Q_M_s2 = estimator.Q_M()
S_M_0 = estimator.project_Q2S(Q_V)
s0 = 0.45
failures = 0
for ndx in range(100): # in case you want to do small increments

    s0_idx = vibly.digitize_s(s0, grids['states'],
                            s_grid_shape, to_bin=False)


    A_slice = np.copy(Q_V[s0_idx, slice(None)])

    thresh_idx = np.array(A_slice, dtype=bool)

    if not thresh_idx.any(): # empty, pick the safest
        if verbose > 1:
            print('WARNING: LEFT SET!')
        a_idx = np.argmax(A_slice)
    else: # not empty, pick one of these
        A_slice[~thresh_idx] = np.nan
        # A_slice_s2[~thresh_idx] = np.nan
        a_idx = np.random.choice(np.where(thresh_idx)[0])


    a = grids['actions'][0][a_idx] #+ (np.random.rand()-0.5)*np.pi/36
    # apply action, get to the next state
    x0, p_true = true_model.mapSA2xp((s0, a), p_true)
    x_next, failed = true_model.poincare_map(x0, p_true)
    if failed:
        failures += 1
        if verbose:
            print("FAILED!")
            print((s0, a))
        # TODO: restart from expected good results
        # Currently, just restart from some magic numbers
        s_next_idx = np.random.choice(np.where(S_M_0 > 0)[0])
        s_next = grids['states'][0][s_next_idx]
        s_next_idx = vibly.digitize_s(s_next, grids['states'],
                                s_grid_shape, to_bin = False)
    else:
        s_next = true_model.map2s(x_next, p_true)
        s_next_idx = vibly.digitize_s(s_next, grids['states'],
                                        s_grid_shape, to_bin=False)

    # take another step
    s0 = s_next
    s0_idx = s_next_idx

print(str(failures))