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

################################################################################
# Compute measure from grid for warm-start
################################################################################
Q_V_proxy, S_V_proxy = vibly.compute_QV(Q_map_proxy, grids)

Q_feas = vibly.get_feasibility_mask(proxy_model.feasible, proxy_model.mapSA2xp_height_angle,
            grids=grids, x0=x0, p0=p_proxy)
S_M_proxy = vibly.project_Q2S(Q_V_proxy, grids, np.mean)
Q_M_proxy = vibly.map_S2Q(Q_map_proxy, S_M_proxy, Q_V_proxy)

estimator = estimation.MeasureEstimation(state_dim=1, action_dim=1, seed=1)

AS_grid = np.meshgrid(grids['actions'][0], grids['states'][0])
estimator.prepare_data(AS_grid=AS_grid, Q_M=Q_M_proxy, Q_V=Q_V_proxy, Q_feas=Q_feas)
estimator.create_prior(load='./model/prior.npy', save='./model/prior.npy')

################################################################################
# Exploration decay
################################################################################

alpha_init = 0.9
alpha_min = 0.9

def active_threshold_f(n, max):

   # Exponential
   k = np.log(1/0.01) / max
   alpha = alpha_min + (alpha_init - alpha_min) * np.exp(-k*n)

   # Linear
   alpha = alpha_min + (alpha_init - alpha_min) * n/max
   return alpha


active_threshold = active_threshold_f(0,10)

################################################################################
# Stuff
################################################################################

X_grid_1, X_grid_2 = np.meshgrid(grids['actions'], grids['states'])
X_grid = np.column_stack((X_grid_1.flatten(), X_grid_2.flatten()))


estimator.set_data_empty()

sets = estimator.estimate_sets(X_grid=X_grid, grids=grids, Q_feas=Q_feas, active_threshold=active_threshold)
Q_M_prior = np.copy(sets.Q_M_est) # make a copy to compare later
S_M_prior = np.copy(sets.S_M_est)
Q_V_prior = np.copy(sets.Q_V_est)


################################################################################
# load ground truth
################################################################################
import slippy.slip as true_model
# import slippy.slip as true_model

# infile = open('../data/nslip_map.pickle', 'rb')
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
true_model.mapSA2xp = lambda q, p: true_model.mapSA2xp_height_angle(q, x0, p)

################################################################################
# Compute measure from grid for warm-start
################################################################################
Q_V_true, S_V_true = vibly.compute_QV(Q_map_true, grids)
Q_feas = vibly.get_feasibility_mask(true_model.feasible, true_model.mapSA2xp_height_angle,
            grids=grids, x0=x0, p0=p_true)
S_M_true = vibly.project_Q2S(Q_V_true, grids, np.mean)
Q_M_true = vibly.map_S2Q(Q_map_true, S_M_true, Q_V_true)

################################################################################
# Active Sampling part
################################################################################

s_grid_shape = list(map(np.size, grids['states']))
s_bin_shape = tuple(dim+1 for dim in s_grid_shape)
a_grid_shape = list(map(np.size, grids['actions']))
a_bin_shape = tuple(dim+1 for dim in a_grid_shape)
#### from GP approximation, choose parts of Q to sample

# pick initial state
s0 = np.random.uniform(0.4, 0.7)
s0_idx = vibly.digitize_s(s0, grids['states'], s_grid_shape, to_bin = False)

verbose = 1
np.set_printoptions(precision=4)

def learn(estimator, s0, p_true, n_samples = 100, X = None, y = None):

    # Init empty Dataset
    if X is None or y is None:
        X = np.empty((0,estimator.input_dim))
        y = np.empty((0,1))

    # In sets are currently Q_M_est, Q_M_est_s2, S_M_est, Q_V_est
    sets = estimator.estimate_sets(X_grid=X_grid, grids=grids, Q_feas=Q_feas, active_threshold=active_threshold)

    s0_idx = vibly.digitize_s(s0, grids['states'],
                            s_grid_shape, to_bin=False)

    failed_samples = [False]*n_samples

    for ndx in range(n_samples):

        if verbose:
            print('iteration '+str(ndx+1))

        # TODO Alex: dont grid states, dont evalutate the whole sets

        Q_V_slice = np.copy(sets.Q_V_est[s0_idx, slice(None)])

        # NOTE: a higher value in active_threshold indicates accepting a LOWER chance of failing
        thresh_idx = np.where(np.greater(Q_V_slice, active_threshold),
                        [True], [False])

        # slice actions available for those states
        A_slice = np.copy(sets.Q_M_est[s0_idx, slice(None)])
        A_slice_s2 = np.copy(sets.Q_M_est_s2[s0_idx, slice(None)])


        # TODO: explore or don't more smartly
        # choose exploration based on uncertainty. Plot this uncertainty first
        if not thresh_idx.any(): # empty, pick the safest
            if verbose > 1:
                print('taking safest')
            a_idx = np.argmax(Q_V_slice)
        else: # not empty, pick one of these
            if verbose > 1:
                print('explore!')
            A_slice[~thresh_idx] = np.nan
            A_slice_s2[~thresh_idx] = np.nan

            a_idx = np.nanargmax(A_slice_s2)


        a = grids['actions'][0][a_idx]
        # apply action, get to the next state
        x0, p_true = true_model.mapSA2xp((s0, a), p_true)
        x_next, failed = true_model.poincare_map(x0, p_true)
        if failed:
            failed_samples[ndx] = True
            if verbose:
                print("FAILED!")

            # TODO: restart from expected good results
            # Currently, just restart from some magic numbers
            s_next = .5

            s_next_idx = vibly.digitize_s(s_next, grids['states'],
                                    s_grid_shape, to_bin = False)

            measure = estimator.failure_value
        else:
            s_next = true_model.map2s(x_next, p_true)
            s_next_idx = vibly.digitize_s(s_next, grids['states'],
                                          s_grid_shape, to_bin=False)

            measure = sets.S_M_est[s_next_idx]

        #Add action state pair to dataset
        q_new = np.array([[a, s0]])
        X = np.concatenate((X, q_new), axis=0)

        y_new = np.array(measure).reshape(-1,1)
        y = np.concatenate((y, y_new))
        estimator.set_data(X=X, Y=y)

        sets = estimator.estimate_sets(X_grid=X_grid, grids=grids, Q_feas=Q_feas,
                                                               active_threshold=active_threshold)
        # take another step
        s0 = s_next
        s0_idx = s_next_idx

    estimator.failed_samples.extend(failed_samples)

    return estimator, s_next


S_M_safe_prior = np.copy(sets.S_M_safe)
print("INITIAL ACCUMULATED ERROR: " + str(np.sum(np.abs(S_M_safe_prior-S_M_true))))


steps = 30
estimator.failed_samples = list()

import plotting.corl_plotters as cplot

np.random.seed(1)
X = None
Y = None

s0 = .5

for ndx in range(steps): # in case you want to do small increments


    set_old = estimator.estimate_sets(X_grid=X_grid, grids=grids, Q_feas=Q_feas, active_threshold=active_threshold)
    active_threshold = active_threshold_f(ndx, steps)

    estimator, s0 = learn(estimator, s0, p_true, n_samples=1, X=X, y=Y)


    sets = estimator.estimate_sets(X_grid=X_grid, grids=grids, Q_feas=Q_feas, active_threshold=active_threshold)


    fig = cplot.plot_Q_S(sets.Q_M_est, (sets.S_M_safe, sets.S_M_est, S_M_true), grids,
                         samples = (estimator.gp.X, estimator.gp.Y),
                         failed_samples = estimator.failed_samples,
                         S_labels=("safe estimate", "viable estimate", "ground truth"))
    plt.savefig('./sample'+str(ndx))
    plt.close('all')
    # plt.show()
    print(str(ndx) + " error: " + str(np.sum(np.abs(sets.S_M_safe-S_M_true))))

    X = estimator.gp.X
    Y = estimator.gp.Y

    # Recompute prior, and restart (naive forgetting)

    # TODO to do this right we should sample from both distributions :(
    # "Error variance"
    # error = np.abs(Q_M_est - Q_M_est_old)
    # er_var = np.mean(error - np.mean(error)**2)
    #
    # #Keep the failures:
    # if np.any(estimator.failed_samples):
    #     X = estimator.gp.X[estimator.failed_samples,:]
    #     Y = estimator.gp.Y[estimator.failed_samples,:]
    #     estimator.failed_samples = [x for x in estimator.failed_samples if x]
    #
    # else:
    #     estimator.set_data_empty()
    #     estimator.failed_samples = list()
    #
    # AS_grid = np.meshgrid(grids['actions'][0], grids['states'][0])


    # estimator.prepare_data(AS_grid=AS_grid, Q_M=Q_M_est, Q_V=Q_V, Q_feas=Q_feas)
    # estimator.create_prior(save=False)

    # estimator.set_data(X=X, Y=Y)


