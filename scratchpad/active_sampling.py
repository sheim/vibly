import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

import GPy

import slippy.viability as vibly
import slippy.slip as proxy_model

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

y = Q_M_proxy.flatten().T
y_train = y.reshape(-1,1)
# y_scale = np.max(np.abs(y_train), axis=0)[0]
# y_train = y_train / y_scale

# TODO This is ugly af
# * actually seems like it's not so bad: https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
X_train_1, X_train_2 = np.meshgrid(grids['actions'], grids['states'])
X = np.column_stack((X_train_1.flatten(), X_train_2.flatten()))

np.random.seed(1)
idx = np.random.choice(9100, size=3000, replace=False)

idx_safe = np.argwhere(Q_V_proxy.flatten()).ravel()
idx_notfeas = np.argwhere(~Q_feas.flatten()).ravel()
idx_unsafe = np.argwhere( Q_feas.flatten() & ~Q_V_proxy.flatten()).ravel()

idx_1 = np.random.choice(idx_safe, size=1000, replace=False)
idx_2 = np.random.choice(idx_unsafe, size=500, replace=False)

idx = np.concatenate((idx_1, idx_2))

y_train = y_train[idx]
X_train = X[idx, :]

#X_train = X_train[0:100,:]
#y_train = y_train[0:100]

kernel_1 = GPy.kern.Matern32(input_dim=2, variance=1., lengthscale=.5,
        ARD=True, name='kern1')
kernel_1.variance.constrain_bounded(1e-3, 1e4)

kernel_2 = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=.5,
        ARD=True, name='kern2')
kernel_2.variance.constrain_bounded(1e-3, 1e4)

# kernel_3 = GPy.kern.ChangePointBasisFuncKernel(input_dim=1, active_dims=0, changepoint=(.3,.6), variance=1, ARD=True, name='kern3')
# kernel_4 = GPy.kern.ChangePointBasisFuncKernel(input_dim=1, active_dims=1, changepoint=(.5,.9), variance=1, ARD=True, name='kern4')

kernel = kernel_1 + kernel_2

# Loading a model
# Model creation, without initialization:
gp_prior = GPy.models.GPRegression(X_train, y_train, kernel=kernel, noise_var=0.01)
gp_prior.update_model(False) # do not call the underlying expensive algebra on load
gp_prior.initialize_parameter() # Initialize the parameters (connect the parameters up)
gp_prior[:] = np.load('../data/model_save.npy') # Load the parameters
gp_prior.update_model(True) # Call the algebra only once

print(gp_prior)

mu, s2 = gp_prior.predict(np.atleast_2d(X_train[np.argmax(y_train),:]))
print('GP prior predictions')
print(mu)
print(np.sqrt(s2) * 2)
print('GP prior true')
print(y_train[np.argmax(y_train),:])

################################################################################
# Setup GP from prior data
################################################################################

def prior_mean(x):
    mu, s2 = gp_prior.predict(np.atleast_2d(x))
    return mu


mf = GPy.core.Mapping(2,1)
mf.f = prior_mean
mf.update_gradients = lambda a, b: None


kernel = kernel.copy()

gp = GPy.models.GPRegression(X_train, y_train, kernel,
                             noise_var=0.01,
                             mean_function=mf)

################################################################################
# Stuff
################################################################################

X_grid_1, X_grid_2 = np.meshgrid(grids['actions'], grids['states'])
X_grid = np.column_stack((X_grid_1.flatten(), X_grid_2.flatten()))

viable_threshold = 0.01 # TODO: tune this! and perhaps make it adaptive...
# TODO: I have a feeling using this more is quite key

def estimate_sets(gp, X_grid):
        Q_M_est, Q_M_est_s2 = gp.predict(X_grid)
        Q_M_est = Q_M_est.reshape(Q_M_proxy.shape)

        Q_M_est[np.logical_not(Q_feas)] = 0 # do not consider infeasible points
        Q_V_est = np.copy(Q_M_est)
        Q_V_est[np.less(Q_V_est, viable_threshold)] = 0
        S_M_est = vibly.project_Q2S(Q_V_est.astype(bool), grids, np.mean)

        Q_M_est_s2 = Q_M_est_s2.reshape(Q_M_proxy.shape)
        Q_M_est_s2[np.logical_not(Q_feas)] = 0 # do not consider infeasible points

        # TODO  perhaps alwas trim Q_M as well?
        # though I guess that damages some properties...
        # Q_M_est[np.less(Q_V_est, viable_threshold)] = 0
        return Q_M_est, Q_M_est_s2, S_M_est

Q_M_est, Q_M_est_s2, S_M_est = estimate_sets(gp=gp_prior, X_grid=X_grid)
Q_M_prior = np.copy(Q_M_est) # make a copy to compare later

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
#### from GP approximation, choose parts of Q to sample
n_samples = 10
active_threshold = 0.2
# pick initial state
s0 = np.random.uniform(0.4, 0.7)
s0_idx = vibly.digitize_s(s0, grids['states'], s_grid_shape, to_bin = False)

verbose = 1
np.set_printoptions(precision=4)

def learn(gp, x0, p_true, n_samples = 100, verbose = 0, tabula_rasa = False):

        # X_observe = np.zeros([n_samples, 2])
        # Y_observe = np.zeros(n_samples)
        if tabula_rasa:
                X = np.empty((0,2))
                y = np.empty((0,1))
        else:
                X = gp.X
                y = gp.Y

        Q_M_est, Q_M_est_s2, S_M_est = estimate_sets(gp, X_grid)

        s0 = np.random.uniform(0.4, 0.7)
        s0_idx = vibly.digitize_s(s0, grids['states'],
                                s_grid_shape, to_bin = False)

        for ndx in range(n_samples):
                if verbose:
                        print('iteration '+str(ndx))
                # slice actions available for those states
                A_slice = Q_M_est[s0_idx, slice(None)]
                A_slice_s2 = Q_M_est_s2[s0_idx, slice(None)]
                thresh_idx = np.where(np.greater_equal(A_slice, active_threshold),
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

                        idxs = np.argwhere(~np.isnan(A_slice))
                        # TODO: There seems to be a bug variance should be all equal when there is no data
                        a_idx = np.argmax(A_slice_s2[idxs])

                        expected_measure = A_slice[a_idx]
                a = grids['actions'][0][a_idx]
                # apply action, get to the next state
                x0, p_true = true_model.mapSA2xp_height_angle((s0, a), x0, p_true)
                x_next, failed = true_model.poincare_map(x0, p_true)
                if failed:
                        if verbose:
                                print("FAILED!")
                        #break
                        q_new = np.array([[s0, a]])
                        X = np.concatenate((X, q_new), axis=0)
                        y_new = np.array(0).reshape(-1, 1)
                        y = np.concatenate((y, y_new))
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
                gp.set_XY(X=X, Y=y)
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

                Q_M_est, Q_M_est_s2, S_M_est = estimate_sets(gp, X_grid)
                # take another step
                s0 = s_next
                s0_idx = s_next_idx
        return gp

plt.imshow(np.abs(Q_M_est-Q_M_true), origin='lower')
plt.show()

gp = learn(gp, x0, p_true, n_samples=1, verbose = 1, tabula_rasa=True)

Q_M_est, Q_M_est_s2, S_M_est = estimate_sets(gp, X_grid)
print("INITIAL ACCUMULATED ERROR: " + str(np.sum(np.abs(Q_M_est-Q_M_true))))
# plt.imshow(np.abs(Q_M_est-Q_M_true), origin='lower')
# plt.show()

n_samples = 50
for ndx in range(5):
        gp = learn(gp, x0, p_true, n_samples=n_samples, verbose = 1, tabula_rasa=False)
        Q_M_est, Q_M_est_s2, S_M_est = estimate_sets(gp, X_grid)
        plt.imshow(np.abs(Q_M_est-Q_M_true), origin='lower')
        plt.show()
        print(str(ndx) + " ACCUMULATED ERROR: " + str(np.sum(np.abs(Q_M_est-Q_M_true))))
        if np.sum(np.abs(Q_M_est-Q_M_true)) > 300:
                break
        # probably actually only want to care about trimmed error, see below

# Good things to plot
# plt.imshow(np.abs(Q_M_est-Q_M_true), origin='lower')
Q_M_trimmed = np.copy(Q_M_est) # see estimate_sets()
Q_M_trimmed[np.less(Q_M_trimmed, viable_threshold)] = 0
# plt.imshow(Q_M_trimmed, origin='lower')
plt.imshow(np.abs(Q_M_trimmed-Q_M_true), origin='lower')
np.sum(np.abs(Q_M_trimmed-Q_M_true))

# TODO plot sampled points and their true values
# TODO check how many sampled points are outside the true viable set, and check what the state of the gp was at that point, to see why it sampled there.

# batch update
# gp_prior.set_XY()


# repeat