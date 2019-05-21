# Run `initialize_active_sampling.py` first to load everything and the prior
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
S_M_proxy = vibly.project_Q2S(Q_V_proxy, grids, np.sum)
#S_M_proxy = S_M_proxy / grids['actions'][0].size
Q_M_proxy = vibly.map_S2Q(Q_map_proxy, S_M_proxy, Q_V_proxy)

y = Q_M_proxy.flatten().T
y_train = y.reshape(-1,1)
y_scale = np.max(np.abs(y_train), axis=0)[0]
y_train = y_train / y_scale

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
print(y_train[np.argmax(y_train),:] / y_scale)


mu, s2 = gp_prior.predict(X)
mu[idx_notfeas] = np.nan
mu = mu.reshape(100,91)

idx_state = 20
X_test = np.ones((500, 2))
X_test[:, 0] = np.linspace(0, np.pi/2, 500)
X_test[:, 1] *= grids['states'][0][idx_state]

mu, s2 = gp_prior.predict(X_test)

# idx_action = 20
# X_test = np.ones((500, 2))
# X_test[:, 0] = np.linspace(0, 1, 500)
# X_test[:, 1] *= grids['actions'][0][idx_action]
sdx_check = 35

# feasible actions for this state:
A_feas = Q_feas[sdx_check, slice(None)]

X_test = np.zeros((grids['actions'][0][A_feas].size, 2))
X_test[:,0] = grids['actions'][0][A_feas]
X_test[:,1] = grids['states'][0][idx_state]

mu, s2 = gp_prior.predict(X_test)

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

X = np.empty((0,2))
y = np.empty((0,1))
gp = GPy.models.GPRegression(X_train, y_train, kernel,
                             noise_var=0.01,
                             mean_function=mf)

################################################################################
# Stuff
################################################################################

X_grid_1, X_grid_2 = np.meshgrid(grids['actions'], grids['states'])
X_grid = np.column_stack((X_grid_1.flatten(), X_grid_2.flatten()))

def estimate_sets(gp, X_grid):
        Q_M_est, Q_M_est_s2 = gp.predict(X_grid)
        Q_M_est = Q_M_est.reshape(Q_M_proxy.shape)
        Q_M_est[np.logical_not(Q_feas)] = 0 # do not consider infeasible points
        # TODO: normalize prop
        S_M_est = vibly.project_Q2S(Q_M_est, grids, np.sum)/y_scale

        return Q_M_est, S_M_est

Q_M_est, S_M_est = estimate_sets(gp=gp_prior, X_grid=X_grid)
Q_M_prior = np.copy(Q_M_est) # make a copy to compare later

################################################################################
# load ground truth
################################################################################
import slippy.slip as true_model

infile = open('../data/slip_map.pickle', 'rb')
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
S_M_true = vibly.project_Q2S(Q_V_true, grids, np.sum)
#S_M_true = S_M_true / grids['actions'][0].size
Q_M_true = vibly.map_S2Q(Q_map_true, S_M_true, Q_V_true)

plt.imshow(Q_M_true - Q_M_proxy, origin='lower')
plt.show()

Q_M_proxy = Q_M_proxy/y_scale
Q_M_true = Q_M_true/y_scale

plt.imshow(Q_M_true - Q_M_est, origin='lower')
plt.show()
np.max(Q_M_est-Q_M_true)

################################################################################
# Active Sampling part
################################################################################

s_grid_shape = list(map(np.size, grids['states']))
s_bin_shape = tuple(dim+1 for dim in s_grid_shape)
#### from GP approximation, choose parts of Q to sample
n_samples = 10
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
                s_next_idx = np.min([s_next_idx, S_M_est.size - 1])
                measure = 0
        else:
                s_next = true_model.map2s(x_next, p_true)
        # compare expected measure with true measure
                s_next_idx = vibly.digitize_s(s_next, grids['states'], s_bin_shape)
                # HACK: digitize_s returns the index of the BIN, not the grid
                s_next_idx = np.min([s_next_idx, S_M_est.size - 1])
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