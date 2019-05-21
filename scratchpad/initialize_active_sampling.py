import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

import GPy

import slippy.viability as vibly

################################################################################
# Load and unpack data
################################################################################
infile = open('../data/slip_map.pickle', 'rb')
data = pickle.load(infile)
infile.close()

Q_map_prior = data['Q_map']
x0 = data['x0']
p_prior = data['p']
grids = data['grids']

################################################################################
# Compute measure from grid for warm-start
################################################################################
Q_V_prior, S_V_prior = vibly.compute_QV(Q_map_prior, grids)

Q_feas = vibly.get_feasibility_mask(feasible, mapSA2xp_height_angle,
            grids=grids, x0=x0, p0=p_prior)
S_M_prior = vibly.project_Q2S(Q_V_prior, grids, np.sum)
#S_M_prior = S_M_prior / grids['actions'][0].size
Q_M_prior = vibly.map_S2Q(Q_map_prior, S_M_prior, Q_V_prior)

y = Q_M_prior.flatten().T
y_train = y.reshape(-1,1)
y_scale = np.max(np.abs(y_train), axis=0)[0]
y_train = y_train / y_scale

# TODO This is ugly af
# * actually seems like it's not so bad: https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
X_train_1, X_train_2 = np.meshgrid(grids['actions'], grids['states'])
X = np.column_stack((X_train_1.flatten(), X_train_2.flatten()))

np.random.seed(1)
idx = np.random.choice(9100, size=3000, replace=False)

idx_safe = np.argwhere(Q_V_prior.flatten()).ravel()
idx_notfeas = np.argwhere(~Q_feas.flatten()).ravel()
idx_unsafe = np.argwhere( Q_feas.flatten() & ~Q_V_prior.flatten()).ravel()

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
        Q_M_est = Q_M_est.reshape(Q_M_prior.shape)
        Q_M_est[np.logical_not(Q_feas)] = 0 # do not consider infeasible points
        # TODO: normalize prop
        S_M_est = vibly.project_Q2S(Q_M_est, grids, np.sum)/y_scale

        return Q_M_est, S_M_est

Q_M_est, S_M_est = estimate_sets(gp=gp_prior, X_grid=X_grid)
Q_M_prior = np.copy(Q_M_est) # make a copy to compare later

################################################################################
# load ground truth
################################################################################
import slippy.nslip as true_model

infile = open('../data/n_slip.pickle', 'rb')
data = pickle.load(infile)
infile.close()

Q_map_true = data['Q_map']
x0 = data['x0']
p_true = data['p']
grids = data['grids']

################################################################################
# Compute measure from grid for warm-start
################################################################################
Q_V_true, S_V_true = vibly.compute_QV(Q_map_true, grids)

Q_feas = vibly.get_feasibility_mask(true_model.feasible, true_model.mapSA2xp_height_angle,
            grids=grids, x0=x0, p0=p_true)
S_M_true = vibly.project_Q2S(Q_V_true, grids, np.sum)/y_scale
#S_M_true = S_M_true / grids['actions'][0].size
Q_M_true = vibly.map_S2Q(Q_map_true, S_M_true, Q_V_true)