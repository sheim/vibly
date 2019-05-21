import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

import GPy

import slippy.slip as slip
import slippy.viability as vibly

################################################################################
# Load and unpack data
################################################################################
infile = open('../data/slip_map.pickle', 'rb')
data = pickle.load(infile)
infile.close()

Q_map = data['Q_map']

Q_F = data['Q_F']
x0 = data['x0']
poincare_map = data['P_map']
p = data['p']
grids = data['grids']

################################################################################
# Compute measure from grid for warm-start
################################################################################
Q_V, S_V = vibly.compute_QV(Q_map, grids)

Q_feas = vibly.get_feasibility_mask(slip.feasible, slip.mapSA2xp_height_angle,
            grids=grids, x0=x0, p0=p)
S_M = vibly.project_Q2S(Q_V, grids, np.sum)
#S_M = S_M / grids['actions'][0].size
Q_M = vibly.map_S2Q(Q_map, S_M, Q_V)
plt.plot(S_M)
plt.show()
plt.imshow(Q_M, origin='lower')
plt.show()

y = Q_M.flatten().T
y_train = y.reshape(-1,1)
y_scale = np.max(np.abs(y_train), axis=0)[0]
y_train = y_train / y_scale

# TODO This is ugly af
# * actually seems like it's not so bad: https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
X_train_1, X_train_2 = np.meshgrid(grids['actions'], grids['states'])
X = np.column_stack((X_train_1.flatten(), X_train_2.flatten()))

np.random.seed(1)
idx = np.random.choice(9100, size=3000, replace=False)

idx_safe = np.argwhere(Q_V.flatten()).ravel()
idx_notfeas = np.argwhere(~Q_feas.flatten()).ravel()
idx_unsafe = np.argwhere( Q_feas.flatten() & ~Q_V.flatten()).ravel()

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


# 2: loading a model
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

plt.imshow(mu, origin='lower')
plt.show()

idx_state = 20
X_test = np.ones((500, 2))
X_test[:, 0] *= grids['states'][0][idx_state]
X_test[:, 1] = np.linspace(0, np.pi/2, 500)

mu, s2 = gp_prior.predict(X_test)

plt.plot(np.linspace(0, np.pi/2, 500), mu)
plt.plot(np.linspace(0, np.pi/2, 500), mu+np.sqrt(s2)*2)
plt.plot(np.linspace(0, np.pi/2, 500), mu-np.sqrt(s2)*2)


plt.plot(grids['actions'][0].reshape(-1,),
        Q_M[idx_state,:].reshape(-1,) / y_scale)

plt.show()

# idx_action = 20
# X_test = np.ones((500, 2))
# X_test[:, 0] = np.linspace(0, 1, 500)
# X_test[:, 1] *= grids['actions'][0][idx_action]
sdx_check = 35

# feasible actions for this state:
A_feas = Q_feas[sdx_check, slice(None)]

X_test = np.zeros((grids['actions'][0][A_feas].size, 2))
X_test[:, 0] = grids['states'][0][sdx_check]
X_test[:,1] = grids['actions'][0][A_feas]

mu, s2 = gp_prior.predict(X_test)

plt.plot(X_test[:,1], mu, color = (0.2,0.6,0.1,1.0))
plt.plot(X_test[:,1], mu+np.sqrt(s2)*2, color = (0.2,0.4,0.1,0.3))
plt.plot(X_test[:,1], mu-np.sqrt(s2)*2, color = (0.2,0.4,0.1,0.3))
# TODO: scaling is different between gp-data and grid data
plt.plot(grids['actions'][0].reshape(-1,),
        Q_M[idx_state, :].reshape(-1,) / y_scale, color = (0.7,0.0,0.0,1))

plt.show()

# numerical projection

def estimate_sets(gp_prior, X):
        # TODO ideally, evaluate an X_feasible
        Q_M_est, Q_M_est_s2 = gp_prior.predict(X)
        Q_M_est = Q_M_est.reshape(Q_M.shape)
        Q_M_est[np.logical_not(Q_feas)] = 0 # do not consider infeasible points
        S_M_est = vibly.project_Q2S(Q_M_est, grids, np.sum)/y_scale # TODO: normalize prop

        return Q_M_est, S_M_est


# TODO: possibly scaling issues in plots?
Q_M_est, S_M_est = estimate_sets(gp_prior=gp_prior, X=X)
Q_M_prior = np.copy(Q_M_est)
plt.plot(S_M)
plt.plot(S_M_est)
plt.show()

################################################################################
# initialize "true" model
################################################################################

import slippy.nslip as true_model

# This if you use nslip instead of slip
p_true = {'mass':80.0, 'stiffness':705.0, 'resting_angle':17/18*np.pi,
        'gravity':9.81, 'angle_of_attack':1/5*np.pi, 'upper_leg':0.5,
        'lower_leg':0.5}
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

for ndx in range(n_samples):
        # slice actions available for those states
        # A_slice = Q_feas[s0_idx, slice(None)] # pick only feasible actions
        A_slice = Q_M_est[s0_idx, slice(None)]
        # pick out indices that are in the active_threshold
        # thresh_idx = np.where( np.logical_and(
        #                         np.greater_equal(A_slice, active_threshold[0]),
        #                         np.less_equal(A_slice, active_threshold[1])),
        #                 [True], [False])
        thresh_idx = np.where(np.greater_equal(A_slice, active_threshold[0]),
                        [True], [False])
        # TODO: explore or don't more smartly
        if not thresh_idx.any(): # empty, pick the safest
                print('taking safest')
                a_idx = np.argmax(A_slice)
                expected_measure = A_slice[a_idx]
        else: # not empty, pick one of these
                print('explore!')
                # A_slice = np.ma.array(A_slice, mask=thresh_idx)
                # TODO: maybe use masked arrays?
                A_slice[~thresh_idx] = np.nan
                a_idx = np.nanargmin(A_slice)
                expected_measure = A_slice[a_idx]

        # apply action, get to the next state
        x0, p_true = true_model.mapSA2xp_height_angle((s0, grids['actions'][0][a_idx]), x0, p_true)
        x_next, failed = true_model.poincare_map(x0, p_true)
        if failed:
                print("FAILED!")
                break
        s_next = true_model.map2s(x_next, p_true)
        # compare expected measure with true measure
        s_next_idx = vibly.digitize_s(s_next, grids['states'], s_bin_shape)
        # HACK: digitize_s returns the index of the BIN, not the grid
        s_next_idx = np.min([s_next_idx, S_M.size - 1])
        measure = S_M_est[s_next_idx]
        print(measure - expected_measure)
        print("s: "+str(s0) + " a:" +str(grids['actions'][0][a_idx]))

        #Add state action pair to dataset
        a = grids['actions'][0][a_idx]
        q_new = np.array([[s0, a]])
        X_train = np.concatenate((X_train, q_new), axis=0)

        y_new = np.array(measure).reshape(-1,1)
        y_train = np.concatenate((y_train, y_new))

        gp_prior.set_XY(X=X_train, Y=y_train)
        Q_M_est, S_M_est = estimate_sets(gp_prior, X)

        # take another step
        s0 = s_next
        s0_idx = s_next_idx

# batch update
# gp_prior.set_XY()


# repeat