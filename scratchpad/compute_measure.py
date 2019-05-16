import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

import GPy

from slippy.slip import *
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

Q_feas = vibly.get_feasibility_mask(feasible, mapSA2xp_height_angle,
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

# warp_f = GPy.util.warping_functions.TanhFunction(n_terms=2)
# gp_prior = GPy.models.WarpedGP(X_train, y_train, kernel=kernel, warping_function=warp_f)
# gp_prior.likelihood.variance = 0.01


gp_prior = GPy.models.GPRegression(X_train, y_train,
        kernel=kernel, noise_var=0.01)

gp_prior.likelihood.variance.constrain_bounded(1e-7, 1e-3)

#lognormal = GPy.priors.Gamma(a=1,b=0.5)
#gp_prior.likelihood.variance.set_prior(lognormal)


gp_prior.optimize_restarts(num_restarts=2)

print(gp_prior)

mu, s2 = gp_prior.predict(np.atleast_2d(X_train[np.argmax(y_train),:]))
print('GP prior predictions')
print(mu)
print(np.sqrt(s2) * 2)
print('GP prior true')
print(y_train[np.argmax(y_train),:] / y_scale)


mu, s2 = gp_prior.predict(X)
mu[idx_notfeas] = -1
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

idx_action = 20
X_test = np.ones((500, 2))
X_test[:, 0] = np.linspace(0, 1, 500)
X_test[:, 1] *= grids['actions'][0][idx_action]

mu, s2 = gp_prior.predict(X_test)

plt.plot(np.linspace(0, 1, 500), mu)
plt.plot(np.linspace(0, 1, 500), mu+np.sqrt(s2)*2)
plt.plot(np.linspace(0, 1, 500), mu-np.sqrt(s2)*2)
plt.plot(grids['states'][0].reshape(-1,),
        Q_M[:,idx_action].reshape(-1,) / y_scale)

plt.show()

# def prior_mean(x):
#     mu, s2 = gp_prior.predict(np.atleast_2d(x))
#
#     return mu
#
#
# mf = GPy.core.Mapping(2,1)
# mf.f = prior_mean
# mf.update_gradients = lambda a, b: None
#
#
# kernel = GPy.kern.Matern52(input_dim=2,
#                       variance=np.array(kernel.variance.copy()),
#                       lengthscale=np.array(kernel.lengthscale.copy()),
#                       ARD=True)
#
# gp = GPy.models.GPRegression(X_train, y_train, kernel,
#                              noise_var=0.01,
#                              mean_function=mf)
#
#
# gp.likelihood.variance.constrain_bounded(1e-3, 1e2)
# gp.kern.variance.constrain_bounded(1e-3, 1e4)
#
# gp.optimize_restarts(num_restarts=1)
#
# print(gp)
#
#
# mu, s2 = gp.predict(np.atleast_2d(X_train[np.argmax(y_train),:]))
# print(mu * y_scale)
# print(np.sqrt(s2) * y_scale * 2)
#
# print(y_train[np.argmax(y_train),:])
