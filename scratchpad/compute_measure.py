import numpy as np
import matplotlib.pyplot as plt
import time

from slippy.slip import *
import slippy.viability as vibly

import GPy

data = np.load('slip_map.npz')
Q_map = data['Q_map']
Q_F = data['Q_F']

grids = {}
temp = data['grids']
grids['states'] = temp.item().get('states')
grids['actions'] = temp.item().get('actions')

Q_V, S_V = vibly.compute_QV(Q_map, grids)
S_M = vibly.project_Q2S(Q_V, grids, np.sum)
#S_M = S_M / grids['actions'][0].size
Q_M = vibly.map_S2Q(Q_map, S_M, Q_V)
plt.plot(S_M)
plt.show()
plt.imshow(Q_M, origin='lower')
plt.show()
# save file
timestr = time.strftime("%Y_%m_%H_%M_%S")
print("Finished at " + timestr)
print(timestr)
# data2save = {"grids": grids, "Q_map": Q_map, "Q_F": Q_F, "p" : p,
#             "P_map" : poincare_map}
# np.savez('slip_constr_', **data2save)

# plt.imshow(Q_V, origin = 'lower')
# plt.show()


y_train = Q_M.flatten().T
y_train = y_train.reshape(-1,1)
y_scale = np.max(np.abs(y_train), axis=0)[0]
y_train = y_train / y_scale

# TODO This is ugly af
X_train_1, X_train_2 = np.meshgrid(grids['states'], grids['actions'])
X_train = np.column_stack((X_train_1.flatten(), X_train_2.flatten()))

np.random.seed(1)
#idx = np.random.choice(9100, size=2000, replace=False)
idx = Q_V.flatten()
y_train = y_train[idx]
X_train = X_train[idx, :]

#X_train = X_train[0:100,:]
#y_train = y_train[0:100]

kernel_1 = GPy.kern.Matern52(input_dim=2, variance=1., lengthscale=1, ARD=True, name='kern1')
kernel_1.variance.constrain_bounded(1e-3, 1e4)

kernel_2 = GPy.kern.RBF(input_dim=2, variance=.5, lengthscale=.2, ARD=True, name='kern2')
kernel_2.variance.constrain_bounded(1e-3, 1e4)

kernel = kernel_1 #+ kernel_2
gp_prior = GPy.models.GPRegression(X_train, y_train, kernel, noise_var=0.01)

gp_prior.likelihood.variance.constrain_bounded(1e-7, 1e1)

#lognormal = GPy.priors.Gamma(a=1,b=0.5)
#gp_prior.likelihood.variance.set_prior(lognormal)

gp_prior.optimize_restarts(num_restarts=1)

print(gp_prior)

mu, s2 = gp_prior.predict(np.atleast_2d(X_train[np.argmax(y_train),:]))
print(mu * y_scale)
print(np.sqrt(s2) * y_scale * 2)

print(y_train[np.argmax(y_train),:])

X_test = np.ones((500, 2))
X_test[:, 0] *= grids['states'][0][40]
X_test[:, 1] = np.linspace(0, np.pi/2, 500)

mu, s2 = gp_prior.predict(X_test)

plt.plot(np.linspace(0, np.pi/2, 500), mu)
plt.plot(np.linspace(0, np.pi/2, 500), mu+np.sqrt(s2)*2)
plt.plot(np.linspace(0, np.pi/2, 500), mu-np.sqrt(s2)*2)


plt.plot(grids['actions'][0].reshape(-1,), Q_M[40,:].reshape(-1,) / y_scale)

plt.show()

X_test = np.ones((500, 2))
X_test[:, 0] = np.linspace(0, 1, 500)
X_test[:, 1] *= grids['actions'][0][40]

mu, s2 = gp_prior.predict(X_test)

plt.plot(np.linspace(0, 1, 500), mu)
plt.plot(np.linspace(0, 1, 500), mu+np.sqrt(s2)*2)
plt.plot(np.linspace(0, 1, 500), mu-np.sqrt(s2)*2)
plt.plot(grids['states'][0].reshape(-1,), Q_M[:,40].reshape(-1,) / y_scale)

plt.show()

def prior_mean(x):
    mu, s2 = gp_prior.predict(np.atleast_2d(x))

    return mu


mf = GPy.core.Mapping(2,1)
mf.f = prior_mean
mf.update_gradients = lambda a, b: None


kernel = GPy.kern.Matern52(input_dim=2,
                      variance=np.array(kernel.variance.copy()),
                      lengthscale=np.array(kernel.lengthscale.copy()),
                      ARD=True)

gp = GPy.models.GPRegression(X_train, y_train, kernel,
                             noise_var=0.01,
                             mean_function=mf)


gp.likelihood.variance.constrain_bounded(1e-3, 1e2)
gp.kern.variance.constrain_bounded(1e-3, 1e4)

gp.optimize_restarts(num_restarts=1)

print(gp)


mu, s2 = gp.predict(np.atleast_2d(X_train[np.argmax(y_train),:]))
print(mu * y_scale)
print(np.sqrt(s2) * y_scale * 2)

print(y_train[np.argmax(y_train),:])
