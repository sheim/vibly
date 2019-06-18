import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import GPy

from slippy.slip import *
import slippy.viability as vibly
from scipy.integrate import simps
from scipy.stats import norm

# Integrates over the actions, to see how a 'good' a state is
def integral_projection(grid):

    # TODO more than one action dimension
    return lambda a, axis: simps(y=a, x=grid, axis=axis[0])

class MeasureEstimation:

    def __init__(self, state_dim, action_dim, seed=None):
        self.prior_data = {
            'AS': None,    # The action state space, TODO: make it state action space?
            'Q': None      # The measure to be learned
        }
        self.prior_kernel = None
        self.prior = None
        self.prior_mean = None

        self.prior_v = None
        self.prior_mean_v = None
        self.prior_kernel_v = None
        self.kernel_v = None
        self.gp_v = None

        self.gp = None
        self.kernel = None

        np.random.seed(seed)
        self.state_dim = action_dim
        self.action_dim = state_dim

    @property
    def input_dim(self):
        return self.action_dim + self.state_dim

    def init_default_kernel(self):

        # Initialize GP with a general kernel and constrain hyperparameter
        # TODO Hyperpriors and kernel choice

        kernel_1 = GPy.kern.Matern32(input_dim=self.input_dim, variance=1., lengthscale=.5,
                                      ARD=True, name='kern1')

        kernel_1.variance.constrain_bounded(1e-3, 1e4)

        kernel_2 = GPy.kern.RBF(input_dim=self.input_dim, variance=1, lengthscale=.5,
            ARD=True, name='kern2')

        kernel_2.variance.constrain_bounded(1e-3, 1e4)

        kernel_3 = GPy.kern.Poly(input_dim=self.input_dim,
                                 variance=1.0,
                                 scale=1.0,
                                 bias=1.0,
                                 order=4.0,
                                 name='poly')

        bias = GPy.kern.Bias(input_dim=self.input_dim, variance=1.0, active_dims=None, name='bias')

        return kernel_1 + kernel_2



    def prepare_data(self, AS_grid, Q_M, Q_V, Q_feas):

        # Expects the AS_grid data to be in a n-d grid (e.g. a (3,5,5,5) ndarray) where n**d is the number of samples
        # To create such a grid from the grid points:
        # np.mgrid[action1, action2, state1, state2]
        # or np.meshgrid(np.linspace(0,3,3), np.linspace(0,3,3), np.linspace(0,3,4))
        # The Q_M,Q_V and Q_feas data needs to be in a corresponding n**d grid

        AS = np.vstack(map(np.ravel, AS_grid)).T
        Q = Q_M.ravel().T
        V = Q_V.ravel().T * 1 # convert to 0,1


        # Sample training points from safe and unsafe prior
        idx_safe = np.argwhere(Q_V.ravel()).ravel()
        idx_notfeas = np.argwhere(~Q_feas.ravel()).ravel()
        idx_unsafe = np.argwhere(Q_feas.ravel() & ~Q_V.ravel()).ravel()

        idx_sample_safe = np.random.choice(idx_safe, size=500, replace=False)
        idx_sample_unsafe = np.random.choice(idx_unsafe, size=250, replace=False)

        idx = np.concatenate((idx_sample_safe, idx_sample_unsafe))

        self.prior_data['AS'] = AS[idx, :]
        self.prior_data['Q'] = Q[idx].reshape(-1, 1)
        self.prior_data['V'] = V[idx].reshape(-1, 1)

    def create_prior(self, kernel=None, load='./model/prior.npy', save='./model/prior.npy', force_new_model=False):

        # Prior kernel is always the default one learned from data
        self.prior_kernel = self.init_default_kernel()

        gp_prior = GPy.models.GPRegression(X=self.prior_data['AS'],
                                           Y=self.prior_data['Q'],
                                           kernel=self.prior_kernel,
                                           noise_var=0.01)

        self.prior_kernel_v = GPy.kern.RBF(input_dim=self.input_dim, variance=1, lengthscale=.5, ARD=True, name='viable')

        gp_prior_v = GPy.models.GPClassification(X=self.prior_data['AS'],
                                                 Y=self.prior_data['V'],
                                                 kernel=self.prior_kernel_v)

        if not force_new_model and load and Path(load).exists():
            gps = np.load(load)

            gp_prior.update_model(False)  # do not call the underlying expensive algebra on load
            gp_prior.initialize_parameter()  # Initialize the parameters (connect the parameters up)
            gp_prior[:] = gps.item().get('gp_prior')  # Load the parameters
            gp_prior.update_model(True)  # Call the algebra only once

            gp_prior_v.update_model(False)  # do not call the underlying expensive algebra on load
            gp_prior_v.initialize_parameter()  # Initialize the parameters (connect the parameters up)
            gp_prior_v[:] = gps.item().get('gp_prior_v')  # Load the parameters
            gp_prior_v.update_model(True)  # Call the algebra only once

        else:
            gp_prior.likelihood.variance.constrain_bounded(1e-7, 1e-3)
            gp_prior.optimize_restarts(num_restarts=3)  # This is expensive
            print(gp_prior)

            gp_prior_v.optimize_restarts(num_restarts=3)
            print(gp_prior_v)

        self.prior_v = gp_prior_v
        self.prior = gp_prior

        if save:
            file = Path(save)
            file.parent.mkdir(parents=True, exist_ok=True)
            gps = {'gp_prior': gp_prior.param_array, 'gp_prior_v': gp_prior_v.param_array}
            np.save(save, gps)

        def prior_mean(x):
            mu, s2 = gp_prior.predict(np.atleast_2d(x))
            return mu

        self.prior_mean = GPy.core.Mapping(self.input_dim, 1)
        self.prior_mean.f = prior_mean
        self.prior_mean.update_gradients = lambda a, b: None

        def prior_mean_v(x):
            mu, s2 = gp_prior_v.predict(np.atleast_2d(x), include_likelihood=False)
            return mu

        self.prior_mean_v = GPy.core.Mapping(self.input_dim, 1)
        self.prior_mean_v.f = prior_mean_v
        self.prior_mean_v.update_gradients = lambda a, b: None

        # If there is no specific kernel supplied, use the prior
        if kernel is None:

            self.kernel = self.prior_kernel.copy()

            # self.kernel.kern1.variance = self.kernel.kern1.variance
            # self.kernel.kern2.variance = self.kernel.kern2.variance

        self.kernel_v = self.prior_kernel_v.copy()

    def set_data(self, X, Y):

        self.gp = GPy.models.GPRegression(X=X,
                                          Y=Y,
                                          kernel=self.kernel,
                                          noise_var=0.01, #self.prior.likelihood.variance, # TODO: Is this a good idea?
                                          mean_function=self.prior_mean)

        self.gp_v = GPy.models.GPClassification(X=X,
                                          Y=(Y>0)*1,  # TODO explicitly give failure
                                          kernel=self.kernel_v,
                                          mean_function=self.prior_mean_v)


    # Estimate the learned sets on a grid, Q_feas gives feasible points as well as the shape of the grid
    # TODO: How to make sure X_grid is in correct oder when reshaped into Q_feas.shape()?
    def estimate_sets(self, X_grid, grids, Q_feas, viable_threshold):
        # TODO @Alex X_grid should be a grid 
        Q_M_est, Q_M_est_s2 = self.gp.predict(X_grid)
        Q_M_est = Q_M_est.reshape(Q_feas.shape)
        Q_M_est[np.logical_not(Q_feas)] = 0  # do not consider infeasible points

        Q_M_est_s2 = Q_M_est_s2.reshape(Q_feas.shape)
        Q_M_est_s2[np.logical_not(Q_feas)] = 1e-10

        Q_V_prob = norm.cdf((Q_M_est - 0) / np.sqrt(Q_M_est_s2)) # TODO @Alex check if you can just change the sign to get
        # TODO @Alex get rid of safety_threshold magic number
        # This is the $\tilde{\alpha}$ parameter
        # this should be $\in [0, 0.5]
        # put an assert into here for good measure
        safety_threshold = 0.1
        # assert(safety_threshold >= 0.0 && safety_threshold <= 0.5)
        Q_V_est = Q_V_prob>(0.5 + safety_threshold)
        # Q_V_est, _ = self.gp_v.predict(X_grid)
        # Q_V_est = Q_V_est.reshape(Q_feas.shape)
        Q_V_est[np.logical_not(Q_feas)] = 0  # do not consider infeasible points

        S_M_est = vibly.project_Q2S(Q_V_est, grids, np.mean)

        return Q_M_est, Q_M_est_s2, S_M_est, Q_V_est


if __name__ == "__main__":
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
    S_M = vibly.project_Q2S(Q_V, grids, np.mean)
    # S_M = vibly.project_Q2S(Q_V, grids, np.mean)

    #S_M = S_M / grids['actions'][0].size
    Q_M = vibly.map_S2Q(Q_map, S_M, Q_V)
    plt.plot(S_M)
    plt.show()
    plt.imshow(Q_M, origin='lower')
    plt.show()

    ################################################################################
    # Create estimation object
    ################################################################################

    AS_grid = np.meshgrid(grids['actions'][0], grids['states'][0])
    estimation = MeasureEstimation(state_dim=1, action_dim=1, seed=1)
    estimation.prepare_data(AS_grid=AS_grid, Q_M=Q_M, Q_V=Q_V, Q_feas=Q_feas)
    estimation.create_prior(load='./model/prior.npy', save=False)
    estimation.set_data(X=np.array([[-100, -100]]), Y=np.array([[1]]))

    X_train_1, X_train_2 = np.meshgrid(grids['actions'], grids['states'])
    X = np.column_stack((X_train_1.flatten(), X_train_2.flatten()))

    Q_M_est, Q_M_est_s2, S_M_est, Q_V_est = estimation.estimate_sets(X_grid=X, grids=grids,
                                                                        Q_feas=Q_feas,
                                                                        viable_threshold=0.1)

    plt.plot(S_M_est)
    plt.plot(S_M)
    plt.show()

    plt.imshow(Q_M_est, origin='lower')
    plt.show()


    plt.imshow(Q_V_est, origin='lower')
    plt.show()

