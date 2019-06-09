import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import GPy

from slippy.slip import *
import slippy.viability as vibly


class MeasureEstimation:

    def __init__(self, state_dim, action_dim, seed=None):
        self.prior_data = {
            'AS': None,    # The action state space, TODO: make it state action space?
            'Q': None      # The measure to be learned
        }
        self.prior_kernel = None
        self.prior = None
        self.prior_mean = None

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

        return kernel_1 + kernel_2


    def prepare_data(self, AS_grid, Q_M, Q_V, Q_feas):

        # Expects the AS_grid data to be in a n-d grid (e.g. a (3,5,5,5) ndarray) where n**d is the number of samples
        # To create such a grid from the grid points:
        # np.mgrid[action1, action2, state1, state2]
        # or np.meshgrid(np.linspace(0,3,3), np.linspace(0,3,3), np.linspace(0,3,4))
        # The Q_M,Q_V and Q_feas data needs to be in a corresponding n**d grid

        AS = np.vstack(map(np.ravel, AS_grid)).T
        Q = Q_M.ravel().T


        # Sample training points from safe and unsafe prior
        idx_safe = np.argwhere(Q_V.ravel()).ravel()
        idx_notfeas = np.argwhere(~Q_feas.ravel()).ravel()
        idx_unsafe = np.argwhere(Q_feas.ravel() & ~Q_V.ravel()).ravel()

        idx_sample_safe = np.random.choice(idx_safe, size=1000, replace=False)
        idx_sample_unsafe = np.random.choice(idx_unsafe, size=500, replace=False)

        idx = np.concatenate((idx_sample_safe, idx_sample_unsafe))

        self.prior_data['AS'] = AS[idx, :]
        self.prior_data['Q'] = Q[idx].reshape(-1, 1)

    def create_prior(self, kernel=None, load='./model/prior.npy', save='./model/prior.npy', force_new_model=False):

        # Prior kernel is always the default one learned from data
        self.prior_kernel = self.init_default_kernel()

        gp_prior = GPy.models.GPRegression(X=self.prior_data['AS'],
                                           Y=self.prior_data['Q'],
                                           kernel=self.prior_kernel,
                                           noise_var=0.01,
                                           normalizer=True)

        if not force_new_model and load and Path(load).exists():
            gp_prior.update_model(False)  # do not call the underlying expensive algebra on load
            gp_prior.initialize_parameter()  # Initialize the parameters (connect the parameters up)
            gp_prior[:] = np.load(load)  # Load the parameters
            gp_prior.update_model(True)  # Call the algebra only once

        else:
            gp_prior.likelihood.variance.constrain_bounded(1e-7, 1e-3)
            gp_prior.optimize_restarts(num_restarts=3)  # This is expensive

        self.prior = gp_prior

        if save:
            file = Path(save)
            file.parent.mkdir(parents=True, exist_ok=True)
            np.save(save, gp_prior.param_array)

        def prior_mean(x):
            mu, s2 = gp_prior.predict(np.atleast_2d(x))
            return mu

        self.prior_mean = GPy.core.Mapping(self.input_dim, 1)
        self.prior_mean.f = prior_mean
        self.prior_mean.update_gradients = lambda a, b: None

        # If there is no specific kernel supplied, use the prior
        if kernel is None:

            self.kernel = self.prior_kernel.copy()

    def set_data(self, X, Y):

        self.gp = GPy.models.GPRegression(X=X,
                                          Y=Y,
                                          kernel=self.kernel,
                                          noise_var=0.01, #self.prior.likelihood.variance, # TODO: Is this a good idea?
                                          mean_function=self.prior_mean)

    # Estimate the learned sets on a grid, Q_feas gives feasible points as well as the shape of the grid
    # TODO: How to make sure X_grid is in correct oder when reshaped into Q_feas.shape()?
    def estimate_sets(self, X_grid, grids, Q_feas, viable_threshold):

        Q_M_est, Q_M_est_s2 = self.gp.predict(X_grid)
        Q_M_est = Q_M_est.reshape(Q_feas.shape)
        Q_M_est[np.logical_not(Q_feas)] = -1e-10  # do not consider infeasible points

        Q_M_est_s2 = Q_M_est_s2.reshape(Q_feas.shape)
        Q_M_est_s2[np.logical_not(Q_feas)] = 0

        # TODO make viable_threshold a function of variance (probability to fail)
        # Q_V_est = np.copy(Q_M_est)
        # Q_V_est[np.less(Q_V_est, viable_threshold)] = 0 # can also use a mask
        # S_M_est = vibly.project_Q2S(Q_V_est.astype(bool), grids, np.mean)
        # * or trim Q_M_est directly
        Q_M_est[np.less(Q_M_est, viable_threshold)] = 0
        S_M_est = vibly.project_Q2S(Q_M_est.astype(bool), grids, np.mean)

        # TODO  perhaps alwas trim Q_M as well?
        # though I guess that damages some properties...
        # Q_M_est[np.less(Q_V_est, viable_threshold)] = 0
        return Q_M_est, Q_M_est_s2, S_M_est


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
    S_M = vibly.project_Q2S(Q_V, grids, np.sum)
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
    estimation.create_prior(load=False, save=False)

    # estimation.estimate_sets()
