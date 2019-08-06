import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import GPy

import numpy as np
# from slippy.slip import *
import viability as vibly # TODO: get rid of this dependency
from scipy.stats import norm


class MeasureEstimation:

    def __init__(self, state_dim, action_dim, grids, seed=None):

        self.prior_kernel = None
        self.prior = None
        self.prior_mean = None

        self.gp = None
        self.kernel = None

        np.random.seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.X_grid = None
        self.Q_shape = None
        self.grids = grids

    def set_grid_shape(self, X_grid, Q_shape):
        self.X_grid = X_grid
        self.Q_shape = Q_shape
        # assert, Q_shape and X_grid make sense.

    @property
    def input_dim(self):
        return self.action_dim + self.state_dim

    # The failure value is chosen such that at the point there is only some probability left that the point is viable
    @property
    def failure_value(self):
        return - 2*np.sqrt(self.gp.likelihood.variance)

    def init_default_kernel(self):

        # Initialize GP with a general kernel and constrain hyperparameter
        # TODO Hyperpriors and kernel choice

        kernel_1 = GPy.kern.Matern52(input_dim=self.input_dim, variance=1., lengthscale=.5,
                                      ARD=True, name='kern1')

        kernel_1.variance.constrain_bounded(1e-3, 1e4)

        kernel_2 = GPy.kern.RBF(input_dim=self.input_dim, variance=1, lengthscale=.5,
            ARD=True, name='kern2')

        kernel_2.variance.constrain_bounded(1e-3, 1e4)

        return kernel_1 # + kernel_2



    def learn_hyperparameter(self, AS_grid, Q_M, Q_V, save='./model/prior.npy'):

        # Expects the AS_grid data to be in a n-d grid (e.g. a (3,5,5,5) ndarray) where n**d is the number of samples
        # To create such a grid from the grid points:
        # np.mgrid[action1, action2, state1, state2]
        # or np.meshgrid(np.linspace(0,3,3), np.linspace(0,3,3), np.linspace(0,3,4))
        # The Q_M,Q_V and Q_feas data needs to be in a corresponding n**d grid

        AS = np.vstack(map(np.ravel, AS_grid)).T
        Q = Q_M.ravel().T

        # Sample training points from safe and unsafe prior
        idx_safe = np.argwhere(Q_V.ravel()).ravel()
        idx_unsafe = np.argwhere(~Q_V.ravel()).ravel()

        if len(idx_safe) > 1000: # or len(idx_unsafe) > 250:
            print('Warning: Dataset to big to learn hyperparameter fast. Using a subset to speed things up.')

        idx_sample_safe = np.random.choice(idx_safe, size=np.min([1000, len(idx_safe)]), replace=False)
        idx_sample_unsafe = np.random.choice(idx_unsafe, size=np.min([250, len(idx_unsafe)]), replace=False)

        #idx = np.concatenate((idx_sample_safe, idx_sample_unsafe))
        idx = idx_sample_safe

        X_train = AS[idx, :]
        y_train = Q[idx].reshape(-1, 1)

        self.prior_kernel = self.init_default_kernel()

        gp_prior = GPy.models.GPRegression(X=X_train,
                                           Y=y_train,
                                           kernel=self.prior_kernel,
                                           noise_var=0.001)

        gp_prior.likelihood.variance.constrain_bounded(1e-7, 1e-3)
        gp_prior.optimize_restarts(num_restarts=3)  # This is expensive

        print(gp_prior)
        print(gp_prior.kern1.lengthscale)

        if save:
            file = Path(save)
            file.parent.mkdir(parents=True, exist_ok=True)
            gps = {'gp_prior': gp_prior.param_array}
            np.save(save, gps)
        else:
            print('Warning: Model NOT saved. All the work was for naught')



    def init_estimator(self, X, y, load='./model/prior.npy'):

        self.prior_kernel = self.init_default_kernel()

        gp_prior = GPy.models.GPRegression(X=X,
                                           Y=y,
                                           kernel=self.prior_kernel,
                                           noise_var=0.001)

        if load and Path(load).exists():
            gps = np.load(load)

            gp_prior.update_model(False)  # do not call the underlying expensive algebra on load
            gp_prior.initialize_parameter()  # Initialize the parameters (connect the parameters up)
            gp_prior[:] = gps.item().get('gp_prior')  # Load the parameters
            gp_prior.update_model(True)  # Call the algebra only once

        else:
            print('WARNING: No model found. Using default kernel parameters. Make sure you really want to do this!')

        self.prior = gp_prior

        def prior_mean(x):
            mu, s2 = gp_prior.predict(np.atleast_2d(x))
            return mu

        self.prior_mean = GPy.core.Mapping(self.input_dim, 1)
        self.prior_mean.f = prior_mean
        self.prior_mean.update_gradients = lambda a, b: None

        print(self.prior)
        print(self.prior_kernel.lengthscale)
        self.kernel = self.prior_kernel.copy()


    def set_data(self, X=None, Y=None):

        if (X is None) or (Y is None):
            self.set_data_empty()

        self.gp = GPy.models.GPRegression(X=X,
                                          Y=Y,
                                          kernel=self.kernel,
                                          noise_var=0.001,  # self.prior.likelihood.variance,
                                          mean_function=self.prior_mean)

    # Utility function to empty out data set
    def set_data_empty(self):
        # GPy fails with empty dataset. So put in a data point far removed from everything
        X = np.ones((1,self.input_dim))*-1000
        y = np.zeros((1,1))
        self.set_data(X=X, Y=y)

    def project_Q2S(self, Q):
        a_axes = tuple(range(Q.ndim - self.action_dim, Q.ndim))
        return np.mean(Q, a_axes)


    def safe_level_set(self, safety_threshold = 0, confidence_threshold = 0.5, current_state=None):
        # assert self.Q_shape != None, "Q_shape was not initialized"
        # assert self.X_grid != None, "X_grid was not initialized"

        if current_state is None:
            Q_est, Q_est_s2 = self.gp.predict(self.X_grid)
        else:
            a_grid = np.meshgrid(*(self.grids['actions']), indexing='ij')
            a_points = np.vstack(map(np.ravel, a_grid)).T

            # TODO:  check math
            state_points = np.ones((a_points.shape[0], len(self.grids['actions']))) * current_state.T

            x_points = np.hstack((state_points, a_points))
            Q_est, Q_est_s2 = self.gp.predict(x_points)



        Q_level_set = norm.cdf((Q_est - safety_threshold) / np.sqrt(Q_est_s2))

        if confidence_threshold is not None:
            Q_level_set[np.where(Q_level_set < confidence_threshold)] = 0
            Q_level_set[np.where(Q_level_set > confidence_threshold)] = 1

        # TODO: Return boolean or int
        if current_state is None:
            return self.prediction_to_grid(Q_level_set)
        else:
            return Q_level_set.reshape(self.Q_shape[-self.action_dim:])

    # TODO: unite with safe_level_set
    def Q_M(self, current_state = None):
        if current_state is None:
            Q_est, Q_est_s2 = self.gp.predict(self.X_grid)
        else:
            a_grid = np.meshgrid(*(self.grids['actions']), indexing='ij')
            a_points = np.vstack(map(np.ravel, a_grid)).T

            # TODO:  check math
            state_points = np.ones((a_points.shape[0], len(self.grids['actions']))) * current_state.T

            x_points = np.hstack((state_points, a_points))
            Q_est, Q_est_s2 = self.gp.predict(x_points)

        if current_state is None:
            return self.prediction_to_grid(Q_est), self.prediction_to_grid(Q_est_s2)
        else:
            return Q_est.reshape(self.Q_shape[-self.action_dim:]), Q_est_s2.reshape(self.Q_shape[-self.action_dim:])

    def prediction_to_grid(self, pred):
        return pred.reshape(self.Q_shape)


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

    # Uncomment if you want to learn the hyperparameters of the GP. This might take a while
    # estimation.learn_hyperparameter(AS_grid, Q_M, Q_V, save='./model/prior.npy')

    X_seed = np.atleast_2d(np.array([38 / (180) * np.pi, .45]))

    initial_measure = .2
    y_seed = np.array([[initial_measure]])

    estimation.init_estimator(X_seed, y_seed, load='./model/prior.npy')


    X_grid_1, X_grid_2 = np.meshgrid(grids['states'], grids['actions'])
    X_grid = np.column_stack((X_grid_1.flatten(), X_grid_2.flatten()))

    estimation.set_grid_shape(X_grid, Q_M.shape)

    # Start from an empty data set
    estimation.set_data_empty()

    X_1, X_2 = np.meshgrid(grids['actions'], grids['states'])
    X = np.column_stack((X_1.flatten(), X_1.flatten()))

    Q_V_est = estimation.safe_level_set(safety_threshold=0, confidence_threshold=0.6)
    S_M_est = estimation.project_Q2S(Q_V_est)
    plt.plot(S_M_est)
    plt.show()

    Q_M_mean, Q_M_S2 = estimation.Q_M()
    plt.imshow(Q_M_mean, origin='lower')
    plt.show()

    # estimation.set_data(X=X_seed, Y=y_seed)
    # Q_V_est = estimation.safe_level_set(safety_threshold=0, confidence_threshold=0.6)
    # S_M_est = estimation.project_Q2S(Q_V_est)
    # plt.plot(S_M_est)
    # plt.show()

