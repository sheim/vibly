import numpy as np
import pickle

import slippy.viability as vibly

import slippy.estimate_measure as estimate_measure


def linear_interpolation(a, b, n):
    # assert n > 0 and n <=1
    return a + n * (b - a)


class MeasureLearner:

    def __init__(self, model_data, model):

        self.current_estimation = None

        self.model_data = model_data

        grids = model_data['grids']
        self.grids = grids

        Q_map_proxy = model_data['Q_map']
        self.grid_shape = Q_map_proxy.shape

        p_true = model_data['p']
        p_true['x0'] = model_data['x0']
        self.p = p_true

        seed_n = np.random.randint(1, 100)

        print("Seed: " + str(seed_n))
        self.seed = np.random.seed(seed_n)

        # A bunch of parameters to adjust. These are just defaults

        self.exploration_confidence_s = 0.95
        self.exploration_confidence_e = 0.95
        self.measure_confidence_s = 0.7
        self.measure_confidence_e = 0.7

        self.safety_threshold_s = 0.0
        self.safety_threshold_e = 0.0

        self.interpolation = linear_interpolation

        self.model = model

        # Sampled measures
        self.X = None
        self.y = None
        self.failed_samples = list()

        self.verbose = 2

    def init_estimation(self, seed_data, prior_model_path = './model/prior.npy', learn_hyperparameters=False):

        grids = self.grids
        state_dim = len(grids['states'])
        action_dim = len(grids['actions'])

        estimation = estimate_measure.MeasureEstimation(state_dim=state_dim, action_dim=action_dim, seed=self.seed)

        AS_grid = np.meshgrid(grids['actions'][0], grids['states'][0])

        if learn_hyperparameters:

            Q_M_proxy = self.model_data['Q_M']
            Q_V_proxy = self.model_data['Q_V']

            estimation.learn_hyperparameter(AS_grid=AS_grid, Q_M=Q_M_proxy, Q_V=Q_V_proxy, save=prior_model_path)

        X_grid_points = np.vstack(map(np.ravel, AS_grid)).T
        estimation.set_grid_shape(X_grid_points, self.grid_shape)
        estimation.set_data_empty()

        X_seed = seed_data['X']
        y_seed = seed_data['y']

        estimation.init_estimator(X_seed, y_seed, load='./model/prior.npy')
        estimation.set_data_empty()

        self.current_estimation = estimation

        # Add the seed to to the initial data set?
        # self.X = X_seed
        # self.Y = y_seed

    def sample(self, s0, measure_confidence, exploration_confidence, ndx, safety_threshold=0):

        estimation = self.current_estimation

        # Init empty Dataset
        if self.X is None or self.y is None:
            self.X = np.empty((0, estimation.input_dim))
            self.y = np.empty((0, 1))

        # TODO Alex: dont grid states, dont evalutate the whole sets
        s_grid_shape = list(map(np.size, self.grids['states']))
        s0_idx = vibly.digitize_s(s0, self.grids['states'],
                                  s_grid_shape, to_bin=False)


        Q_V = estimation.safe_level_set(safety_threshold=0,
                                        confidence_threshold=measure_confidence)

        S_M_0 = estimation.project_Q2S(Q_V)

        Q_M, Q_M_s2 = estimation.Q_M()
        Q_V_explore = estimation.safe_level_set(safety_threshold=safety_threshold,
                                                confidence_threshold=exploration_confidence)

        # slice actions available for those states
        A_slice = np.copy(Q_V_explore[s0_idx, slice(None)])

        A_slice_s2 = np.copy(Q_M_s2[s0_idx, slice(None)])

        thresh_idx = np.array(A_slice, dtype=bool)

        if not thresh_idx.any():  # empty, pick the safest
            if self.verbose > 1:
                print('taking safest on iteration ' + str(ndx + 1))

            Q_V_prop = estimation.safe_level_set(safety_threshold=0, confidence_threshold=None)
            Q_prop_slice = np.copy(Q_V_prop[s0_idx, slice(None)])
            a_idx = np.argmax(Q_prop_slice + np.random.randn(A_slice.shape[0]) * 0.01)

        else:  # not empty, pick one of these

            A_slice[~thresh_idx] = np.nan
            A_slice_s2[~thresh_idx] = np.nan

            exploration_heuristic = np.sqrt(A_slice_s2)
            a_idx = np.nanargmax(exploration_heuristic)

        a = self.grids['actions'][0][a_idx]  # + (np.random.rand()-0.5)*np.pi/36
        # apply action, get to the next state
        x0, p_true = self.model.mapSA2xp((s0, a), self.p)
        x_next, failed = self.model.p_map(x0, p_true)

        if failed:
            self.failed_samples.append(True)
            if self.verbose:
                print('FAILED on iteration ' + str(ndx + 1))

            S_M_safe = estimation.project_Q2S(Q_V_explore)

            # TODO make dimensions work
            s_next_idx = np.random.choice(np.where(S_M_safe > 0)[0])
            s_next = self.grids['states'][0][s_next_idx]

            measure = self.current_estimation.failure_value
        else:
            self.failed_samples.append(False)

            s_next = self.model.map2s(x_next, p_true)
            s_next_idx = vibly.digitize_s(s_next, self.grids['states'],
                                          s_grid_shape, to_bin=False)

            measure = S_M_0[s_next_idx]


        # Add action state pair to dataset
        q_new = np.array([[a, s0]])
        self.X = np.concatenate((self.X, q_new), axis=0)

        y_new = np.array(measure).reshape(-1, 1)
        self.y = np.concatenate((self.y, y_new))
        estimation.set_data(X=self.X, Y=self.y)

        self.current_estimation = estimation

        return s_next, s_next_idx

    def run(self, n_samples, s0, callback=None):

        # Callback for e.g. plotting
        if callable(callback):
            thresholds = {
                'exploration_confidence': self.exploration_confidence_s,
                'measure_confidence': self.measure_confidence_s,
                'safety_threshold': self.safety_threshold_s
            }
            callback(self, -1, thresholds)

        for ndx in range(n_samples):

            exploration_confidence = self.interpolation(self.exploration_confidence_s, self.exploration_confidence_e, ndx / n_samples)

            measure_confidence = self.interpolation(self.measure_confidence_s, self.measure_confidence_e, ndx / n_samples)

            safety_threshold = self.interpolation(self.safety_threshold_s, self.safety_threshold_e, ndx / n_samples)

            s0, s0_idx = self.sample(s0,
                                     measure_confidence=measure_confidence,
                                     exploration_confidence=exploration_confidence,
                                     safety_threshold=safety_threshold,
                                     ndx=ndx)

            # Callback for e.g. plotting
            if callable(callback):
                thresholds = {
                    'exploration_confidence': exploration_confidence,
                    'measure_confidence': measure_confidence,
                    'safety_threshold': safety_threshold
                }
                callback(self, ndx, thresholds)




if __name__ == "__main__":

    import slippy.slip as true_model

    #TODO Steve
    true_model.mapSA2xp = true_model.mapSA2xp_height_angle
    true_model.p_map = true_model.poincare_map

    ################################################################################
    # Load model data
    ################################################################################
    infile = open('../data/slip_map.pickle', 'rb')
    data = pickle.load(infile)
    infile.close()

    # Start from bad prior

    # This comes from knowledge of the system
    # TODO clean this up
    X_seed = np.atleast_2d(np.array([38 / (180) * np.pi, .45]))
    y_seed = np.array([[.2]])
    seed_data = {'X': X_seed, 'y': y_seed}

    sampler = MeasureLearner(model=true_model, model_data=data)
    sampler.init_estimation(seed_data=seed_data, prior_model_path='./model/prior.npy')

    s0 = .45

    sampler.run(n_samples=100, s0=s0)