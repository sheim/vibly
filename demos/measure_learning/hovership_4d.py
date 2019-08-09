import models.spaceship4 as true_model
import numpy as np
import pickle
import matplotlib.pyplot as plt

import plotting.corl_plotters as cplot
import measure.active_sampling as sampling

def run_demo(dynamics_model_path = './data/dynamics/', gp_model_path='./data/gp_model/', results_path='./results/'):

    ################################################################################
    # Load model data
    ################################################################################

    dynamics_file = dynamics_model_path + 'spaceship4_map.pickle'
    # use 'slip_prior_proxy.npy' for incorrect prior
    # use 'slip_prior_true.npy' for prior regressed over ground truth
    gp_model_file = gp_model_path + 'spaceship4.npy'

    infile = open(dynamics_file, 'rb')
    data = pickle.load(infile)
    infile.close()

    # A prior action state pair that is considered safe (from system knowledge)
    safest_idx = np.unravel_index(np.argmax(data['Q_M']), data['Q_M'].shape)

    grids = data['grids']
    X_seed = np.atleast_2d(np.array([grids['states'][0][safest_idx[0]],
                                     grids['states'][1][safest_idx[1]],
                                     grids['actions'][0][safest_idx[2]],
                                     grids['actions'][1][safest_idx[3]]],
                                    ))

    SA_grid = np.meshgrid(*(grids['states']), *(grids['actions']), indexing='ij')
    X_grid_points = np.vstack(map(np.ravel, SA_grid)).T

    X_seed = X_grid_points
    y_seed = data['Q_M'].ravel().T

    idx = list(range(len(y_seed)))
    idx = np.random.choice(idx, size=np.min([10, len(y_seed)]), replace=False)

    seed_data = {'X': X_seed[idx,:], 'y': y_seed.reshape(-1,1)[idx,:]}


    X_seed = np.atleast_2d(np.array([1, 0, .5, 0]))
    y_seed = np.array([[1]])
    seed_data = {'X': X_seed, 'y': y_seed.reshape(-1, 1)}

    sampler = sampling.MeasureLearner(model=true_model, model_data=data)
    sampler.init_estimation(seed_data=seed_data, prior_model_path=gp_model_file, learn_hyperparameters=False)

    sampler.exploration_confidence_s = 0.85
    sampler.exploration_confidence_e = 0.85
    sampler.measure_confidence_s = 0.60
    sampler.measure_confidence_e = 0.60
    sampler.safety_threshold_s = 0.0
    sampler.safety_threshold_e = 0.0

    # randomize, but keep track of it in case you want to reproduce
    sampler.seed = np.random.randint(1, 100)
    print('Seed: ' + str(sampler.seed))

    n_samples = 500

    s0 = X_seed[0,0:2].T

    def plot_callback(sampler, ndx, thresholds):
        # Plot every n-th iteration
        if ndx % 100 == 0 or ndx + 1 == n_samples or ndx == -1:

            extent = [grids['states'][1][0],
                      grids['states'][1][-1],
                      grids['states'][0][0],
                      grids['states'][0][-1]]

            S_M_true = sampler.model_data['S_M']

            Q_V = sampler.current_estimation.safe_level_set(safety_threshold=0,
                                                            confidence_threshold=thresholds['measure_confidence'])
            S_M_0 = sampler.current_estimation.project_Q2S(Q_V)

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            samples = (sampler.X, sampler.y)
            failed_samples = sampler.failed_samples
            if samples is not None and samples[0] is not None:
                action = samples[0][:, 1]
                state = samples[0][:, 0]
                if failed_samples is not None and len(failed_samples) > 0:
                    ax1.scatter(action[failed_samples], state[failed_samples],
                                 marker='x', edgecolors=[[0.9, 0.3, 0.3]], s=100,
                                 facecolors=[[0.9, 0.3, 0.3]])
                    ax2.scatter(action[failed_samples], state[failed_samples],
                                 marker='x', edgecolors=[[0.9, 0.3, 0.3]], s=100,
                                 facecolors=[[0.9, 0.3, 0.3]])
                    failed_samples = np.logical_not(failed_samples)
                    ax1.scatter(action[failed_samples], state[failed_samples],
                                 facecolors=[[0.9, 0.3, 0.3]], s=100,
                                 marker='.', edgecolors='none')
                    ax2.scatter(action[failed_samples], state[failed_samples],
                                 facecolors=[[0.9, 0.3, 0.3]], s=100,
                                 marker='.', edgecolors='none')
                else:
                    ax1.scatter(action, state,
                                 facecolors=[[0.9, 0.3, 0.3]], s=100,
                                 marker='.', edgecolors='none')
                    ax2.scatter(action, state,
                                 facecolors=[[0.9, 0.3, 0.3]], s=100,
                                 marker='.', edgecolors='none')

            ax1.imshow(S_M_0, origin='lower', interpolation='none', extent=extent)
            ax2.imshow(S_M_true, origin='lower', interpolation='none', extent=extent)

            plt.show()

            if sampler.y is not None:
                print(str(ndx) + " ACCUMULATED ERROR: "
                      + str(np.sum(np.abs(S_M_0 - S_M_true)))
                      + " Failure rate: " + str(np.mean(sampler.y < 0)))

    sampler.run(n_samples=n_samples, s0=s0, callback=plot_callback)



if __name__ == "__main__":
    dynamics_model_path = '../../data/dynamics/'
    gp_model_path = '../../data/gp_model/'
    results_path = '../../results/'

    run_demo(dynamics_model_path=dynamics_model_path,
                        gp_model_path=gp_model_path,
                        results_path=results_path)
