import slippy.hovership as true_model
import numpy as np
import pickle
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import plotting.corl_plotters as cplot

import scratchpad.active_sampling as sampling

################################################################################
# Load model data
################################################################################
infile = open('../data/hover_map.pickle', 'rb')
data = pickle.load(infile)
infile.close()

# TODO Steve
true_model.mapSA2xp = true_model.sa2xp
true_model.map2s = true_model.xp2s

# This comes from knowledge of the system
X_seed = np.atleast_2d(np.array([.3, 2]))
y_seed = np.array([[.5]])

seed_data = {'X': X_seed, 'y': y_seed}

sampler = sampling.MeasureLearner(model=true_model, model_data=data)
sampler.init_estimation(seed_data=seed_data, prior_model_path='./model/hover_prior.npy', learn_hyperparameters=False)

sampler.seed = 69

sampler.exploration_confidence_s = 0.98
sampler.exploration_confidence_e = 0.9
sampler.measure_confidence_s = 0.7
sampler.measure_confidence_e = 0.9

n_samples = 500

random_string = str(np.random.randint(1, 10000))

def plot_callback(sampler, ndx, thresholds):
    # Plot every n-th iteration
    if ndx % 50 == 0 or ndx + 1 == n_samples:

        Q_map_true = sampler.model_data['Q_map']
        grids = sampler.grids

        Q_M_true = sampler.model_data['Q_M']
        Q_V_true = sampler.model_data['Q_V']
        S_M_true = sampler.model_data['S_M']
        Q_F = sampler.model_data['Q_F']



        Q_V = sampler.current_estimation.safe_level_set(safety_threshold=thresholds['safety_threshold'],
                                                     confidence_threshold=thresholds['measure_confidence'])
        S_M_0 = sampler.current_estimation.project_Q2S(Q_V)

        Q_V_exp = sampler.current_estimation.safe_level_set(safety_threshold=0,
                                                         confidence_threshold=thresholds['exploration_confidence'])


        today = [datetime.date.today()]
        time = datetime.datetime.now()
        time_string = time.strftime('%H:%M:%S')
        folder_name = 'hovership_experiment' + '_' + str(today[0]) + '_random' + random_string
        filename = 'hovership_experiment' + '_' + str(today[0]) + '_' + time_string + '_' + 'ndx' + str(ndx) + '_data'


        data2save = {
            'Q_V_true': Q_V_true,
            'Q_V_exp': Q_V_exp,
            'Q_V': Q_V,
            'S_M_0': S_M_0,
            'S_M_true': S_M_true,
            'grids': grids,
            'sampler.failed_samples': sampler.failed_samples,
            'ndx': ndx,
            'threshold': thresholds
        }

        path = './data/experiments/' + folder_name + '/'
        file = Path(path)
        file.mkdir(parents=True, exist_ok=True)

        outfile = open(path + filename, 'wb')
        pickle.dump(data2save, outfile)
        outfile.close()

        fig = cplot.plot_Q_S(Q_V_true, Q_V_exp, Q_V, S_M_0, S_M_true, grids,
                             samples=(sampler.X, sampler.y),
                             failed_samples=sampler.failed_samples,
                             S_labels=("safe estimate", "ground truth"), Q_F=Q_F)

        plt.savefig(path + filename + '_fig', format='pdf')
        plt.tight_layout()
        plt.show()
        plt.close('all')
        # plt.show()

        if sampler.y is not None:
            print(str(ndx) + " ACCUMULATED ERROR: "
                  + str(np.sum(np.abs(S_M_0 - S_M_true)))
                  + " Failure rate: " + str(np.mean(sampler.y < 0)))


s0 = 2

sampler.run(n_samples=n_samples, s0=s0, callback=plot_callback)
