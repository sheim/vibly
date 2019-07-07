

dynamics_model_path = './data/dynamics/'
gp_model_path = './data/gp_model/'
results_path = './results/'


# import demos.measure_learning.hovership as hovership_experiment
#
# hovership_experiment.run_demo(dynamics_model_path=dynamics_model_path,
#                               gp_model_path=gp_model_path,
#                               results_path=results_path)
#
#
# import demos.measure_learning.hovership_unviable_start as hovership_unviable_start_experiment
#
# hovership_unviable_start_experiment.run_demo(dynamics_model_path=dynamics_model_path,
#                                              gp_model_path=gp_model_path,
#                                              results_path=results_path)
#
# import demos.measure_learning.slip_cautious as slip_cautious_experiment
#
# slip_cautious_experiment.run_demo(dynamics_model_path=dynamics_model_path,
#                                   gp_model_path=gp_model_path,
#                                   results_path=results_path)

import demos.measure_learning.slip as slip_experiment

slip_experiment.run_demo(dynamics_model_path=dynamics_model_path,
                         gp_model_path=gp_model_path,
                         results_path=results_path)

# import demos.measure_learning.slip_optimistic as slip_optimistic_experiment
#
# slip_optimistic_experiment.run_demo(dynamics_model_path=dynamics_model_path,
#                                   gp_model_path=gp_model_path,
#                                   results_path=results_path)
