dynamics_model_path = './data/dynamics/'
gp_model_path = './data/gp_model/'
results_path = './results/'


# import demos.measure_learning.hovership_default as experiment
import demos.measure_learning.hovership_unviable_start as experiment #<--- This is described in the paper
# import demos.measure_learning.slip_cautious as experiment
# import demos.measure_learning.slip_default as experiment  #<--- This is described in the paper
# import demos.measure_learning.slip_optimistic as experiment
# import demos.measure_learning.high_dimension as experiment

experiment.run_demo(dynamics_model_path=dynamics_model_path,
                                  gp_model_path=gp_model_path,
                                  results_path=results_path)
