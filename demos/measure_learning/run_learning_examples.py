import os

if os.path.exists("data"):
    # if we are in the vibly root folder:
    path_to_data = "data/"
else:
    # else we assume this is being run from the /demos/measure_learning folder.
    path_to_data = "../../data/"

dynamics_model_path = path_to_data + "dynamics/"
gp_model_path = path_to_data + "gp_model/"
results_path = path_to_data + "results/"


# import demos.measure_learning.hovership_default as experiment
# import demos.measure_learning.hovership_unviable_start as experiment #<--- This is described in the paper
import demos.measure_learning.slip_cautious as experiment
# import demos.measure_learning.slip_default as experiment  #<--- This is described in the paper
# import demos.measure_learning.slip_optimistic as experiment
# import demos.measure_learning.slip_prior as experiment

try:
    experiment.run_demo(
        dynamics_model_path=dynamics_model_path,
        gp_model_path=gp_model_path,
        results_path=results_path,
    )
except FileNotFoundError:
    print(
        "ERROR: Ground-truth data for comparison not available. Please generate the data with `/demos/computeQ_hovership.py` or `/demos/computeQ_slip.py`, as appropriate for the experiment chosen. This should generate a `[model]_map.pickle` file in `vibly/data/dynamics/`"
    )
