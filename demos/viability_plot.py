import numpy as np
import matplotlib.pyplot as plt
from plotting.visualise_viability import visualise

data               = np.load('..\data\low_res_slip.npz')
initial_conditions = np.array([0, 0.85, 5.5, 0, 0, 0]) 

visualise(data)
visualise(data, initial_conditions = initial_conditions)
visualise(data, initial_conditions = initial_conditions, include_end_state = True)