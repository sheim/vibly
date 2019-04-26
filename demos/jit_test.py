from slip import *
import numpy as np
import matplotlib.pyplot as plt
import timeit

p = {'mass':80.0, 'stiffness':8200.0, 'resting_length':1.0, 'gravity':9.81,
    'angle_of_attack':1/5*np.pi}
x0 = np.array([0, 0.85, 5.5, 0, 0, 0])
p['total_energy'] = compute_total_energy(x0, p)
x0 = reset_leg(x0, p)
sol = step(x0, p)

print(timeit.timeit('lambda x0, p: step', globals=globals()),number=10000)

#################