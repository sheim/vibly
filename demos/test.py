from slippy.slip import *
import numpy as np
import matplotlib.pyplot as plt

p = {'mass':80.0, 'stiffness':8200.0, 'resting_length':1.0, 'gravity':9.81,
    'angle_of_attack':1/5*np.pi, 'damping':0.0, 
    'open_loop_time_force_series':[]}

x0 = np.array([0, 0.85, 5.5, 0, 0, 0])
x0 = reset_leg(x0, p)
p['total_energy'] = compute_total_energy(x0, p)
(p_limit_cycle, limit_cycle_found) = limit_cycle(x0,p,'angle_of_attack',np.pi*0.25)

x0 = reset_leg(x0, p_limit_cycle)
p['total_energy'] = compute_total_energy(x0, p_limit_cycle)
sol = step(x0, p_limit_cycle)

state_labels = [' x_com',' y_com','vx_com','vy_com','x_foot','y_foot']

print('Limit Cycle State Error: x(start) - x(end)')
print('  Note: state erros in x_com and x_foot will not go to zero')
for i in range(0,6):
  print('  '+state_labels[i]+'  '+str(sol.y[i,0]-sol.y[i,-1]))

plt.plot(sol.y[0], sol.y[1], color='orange')
plt.show()
