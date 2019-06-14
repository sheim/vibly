import numpy as np
import matplotlib.pyplot as plt
import slippy.ardyn as model
import slippy.viability as vibly

# define nonlinear dynamics

def f_nl(x, p):
    x += ( # TODO use a list of coeffs and funcs, and second coeffs
          p['a1']*np.sin(x) +
          p['a2']*np.cos(x) +
          p['a3']*np.cos(x)*np.sin(x) +
          p['a4']*np.power(x, p['b4']) +
          p['ctrl_mat']*p['actions']
    )
    return x
# define parameters

n_states = 1
m_actions = 1
np.random.seed(2)
a1 = 5*(np.random.random()-0.5)
a2 = 5*(np.random.random()-0.5)
a3 = 5*(np.random.random()-0.5)
a4 = 5*(np.random.random()-0.5)
b4 = np.random.randint(1, 5)
p = {'a1':a1,
    'a2':a2,
    'a3':a3,
    'a4':a4,
    'b4':b4,
    'ctrl_mat': np.eye(m_actions),
    'nonlinear':f_nl,
    'fail_bound': 5,
    'n_states': n_states,
    'm_actions': m_actions,
    'actions': np.zeros(m_actions)}

x0 = np.zeros([n_states,1])

p_map = model.p_map
p_map.p = p
p_map.x = x0
p_map.sa2xp = model.sa2xp
p_map.xp2s = model.xp2s

s_grid = np.linspace(-6, 6, 50)

a_bound_upp = np.array([1])
a_bound_low = - a_bound_upp
a_grid = np.linspace(a_bound_low, a_bound_upp, 51)
grids = {'states':(s_grid,), 'actions':(a_grid,)}
Q_map, Q_F = vibly.compute_Q_map(grids, p_map)

# save file
# data2save = {"s_grid": s_grid, "a_grid": a_grid, "Q_map": Q_map, "Q_F": Q_F,
#             "p_map":p_map, "p":p}
# np.savez('test.npz', **data2save)

plt.imshow(Q_map, origin = 'lower')
plt.show()