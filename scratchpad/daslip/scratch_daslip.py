import numpy as np
import matplotlib.pyplot as plt

from ttictoc import TicToc

import models.daslip as model
import viability as vibly

# * First, solve for the operating point to get an open-loop force traj
# Model parameters for both slip/daslip. Parameters only used by daslip are *
p = {'mass': 80,                          # kg
     'stiffness': 8200.0,                 # K : N/m
     'spring_resting_length': 0.9,        # m
     'gravity': 9.81,                     # N/kg
     'angle_of_attack': 1/5*np.pi,        # rad
     'actuator_resting_length': 0.1,      # m
     'actuator_force': [],                # * 2 x M matrix of time and force
     'actuator_force_period': 10,         # * s
     'activation_amplification': 1,
     'activation_delay': 0.0,  # * a delay for when to start activation
     'constant_normalized_damping': 0.75,          # *    s : D/K : [N/m/s]/[N/m]
     'linear_normalized_damping_coefficient': 3.5,  # * A: s/m : D/F : [N/m/s]/N : 0.0035 N/mm/s -> 3.5 1/m/s from Kirch et al. Fig 12
     'linear_minimum_normalized_damping': 0.05,    # *   1/A*(kg*N/kg) :
     'swing_leg_norm_angular_velocity':  0,  # [1/s]/[m/s] (omega/(vx/lr))
     'swing_velocity': 0,   # rad/s (set by calculation)
     'angle_of_attack_offset': 0}        # rad   (set by calculation)
# * linear_normalized_damping_coefficient:
# * A: s/m : D/F : [N/m/s]/N : 0.0035 N/mm/s -> 3.5 1/m/s (Kirch et al. Fig 12)

x0 = np.array([0, 1.00, 5.5, 0, 0, 0, p['actuator_resting_length'], 0, 0, 0])
x0 = model.reset_leg(x0, p)
p['total_energy'] = model.compute_total_energy(x0, p)
x0, p = model.create_open_loop_trajectories(x0, p)
p['x0'] = x0
p['activation_amplification'] = 1.5
# initialize default x0_daslip

p_map = model.poincare_map
p_map.p = p
p_map.x = x0
p_map.sa2xp = model.sa2xp_y_xdot_timedaoa
# p_map.sa2xp = model.sa2xp_y_xdot_aoa
p_map.xp2s = model.xp2s_y_xdot

s_grid_height = np.linspace(0.5, 1.5, 11)
s_grid_velocity = np.linspace(3, 8, 11)
s_grid = (s_grid_height, s_grid_velocity)
a_grid_aoa = np.linspace(00/180*np.pi, 70/180*np.pi, 71)
# a_grid = (a_grid_aoa, )
a_grid_amp = np.linspace(0.9, 1.2, 31)
a_grid = (a_grid_aoa, a_grid_amp)

grids = {'states': s_grid, 'actions': a_grid}
t = TicToc()
t.tic()
Q_map, Q_F, Q_reach = vibly.compute_Q_map(grids, p_map, keep_coords=True,
                                          verbose=2)
t.toc()
print("time elapsed: " + str(t.elapsed/60))
Q_V, S_V = vibly.compute_QV(Q_map, grids)
S_M = vibly.project_Q2S(Q_V, grids, proj_opt=np.mean)
Q_M = vibly.map_S2Q(Q_map, S_M, s_grid, Q_V=Q_V)
# plt.scatter(Q_map[1], Q_map[0])
print("non-failing portion of Q: " + str(np.sum(~Q_F)/Q_F.size))
print("viable portion of Q: " + str(np.sum(Q_V)/Q_V.size))

import itertools as it
# Q0 = np.zeros((len(grids['states']), total_gridpoints))
# def create_x0(grids):
#     for idx, state_action in enumerate(np.array(list(
#             it.product(*grids['states'], *grids['actions'])))):


def color_generator(n=1):
    # colors = list()
    colors = np.zeros((n, 3))
    for n in range(n):
        colors[n, :] = np.array([np.random.randint(0, 255),
                                 np.random.randint(0, 255),
                                 np.random.randint(0, 255)])/256
    return colors


Q0 = np.array(list(it.product(*grids['states'], *grids['actions']))).T
R_map = Q_reach[:, ~Q_F.flatten()]
R0 = Q0[:, ~Q_F.flatten()]

# for adx, a in enumerate(a_grid[0]):
#     idx = np.where(R0[2] == a)
#     # for i in range(0, R0.shape[1]):
#     plt.figure(adx)
#     for i in idx[0]:
#         # if i > 0:
#         #     if not np.allclose(R0[0:2, i], R0[0:2, i-1]):
#         #         c = color_generator()
#         #         # print(i)
#         cdx = np.where(np.equal(a_grid[0], R0[2, i]))
#         col = tuple(c[cdx].squeeze())
#         plt.plot([R_map[1, i], R0[1, i]], [R_map[0, i], R0[0, i]], color=col, alpha=0.5)
#         plt.scatter(R_map[1, i], R_map[0, i], color=col)
# plt.show()

# for adx, a in enumerate(a_grid[0]):
#     idx = np.where(R0[2] == a)
R_diff = R_map - R0[0:2, :]
R_dist = np.linalg.norm(R_diff, axis=0)
max_dist = np.max(R_dist)
min_dist = np.min(R_dist)
max_dist = 0.8
if False:
    extent = [grids['states'][1][0],
            grids['states'][1][-1],
            grids['states'][0][0],
            grids['states'][0][-1]]

    S_F = np.mean(~Q_F, axis=2)
    plt.imshow(S_F, origin='lower', extent=extent, alpha=0.5)

    for i in range(0, R0.shape[1], 4):
        # plt.figure(adx)
        # for i in idx[0]:
            # if i > 0:
            #     if not np.allclose(R0[0:2, i], R0[0:2, i-1]):
            #         c = color_generator()
            #         # print(i)
        # cdx = np.where(np.equal(a_grid[0], R0[2, i]))
        # col = tuple(c[cdx].squeeze())
        if R_dist[i] < max_dist:
            col = np.array([1, 0, 0])*((max_dist-R_dist[i])/(max_dist))**2
            # col += np.array([0.8, 0, 0.0])*(max_dist - R_dist[i])/(max_dist-min_dist)
            al = (max_dist-R_dist[i])/max_dist
            plt.plot([R_map[1, i], R0[1, i]], [R_map[0, i], R0[0, i]], color=col,
                    alpha=al)
            plt.scatter(R_map[1, i], R_map[0, i], color=col)
    plt.axis('equal')
    plt.show()


# Q_V, S_V = vibly.compute_QV(Q_map, grids)
# S_M = vibly.project_Q2S(Q_V, grids, proj_opt=np.mean)
# Q_M = vibly.map_S2Q(Q_map, S_M, s_grid=s_grid, Q_V=Q_V)
# print("non-failing portion of Q: " + str(np.sum(~Q_F)/Q_F.size))
# print("viable portion of Q: " + str(np.sum(Q_V)/Q_V.size))

###############################################################################
# save data as pickle
###############################################################################
import pickle
filename = 'daslip_ha1.pickle'
data2save = {"grids": grids, "Q_map": Q_map, "Q_F": Q_F, "Q_V": Q_V,
             "Q_M": Q_M, "S_M": S_M, "p": p, "x0": x0}
outfile = open(filename, 'wb')
pickle.dump(data2save, outfile)
outfile.close()
# to load this data, do:
# infile = open(filename, 'rb')
# data = pickle.load(infile)
# infile.close()

# plt.imshow(S_V, origin='lower')
# plt.show()

# plt.imshow(S_M, origin='lower')
# plt.show()
