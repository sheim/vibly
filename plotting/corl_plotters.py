import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_Q_S(Q, S, grids, samples = None, failed_samples = None, S_true = None):
    # TODO change S_true, simply have S as a tuple of Ss, and add names
    extent = [grids['actions'][0][0],
              grids['actions'][0][-1],
              grids['states'][0][0],
              grids['states'][0][-1]]

    fig = plt.figure(constrained_layout=True, figsize=(10, 5))
    gs = fig.add_gridspec(1, 2)

    ax_Q = fig.add_subplot(gs[0, 0])
    ax_S = fig.add_subplot(gs[0, 1])
    ax_Q.tick_params(direction='in', top=True, right=True)
    ax_S.tick_params(direction='in', left=False)

    ax_S.plot(S, grids['states'][0])
    ax_S.set_ylim(ax_Q.get_ylim())
    s_max = np.max(S)
    if S_true is not None:
        ax_S.plot(S_true, grids['states'][0])
        s_max = np.max([s_max, np.max(S_true)])
        ax_S.legend(['safe estimate', 'ground truth'])
    # print(s_max)
    # ax_S.set_xlim((0, s_max + 0.1))

    ax_S.set_xlim(ax_Q.get_xlim())
    ax_S.set_xlabel('safe actions available')
    # ax_S.set_ylabel('state space: height at apex')
    aspect_ratio_Q = 'auto'# 1.5
    # aspect_ratio_S = ax_Q.get_xlim() / s_max
    # ax_S.set_aspect(aspect_ratio_S)
    if samples is not None:
        action = samples[0][:,0]
        state = samples[0][:,1]
        if failed_samples is not None:
            ax_Q.scatter(action[failed_samples], state[failed_samples],
                        marker='x', edgecolors=[[0.9,0.3,0.3]], s=30,
                        facecolors=[[0.9,0.3,0.3]])
            failed_samples = np.logical_not(failed_samples)
            ax_Q.scatter(action[failed_samples], state[failed_samples],
                        edgecolors=[[0.9,0.3,0.3]], s=30,
                        marker='o', facecolors='none')
        else:
            ax_Q.scatter(action, state,
                        edgecolors=[[0.9,0.3,0.3]], s=30,
                        marker='o', facecolors='none')

    # this needs to happen after the scatter plot
    ax_Q.imshow(Q, origin='lower',extent=extent, aspect=aspect_ratio_Q,
            interpolation='gaussian')
    ax_Q.set_xlabel('action space: angle of attack')
    ax_Q.set_ylabel('state space: height at apex')

    return fig