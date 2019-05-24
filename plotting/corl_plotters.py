import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_Q_S(Q, S, grids):
    # grid coordinates for Q
    real_x=grids['actions'][0]
    real_y=grids['states'][0]
    dx = (real_x[1]-real_x[0])/2.
    dy = (real_y[1]-real_y[0])/2.
    extent = [real_x[0]-dx, real_x[-1]+dx, real_y[0]-dy, real_y[-1]+dy]

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(1, 2)

    ax_Q = fig.add_subplot(gs[0, 0])
    ax_S = fig.add_subplot(gs[0, 1])
    ax_Q.tick_params(direction='in', top=True, right=True)
    ax_S.tick_params(direction='in', labelleft=False)

    ax_Q.imshow(Q, origin='lower', extent=extent, aspect="auto")

    ax_S.plot(S, real_y)
    ax_S.set_ylim(ax_Q.get_ylim())

    return fig