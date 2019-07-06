import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm

matplotlib.rcParams['figure.figsize'] = 5.5, 7

font = {'size'   : 8}

matplotlib.rc('font', **font)

# yellow = [255, 225, 25]
# blue = [0, 130, 200]  # Viable
# lavender = [230, 190, 255]  # optimistic
# maroon = [128, 0, 0]  # cautious
# navy = [0, 0, 128]
# grey = [128, 128, 128]  # unviable, not yet failed
# red = [230, 25, 75]  # failed
# green = [60, 180, 75]
# lime = [210, 245, 60]
# mint = [170, 255, 195]
# red = [230, 25, 75]

# brewer2
dark_blue = [31,120,180]
light_blue = [166,206,227]
green = [178,223,138]

light_orange = [253,205,172]
light_purple = [203,213,232]

optimistic_color = light_blue
explore_color = dark_blue
truth_color = green

failure_color = light_orange
unviable_color = light_purple

def create_set_colormap():

    colors = np.array([
        unviable_color,  # Q_V_true      0 - 0
        failure_color,  # failure 1 - 1
        truth_color,  # Q_V_true      1 - 2
        optimistic_color,  # Q_V_safe       1 - 3
        explore_color   # Q_V_explore      1 - 4
    ])/256
    return ListedColormap(colors)


def frame_image(img, frame_width):
    b = frame_width  # border size in pixel
    # resolution / number of pixels in x and y
    ny, nx = img.shape[0], img.shape[1]
    if img.ndim == 3:  # rgb or rgba array
        framed_img = np.zeros((b+ny+b, b+nx+b, img.shape[2]))
    elif img.ndim == 2:  # grayscale image
        framed_img = np.zeros((b+ny+b, b+nx+b))
    framed_img[b:-b, b:-b] = img
    return framed_img


def plot_Q_S(Q_V_true, Q_V_explore, Q_V_safe, S_M_0, S_M_true, grids,
             samples=None, failed_samples=None, S_labels=[], Q_F=None):
    # TODO change S_true, simply have S as a tuple of Ss, and add names
    extent = [grids['actions'][0][0],
              grids['actions'][0][-1],
              grids['states'][0][0],
              grids['states'][0][-1]]

    fig = plt.figure(constrained_layout=True, figsize=(5.5/2, 2.4))
    gs = fig.add_gridspec(1, 2,  width_ratios=[3, 1])
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.01, hspace=0.01)

    ax_Q = fig.add_subplot(gs[0, 0])
    ax_S = fig.add_subplot(gs[0, 1], sharey=ax_Q)
    ax_Q.tick_params(direction='in', top=True, right=True)
    ax_S.tick_params(direction='in', left=False)

    # ax_S.plot(S, grids['states'][0])
    # ax_S.set_ylim(ax_Q.get_ylim())
    # s_max = np.max(S)

    # if S_true is not None:
    # ax_S.plot(S_true, grids['states'][0])
    # s_max = np.max([s_max, np.max(S_true)])
    # ax_S.legend(['safe estimate', 'ground truth'])
    # print(s_max)
    # ax_S.set_xlim((0, s_max + 0.1))

    ax_S.plot(S_M_0, grids['states'][0],
              color=tuple(c/256 for c in optimistic_color))
    ax_S.plot(S_M_true, grids['states'][0],
              color=tuple(c/256 for c in truth_color))

    # if S_labels:
    #     ax_S.legend(S_labels, loc=2, ncol=1)

    ax_S.set_xlim((0, max(S_M_true)*1.2))
    ax_S.get_yaxis().set_visible(False)
    ax_S.set_xlabel('$\Lambda$')
    # ax_S.set_ylabel('state space: height at apex')
    aspect_ratio_Q = 'auto'  # 1.5
    # aspect_ratio_S = ax_Q.get_xlim() / s_max
    # ax_S.set_aspect(aspect_ratio_S)
    if samples is not None and samples[0] is not None:
        action = samples[0][:, 0]
        state = samples[0][:, 1]
        if failed_samples is not None and len(failed_samples) > 0:
            ax_Q.scatter(action[failed_samples], state[failed_samples],
                         marker='x', edgecolors=[[0.9, 0.3, 0.3]], s=30,
                         facecolors=[[0.9, 0.3, 0.3]])
            failed_samples = np.logical_not(failed_samples)
            ax_Q.scatter(action[failed_samples], state[failed_samples],
                         edgecolors=[[0.9, 0.3, 0.3]], s=30,
                         marker='o', facecolors='none')
        else:
            ax_Q.scatter(action, state,
                         edgecolors=[[0.9, 0.3, 0.3]], s=30,
                         marker='o', facecolors='none')

    X, Y = np.meshgrid(grids['actions'][0], grids['states'][0])
    ax_Q.contour(X, Y, Q_V_true, [.5], colors='k')
    # ax_Q.contour(X, Y, Q_V_explore, [.5], colors='k')
    # ax_Q.contour(X, Y, Q_V_safe, [.5], colors='k')

    # Build image from sets

    img = np.zeros(Q_V_true.shape)
    img[Q_V_true == 1] = 0.5
    if Q_F is not None:
        img[Q_F] = 1.5
    img[Q_V_true == 1] = 2.5
    img[Q_V_safe == 1] = 3.5
    img[Q_V_explore == 1] = 4.5

    # img = frame_image(img, 10)

    cmap = create_set_colormap()
    bounds = [0, 1, 2, 3, 4,5]
    norm = BoundaryNorm(bounds, cmap.N)

    # this needs to happen after the scatter plot
    ax_Q.imshow(img, origin='lower', extent=extent, aspect=aspect_ratio_Q,
                interpolation='none', cmap=cmap, norm=norm)
    ax_Q.set_xlabel('action space $A$')
    ax_Q.set_ylabel('state space $S$')

    extent = [grids['actions'][0][0],
              grids['actions'][0][-1],
              grids['states'][0][0],
              grids['states'][0][-1]]

    frame_width_x = grids['actions'][0][-1]*.03
    ax_Q.set_xlim((grids['actions'][0][0] - frame_width_x,
                   grids['actions'][0][-1] + frame_width_x))

    frame_width_y = grids['states'][0][-1]*.03
    ax_Q.set_ylim((grids['states'][0][0] - frame_width_y,
                   grids['states'][0][-1] + frame_width_y))

    fig.subplots_adjust(0, 0, 1, 1)
    return fig
