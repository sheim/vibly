import numpy as np

import matplotlib
# matplotlib.use("TkAgg")

import matplotlib.colors as colors
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

font = {"size": 8}
matplotlib.rc("font", **font)

# import seaborn as sns
# sns.set_style('dark', {'axes.grid': False})
# sns.set_style("whitegrid", {'axes.grid' : False})
# sns.set_context("paper", font_scale=1.5)

# * set up helper functions


def set_size(w, h, ax=None):
    """
    w, h: width, height in inches
    https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 1.0.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.0 and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


# ground_truth_color
ground_truth_color = "gold"


def value_function(
    ax,
    X_value,
    grids,
    XV=None,
    cont_lvls=10,
    mynorm=None,
    mymap=None,
    viability_threshold=None,
):
    if mynorm is None:
        mynorm = colors.CenteredNorm()
    if mymap is None:
        mymap = plt.get_cmap("bwr_r")

    extent = [
        grids["states"][1][0],
        grids["states"][1][-1],
        grids["states"][0][0],
        grids["states"][0][-1],
    ]

    # * Value Function

    pc = ax.imshow(
        X_value,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="none",
        norm=mynorm,
        cmap=mymap,
    )
    if viability_threshold:
        cont_vibl = ax.contour(
            grids["states"][1],
            grids["states"][0],
            X_value,
            levels=[
                viability_threshold,
            ],
            colors="k",
        )
    cont1 = ax.contour(
        grids["states"][1], grids["states"][0], X_value, levels=cont_lvls, colors="k"
    )
    if XV is not None:
        ax.contour(
            grids["states"][1],
            grids["states"][0],
            XV.astype(float),
            colors=ground_truth_color,
            levels=[
                0.0,
            ],
            extend="both",
            linewidths=2.5,
            alpha=0.75,
        )

    return pc


def reward_function(
    ax, RX_value, grids, XV=None, mynorm=None, mymap=None, viability_threshold=None
):
    if mynorm is None:
        mynorm = colors.CenteredNorm()
    if mymap is None:
        mymap = plt.get_cmap("bwr_r")

    extent = [
        grids["states"][1][0],
        grids["states"][1][-1],
        grids["states"][0][0],
        grids["states"][0][-1],
    ]

    pc = ax.imshow(
        RX_value,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="none",
        norm=mynorm,
        cmap=mymap,
    )

    if XV is not None:
        cs0 = ax.contourf(
            grids["states"][1],
            grids["states"][0],
            XV,
            levels=2,
            hatches=["/", None],
            colors="none",
            extend="both",
            alpha=0.0,
        )
        ax.contour(
            grids["states"][1],
            grids["states"][0],
            XV.astype(float),
            colors=ground_truth_color,
            levels=[
                0.0,
            ],
            extend="both",
            linewidths=2.5,
            alpha=0.75,
        )
        # ax.contour(grids['states'][1], grids['states'][0], XV.astype(float),
        #             colors=ground_truth_color, levels105=[0.,], extend='both')

    for c in cs0.collections:
        c.set_edgecolor("face")

    # plt.setp(ax.get_xticklabels(), visible=False)

    return pc


def get_vmap(center=8):
    sides = int((256 - center) / 2)
    # mymap = np.vstack((plt.cm.Reds_r(np.linspace(0.2, 0.8, 120)),
    #                plt.cm.Greys(np.linspace(0., 0., 16)),
    #                plt.cm.Blues(np.linspace(0.2, 0.8, 120))))
    mymap = np.vstack(
        (
            plt.cm.Blues_r(np.linspace(0.2, 0.7, sides)),
            plt.cm.Greys(np.linspace(0.0, 0.0, center)),
            plt.cm.Oranges(np.linspace(0.3, 0.8, sides)),
        )
    )
    mymap = colors.LinearSegmentedColormap.from_list("my_colormap", mymap)

    return mymap
