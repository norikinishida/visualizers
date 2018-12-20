# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd

COLORMAPS = [
        None,
        "viridis", "plasma", "inferno", "magma",
        "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
        "Y|OrBr", "Y|OrRd", "OrRd", "PuRd", "RdPu", "BuPu",
        "GnBu", "PuBu", "Y|GnBu", "PuBuGn", "BuGn", "Y|Gn",
        "binary", "gist_yarg", "gist_gray", "gray",
        "bone", "pink", "spring", "summer", "autumn", "winter",
        "cool", "Wistia", "hot", "afmhot", "gist_heat", "copper",
        "PiYg", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdY|Bu",
        "RdY|Gn", "Spectral", "coolwarm", "bwr", "seismic",
        ]


def _prepare_matplotlib():
    # Presetting for matplotlib
    rc("font", **{"family": "sans-serif",
                  "sans-serif": ["Helvetica"]})
    rc("text", usetex=True)
    matplotlib.rcParams.update({"errorbar.capsize": 2})

    # Presetting for seaborn
    sns.set()
    sns.set_style(style="darkgrid",
                  rc={"grid.linestyle": "--"})
    sns.set_context(context="paper", font_scale=1.5,
                    rc={"lines.linewidth": 4})
    sns.set_palette(palette="winter", n_colors=8, desat=1)
    sns.set(context="talk", style="darkgrid", palette="deep",
            font_scale=1.5,
            rc={"lines.linewidth": 2, "grid.linestyle": "--"})

def plot(
        data,
        xticks, xlabel, ylabels,
        legend_names, legend_anchor, legend_location,
        marker="o", linestyle="-", markersize=10,
        fontsize=30,
        savepaths=None, figsize=(8,6), dpi=100):
    """
    :type data: {str: list of list of float}
    :type xticks: list of str
    :type xlabel: str
    :type ylabels: list of str
    :type legend_names: list of str
    :type legend_anchor: (int, int)
    :type legend_location: str
    :type marker: str
    :type linestyle: str
    :type markersize: int
    :type fontsize: int
    :type savepaths: list of str
    :type figsize: (int, int)
    :type dpi: int
    :rtype: None
    """
    assert len(set(data.keys()) - set(ylabels)) == 0 # data.keys() should be equivalent to ylabels
    assert legend_location in ["upper left", "upper right", "lower left", "lower right", "center left", "center right"]
    if savepaths is not None:
        assert len(ylabels) == len(savepaths)

    # Preparation
    _prepare_matplotlib()

    if savepaths is None:
        savepaths = [None for _ in range(len(ylabels))]

    # Visualization
    for ylabel, savepath in zip(ylabels, savepaths):
        _plot(data[ylabel], xticks=xticks, xlabel=xlabel, ylabel=ylabel,
                  legend_names=legend_names,
                  legend_anchor=legend_anchor, legend_location=legend_location,
                  marker=marker, linestyle=linestyle, markersize=markersize,
                  fontsize=fontsize,
                  savepath=savepath, figsize=figsize, dpi=dpi)

def _plot(list_ys, xticks, xlabel, ylabel,
          legend_names, legend_anchor, legend_location,
          marker, linestyle, markersize,
          fontsize,
          savepath, figsize, dpi):
    """
    :type list_ys: list of list of float
    :type xticks: list of str
    :type xlabel: str
    :type ylabel: str
    :type legend_names: list of str
    :type legend_anchor: (int, int)
    :type legend_location: str
    :type marker: str
    :type linestyle: str
    :type markersize: int
    :type fontsize: int
    :type savepath: str
    :type figsize: (int, int)
    :type dpi: int
    :rtype: None
    """
    assert len(list_ys) == len(legend_names)

    plt.figure(figsize=figsize, dpi=dpi)
    for ys, legend_name in zip(list_ys, legend_names):
        plt.plot(ys,
                 marker=marker, linestyle=linestyle, ms=markersize,
                 label=r"%s" % legend_name)
    if xticks is None:
        plt.xticks(fontsize=fontsize-5)
    else:
        plt.xticks(np.arange(len(xticks)), xticks, fontsize=fontsize-5)
    plt.yticks(fontsize=fontsize-5)
    plt.xlabel(r"%s" % xlabel, fontsize=fontsize)
    plt.ylabel(r"%s" % ylabel, fontsize=fontsize)
    plt.grid(True)
    plt.legend(bbox_to_anchor=legend_anchor, loc=legend_location,
               borderaxespad=0, fontsize=fontsize-5)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches="tight")
        print("Saved a figure to %s" % savepath)
    plt.clf()

def _errorbar(list_ys, list_es, xticks, xlabel, ylabel,
          legend_names, legend_anchor, legend_location,
          marker, linestyle, markersize,
          capsize, capthick,
          fontsize,
          savepath, figsize, dpi):
    """
    :type list_ys: list of list of float
    :type list_es: list of list of float
    :type xticks: lsit of str
    :type xlabel: str
    :type ylabel: str
    :type legend_names: list of str
    :type legend_anchor: (int, int)
    :type legend_location: str
    :type marker: str
    :type linestyle: str
    :type markersize: int
    :type capsize: float,
    :type capthick: float,
    :type fontsize: int
    :type savepath: str
    :type figsize: (int, int)
    :type dpi: int
    :rtype: None
    """
    assert len(list_ys) == len(legend_names)

    plt.figure(figsize=figsize, dpi=dpi)
    for ys, es, legend_name in zip(list_ys, list_es, legend_names):
        plt.errorbar([i for i in range(len(ys))],
                     ys, es,
                     marker=marker, linestyle=linestyle, ms=markersize,
                     capsize=capsize, capthick=capthick,
                     label=r"%s" % legend_name)
    if xticks is None:
        plt.xticks(fontsize=fontsize-5)
    else:
        plt.xticks(np.arange(len(xticks)), xticks, fontsize=fontsize-5)
    plt.yticks(fontsize=fontsize-5)
    plt.xlabel(r"%s" % xlabel, fontsize=fontsize)
    plt.ylabel(r"%s" % ylabel, fontsize=fontsize)
    plt.legend(bbox_to_anchor=legend_anchor, loc=legend_location,
               borderaxespad=0, fontsize=fontsize-5)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches="tight")
        print("Saved a figure to %s" % savepath)
    plt.clf()

def bar(list_ys, xticks, xlabel, ylabel,
        legend_names, legend_anchor, legend_location,
        fontsize=30,
        savepath=None, figsize=(8,6), dpi=100):
    """
    :type list_ys: list of list of float
    :type xticks: list of str
    :type xlabel: str
    :type ylabel: str
    :type legend_names: list of str
    :type legend_anchor: (int, int)
    :type legend_location: str
    :type fontsize: int
    :type savepath: str
    :type figsize: (int, int)
    :type dpi: int
    :rtype: None
    """
    assert len(list_ys) == len(legend_names)
    for ys in list_ys:
        assert len(ys) == len(xticks)

    # Preparation
    _prepare_matplotlib()

    # Convert lists to pandas.DataFrame
    dictionary = {"X": [], "Y": [], "hue": []}
    dictionary["Y"] = [y for ys in list_ys for y in ys]
    for _ in list_ys:
        dictionary["X"].extend(xticks)
    for i in range(len(list_ys)):
        dictionary["hue"].extend([legend_names[i]] * len(list_ys[i]))
    df = pd.DataFrame(dictionary)

    # Visualization
    plt.figure(figsize=figsize, dpi=dpi)
    sns.barplot(data=df, x="X", y="Y", hue="hue")
    plt.tight_layout()
    plt.xlabel(r"%s" % xlabel, fontsize=fontsize)
    plt.ylabel(r"%s" % ylabel, fontsize=fontsize)
    plt.legend(bbox_to_anchor=legend_anchor, loc=legend_location,
               borderaxespad=0)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches="tight")
        print("Saved a figure to %s" % savepath)
    plt.clf()

def heatmap(matrix, xticks, yticks, xlabel, ylabel,
          vmin=None, vmax=None,
          annotate_counts=True, show_colorbar=True, colormap=None,
          linewidths=0, fmt=".2g",
          fontsize=30,
          savepath=None, figsize=(8,6), dpi=100):
    """
    :type matrix: list of list of float
    :type xticks: list of str
    :type yticks: list of str
    :type xlabel: str
    :type ylabel: str
    :type vmin: float
    :type vmax: float
    :type annotate_counts: bool
    :type show_colorbar: bool
    :type colormap: str
    :type linewidths: int
    :type fmt: str
    :type fontsize: int
    :type savepath: str
    :type figsize: int
    :type dpi: int
    """
    assert len(matrix) == len(yticks)
    for row in matrix:
        assert len(row) == len(xticks)
    assert colormap in COLORMAPS

    # Preparation
    _prepare_matplotlib()

    # Convert lists to pandas.DataFrame
    matrix = np.asarray(matrix)
    df = pd.DataFrame(data=matrix, index=yticks, columns=xticks)

    # Visualization
    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(df, vmin=vmin, vmax=vmax,
                annot=annotate_counts, cbar=show_colorbar, cmap=colormap,
                linewidths=linewidths, fmt=fmt)
    plt.tight_layout()
    plt.xlabel(r"%s" % xlabel, fontsize=fontsize)
    plt.ylabel(r"%s" % ylabel, fontsize=fontsize)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches="tight")
        print("Saved a figure to %s" % savepath)
    plt.clf()


