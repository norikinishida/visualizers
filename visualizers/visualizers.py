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

LEGEND_LOCATIONS = [
        "upper left",
        "upper right",
        "lower left",
        "lower right",
        "center left",
        "center right",
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
        list_ys, list_xs,
        xticks, xlabel, ylabel,
        legend_names, legend_anchor, legend_location,
        marker="o", linestyle="-", markersize=10,
        fontsize=30,
        savepath=None, figsize=(8,6), dpi=100):
    """
    :type list_ys: list of list of float
    :type list_xs: list of list of float
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
    if list_xs is not None:
        assert len(list_xs) == len(legend_names)
        assert xticks is None
    if xticks is not None:
        assert list_xs is None
    assert legend_location in LEGEND_LOCATIONS

    # Preparation
    _prepare_matplotlib()

    plt.figure(figsize=figsize, dpi=dpi)
    if list_xs is None:
        for ys, legend_name in zip(list_ys, legend_names):
            plt.plot(ys,
                     marker=marker, linestyle=linestyle, ms=markersize,
                     label=r"%s" % legend_name)
    else:
        for xs, ys, legend_name in zip(list_xs, list_ys, legend_names):
            plt.plot(xs, ys,
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
    plt.close()

def errorbar(
        list_ys, list_es, list_xs,
        xticks, xlabel, ylabel,
        legend_names, legend_anchor, legend_location,
        marker="o", linestyle="-", markersize=10,
        capsize=4.0, capthick=2.0,
        fontsize=30,
        savepath=None, figsize=(8,6), dpi=100):
    """
    :type list_ys: list of list of float
    :type list_es: list of list of float
    :type list_xs: list of list of float
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
    assert len(list_ys) == len(list_es) == len(legend_names)
    if list_xs is not None:
        assert len(list_xs) == len(legend_names)
        assert xticks is None
    if xticks is not None:
        assert list_xs is None
    assert legend_location in LEGEND_LOCATIONS

    # Preparation
    _prepare_matplotlib()

    plt.figure(figsize=figsize, dpi=dpi)
    if list_xs is None:
        for ys, es, legend_name in zip(list_ys, list_es, legend_names):
            plt.errorbar([i for i in range(len(ys))],
                         ys, es,
                         marker=marker, linestyle=linestyle, ms=markersize,
                         capsize=capsize, capthick=capthick,
                         label=r"%s" % legend_name)
    else:
        for xs, ys, es, legend_name in zip(list_xs, list_ys, list_es, legend_names):
            plt.errorbar(xs,
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
    plt.close()

def scatter(
        vectors,
        categories, category_name, category_order,
        category_centers, category_covariances,
        xlabel, ylabel,
        fontsize=30,
        savepath=None, figsize=(8,6), dpi=100):
    """
    :type vectors: numpy.ndarray(shape=(N,2), dtype=float)
    :type categories: numpy.ndarray(shape=(N,), dtype=str)
    :type category_name: str
    :type category_order: list of str
    :type category_centers: numpy.ndarray(shape=(n_categories, 2), dtype=float)
    :type category_covariances: numpy.ndarray(shape=(n_categories, 2, 2), dtype=float)
    :type xlabel: str
    :type ylabel: str
    :type fontsize: int
    :type savepath: str
    :type figsize: (int, int)
    :type dpi: int
    :rtype: None
    """
    assert len(vectors.shape) == 2
    assert len(categories.shape) == 1
    assert vectors.shape[0] == categories.shape[0]
    assert vectors.shape[1] == 2

    # Preparation
    _prepare_matplotlib()

    dictionary = {"x": [], "y": [], category_name: []}
    dictionary["x"] = vectors[:,0]
    dictionary["y"] = vectors[:,1]
    dictionary[category_name] = categories
    df = pd.DataFrame(dictionary)

    # Visualization
    plt.figure(figsize=figsize, dpi=dpi)
    ax = sns.scatterplot(data=df, x="x", y="y",
                         hue=category_name, hue_order=category_order)
    if category_centers is not None:
        for c_i in range(len(category_order)):
            mean = category_centers[c_i]
            ax.scatter(x=mean[0], y=mean[1], s=300,
                       marker="*", linewidth=2,
                       c="yellow", edgecolors="orange")
    if category_covariances is not None:
        palette = sns.color_palette()
        for c_i in range(len(category_order)):
            mean = category_centers[c_i]
            cov = category_covariances[c_i]
            v, w = np.linalg.eigh(cov)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi # convert to degrees
            ell = matplotlib.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=palette[c_i])
            ell.set_alpha(0.5)
            ax.add_artist(ell)
    plt.tight_layout()
    plt.xlabel(r"%s" % xlabel, fontsize=fontsize)
    plt.ylabel(r"%s" % ylabel, fontsize=fontsize)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches="tight")
        print("Saved a figure to %s" % savepath)
    plt.close()

def bar(list_ys,
        xticks, xlabel, ylabel,
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
    assert legend_location in LEGEND_LOCATIONS

    # Preparation
    _prepare_matplotlib()

    # Convert lists to pandas.DataFrame
    dictionary = {"x": [], "y": [], "hue": []}
    dictionary["y"] = [y for ys in list_ys for y in ys]
    for _ in list_ys:
        dictionary["x"].extend(xticks)
    for i in range(len(list_ys)):
        dictionary["hue"].extend([legend_names[i]] * len(list_ys[i]))
    df = pd.DataFrame(dictionary)

    # Visualization
    plt.figure(figsize=figsize, dpi=dpi)
    sns.barplot(data=df, x="x", y="y", hue="hue")
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
    plt.close()

def heatmap(
        matrix,
        xticks, yticks, xlabel, ylabel,
        vmin=None, vmax=None,
        annotate_counts=True, show_colorbar=True, colormap="Blues",
        linewidths=0.5, fmt=".2g",
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
    plt.close()


