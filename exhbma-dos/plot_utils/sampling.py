import numpy as np
from matplotlib import pyplot as plt


def plot_sampling_sequence(
    seq,
    ax=None,
    figsize=(12, 8),
    color="black",
    linewidth=1.0,
    xlabel=None,
    ylabel=None,
    label_size=14,
    xlim=None,
    ylim=None,
    **plot_kwargs,
):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    n_samples = len(seq)
    ax.plot(
        np.arange(n_samples),
        seq,
        color=color,
        linewidth=linewidth,
        **plot_kwargs,
    )

    ax.set_xlim([0, n_samples])
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_size)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax


def plot_sampling_autocorrelation(
    seq,
    ax=None,
    figsize=(12, 8),
    color="black",
    linewidth=1.0,
    xlabel=None,
    ylabel=None,
    label_size=14,
    xlim=None,
    ylim=None,
    **plot_kwargs,
):
    np_seq = np.array(seq)
    seq_bar = np_seq - np_seq.mean()
    full_cor = np.correlate(seq_bar, seq_bar, mode="full")
    cor = full_cor[len(full_cor) // 2 :] / np.arange(len(seq), 0, -1)
    cor = cor / cor[0]

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    ax.plot(
        np.arange(len(cor)),
        cor,
        color=color,
        linewidth=linewidth,
        **plot_kwargs,
    )

    ax.set_yscale("log")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_size)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax


def plot_two_parameter_trajectory(
    seq1,
    seq2,
    ax=None,
    figsize=(12, 8),
    color="black",
    alpha=0.5,
    linewidth=1.0,
    marker="x",
    markersize=2.0,
    xlabel=None,
    ylabel=None,
    label_size=18,
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None,
    title=None,
    title_size=24,
    **plot_kwargs,
):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    ax.plot(
        seq1,
        seq2,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
        **plot_kwargs,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_size)

    if title is not None:
        ax.set_title(title, fontsize=title_size)

    return ax
