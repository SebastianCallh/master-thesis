"""This module contains useful plotting functionality.
"""
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from numpy import ndarray
from scipy.stats import norm


COLOR_MAP_NAME = 'tab20'
COLOR_MAP = plt.cm.get_cmap(COLOR_MAP_NAME)


def default_color(i):
    cols = COLOR_MAP.colors
    return cols[i % len(cols)]


def grid_for(
        mus: ndarray,
        sigmas: ndarray,
        grid_res=50,
        grid_pad=3) -> ndarray:
    # Create grid that covers all components
    asc_preds = sorted(zip(mus, sigmas))
    smallest_pred_mean = float(asc_preds[0][0])
    smallest_pred_var = float(asc_preds[0][1])
    xmin = max(0, np.floor(smallest_pred_mean - smallest_pred_var * grid_pad))
    biggest_pred_mean = float(asc_preds[-1][0])
    biggest_pred_var = float(asc_preds[-1][1])
    xmax = np.ceil(biggest_pred_mean + biggest_pred_var * grid_pad)

    return np.linspace(xmin, xmax, (xmax - xmin) * grid_res)


def plot_grid(n_rows, n_cols):
    fig_size = 8
    return plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(
            fig_size * n_cols,
            fig_size * n_rows
        )
    )


def plot_data(ax, grid, data):
    ax.scatter(
        grid, data,
        marker='x',
        color='#202020',
        label='Observations'
    )


def plot_marker(ax, x, label, color='red'):
    ax.plot(
        np.array([x]),
        np.array([0]),
        '^', color=color,
        clip_on=False,
        label=label
    )


# def plot_prediction_distribution

def plot_mixture(
        ax, grid,
        component_ixs,
        component_mus,
        component_sigmas,
        component_weights,
        label, linestyle,
        color):
    seq = zip(
        component_ixs,
        component_mus,
        component_sigmas,
        component_weights
    )

    for i, mu, sigma, w in seq:
        ax.plot(
            grid,
            w * norm.pdf(grid, mu, sigma),
            label=label(i),
            color=color(i),
            linestyle=linestyle(i)
        )




    # prediction_dist = norm.pdf(t_grid, t_mu, np.sqrt(t_sigma))
    # prediction = np.sum(prediction_dist*t_grid) / np.sum(prediction_dist)

    # predictive_dist_ax.set_title(r'$\mu_t$ Prediction Distribution')
    # predictive_dist_ax.set_xlabel('Seconds')
    # predictive_dist_ax.set_ylabel('Density')
    # plot_marker(predictive_dist_ax, prediction, 'Predicted arrival time', color='blue')
    # plot_marker(predictive_dist_ax, time_left, 'Arrival time')
    # plot_mixture(
    #     predictive_dist_ax, t_grid,
    #     [most_probable_model_index],
    #     [t_mu], [t_sigma],
    #     [1], label=label,
    #     color=default_color,
    #     linestyle=linestyle
    # )
    # predictive_dist_ax.legend()






