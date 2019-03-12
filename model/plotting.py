"""This module contains useful plotting functionality.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from functools import reduce
from predict import predict
from numpy import ndarray
from scipy.stats import norm
from model import TrajectoryModel, FunctionModel
from pandas import DataFrame
import pandas as pd

COLOR_MAP_NAME = 'tab10'
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
    xmin = max(0, np.floor(smallest_pred_mean - smallest_pred_var*grid_pad))
    biggest_pred_mean = float(asc_preds[-1][0])
    biggest_pred_var = float(asc_preds[-1][1])
    xmax = np.ceil(biggest_pred_mean + biggest_pred_var*grid_pad)

    return np.linspace(xmin, xmax, (xmax-xmin)*grid_res)


def plot_grid(n_rows, n_cols):
    fig_size = 8
    return plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(
            fig_size*n_cols,
            fig_size*n_rows
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


def plot_function(func: FunctionModel, ax=None) -> ():
    if ax:
        func.model.plot(ax=ax)
    else:
        func.model.plot()


def plot_model(m: TrajectoryModel, data: DataFrame, f_p_codomain, f_v_codomain):
    n_rows = 4
    n_cols = 2
    fig_size = 8
    _, axs = plot_grid(n_rows, n_cols)

    # Input data
    sns.scatterplot(
        data=data, 
        x=f_p_codomain[0], 
        y=f_p_codomain[1], 
        ax=axs[0][0]
    )
    axs[0][0].set_title('Input data')

    # h for input data
    x = data[f_p_codomain].values
    mean, _  = predict(m.g, x)
    df = pd.DataFrame({
        'prediction': mean.T[0], 
        'progress': data['tau'].values
    })
    sns.scatterplot(
        data=df, 
        x='progress', 
        y='prediction', 
        ax=axs[1][0]
    )
    axs[1][0].set_title(r'Predicted $\tau$ of input data')
    #axs[1][0].set_aspect('equal', 'datalim')

    # Training data
    plot_function(m.g, ax=axs[0][1])
    axs[0][1].set_title(r'Inverse model function $g: (lat,lon) \mapsto \tau$')
    axs[0][1].set_xlabel('x')
    axs[0][1].set_ylabel('y')
    #axs[0][1].axis('scaled')
    
    # H for training data
    mean, _  = m.g.model.predict(m.g.model.X)
    df = pd.DataFrame({
        'prediction': mean.T[0], 
        'progress': m.g.model.Y.flatten()
    })
    sns.scatterplot(
        data=df, 
        x='progress', 
        y='prediction', 
        ax=axs[1][1]
    )
    axs[1][1].set_title(r'Prediction $\tau$ of training data')

    # f_p
    plot_function(m.f_p_x, ax=axs[2][0])
    plot_function(m.f_p_y, ax=axs[2][0])
    axs[2][0].set_title(r'Model function $f_p: \tau \mapsto (p_x, p_y)$')
    
    # f_v
    plot_function(m.f_v_x, ax=axs[2][1])
    plot_function(m.f_v_y, ax=axs[2][1])
    axs[2][1].set_title(r'Model function $f_v: \tau \mapsto (v_x, v_y)$')

    # h
    plot_function(m.h, ax=axs[3][0])
    axs[3][0].set_title(r'Prediction function $h: \tau \mapsto t$')


def plot_with_credible_bands(
        ax, grid, function,
        label, color, linestyle):
    N = len(grid)
    mu, sigma = predict(function, grid.reshape(-1, 1))
    sd = np.sqrt(sigma)
    upper_env = mu + sd*2
    lower_env = mu - sd*2
    ax.plot(
        grid, mu,
        label=label,
        color=color,
        linestyle=linestyle
    )

    ax.fill_between(
        grid,
        upper_env.reshape(N),
        lower_env.reshape(N),
        color=color,
        alpha=.3
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
            w*norm.pdf(grid, mu, sigma),
            label=label(i),
            color=color(i),
            linestyle=linestyle(i)
        )


def plot_mu_t(
        t_grid,
        model_ixs,
        mus, sigmas,
        weights,
        time_left,
        linestyle):

    _, axs = plot_grid(2, 2)

    prior_component_ax = axs[0][0]
    posterior_component_ax = axs[0][1]
    prior_ax = axs[1][0]
    posterior_ax = axs[1][1]
    label = lambda i: 'Trajectory {}'.format(i)

    #print('mus', mus)
    #print(np.vstack([mus, sigmas])) #.reshape(-1, 2))
    prior_components = [
        norm.pdf(t_grid, mu, np.sqrt(sigma))
        for mu, sigma in
        zip(mus, sigmas)
    ]

    posterior_components = [
        w*norm.pdf(t_grid, mu, np.sqrt(sigma))
        for mu, sigma, w in
        zip(mus, sigmas, weights)
    ]

    prior = reduce(np.add, prior_components).reshape(t_grid.shape[0])
    prior_sum = np.sum(prior)
    prior = prior / prior_sum
    prior_mean = np.sum(prior*t_grid)
    uniform_weights = np.repeat(1, len(mus))

    prior_ax.plot(t_grid, prior, label='Prior density', color=default_color(0))
    prior_ax.set_title(r'$\mu_t$ Prior')
    prior_ax.set_xlabel('Seconds')
    prior_ax.set_ylabel('Density')
    plot_marker(prior_ax, time_left, 'Arrival time')
    plot_marker(prior_ax, prior_mean, 'Prior mean', color='blue')
    prior_ax.legend()

    prior_component_ax.set_title(r'$\mu_t$ Prior Components')
    prior_component_ax.set_xlabel('Seconds')
    prior_component_ax.set_ylabel('Density')
    plot_mixture(
        prior_component_ax, t_grid, model_ixs,
        mus, sigmas,
        uniform_weights,
        label, linestyle,
        default_color
    )
    plot_marker(prior_component_ax, time_left, 'Arrival time')
    #prior_component_ax.legend()

    posterior = reduce(np.add, posterior_components).reshape(t_grid.shape[0])
    posterior_sum = np.sum(posterior)
    posterior = posterior / posterior_sum
    posterior_mean = np.sum(posterior*t_grid)

    posterior_ax.plot(t_grid, posterior, label='Posterior density', color=default_color(0))
    plot_marker(posterior_ax, time_left, 'Arrival time')
    plot_marker(posterior_ax, posterior_mean, 'Posterior mean', color='blue')
    posterior_ax.set_title(r'$\mu_t$ Posterior')
    posterior_ax.set_xlabel('Seconds')
    posterior_ax.set_ylabel('Density')
    posterior_ax.legend()

    plot_mixture(
        posterior_component_ax, t_grid, model_ixs,
        mus, sigmas, weights,
        label, linestyle,
        default_color
    )
    plot_marker(posterior_component_ax, time_left, 'Arrival time')
    posterior_component_ax.set_title(r'$\mu_t$ Posterior Components')
    posterior_component_ax.set_xlabel('Seconds')
    posterior_component_ax.set_ylabel('Density')
    #posterior_component_ax.legend()


def plot_t(
        t_grid,
        model_ixs,
        t_mus,
        t_sigmas,
        weights,
        time_left,
        linestyle):

    _, axs = plot_grid(2, 2)
    prior_t_components_ax = axs[0][0]
    posterior_t_components_ax = axs[0][1]
    prior_t_ax = axs[1][0]
    posterior_t_ax = axs[1][1]

    label = lambda i: r'$\mathcal{M}_ {' + str(i) + '}$'
    uniform_weights = np.repeat(1, len(t_mus))


    plot_mixture(
        prior_t_components_ax,
        t_grid, model_ixs,
        t_mus, t_sigmas, uniform_weights,
        label, linestyle,
        default_color
    )

    prior_t_components_ax.set_title(r'$t$ prior components (fixed variance)')
    prior_t_components_ax.set_xlabel(r'$t$')
    prior_t_components_ax.set_ylabel('Density') #p(t | \mathcal{M}_k)')
    plot_marker(prior_t_components_ax, time_left, 'Arrival time')
    #prior_t_components_ax.legend()

    _, most_probable_model_mean, _, _ = max(
        zip(weights, t_mus, t_sigmas, range(len(weights))),
        key=lambda x: x[0]
    )

    plot_mixture(
        posterior_t_components_ax,
        t_grid, model_ixs,
        t_mus, t_sigmas, weights,
        label, linestyle,
        default_color
    )

    posterior_t_components_ax.set_title(r'$t$ posterior components (fixed variance)')
    posterior_t_components_ax.set_xlabel(r'$t$')
    posterior_t_components_ax.set_ylabel('Density')
    plot_marker(posterior_t_components_ax, most_probable_model_mean, 'Most probable model mean', color='blue')
    plot_marker(posterior_t_components_ax, time_left, 'Arrival time')
    #posterior_t_components_ax.legend()

    prior_components = [
        norm.pdf(t_grid, mu, np.sqrt(sigma))
        for mu, sigma in
        zip(t_mus, t_sigmas)
    ]
    prior = reduce(np.add, prior_components)
    prior = prior / (np.sum(prior))
    prior_mean = np.sum(prior*t_grid) / np.sum(prior)
    prior_t_ax.plot(t_grid, prior, color=default_color(0))

    posterior_t_ax.set_xlabel(r'$\tau$')
    posterior_t_ax.set_ylabel('Density')
    posterior_t_ax.set_title(r'$t$ prior')
    plot_marker(prior_t_ax, time_left, 'Arrival time')
    plot_marker(prior_t_ax, prior_mean, 'Prior mean', color='blue')
    prior_t_ax.legend()

    posterior_components = [
        w*norm.pdf(t_grid, mu, np.sqrt(sigma))
        for mu, sigma, w in
        zip(t_mus, t_sigmas, weights)
    ]
    posterior = reduce(np.add, posterior_components).reshape(t_grid.shape[0])
    posterior = posterior / np.sum(posterior)
    posterior_mean = np.sum(posterior*t_grid)

    posterior_t_ax.plot(t_grid, posterior, color=default_color(0))
    posterior_t_ax.set_xlabel(r'$\tau$')
    posterior_t_ax.set_ylabel('Density')
    posterior_t_ax.set_title(r'$t$ posterior')
    plot_marker(posterior_t_ax, time_left, 'Arrival time')
    plot_marker(posterior_t_ax, posterior_mean, 'Posterior mean', color='blue')
    posterior_t_ax.legend()

    #prediction_dist = norm.pdf(t_grid, t_mu, np.sqrt(t_sigma))
    #prediction = np.sum(prediction_dist*t_grid) / np.sum(prediction_dist)

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

def plot_model_probabilities(
        tau_grid,
        model_probs,
        model_ixs,
        cum_model_probs,
        linestyle):

    _, axs = plot_grid(1, 2)
    model_prob_ax = axs[0]
    cum_model_prob_ax = axs[1]

    seq = (zip(model_ixs, model_probs, cum_model_probs))
    for i, model_prob, cum_model_prob in seq:
        model_prob_ax.set_title('Model probabilities')
        model_prob_ax.set_xlabel(r'$\tau$')
        model_prob_ax.set_ylabel(r'$p(\mathcal{m}_k | X_{obs}$')
        model_prob_ax.plot(
            tau_grid, model_prob,
            color=default_color(i),
            label='Trajectory' + str(i),
            linestyle=linestyle(i)
        )

        cum_model_prob_ax.set_title(r'Cumulative model probabilities')
        cum_model_prob_ax.set_xlabel(r'$\tau$')
        cum_model_prob_ax.set_ylabel(r'$p(\mathcal{m}_k | X_{obs})$')
        cum_model_prob_ax.plot(
            tau_grid, cum_model_prob,
            color=default_color(i),
            label='Trajectory' + str(i),
            linestyle=linestyle(i)
        )

def plot_state_model_means(
        models,
        model_ixs,
        seg, progression,
        linestyle):

    _, axs = plot_grid(3, 2)
    tau_pred_ax = axs[0][0]
    time_left_pred_ax = axs[0][1]
    pos_x_pred_ax = axs[1][0]
    pos_y_pred_ax = axs[1][1]
    vel_x_pred_ax = axs[2][0]
    vel_y_pred_ax = axs[2][1]

    pos = seg[['x', 'y']].values.reshape(-1, 2)
    for i, model in zip(model_ixs, models):
        tau, _ = predict(model.g, pos)
        tau_pred_ax.plot(
            progression, tau.T[0],
            color=default_color(i),
            label='Trajectory ' + str(i),
            linestyle=linestyle(i)
        )

        time_left, _ = predict(model.h, tau)
        time_left_pred_ax.plot(
            tau, time_left,
            color=default_color(i),
            label='Trajectory ' + str(i),
            linestyle=linestyle(i)
        )

        p_x, _ = predict(model.f_p_x, tau)
        pos_x_pred_ax.plot(
            tau, p_x,
            color=default_color(i),
            label='Trajectory ' + str(i),
            linestyle=linestyle(i)
        )

        p_y, _ = predict(model.f_p_y, tau)
        pos_y_pred_ax.plot(
            tau, p_y,
            color=default_color(i),
            label='Trajectory ' + str(i),
            linestyle=linestyle(i)
        )

        v_x, _ = predict(model.f_v_x, tau)
        vel_x_pred_ax.plot(
            tau, v_x,
            color=default_color(i),
            label='Trajectory ' + str(i),
            linestyle=linestyle(i)
        )

        v_y, _ = predict(model.f_v_y, tau)
        vel_y_pred_ax.plot(
            tau, v_y,
            color=default_color(i),
            label='Trajectory ' + str(i),
            linestyle=linestyle(i)
        )

    tau_pred_ax.set_title(r'Predicted $\tau$ of input data')
    tau_pred_ax.set_xlabel('Normalised progression')
    tau_pred_ax.set_ylabel(r'$\tau$')
    tau_pred_ax.plot(
        progression, seg.tau,
        label='True trajectory',
        color='red'
    )

    time_left_pred_ax.set_title(r'Predicted $\mu_t$ of input data')
    time_left_pred_ax.set_xlabel(r'$\tau$')
    time_left_pred_ax.set_ylabel(r'$\mu_t$')
    time_left_pred_ax.plot(
        seg.tau, seg.time_left,
        label='True trajectory',
        color='red'
    )

    pos_x_pred_ax.set_title(r'Predicted $x$-position of input data')
    pos_x_pred_ax.set_xlabel(r'$\tau$')
    pos_x_pred_ax.set_ylabel(r'$p_x$')
    pos_x_pred_ax.plot(
        seg.tau, seg.x,
        label='True trajectory',
        color='red'
    )

    pos_y_pred_ax.set_title(r'Predicted $y$-position of input data')
    pos_y_pred_ax.set_xlabel(r'$\tau$')
    pos_y_pred_ax.set_ylabel(r'$p_y$')
    pos_y_pred_ax.plot(
        seg.tau, seg.y,
        label='True trajectory',
        color='red'
    )

    vel_x_pred_ax.set_title(r'Predicted $x$-velocity of input data')
    vel_x_pred_ax.set_xlabel(r'$\tau$')
    vel_x_pred_ax.set_ylabel(r'$v_x$')
    vel_x_pred_ax.plot(
        seg.tau, seg.dx,
        label='True trajectory',
        color='red'
    )

    vel_y_pred_ax.set_title(r'Predicted $y$-velocity of input data')
    vel_y_pred_ax.set_xlabel(r'$\tau$')
    vel_y_pred_ax.set_ylabel(r'$v_y$')
    vel_y_pred_ax.plot(
        seg.tau, seg.dy,
        label='True trajectory',
        color='red'
    )


def plot_most_probable_models(
        models,
        model_weights,
        cum_model_probs,
        data,
        linestyle,
        n_models):

    n_models = 3
    model_indicies = range(len(models))
    most_probable_models = sorted(
        zip(model_weights, model_indicies, models),
        key=lambda x: x[0],
        reverse=True
    )

    _, ax = plot_grid(3, 2)
    tau_grid = np.linspace(0, data.iloc[-1].tau, data.shape[0])
    p_x_ax = ax[0][0]
    p_y_ax = ax[0][1]
    v_x_ax = ax[1][0]
    v_y_ax = ax[1][1]
    t_ax = ax[2][0]
    probs_ax = ax[2][1]
    label = lambda i: r'$\mathcal{M}_{'+ str(i) + '}$'

    for w, i, m in most_probable_models[:n_models]:
        plot_with_credible_bands(
            p_x_ax, tau_grid, m.f_p_x,
            label(i), default_color(i), linestyle(i)
        )
        plot_with_credible_bands(
            p_y_ax, tau_grid, m.f_p_y,
            label(i), default_color(i), linestyle(i)
        )
        plot_with_credible_bands(
            v_x_ax, tau_grid, m.f_v_x,
            label(i), default_color(i), linestyle(i)
        )
        plot_with_credible_bands(
            v_y_ax, tau_grid, m.f_v_y,
            label(i), default_color(i), linestyle(i)
        )
        plot_with_credible_bands(
            t_ax, tau_grid, m.h,
            label(i), default_color(i), linestyle(i)
        )
    for w, i, m in most_probable_models[:n_models*2]:
        probs = cum_model_probs[i]
        probs_ax.plot(
            tau_grid, probs,
            label=label(i),
            color=default_color(i),
            linestyle=linestyle(i)
        )

    suffix = lambda n: ' for {} most probable models'.format(n)
    plot_data(p_x_ax, data.tau, data.x)
    p_x_ax.set_title(r'Predicted $x$-position' + suffix(n_models))
    p_x_ax.set_xlabel(r'$\tau$')
    p_x_ax.set_ylabel(r'$p(p_x | \mathcal{M}_k, X_{obs})$')
    p_x_ax.legend()

    plot_data(p_y_ax, data.tau, data.y)
    p_y_ax.set_title(r'Predicted $y$-position' + suffix(n_models))
    p_y_ax.set_xlabel(r'$\tau$')
    p_y_ax.set_ylabel(r'$p(p_y | \mathcal{M}_k, X_{obs})$')
    p_y_ax.legend()

    plot_data(v_x_ax, data.tau, data.dx)
    v_x_ax.set_title(r'Predicted $x$-velocity' + suffix(n_models))
    v_x_ax.set_xlabel(r'$\tau$')
    v_x_ax.set_ylabel(r'$p(v_x | \mathcal{M}_k, X_{obs})$')
    v_x_ax.legend()

    plot_data(v_y_ax, data.tau, data.dy)
    v_y_ax.set_title(r'Predicted $y$-velocity' + suffix(n_models))
    v_y_ax.set_xlabel(r'$\tau$')
    v_y_ax.set_ylabel(r'$p(v_y | \mathcal{M}_k, X_{obs})$')
    v_y_ax.legend()

    plot_data(t_ax, data.tau, data.time_left)
    t_ax.set_title(r'Predicted $\mu_t$' + suffix(n_models))
    t_ax.set_xlabel(r'$\tau$')
    t_ax.set_ylabel(r'$p(\mu_t | \mathcal{M}_k, X_{obs})$')
    t_ax.legend()

    probs_ax.set_title(r'Cumulative model probabilities' + suffix(n_models*2))
    probs_ax.set_xlabel(r'$\tau$')
    probs_ax.set_ylabel(r'$p(\mathcal{M}_k | X_{obs})$')
    probs_ax.legend()
