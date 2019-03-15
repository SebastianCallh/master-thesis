"""This module defines the "inverse-Gaussian Process" trajectory model.

The model consist of three Gaussian Processes.
f: tau        -> state
g: (lat, lon) -> tau
h: tau        -> time
"""
import math
import numpy as np
from functools import reduce
from typing import NamedTuple
from pandas import DataFrame
from numpy import ndarray
from numpy.linalg import inv, det
from scipy.stats import norm
import GPy

import pandas as pd
import seaborn as sns
from .function_model import FunctionModel, plot_function, predict, \
    plot_with_credible_bands, learn_function, gamma_prior
from .plotting import plot_grid, default_color, plot_data, plot_marker, \
    plot_mixture

F_CODOMAIN = ['x', 'y', 'dx', 'dy']


class TrajectoryModel(NamedTuple):
    route: int
    segment: int
    f_p_x: FunctionModel
    f_p_y: FunctionModel
    f_v_x: FunctionModel
    f_v_y: FunctionModel
    g: FunctionModel
    h: FunctionModel


def model_cum_loglik(
    model: TrajectoryModel,
    X_obs: ndarray, tau: ndarray) -> ndarray:

    inc_tau, inc_X_obs = zip(*[
        (tau[:n], X_obs[:n])
        for n in range(len(tau))
    ])
    return np.array([
        model_data_loglik(model, obs, tau)
        for tau, obs in zip(inc_tau, inc_X_obs)
    ]).reshape(-1, 1)


def model_pred(model: TrajectoryModel, tau: ndarray):
    # predict for the last point
    last_point = tau[-1].reshape(1, 1)
    return predict(model.h, last_point)


def model_data_loglik(
        model: TrajectoryModel,
        X_obs: ndarray, tau: ndarray) -> ndarray:

    def loglik(x: ndarray, mu: ndarray, sigma: ndarray):
        """Assumes sigma is the diagonal of the covariance matrix."""
        return -0.5*((x-mu).T).dot(inv(sigma)).dot(x-mu) \
            -0.5*np.log(det(sigma))

    X_k_p_x, sigma_p_x = predict(model.f_p_x, tau)
    X_k_p_y, sigma_p_y = predict(model.f_p_y, tau)
    X_k_v_x, sigma_v_x = predict(model.f_v_x, tau)
    X_k_v_y, sigma_v_y = predict(model.f_v_y, tau)
    pos = X_obs[:,0:2]
    vel = X_obs[:,2:4]

    loglik_pos_x = loglik(pos[:,0].reshape(-1, 1), X_k_p_x, np.diag(sigma_p_x.T[0]))
    loglik_pos_y = loglik(pos[:,1].reshape(-1, 1), X_k_p_y, np.diag(sigma_p_y.T[0]))
    loglik_vel_x = loglik(vel[:,0].reshape(-1, 1), X_k_v_x, np.diag(sigma_v_x.T[0]))
    loglik_vel_y = loglik(vel[:,1].reshape(-1, 1), X_k_v_y, np.diag(sigma_v_y.T[0]))


    loglik_sum = \
        loglik_pos_x + \
        loglik_pos_y + \
        loglik_vel_x + \
        loglik_vel_y

    if math.isinf(loglik_sum) or math.isnan(loglik_sum):
        print('inf loglik', loglik_pos_x, loglik_pos_y, loglik_vel_x, loglik_vel_y)
        # m
        u = X_k_v_x
    return loglik_sum


def normalise_logliks(logliks):
    """Normalise model logliks for an observation."""
    # Scale to avoid numerical errors due to small numbers
    c = 1/max(logliks)
    loglik_sum = np.sum(logliks)
    f = np.vectorize(lambda l: l - loglik_sum)
    return f(logliks)


# LEARN MODEL


def create_support_data(
        pos: ndarray,
        tau: ndarray,
        f_p_x, f_p_y, f_v_x, f_v_y,
        n_samples: int,
        delta: float,
        sigma: float) -> DataFrame:
    def orth_comp(v):
        return np.array([-v[1], v[0]])

    tau_grid = np.linspace(
        np.min(np.min(tau)),
        np.min(np.max(tau)),
        math.round(1 / delta)
    )

    tau = tau_grid.reshape(len(tau_grid), 1)
    x, _ = predict(f_p_x, tau)
    y, _ = predict(f_p_y, tau)
    dx, _ = predict(f_v_x, tau)
    dy, _ = predict(f_v_y, tau)
    acc = []
    pos = np.hstack([x, y])
    vel = np.hstack([dx, dy])
    for n in range(len(tau_grid) - 1):
        cur_vel = vel[n]
        cur_pos = pos[n]
        nxt_pos = pos[n + 1]
        orth_delta = orth_comp(nxt_pos - cur_pos)
        orth_delta = orth_delta / np.linalg.norm(orth_delta)

        acc.extend([
            {'x': x[0],
             'y': x[1],
             # 'dx': cur_vel[0],
             # 'dy': cur_vel[1],
             'tau': tau_grid[n]}
            for x in [
                cur_pos + orth_delta * x
                for x in np.random.normal(0, sigma, n_samples)
            ]
        ])

    return pd.DataFrame(acc)


def learn_model(
        data: DataFrame, route_n: int, seg_n: int,
        f_p_codomain, f_v_codomain,
        f_p_likelihood_noise,
        f_v_likelihood_noise,
        seg_normalisers) -> TrajectoryModel:
    tau = data['tau'].values.reshape(-1, 1)
    normaliser = seg_normalisers[seg_n]


    def apply_f_p_x_priors(m):
        # kern.rbf.lengthscale
        m.kern.lengthscale.set_prior(gamma_prior(0.15, 0.1))
        m.kern.variance.set_prior(gamma_prior(0.25, 0.1))
        # m.kern.linear.variances.set_prior(gamma_prior(0.15, 0.1))
        m.likelihood.variance = f_p_likelihood_noise / normaliser.p_scale
        m.likelihood.variance.fix()

    def apply_f_p_y_priors(m):
        m.kern.lengthscale.set_prior(gamma_prior(0.15, 0.1))
        m.kern.variance.set_prior(gamma_prior(0.25, 0.1))
        # m.kern.linear.variances.set_prior(gamma_prior(0.15, 0.1))
        m.likelihood.variance = f_p_likelihood_noise / normaliser.p_scale
        m.likelihood.variance.fix()

    def apply_f_v_x_priors(m):
        # m.kern.lengthscale.set_prior(gamma_prior(0.1, 0.04))
        # m.kern.variance.set_prior(gamma_prior(75, 5))
        # m.likelihood_noise.set_prior(gamma_prior(0.4, 0.4))
        m.likelihood.variance = f_v_likelihood_noise / normaliser.v_scale
        m.likelihood.variance.fix()

    def apply_f_v_y_priors(m):
        # m.kern.lengthscale.set_prior(gamma_prior(0.1, 0.04))
        # m.kern.variance.set_prior(gamma_prior(75, 5))
        # m.likelihood_noise.set_prior(gamma_prior(0.4, 0.4))
        m.likelihood.variance = f_v_likelihood_noise / normaliser.v_scale
        m.likelihood.variance.fix()

    def apply_g_priors(m):
        return
        # m.kern.rbf.lengthscale.set_prior(gamma_prior(0.10, 0.05))
        # m.kern.rbf.variance.set_prior(gamma_prior(0.25, 0.1))
        # m.kern.linear.variances.set_prior(gamma_prior(0.15, 0.1))
        # m.likelihood.variance = 0.1
        # m.likelihood.variance.fix()

    def apply_h_priors(m):
        m.likelihood.variance = 0.1
        m.likelihood.variance.fix()

    f_p_x = learn_function(
        tau, data['x'].values.reshape(-1, 1),
        apply_f_p_x_priors, 'f_p_x',
        route_n, seg_n, kernel=GPy.kern.RBF(
            input_dim=1,
            ARD=False
        ), z=np.random.rand(15, 1)
    )

    f_p_y = learn_function(
        tau, data['y'].values.reshape(-1, 1),
        apply_f_p_y_priors, 'f_p_y',
        route_n, seg_n, kernel=GPy.kern.RBF(
            input_dim=1,
            ARD=False
        ), z=np.random.rand(15, 1)
    )

    f_v_x = learn_function(
        tau, data['dx'].values.reshape(-1, 1),
        apply_f_v_x_priors, 'f_v_x',
        route_n, seg_n, kernel=GPy.kern.Matern52(
            input_dim=1,
            ARD=False
        ), z=np.random.rand(15, 1)
    )

    f_v_y = learn_function(
        tau, data['dy'].values.reshape(-1, 1),
        apply_f_v_y_priors, 'f_v_y',
        route_n, seg_n, kernel=GPy.kern.Matern52(
            input_dim=1,
            ARD=False
        ), z=np.random.rand(15, 1)
    )

    n_augment_samples = 10
    augment_sigma = 8 / normaliser.p_scale  # meters
    augment_delta = 0.015
    support_data = create_support_data(
        data[['x', 'y']].values.reshape(-1, 1),
        tau, f_p_x, f_p_y, f_v_x, f_v_y,
        n_augment_samples,
        augment_delta, augment_sigma
    )
    augmented_data = \
        data[['x', 'y', 'tau']] \
            .append(support_data)

    g_pos = augmented_data[['x', 'y']].values.reshape(-1, 2)
    g_tau = augmented_data['tau'].values.reshape(-1, 1)

    g = learn_function(
        g_pos, g_tau,
        apply_g_priors, 'g',
        route_n, seg_n, kernel=GPy.kern.RBF(
            input_dim=2,
            ARD=False
        ) + GPy.kern.Linear(
            input_dim=2,
            ARD=False
        ), z=np.random.rand(50, 2)
    )

    time_left = data.time_left.values.reshape(-1, 1)
    h = learn_function(
        tau, time_left,
        apply_h_priors, 'h',
        route_n, seg_n, kernel=GPy.kern.RBF(
            input_dim=1,
            ARD=False
        ) + GPy.kern.Linear(
            input_dim=1,
            ARD=False
        )
    )

    return TrajectoryModel(
        route_n, seg_n, f_p_x, f_p_y, f_v_x, f_v_y, g, h
    )


# MODEL PLOTTING


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
    mean, _ = predict(m.g, x)
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
    # axs[1][0].set_aspect('equal', 'datalim')

    # Training data
    plot_function(m.g, ax=axs[0][1])
    axs[0][1].set_title(r'Inverse model function $g: (lat,lon) \mapsto \tau$')
    axs[0][1].set_xlabel('x')
    axs[0][1].set_ylabel('y')
    # axs[0][1].axis('scaled')

    # H for training data
    mean, _ = m.g.model.predict(m.g.model.X)
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
    label = lambda i: r'$\mathcal{M}_{' + str(i) + '}$'

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
    for w, i, m in most_probable_models[:n_models * 2]:
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

    probs_ax.set_title(r'Cumulative model probabilities' + suffix(n_models * 2))
    probs_ax.set_xlabel(r'$\tau$')
    probs_ax.set_ylabel(r'$p(\mathcal{M}_k | X_{obs})$')
    probs_ax.legend()


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
    prior_t_components_ax.set_ylabel('Density')  # p(t | \mathcal{M}_k)')
    plot_marker(prior_t_components_ax, time_left, 'Arrival time')
    # prior_t_components_ax.legend()

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

    posterior_t_components_ax.set_title(
        r'$t$ posterior components (fixed variance)')
    posterior_t_components_ax.set_xlabel(r'$t$')
    posterior_t_components_ax.set_ylabel('Density')
    plot_marker(posterior_t_components_ax, most_probable_model_mean,
                'Most probable model mean', color='blue')
    plot_marker(posterior_t_components_ax, time_left, 'Arrival time')
    # posterior_t_components_ax.legend()

    prior_components = [
        norm.pdf(t_grid, mu, np.sqrt(sigma))
        for mu, sigma in
        zip(t_mus, t_sigmas)
    ]
    prior = reduce(np.add, prior_components)
    prior = prior / (np.sum(prior))
    prior_mean = np.sum(prior * t_grid) / np.sum(prior)
    prior_t_ax.plot(t_grid, prior, color=default_color(0))

    posterior_t_ax.set_xlabel(r'$\tau$')
    posterior_t_ax.set_ylabel('Density')
    posterior_t_ax.set_title(r'$t$ prior')
    plot_marker(prior_t_ax, time_left, 'Arrival time')
    plot_marker(prior_t_ax, prior_mean, 'Prior mean', color='blue')
    prior_t_ax.legend()

    posterior_components = [
        w * norm.pdf(t_grid, mu, np.sqrt(sigma))
        for mu, sigma, w in
        zip(t_mus, t_sigmas, weights)
    ]
    posterior = reduce(np.add, posterior_components).reshape(t_grid.shape[0])
    posterior = posterior / np.sum(posterior)
    posterior_mean = np.sum(posterior * t_grid)

    posterior_t_ax.plot(t_grid, posterior, color=default_color(0))
    posterior_t_ax.set_xlabel(r'$\tau$')
    posterior_t_ax.set_ylabel('Density')
    posterior_t_ax.set_title(r'$t$ posterior')
    plot_marker(posterior_t_ax, time_left, 'Arrival time')
    plot_marker(posterior_t_ax, posterior_mean, 'Posterior mean', color='blue')
    posterior_t_ax.legend()
