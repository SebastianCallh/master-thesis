"""This module defines the "inverse-Gaussian Process" trajectory model.
"""

import math
from functools import reduce
from typing import NamedTuple, Tuple, List, Callable

import GPy
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import ndarray
from numpy.linalg import inv, det
from pandas import DataFrame
from scipy.stats import norm

from .function_model import FunctionModel, plot_function, predict, \
    plot_with_credible_bands, learn_function, plot_posterior, gamma_prior
from .metric import time
from .plotting import plot_grid, default_color, plot_data, plot_marker, \
    plot_mixture
from .pre_process import duplicate, SegmentNormaliser

F_CODOMAIN = ['x', 'y', 'dx', 'dy']
KM_H_RATIO = 3.6
KM_RATIO = 1000

class TrajectoryModel(NamedTuple):
    route: int
    segment: int
    traj: int
    f_p_x: FunctionModel
    f_p_y: FunctionModel
    f_v_x: FunctionModel
    f_v_y: FunctionModel
    f_p_x_1: FunctionModel
    f_p_x_2: FunctionModel
    f_p_y_1: FunctionModel
    f_p_y_2: FunctionModel
    f_v_x_1: FunctionModel
    f_v_x_2: FunctionModel
    f_v_y_1: FunctionModel
    f_v_y_2: FunctionModel
    g: FunctionModel
    h: FunctionModel


def model_pred(model: TrajectoryModel, tau: ndarray):
    # predict for the last point
    last_point = tau[-1].reshape(1, 1)
    return predict(model.h, last_point)


def combine(mu: ndarray, var: ndarray):
    """mu and sigma is is M x N where M is number of data points
     and N is number of models.
    """
    N = var.shape[1]
    mu_prime = mu.sum(axis=1) / N
    var_prime = (var + mu**2).sum(axis=1) / N - mu_prime**2
    return mu_prime.reshape(-1, 1), var_prime.reshape(-1, 1)


def loglik(x: ndarray, mu: ndarray, sigma: ndarray):
    #z = x - mu
    #return -0.5 * z * (1 / sigma) * z \
    #       - 0.5 * np.log(np.prod(sigma))
    z = x - mu
    return -0.5*z.T.dot(inv(sigma)).dot(z) \
        - 0.5*np.log(det(sigma))


def model_data_loglik(
        model: TrajectoryModel,
        tau: ndarray, X_obs: ndarray) -> ndarray:

    #print('computing loglik fotr', tau.shape, X_obs.shape)

    def loglik(x: ndarray, mu: ndarray, sigma: ndarray):
        #z = x - mu
        #return -0.5 * z * (1 / sigma) * z \
        #       - 0.5 * np.log(np.prod(sigma))

        return -0.5*((x-mu).T).dot(inv(sigma)).dot(x-mu) \
            -0.5*np.log(det(sigma))

    mu_p_x1, var_p_x1 = predict(model.f_p_x_1, tau)
    mu_p_x2, var_p_x2 = predict(model.f_p_x_2, tau)
    mu_p_x, var_p_x = combine(
        np.hstack((mu_p_x1, mu_p_x2)),
        np.hstack((var_p_x1, var_p_x2))
    )

    mu_p_y1, var_p_y1 = predict(model.f_p_y_1, tau)
    mu_p_y2, var_p_y2 = predict(model.f_p_y_2, tau)
    mu_p_y, var_p_y = combine(
        np.hstack((mu_p_y1, mu_p_y2)),
        np.hstack((var_p_y1, var_p_y2))
    )

    mu_v_x1, var_v_x1 = predict(model.f_v_x_1, tau)
    mu_v_x2, var_v_x2 = predict(model.f_v_x_2, tau)
    mu_v_x, var_v_x = combine(
        np.hstack((mu_v_x1, mu_v_x2)),
        np.hstack((var_v_x1, var_v_x2))
    )

    mu_v_y1, var_v_y1 = predict(model.f_v_y_1, tau)
    mu_v_y2, var_v_y2 = predict(model.f_v_y_2, tau)
    mu_v_y, var_v_y = combine(
        np.hstack((mu_v_y1, mu_v_y2)),
        np.hstack((var_v_y1, var_v_y2))
    )

    pos = X_obs[:, 0:2]
    vel = X_obs[:, 2:4]

    loglik_pos_x = loglik(pos[:,0].reshape(-1, 1), mu_p_x,
                          np.diag(var_p_x.reshape(-1)))
    loglik_pos_y = loglik(pos[:,1].reshape(-1, 1), mu_p_y,
                          np.diag(var_p_y.reshape(-1)))
    loglik_vel_x = loglik(vel[:,0].reshape(-1, 1), mu_v_x,
                          np.diag(var_v_x.reshape(-1)))
    loglik_vel_y = loglik(vel[:,1].reshape(-1, 1), mu_v_y,
                          np.diag(var_v_y.reshape(-1)))

    loglik_sum = \
        loglik_pos_x + \
        loglik_pos_y + \
        loglik_vel_x + \
        loglik_vel_y

    if math.isinf(loglik_sum) or math.isnan(loglik_sum):
        print('inf loglik', loglik_pos_x, loglik_pos_y, loglik_vel_x, loglik_vel_y)

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
        tau: ndarray,
        f_p_x_1, f_p_x_2,
        f_p_y_1, f_p_y_2,
        delta: float) -> DataFrame:

    def orth_comp(v):
        return np.array([-v[1], v[0]])

    tau_grid = np.linspace(
        np.min(np.min(tau)),
        np.min(np.max(tau)),
        round(1 / delta)
    )

    tau = tau_grid.reshape(-1, 1)
    x1, _ = predict(f_p_x_1, tau)
    x2, _ = predict(f_p_x_2, tau)
    y1, _ = predict(f_p_y_1, tau)
    y2, _ = predict(f_p_y_2, tau)
    data = np.vstack([
        np.hstack([tau, x1, y1]),
        np.hstack([tau, x2, y2])
    ])
    return pd.DataFrame(data, columns=['tau', 'x', 'y'])

    acc = []
    pos = np.hstack([x, y])
    for n in range(len(tau_grid) - 1):

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


def offset_by(df: DataFrame, offset: float) -> DataFrame:
    df.x += offset
    df.y += offset
    return df


def equidistant_points(
        X: ndarray,
        Y: ndarray,
        delta: float) -> ndarray:

    def distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    points = np.hstack((X, Y))
    equidistant_points = [points[0]]
    for point in points:
        last_x = equidistant_points[-1][0]
        last_y = equidistant_points[-1][1]
        if distance((point[0], point[1]), (last_x, last_y)) >= delta:
            #print('distance is', distance((point[1], point[2]), (last_x,
            # last_y)))
            equidistant_points.append(point)

    equidistant_points.append(points[-1])
    return np.vstack(equidistant_points)


def learn_model(
        data: DataFrame,
        route_n: int, seg_n: int, traj_n,
        f_p_codomain: List[str],
        f_v_codomain: [str],
        f_p_likelihood_noise: float,
        f_v_likelihood_noise: float,
        normaliser: SegmentNormaliser,
        delta_xy: float,
        delta_p: float,
        delta_v: float,
        ) -> TrajectoryModel:

    def apply_f_p_x_priors(m):
        # kern.rbf.lengthscale
        #return
        #m.kern.lengthscale.set_prior(gamma_prior(0.15, 0.1))
        #m.kern.variance.set_prior(gamma_prior(0.25, 0.1))
        # m.kern.linear.variances.set_prior(gamma_prior(0.15, 0.1))
        m.likelihood.variance = f_p_likelihood_noise / normaliser.p_scale
        m.likelihood.variance.fix()

    def apply_f_p_y_priors(m):
        #m.kern.lengthscale.set_prior(gamma_prior(0.15, 0.1))
        #m.kern.variance.set_prior(gamma_prior(0.25, 0.1))
        # m.kern.linear.variances.set_prior(gamma_prior(0.15, 0.1))
        m.likelihood.variance = f_p_likelihood_noise / normaliser.p_scale
        m.likelihood.variance.fix()

    def apply_f_v_x_priors(m):
        # m.kern.lengthscale.set_prior(gamma_prior(.01, 1))
        m.kern.variance.set_prior(gamma_prior(5, 3))
        # m.likelihood_noise.set_prior(gamma_prior(0.4, 0.4))
        m.likelihood.variance = f_v_likelihood_noise / normaliser.v_scale
        m.likelihood.variance.fix()

    def apply_f_v_y_priors(m):
        # m.kern.lengthscale.set_prior(gamma_prior(0.1, 0.04))
        m.kern.variance.set_prior(gamma_prior(5, 3))
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
        return
        #m.likelihood.variance = 0.1
        #m.likelihood.variance.fix()

    def learn_tau_rbf_func(x, y, n_inducing, priors, name):
        return learn_function(
            x, y, priors, name,
            kernel=GPy.kern.Matern52(
                input_dim=1,
                ARD=False
            ), num_inducing=n_inducing  # z=np.random.rand(n_inducing, 1)
        )

    f_p_inducing = 20
    f_v_inducing = 131
    f_g_inducing = 50
    h_inducing = 20

    # Learn an interpolation function first
    tau = data.tau.values.reshape(-1, 1)
    p_x = data[f_p_codomain[0]].values.reshape(-1, 1)
    p_y = data[f_p_codomain[1]].values.reshape(-1, 1)
    f_p_x = learn_tau_rbf_func(tau, p_x, f_p_inducing, apply_f_p_x_priors, 'f_p_x')
    f_p_y = learn_tau_rbf_func(tau, p_y, f_p_inducing, apply_f_p_y_priors, 'f_p_y')

    # Interpolate
    delta_tau = 0.005
    tau_grid = np.linspace(0, 1, round(1/delta_tau)).reshape(-1, 1)
    mu_p_x, _ = f_p_x.model.predict(tau_grid)
    mu_p_y, _ = f_p_y.model.predict(tau_grid)

    # Compute equidistant spatial steps
    # too small steps makes for singular covariance matrix
    #points_p_x = equidistant_points(tau_grid, mu_p_x, epsilon_p)
    #equidist_p_x_tau = tau_grid  # points_p_x[:, 0].reshape(-1, 1)
    #equidist_p_x = mu_p_x  # points_p_x[:, 1].reshape(-1, 1)

    #equidist_p_y = points_p[:, 2].reshape(-1, 1)

    # Project orthogonally and create support GPs
    #p_x_1, p_x_2 = duplicate(equidist_p_x_tau, equidist_p_x, delta_p)
    p_x_1 = mu_p_x + delta_xy
    p_x_2 = mu_p_x - delta_xy

    f_p_x_1 = learn_tau_rbf_func(
        tau_grid, p_x_1.reshape(-1, 1),
        f_p_inducing, apply_f_p_x_priors, 'f_p_x_1'
    )
    f_p_x_2 = learn_tau_rbf_func(
        tau_grid, p_x_2.reshape(-1, 1),
        f_p_inducing, apply_f_p_x_priors, 'f_p_x_2'
    )

    #points_p_y = equidistant_points(tau_grid, mu_p_y, epsilon_p)
    p_y_1 = mu_p_y + delta_p
    p_y_2 = mu_p_y - delta_p
    f_p_y_1 = learn_tau_rbf_func(
        tau_grid, p_y_1.reshape(-1, 1),
        f_p_inducing, apply_f_p_y_priors, 'f_p_y_1'
    )
    f_p_y_2 = learn_tau_rbf_func(
        tau_grid, p_y_2.reshape(-1, 1),
        f_p_inducing, apply_f_p_y_priors, 'f_p_y_2'
    )

    # Do the same for velocity
    v_x = data[f_v_codomain[0]].values.reshape(-1, 1)
    v_y = data[f_v_codomain[1]].values.reshape(-1, 1)
    f_v_x = learn_tau_rbf_func(tau, v_x, f_v_inducing, apply_f_v_x_priors, 'f_v_x')
    f_v_y = learn_tau_rbf_func(tau, v_y, f_v_inducing, apply_f_v_y_priors, 'f_v_y')
    mu_v_x, _ = f_v_x.model.predict(tau_grid)
    mu_v_y, _ = f_v_y.model.predict(tau_grid)

    #points_v_x = equidistant_points(tau_grid, mu_v_x, epsilon_v)
    #equidist_v_x_tau = points_v_x[:, 0].reshape(-1, 1)
    # equidist_v_x = points_v_x[:, 1].reshape(-1, 1)
    #equidist_v_y = points_v[:, 2].reshape(-1, 1)

    v_x_1 = mu_v_x + delta_v
    v_x_2 = mu_v_x - delta_v
    f_v_x_1 = learn_tau_rbf_func(
        tau_grid, v_x_1.reshape(-1, 1),
        f_v_inducing, apply_f_v_x_priors, 'f_v_x_1'
    )
    f_v_x_2 = learn_tau_rbf_func(
        tau_grid, v_x_2.reshape(-1, 1),
        f_v_inducing, apply_f_v_x_priors, 'f_v_x_2'
    )

    #points_v_y = equidistant_points(tau_grid, mu_v_y, epsilon_v)
    #equidist_v_y_tau = points_v_y[:, 0].reshape(-1, 1)
    # equidist_v_y = points_v_y[:, 1].reshape(-1, 1)
    v_y_1 = mu_v_y + delta_v
    v_y_2 = mu_v_y - delta_v
        #, v_y_2 = duplicate(equidist_v_y_tau, equidist_v_y,
        # delta_v)

    f_v_y_1 = learn_tau_rbf_func(
        tau_grid, v_y_1.reshape(-1, 1),
        f_v_inducing, apply_f_v_y_priors, 'f_v_y_1'
    )
    f_v_y_2 = learn_tau_rbf_func(
        tau_grid, v_y_2.reshape(-1, 1),
        f_v_inducing, apply_f_v_y_priors, 'f_v_y_2'
    )

    # augment_delta = 0.015
    # support_data = create_support_data(
    #     tau, f_p_x_1, f_p_x_2,
    #     f_p_y_1, f_p_y_2,
    #     augment_delta
    # )

    points_spatial_1, points_spatial_2 = duplicate(
        data.x.values, data.y.values, delta_xy
    )

    support_data = pd.DataFrame(np.vstack((
        np.hstack((points_spatial_1, data.tau.values.reshape(-1, 1))),
        np.hstack((points_spatial_2, data.tau.values.reshape(-1, 1)))
    )), columns=['x', 'y', 'tau'])

    augmented_data = data[['x', 'y', 'tau']].append(support_data)
    g_pos = augmented_data[['x', 'y']].values.reshape(-1, 2)
    g_tau = augmented_data['tau'].values.reshape(-1, 1)

    g = learn_function(
        g_pos, g_tau,
        apply_g_priors, 'g',
        kernel=GPy.kern.RBF(
            input_dim=2,
            ARD=False
        ) + GPy.kern.Linear(
            input_dim=2,
            ARD=False
        ), num_inducing=f_g_inducing
    )

    time_left = data.time_left.values.reshape(-1, 1)
    h = learn_function(
        tau, time_left,
        apply_h_priors, 'h',
        kernel=GPy.kern.RBF(
            input_dim=1,
            ARD=False
        ) + GPy.kern.Linear(
            input_dim=1,
            ARD=False
        ), num_inducing=h_inducing
    )

    return TrajectoryModel(
        route_n, seg_n, traj_n,
        f_p_x, f_p_y,
        f_v_x, f_v_y,
        f_p_x_1, f_p_x_2,
        f_p_y_1, f_p_y_2,
        f_v_x_1, f_v_x_2,
        f_v_y_1, f_v_y_2,
        g, h
    )


def model_observations_loglik(
        model: TrajectoryModel, X_obs: ndarray,
        tau: ndarray) -> Tuple[ndarray, ndarray]:

    tau_obs = np.hstack((tau, X_obs))
    return np.apply_along_axis(
        lambda row: model_data_loglik(
            model,
            row[0].reshape(1, -1),
            row[1:].reshape(1, -1)
        ), 1, tau_obs
    )


def to_prob(x):
    x = np.exp(x - x.max())
    return x / x.sum()


def to_probabilities(likelihoods):
    likelihoods = np.apply_along_axis(np.cumsum, 2, likelihoods)
    likelihoods = np.apply_along_axis(to_prob, 1, likelihoods)
    return likelihoods


# this is the divide by zero guy
def model_weights(models: List[TrajectoryModel], X_obs: ndarray) -> ndarray:
    n_models = 4
    logliks = np.empty((n_models, len(models)))
    pos, vel = X_obs[:, 0:2].reshape(-1, 2), X_obs[:, 2:4].reshape(-1, 2)
    for i, m in enumerate(models):
        tau, _ = predict(m.g, pos)
        p_x, p_y = pos[:, 0].reshape(-1, 1), pos[:, 1].reshape(-1, 1)
        v_x, v_y = vel[:, 0].reshape(-1, 1), vel[:, 1].reshape(-1, 1)
        print(v_x.shape, p_x.shape)
        mu_p_x, var_p_x = combined_prediction(tau, m.f_p_x_1, m.f_p_x_2)
        print(var_p_x.T[0].shape)
        logliks[0][i] = loglik(p_x, mu_p_x, np.diag(var_p_x.T[0]))
        mu_p_y, var_p_y = combined_prediction(tau, m.f_p_y_1, m.f_p_y_2)
        logliks[1][i] = loglik(p_y, mu_p_y, np.diag(var_p_y.T[0]))
        mu_v_x, var_v_x = combined_prediction(tau, m.f_v_x_1, m.f_v_x_2)
        logliks[2][i] = loglik(v_x, mu_v_x, np.diag(var_v_x.T[0]))
        mu_v_y, var_v_y = combined_prediction(tau, m.f_v_y_1, m.f_v_y_2)
        logliks[3][i] = loglik(v_y, mu_v_y, np.diag(var_v_y.T[0]))

    probs = np.apply_along_axis(to_prob, 1, logliks)
    print(probs.shape)
    probs = np.apply_along_axis(np.sum, 0, probs)
    return probs / probs.sum()

    #
    # pos = X_obs[:, :2]
    # for i, model in enumerate(models):
    #     tau, _ = predict(model.g, pos)
    #     logliks[i] = model_data_loglik(model, tau, X_obs)
    # return to_probabilities(logliks)


def model_cum_probs(models, X_obs):
    pos = X_obs[:, 0:2]

    logliks = np.empty((len(models), X_obs.shape[0]))
    for i, m in enumerate(models):
        taus = predict(m.g, pos)[0]
        logliks[i, :] = [
            model_data_loglik(m, tau.reshape(1, -1), x.reshape(1, -1))
            for tau, x in zip(taus, X_obs)
        ]

    cum_logliks = np.apply_along_axis(np.cumsum, 1, logliks)
    #print(cum_logliks.shape, 'cum logliks')
    #print(cum_logliks[:, :4])
    larges_loglik = cum_logliks.max()
    to_probs = lambda loglik: to_probabilities(loglik, larges_loglik)
    probs = np.apply_along_axis(to_probs, 0, cum_logliks)
    #print('probs')
    #print(probs[:, :4])
    return probs
    #
    # model_data_logliks = np.hstack([
    #     np.apply_along_axis(
    #         lambda x: model_data_loglik(m, tau, x.reshape(1, -1)),
    #         1, X_obs)
    #     for m, tau in zip(models, taus)
    # ]).reshape(X_obs.shape[0], len(models))
    #
    # print(model_data_logliks.shape)
    #
    # cum_model_logliks = np.apply_along_axis(np.cumsum, 0, model_data_logliks)
    # cum_model_probs = np.apply_along_axis(to_probs, 1, cum_model_logliks)
    # return cum_model_probs


def model_t_mu_predictions(
        models: List[TrajectoryModel],
        X_obs: ndarray
        ) -> Tuple[ndarray, ndarray]:

    pos = X_obs[:, 0:2]
    taus = [
        predict(m.g, pos)[0]
        for m in models
    ]
    t_mu_mus, t_mu_sigmas = zip(*[
        predict(m.h, tau.reshape(-1, 1))
        for m, tau in
        zip(models, taus)
    ])

    t_mu_mus = np.hstack(t_mu_mus)
    t_mu_sigmas = np.hstack(t_mu_sigmas)
    return t_mu_mus, t_mu_sigmas


def t_prediction(t_mu_mus: ndarray, t_mu_sigmas: ndarray):
    sigma_h = 1
    t_mus = t_mu_mus
    t_sigmas = np.repeat(sigma_h, t_mu_sigmas.shape[0]).reshape(-1, 1)
    return t_mus, t_sigmas


def most_probable_model_predictor(
        models: List[TrajectoryModel], X_obs: ndarray) -> Tuple[ndarray, int]:
    cum_probs = model_cum_probs(models, X_obs)
    weights = cum_probs[:, -1]  # model_weights(models, X_obs)
    most_probable_model = models[weights.argmax()]
    print('most probable is', weights.argmax())
    latest_pos = X_obs[-1, 0:2].reshape(-1, 2)
    t_mu_mu, t_mu_sigma = model_t_mu_predictions(
        [most_probable_model], latest_pos
    )
    t_pred, _ = t_prediction(t_mu_mu, t_mu_sigma)
    return t_pred, most_probable_model.traj


def model_p_y_loglik(model: TrajectoryModel, tau: ndarray, vel: ndarray):
    mu_p_y1, var_p_y1 = predict(model.f_p_y_1, tau)
    mu_p_y2, var_p_y2 = predict(model.f_p_y_2, tau)
    mu_p_y, var_p_y = combine(
        np.hstack((mu_p_y1, mu_p_y2)),
        np.hstack((var_p_y1, var_p_y2))
    )

    v_y = vel[:, 1].reshape(-1, 1)
    sigma = np.diag(var_p_y.reshape(-1))
    return loglik(v_y, mu_p_y, sigma)


def model_cum_gp_likelihoods(models: List[TrajectoryModel], X_obs: ndarray):
    n_models = 4
    logliks = np.empty((n_models, len(models), X_obs.shape[0]))
    for i, m in enumerate(models):
        for j in range(X_obs.shape[0]):
            pos, vel = X_obs[j, 0:2].reshape(-1, 2), X_obs[j, 2:4]
            tau, _ = predict(m.g, pos)
            p_x, p_y = pos.T[0].reshape(-1, 1), pos.T[1].reshape(-1, 1)
            v_x, v_y = vel[0].reshape(-1, 1), vel[1].reshape(-1, 1)

            mu_p_x, var_p_x = combined_prediction(tau, m.f_p_x_1, m.f_p_x_2)
            logliks[0][i][j] = loglik(p_x, mu_p_x, var_p_x)
            mu_p_y, var_p_y = combined_prediction(tau, m.f_p_y_1, m.f_p_y_2)
            logliks[1][i][j] = loglik(p_y, mu_p_y, var_p_y)
            mu_v_x, var_v_x = combined_prediction(tau, m.f_v_x_1, m.f_v_x_2)
            logliks[2][i][j] = loglik(v_x, mu_v_x, var_v_x)
            mu_v_y, var_v_y = combined_prediction(tau, m.f_v_y_1, m.f_v_y_2)
            logliks[3][i][j] = loglik(v_y, mu_v_y, var_v_y)

    return logliks


# MODEL PLOTTING


def plot_with_combination(
        ax, grid: ndarray,
        f1: FunctionModel,
        f2: FunctionModel,
        scale: Callable[[ndarray], ndarray]):
    color = default_color(0)
    fuse_color = default_color(6)

    mu1, var1 = predict(f1, grid.reshape(-1, 1))
    mu2, var2 = predict(f2, grid.reshape(-1, 1))
    mu1, mu2 = scale(mu1), scale(mu2)
    combined_mu_fpx, combined_var_fpx = combine(
        np.hstack((mu1, mu2)),
        np.hstack((var1, var2))
    )

    plot_posterior(
        ax, grid, combined_mu_fpx, np.sqrt(combined_var_fpx),
        label=r'Combination of pseudo $f_{p_x}$', color=fuse_color
    )

    plot_posterior(
        ax, grid, mu1, np.sqrt(var1),
        label=r'Pseudo $f_{p_x}$', color=color
    )

    plot_posterior(
        ax, grid, mu2, np.sqrt(var2),
        label=None, color=color
    )


def make_grid(xmin, xmax, res=100, padding=0):
    delta = xmax - xmin
    return np.linspace(
        xmin - padding * delta,
        xmax + padding * delta,
        round(delta * res)
    )


def plot_gp_likelihoods(tau_grid: ndarray, likelihoods: ndarray, linestyle):
    n_gps = 2
    _, axs = plot_grid(n_gps, n_gps)
    for model_n in range(likelihoods.shape[1]):
        axs[0, 0].plot(
            tau_grid, likelihoods[0][model_n],
            color=default_color(model_n),
            linestyle=linestyle(model_n))
        axs[0, 1].plot(
            tau_grid, likelihoods[1][model_n],
            color=default_color(model_n),
            linestyle=linestyle(model_n))
        axs[1, 0].plot(
            tau_grid, likelihoods[2][model_n],
            color=default_color(model_n),
            linestyle=linestyle(model_n))
        axs[1, 1].plot(
            tau_grid, likelihoods[3][model_n],
            color=default_color(model_n),
            linestyle=linestyle(model_n))

    for ax1 in axs:
        for ax2 in ax1:
            ax2.set_xlabel(r'$\tau$')
            ax2.set_ylabel(r'$p(\mathcal{M}_k \vert X_{1:n})$')

    title = r'Cumulative probability of {}'
    axs[0, 0].set_title(title.format('$f_{p_x}$'))
    axs[0, 1].set_title(title.format('$f_{p_y}$'))
    axs[1, 0].set_title(title.format('$f_{v_x}$'))
    axs[1, 1].set_title(title.format('$f_{v_y}$'))


def plot_model_functions(m: TrajectoryModel, normaliser: SegmentNormaliser):
    n_rows, n_cols = 5, 2
    _, axs = plot_grid(n_rows, n_cols)

    f_p_x_ax = axs[0][0]
    f_p_y_ax = axs[0][1]
    f_v_x_ax = axs[1][0]
    f_v_y_ax = axs[1][1]
    f_p_x_1_ax = axs[2][0]
    f_p_x_2_ax = axs[2][1]
    f_v_x_1_ax = axs[3][0]
    f_v_x_2_ax = axs[3][1]
    g_ax = axs[4][0]

    plot_function(m.f_p_x, normaliser=normaliser, ax=f_p_x_ax)
    f_p_x_ax.set_title(r'$f_{p_x}$')
    plot_function(m.f_p_y, normaliser=normaliser, ax=f_p_y_ax)
    f_p_y_ax.set_title(r'$f_{p_y}$')
    plot_function(m.f_v_x, normaliser=normaliser, ax=f_v_x_ax)
    f_v_x_ax.set_title(r'$f_{v_x}$')
    plot_function(m.f_v_y, normaliser=normaliser, ax=f_v_y_ax)
    f_v_y_ax.set_title(r'$f_{v_y}$')
    #plot_function(m.f_p_x_1, f_p_x_1_ax)
    #plot_function(m.f_p_x_2, f_p_x_2_ax)
    #plot_function(m.f_v_x_1, f_v_x_1_ax)
    #plot_function(m.f_v_x_2, f_v_x_2_ax)
    g_ax.set_title(r'$g$')
    plot_function(m.g, g_ax)

def plot_model(
        m: TrajectoryModel,
        data: DataFrame,
        normaliser: SegmentNormaliser):

    n_rows = 5
    n_cols = 2
    _, axs = plot_grid(n_rows, n_cols)
    x_y_ax = axs[0][0]
    g_ax = axs[0][1]
    f_p_x_ax = axs[2][0]
    f_p_y_ax = axs[2][1]
    f_v_x_ax = axs[3][0]
    f_v_y_ax = axs[3][1]
    h_ax = axs[4][0]

    # Input data
    pos_x = normaliser.unnormalise_x(data[F_CODOMAIN[0]])
    pos_y = normaliser.unnormalise_y(data[F_CODOMAIN[1]])
    x_y_ax.scatter(pos_x, pos_y)
    x_y_ax.set_title('Input data')
    x_y_ax.set_aspect('equal', 'box')
    x_y_ax.set_xlabel(r'$p_x$')
    x_y_ax.set_ylabel(r'$p_y$')

    # h for input data
    x = data[F_CODOMAIN].values[:, :2]
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
    #axs[1][0].set_aspect('equal', 'box')

    # Training data
    plot_function(m.g, ax=g_ax)
    g_ax.set_aspect('equal', 'box')
    g_ax .set_title(r'Inverse model function $g: (x, y) \mapsto \tau$')
    g_ax .set_xlabel(r'$p_x$')
    g_ax .set_ylabel(r'$p_y$')
    #axs[0][1].axis('scaled')

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
    axs[1][1].set_title(r'Predicted $\tau$ of training data')

    # Tau grid
    xmin, xmax = m.f_p_x_1.model.X.min(), m.f_p_x_1.model.X.max()
    tau_grid = make_grid(xmin, xmax, 100)

    plot_with_combination(
        f_p_x_ax, tau_grid,
        m.f_p_x_1, m.f_p_x_2,
        lambda x: normaliser.unnormalise_x(x)*KM_RATIO
    )
    p_x_1 = normaliser.unnormalise_x(m.f_p_x_1.model.Y)*KM_RATIO
    p_x_2 = normaliser.unnormalise_x(m.f_p_x_2.model.Y)*KM_RATIO
    plot_data(f_p_x_ax, m.f_p_x_1.model.X, p_x_1, label=None)
    plot_data(f_p_x_ax, m.f_p_x_2.model.X, p_x_2)
    f_p_x_ax.set_title(r'Model function $f_{p_x}: \tau \mapsto p_x$')
    f_p_x_ax.set_xlabel(r'$\tau$')
    f_p_x_ax.set_ylabel(r'$p_x$')
    f_p_x_ax.legend()

    plot_with_combination(
        f_p_y_ax, tau_grid,
        m.f_p_y_1, m.f_p_y_2,
        lambda x: normaliser.unnormalise_y(x)*KM_RATIO
    )
    p_y_1 = normaliser.unnormalise_y(m.f_p_y_1.model.Y)*KM_RATIO
    p_y_2 = normaliser.unnormalise_y(m.f_p_y_2.model.Y)*KM_RATIO
    plot_data(f_p_y_ax, m.f_p_y_1.model.X, p_y_1, label=None)
    plot_data(f_p_y_ax, m.f_p_y_2.model.X, p_y_2)
    f_p_y_ax.set_title(r'Model function $f_{p_y}: \tau \mapsto p_y$')
    f_p_y_ax.set_xlabel(r'$\tau$')
    f_p_y_ax.set_ylabel(r'$p_y$')
    f_p_y_ax.legend()

    plot_with_combination(
        f_v_x_ax, tau_grid,
        m.f_v_x_1, m.f_v_x_2,
        lambda x: normaliser.unnormalise_dx(x)*KM_H_RATIO
    )
    v_x_1 = normaliser.unnormalise_dx(m.f_v_x_1.model.Y)*KM_H_RATIO
    v_x_2 = normaliser.unnormalise_dx(m.f_v_x_2.model.Y)*KM_H_RATIO
    plot_data(f_v_x_ax, m.f_v_x_1.model.X, v_x_1, label=None)
    plot_data(f_v_x_ax, m.f_v_x_2.model.X, v_x_2)
    f_v_x_ax.set_title(r'Model function $f_{v_x}: \tau \mapsto v_x$')
    f_v_x_ax.set_xlabel(r'$\tau$')
    f_v_x_ax.set_ylabel(r'$v_x$')
    f_v_x_ax.legend()

    plot_with_combination(
        f_v_y_ax, tau_grid,
        m.f_v_y_1, m.f_v_y_2,
        lambda x: normaliser.unnormalise_dy(x)*KM_H_RATIO
    )
    v_y_1 = normaliser.unnormalise_dy(m.f_v_y_1.model.Y)*KM_H_RATIO
    v_y_2 = normaliser.unnormalise_dy(m.f_v_y_2.model.Y)*KM_H_RATIO
    plot_data(f_v_y_ax, m.f_v_y_1.model.X, v_y_1, label=None)
    plot_data(f_v_y_ax, m.f_v_y_2.model.X, v_y_2)
    f_v_y_ax.set_title(r'Model function $f_{v_y}: \tau \mapsto v_y$')
    f_v_y_ax.set_xlabel(r'$\tau$')
    f_v_y_ax.set_ylabel(r'$v_y$')
    f_v_y_ax.legend()

    # h
    t_mu, t_var = predict(m.h, tau_grid.reshape(-1, 1))
    plot_posterior(
        h_ax, tau_grid, t_mu, np.sqrt(t_var),
        color=default_color(0), label=r'$h$'
    )
    plot_data(h_ax, m.h.model.X, m.h.model.Y)
    h_ax.set_title(r'Arrival time function $h: \tau \mapsto y$')
    h_ax.set_xlabel(r'$\tau$')
    h_ax.set_ylabel(r'$t$')
    h_ax.legend()


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

    def plotter(i):
        def plot(ax, x, y):
            ax.plot(
                x, y,
                color=default_color(i),
                label='Trajectory ' + str(i),
                linestyle=linestyle(i)
            )
        return plot

    pos = seg[['x', 'y']].values.reshape(-1, 2)
    for i, model in zip(model_ixs, models):
        plot = plotter(i)
        tau, _ = predict(model.g, pos)
        plot(tau_pred_ax, progression, tau.T[0])

        time_left, _ = predict(model.h, tau)
        plot(time_left_pred_ax, tau, time_left)

        p_x_1, _ = predict(model.f_p_x_1, tau)
        plot(pos_x_pred_ax, tau, p_x_1)

        p_x_2, _ = predict(model.f_p_x_2, tau)
        plot(pos_x_pred_ax, tau, p_x_2)

        p_y_1, _ = predict(model.f_p_y_1, tau)
        plot(pos_y_pred_ax, tau, p_y_1)

        v_x_1, _ = predict(model.f_v_x_1, tau)
        plot(vel_x_pred_ax, tau, v_x_1)

        v_y_1, _ = predict(model.f_v_y_1, tau)
        plot(vel_y_pred_ax, tau, v_y_1)

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


def combined_prediction(grid, f1, f2):
    mu_1, var_1 = predict(f1, grid)
    mu_2, var_2 = predict(f2, grid)
    return combine(
        np.hstack((mu_1, mu_2)),
        np.hstack((var_1, var_2))
    )


def plot_most_probable_models(
        models,
        model_weights,
        cum_model_probs,
        data,
        linestyle,
        n_models=3):

    model_indicies = range(len(models))
    most_probable_models = sorted(
        zip(model_weights, model_indicies, models),
        key=lambda x: x[0],
        reverse=True
    )

    _, ax = plot_grid(3, 2)
    xmin, xmax = 0, data.iloc[-1].tau
    tau_grid = make_grid(xmin, xmax, 100, padding=0)
    p_x_ax = ax[0][0]
    p_y_ax = ax[0][1]
    v_x_ax = ax[1][0]
    v_y_ax = ax[1][1]
    t_ax = ax[2][0]
    probs_ax = ax[2][1]
    label = lambda i: r'$\mathcal{M}_{' + str(i) + '}$'

    tau = tau_grid.reshape(-1, 1)
    for w, i, m in most_probable_models[:n_models]:
        mu_p_x, var_p_x = combined_prediction(
            tau, m.f_p_x_1, m.f_p_x_2
        )
        plot_with_credible_bands(
            p_x_ax, tau_grid, mu_p_x, var_p_x,
            label(i), default_color(i), linestyle(i)
        )

        mu_p_y, var_p_y = combined_prediction(
            tau, m.f_p_y_1, m.f_p_y_2
        )
        plot_with_credible_bands(
            p_y_ax, tau_grid, mu_p_y, var_p_y,
            label(i), default_color(i), linestyle(i)
        )

        mu_v_x, var_v_x = combined_prediction(
            tau, m.f_v_x_1, m.f_v_x_2
        )
        plot_with_credible_bands(
            v_x_ax, tau_grid, mu_v_x, var_v_x,
            label(i), default_color(i), linestyle(i)
        )

        mu_v_y, var_v_y = combined_prediction(
            tau, m.f_v_y_1, m.f_v_y_2
        )
        plot_with_credible_bands(
            v_y_ax, tau_grid, mu_v_y, var_v_y,
            label(i), default_color(i), linestyle(i)
        )

        mu_h, var_h = predict(m.h, tau)
        plot_with_credible_bands(
            t_ax, tau_grid, mu_h, var_h,
            label(i), default_color(i), linestyle(i)
        )

    # prob_grid = np.linspace(xmin, xmax, data.shape[0])
    # for w, i, m in most_probable_models[:n_models * 2]:
    #     probs = cum_model_probs[i, :]
    #     probs_ax.plot(
    #         prob_grid, probs,
    #         label=label(i),
    #         color=default_color(i),
    #         linestyle=linestyle(i)
    #     )

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
        #model_probs,
        model_ixs,
        cum_model_probs,
        linestyle):
    _, axs = plot_grid(1, 1)
    #model_prob_ax = axs[0]
    cum_model_prob_ax = axs
    print(len(model_ixs), cum_model_probs.T.shape)
    for i, cum_prob in zip(model_ixs, cum_model_probs):
        print(tau_grid.shape, cum_prob.shape)
        # model_prob = model_probs[i, :]
        # model_cum_prob = cum_model_probs[i, :]
        # print(i, model_prob.shape, cum_model_probs.shape)
        # model_prob_ax.set_title('Model probabilities')
        # model_prob_ax.set_xlabel(r'$\tau$')
        # model_prob_ax.set_ylabel(r'$p(\mathcal{m}_k | X_{obs}$')
        # model_prob_ax.plot(
        #     tau_grid, model_prob,
        #     color=default_color(i),
        #     label='Trajectory' + str(i),
        #     linestyle=linestyle(i)
        # )

        cum_model_prob_ax.set_title(r'Cumulative model probabilities')
        cum_model_prob_ax.set_xlabel(r'$\tau$')
        cum_model_prob_ax.set_ylabel(r'$p(\mathcal{m}_k | X_{obs})$')
        cum_model_prob_ax.plot(
            tau_grid, cum_prob,
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

    prior_components = [
        norm.pdf(t_grid, mu, np.sqrt(sigma))
        for mu, sigma in
        zip(mus, sigmas)
    ]

    posterior_components = [
        w * norm.pdf(t_grid, mu, np.sqrt(sigma))
        for mu, sigma, w in
        zip(mus, sigmas, weights)
    ]

    prior = reduce(np.add, prior_components).reshape(t_grid.shape[0])
    prior_sum = np.sum(prior)
    prior = prior / prior_sum
    prior_mean = np.sum(prior * t_grid)
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
    # prior_component_ax.legend()

    posterior = reduce(np.add, posterior_components).reshape(t_grid.shape[0])
    posterior_sum = np.sum(posterior)
    posterior = posterior / posterior_sum
    posterior_mean = np.sum(posterior * t_grid)

    posterior_ax.plot(t_grid, posterior, label='Posterior density',
                      color=default_color(0))
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
    # posterior_component_ax.legend()


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