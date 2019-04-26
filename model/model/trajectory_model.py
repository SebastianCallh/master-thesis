"""A single trajectory model, consisting of many function models.
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
    plot_posterior, learn_function, learn_sparse

from .plotting import plot_grid, default_color, plot_data, plot_marker, \
    plot_mixture, grid_for
from .pre_process import duplicate, SegmentNormaliser

F_CODOMAIN = ['x', 'y', 'dx', 'dy']
KM_H_RATIO = 3.6
KM_RATIO = 1e-3
INTEGRATION_DELTA = 1e-3

def model_label(k):
    return r'$\mathcal{M}_{' + str(k) + '}$'

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
    z = x - mu
    return -0.5 * z * (1 / sigma) * z \
           - 0.5 * np.log(sigma)

    #z = x - mu
    #return -0.5*z.T.dot(inv(sigma)).dot(z) \
    #    - 0.5*np.log(det(sigma))


def model_data_loglik(
        model: TrajectoryModel,
        tau: ndarray, X_obs: ndarray) -> ndarray:

    #print('computing loglik fotr', tau.shape, X_obs.shape)

    def loglik(x: ndarray, mu: ndarray, sigma: ndarray):
        z = x - mu
        #return -0.5 * z * (1 / sigma) * z \
        #       - 0.5 * np.log(np.prod(sigma))

        return -0.5*z.T.dot(inv(sigma)).dot(z) \
            - 0.5*np.log(det(sigma))

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
        route_n: int, seg_n: int, traj_n: int,
        f_p_sigma_n: float,
        f_v_sigma_n: float,
        g_sigma_n: float,
        h_sigma_n: float,
        delta_xy: float,
        delta_p: float,
        delta_v: float,
        hyperparams: ndarray,
        # var_hyperparams: ndarray,
        # lin_hyperparams: ndarray,
        n_restarts=3) -> TrajectoryModel:

    rbf_kernel = lambda: GPy.kern.RBF(1)
    matern_kernel = lambda: GPy.kern.Matern32(1)  # + GPy.kern.Bias(1)

    N = data.shape[0]
    f_p_inducing = round(N*.3)
    f_v_inducing = round(N*.3)
    g_inducing = round(N*.15)
    h_inducing = round(N*.3)

    def fpx_kernel():
        k = matern_kernel()
        # k.lengthscale.set_prior(
        #     gamma_prior(hyperparams['fpx_ls'][0], hyperparams['fpx_ls'][2])
        # )
        # k.variance.set_prior(
        #     gamma_prior(hyperparams['fpx_var'][0], hyperparams['fpx_var'][2])
        # )
        return k

    # p_x
    tau = data.tau.values.reshape(-1, 1)
    p_x = data.x.values.reshape(-1, 1)
    f_p_x = learn_function(
        tau, p_x, fpx_kernel(),
        f_p_sigma_n, n_restarts
    )

    p_x_grid = f_p_x.model.X
    # [
    #     (f_p_x.model.Z >= 0.0) & (f_p_x.model.Z <= 1.0)
    # ].values.reshape(-1, 1)
    mu_p_x, _ = predict(f_p_x, p_x_grid)

    p_x_1 = (mu_p_x + delta_p).reshape(-1, 1)
    f_p_x_1 = learn_function(
        p_x_grid, p_x_1, fpx_kernel(),
        f_p_sigma_n, n_restarts
    )
    p_x_2 = (mu_p_x - delta_p).reshape(-1, 1)
    f_p_x_2 = learn_function(
        p_x_grid, p_x_2,  fpx_kernel(),
        f_p_sigma_n, n_restarts
    )

    def fpy_kernel():
        k = matern_kernel()
        # k.lengthscale.set_prior(
        #     gamma_prior(hyperparams['fpy_ls'][0], hyperparams['fpy_ls'][2])
        # )
        # k.variance.set_prior(
        #     gamma_prior(hyperparams['fpy_var'][0], hyperparams['fpy_var'][2])
        # )
        return k

    # p_x
    p_y = data.y.values.reshape(-1, 1)
    f_p_y = learn_function(
        tau, p_y, fpy_kernel(),
        f_p_sigma_n, n_restarts
    )
    p_y_grid = f_p_y.model.X
    # [
    #     (f_p_y.model.Z >= 0) & (f_p_y.model.Z <= 1.0)
    # ].values.reshape(-1, 1)
    mu_p_y, _ = predict(f_p_y, p_y_grid)

    p_y_1 = mu_p_y + delta_p
    f_p_y_1 = learn_function(
        p_y_grid, p_y_1, fpy_kernel(),
        f_p_sigma_n, n_restarts
    )

    p_y_2 = mu_p_y - delta_p
    f_p_y_2 = learn_function(
        p_y_grid, p_y_2, fpy_kernel(),
        f_p_sigma_n, n_restarts
    )

    def fvx_kernel():
        k = matern_kernel()
        # k.lengthscale.set_prior(
        #     gamma_prior(hyperparams['fvx_ls'][0], hyperparams['fvx_ls'][2])
        # )
        # k.variance.set_prior(
        #     gamma_prior(hyperparams['fvx_var'][0], hyperparams['fvx_var'][2])
        # )
        return k

    # Do the same for velocity
    v_x = data.dx.values.reshape(-1, 1)
    f_v_x = learn_function(
        tau, v_x, fvx_kernel(),
        f_v_sigma_n, n_restarts
    )
    v_x_grid = f_v_x.model.X
    mu_v_x, _ = predict(f_v_x, v_x_grid)

    v_x_1 = mu_v_x + delta_v
    f_v_x_1 = learn_function(
        v_x_grid, v_x_1, fvx_kernel(),
        f_v_sigma_n, n_restarts
    )
    v_x_2 = mu_v_x - delta_v
    f_v_x_2 = learn_function(
        v_x_grid, v_x_2, fvx_kernel(),
        f_v_sigma_n, n_restarts
    )

    def fvy_kernel():
        k = matern_kernel()
        # k.lengthscale.set_prior(
        #     gamma_prior(hyperparams['fvy_ls'][0], hyperparams['fvy_ls'][2])
        # )
        # k.variance.set_prior(
        #     gamma_prior(hyperparams['fvy_var'][0], hyperparams['fvy_var'][2])
        # )
        return k
    print(fvy_kernel())
    v_y = data.dy.values.reshape(-1, 1)
    f_v_y = learn_function(
        tau, v_y, fvy_kernel(),
        f_v_sigma_n, n_restarts
    )
    v_y_grid = f_v_y.model.X
    mu_v_y, _ = predict(f_v_y, v_y_grid)

    v_y_1 = mu_v_y + delta_v
    f_v_y_1 = learn_function(
        v_y_grid, v_y_1, fvy_kernel(),
        f_v_sigma_n, n_restarts
    )

    v_y_2 = mu_v_y - delta_v
    f_v_y_2 = learn_function(
        v_y_grid, v_y_2, fvy_kernel(),
        f_v_sigma_n, n_restarts
    )

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

    def g_kernel():
        k_mat = GPy.kern.RBF(input_dim=2, ARD=False)
        k_lin = GPy.kern.Linear(input_dim=2, ARD=False)
        # k_mat.lengthscale.set_prior(
        #    gamma_prior(hyperparams['g_ls'][0], hyperparams['g_ls'][2])
        # )
        # k_mat.variance.set_prior(
        #    gamma_prior(hyperparams['g_var'][0], hyperparams['g_var'][2])
        # )
        # k_lin.variances.set_prior(
        #    gamma_prior(hyperparams['g_lin'][0], hyperparams['g_lin'][2])
        # )
        return k_mat + k_lin

    g = learn_sparse(
        g_pos, g_tau, g_kernel(),
        g_inducing, g_sigma_n, n_restarts
    )

    time_left = data.time_left.values.reshape(-1, 1)

    def h_kernel():
        k_mat = GPy.kern.Matern32(input_dim=1, ARD=False)
        k_lin = GPy.kern.Linear(input_dim=1, ARD=False)
        # k_mat.lengthscale.set_prior(
        #     gamma_prior(hyperparams['h_ls'][0], hyperparams['h_ls'][2])
        # )
        # k_mat.variance.set_prior(
        #     gamma_prior(hyperparams['h_var'][0], hyperparams['h_var'][2])
        # )
        # k_lin.variances.set_prior(
        #     gamma_prior(hyperparams['h_lin'][0], hyperparams['h_lin'][2])
        # )
        return k_mat + k_lin

    h = learn_function(
        tau, time_left, h_kernel(),
        h_sigma_n, n_restarts
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

#
# def model_weights(models: List[TrajectoryModel], X_obs: ndarray) -> ndarray:
#     n_models = 4
#     logliks = np.empty((n_models, len(models)))
#     pos, vel = X_obs[:, 0:2].reshape(-1, 2), X_obs[:, 2:4].reshape(-1, 2)
#     for i, m in enumerate(models):
#         tau, _ = predict(m.g, pos)
#         p_x, p_y = pos[:, 0].reshape(-1, 1), pos[:, 1].reshape(-1, 1)
#         v_x, v_y = vel[:, 0].reshape(-1, 1), vel[:, 1].reshape(-1, 1)
#         mu_p_x, var_p_x = combined_prediction(tau, m.f_p_x_1, m.f_p_x_2)
#         logliks[0][i] = loglik(p_x, mu_p_x, np.diag(var_p_x.T[0]))
#         mu_p_y, var_p_y = combined_prediction(tau, m.f_p_y_1, m.f_p_y_2)
#         logliks[1][i] = loglik(p_y, mu_p_y, np.diag(var_p_y.T[0]))
#         mu_v_x, var_v_x = combined_prediction(tau, m.f_v_x_1, m.f_v_x_2)
#         logliks[2][i] = loglik(v_x, mu_v_x, np.diag(var_v_x.T[0]))
#         mu_v_y, var_v_y = combined_prediction(tau, m.f_v_y_1, m.f_v_y_2)
#         logliks[3][i] = loglik(v_y, mu_v_y, np.diag(var_v_y.T[0]))
#
#     probs = np.apply_along_axis(to_prob, 1, logliks).sum(axis=0)
#     #probs = np.apply_along_axis(np.sum, 0, probs)
#     return probs / probs.sum()


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
    to_probs = lambda loglik: to_probabilities(loglik)
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


def trajectory_arrival_time_prior(
        m: TrajectoryModel) -> ndarray:
    tau0 = np.array([0]).reshape(1, 1)
    mu, pred_var = predict(m.h, tau0)
    var = pred_var + segment_uncertainty(m.segment)
    return np.array((mu, var))


def arrival_time_prediction(
        m: TrajectoryModel, X_obs: ndarray
        ) -> Tuple[float, float]:
    pos = X_obs[:, :2].reshape(-1, 2)
    tau, _ = predict(m.g, pos)
    mu_t, var_t = predict(m.h, tau.reshape(-1, 1))
    var_mp = model_uncertainty(m, tau[-1], INTEGRATION_DELTA)
    return mu_t[-1], (var_t + var_mp)[-1]


def segment_uncertainty(seg_n: int) -> float:
    n_trajs_estimated_from = 60
    sigmas = [40, 13, 14, 12, 11, 12, 21, 12, 25, 24, 20]
    return sigmas[seg_n - 1] / n_trajs_estimated_from


def model_uncertainty(m: TrajectoryModel, tau0: float, dx: float) -> float:
    return segment_uncertainty(m.segment)

    # N = round(1/dx)
    # tau = np.linspace(0, tau0, N).reshape(-1, 1)
    # mu_v_x, var_v_x = combined_prediction(tau, m.f_v_x_1, m.f_v_x_2)
    # mu_v_y, var_v_y = combined_prediction(tau, m.f_v_y_1, m.f_v_y_2)
    # g = lambda y: np.gradient(y.reshape(-1))
    # s = lambda grad: np.sum(np.sqrt(1 + grad**2))
    #
    # #"import matplotlib.pyplot as plt
    # #_, ax = plt.subplots(1, 1)
    # g_v_x, g_v_y = g(mu_v_x), g(mu_v_y)
    # s_v_x, s_v_y = s(g_v_x), s(g_v_y)
    # sigma_mp = segment_uncertainty(m.segment)
    # tau_grid = tau.reshape(-1)
    # #ax.plot(tau_grid, mu_v_x.reshape(-1), label='predicted v_x')
    # #ax.plot(tau_grid, g_v_x.reshape(-1), label='gradients')
    # #ax.plot(tau.reshape(-1), cum_s(g_v_x).reshape(-1), label='arc length')
    # #ax.legend()
    # return sigma_mp # float(s_v_x / sigma_mp)



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




# MODEL PLOTTING


def plot_with_combination(
        ax, grid: ndarray,
        f1: FunctionModel,
        f2: FunctionModel,
        label: str,
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
        label=r"$f^{'}_{" + label + "}$", color=fuse_color
    )

    plot_posterior(
        ax, grid, mu1, np.sqrt(var1),
        label=r"Pseudo $f_{" + label + "}$", color=color
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


def plot_gp_likelihoods(tau_grid: ndarray, gp_logliks: ndarray, linestyle):
    _, axs = plot_grid(2, 2)
    for model_n in range(gp_logliks.shape[1]):
        axs[0, 0].plot(
            tau_grid, gp_logliks[0][model_n],
            color=default_color(model_n),
            linestyle=linestyle(model_n))
        axs[0, 1].plot(
            tau_grid, gp_logliks[1][model_n],
            color=default_color(model_n),
            linestyle=linestyle(model_n))
        axs[1, 0].plot(
            tau_grid, gp_logliks[2][model_n],
            color=default_color(model_n),
            linestyle=linestyle(model_n))
        axs[1, 1].plot(
            tau_grid, gp_logliks[3][model_n],
            color=default_color(model_n),
            linestyle=linestyle(model_n))

    fs = [
        r"f^{'}_{p_x}", r"f^{'}_{p_y}",
        r"f^{'}_{v_x}", r"f^{'}_{v_y}",
        " "
    ]
    title = lambda x: r'Pointwise log likelihood of $' + x + '$'
    for f, ax in zip(fs, axs.flatten()):
        ax.set_title(title(f))
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$p(\mathcal{M}_k \vert ' + f + 'X_{1:n})$')
        ax.plot([], [], label='True model', color="#202020")
        ax.plot([], [], linestyle='--', label='Other models', color="#202020")
        ax.legend()

    # Plot totals
    # model_probs = to_model_probabilities(gp_logliks)
    # for i, p in enumerate(model_probs):
    #     axs[2, 0].plot(
    #         tau_grid, model_probs[i],
    #         label=model_label(i),
    #         color=default_color(i),
    #         linestyle=linestyle(i)
    #     )
    # axs[2, 0].set_title('Cumulative model probabilities')
    #axs[2, 1].axis('off')


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
    pos = data[['x', 'y']].values
    tau, _ = predict(m.g, pos)
    df = pd.DataFrame({
        'prediction': tau.T[0],
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
    g_ax.set_title(r'Inverse model function $g: (x, y) \mapsto \tau$')
    g_ax.set_xlabel(r'$p_x$')
    g_ax.set_ylabel(r'$p_y$')
    #axs[0][1].axis('scaled')

    # H for training data
    tau, _ = m.g.model.predict(m.g.model.X)
    df = pd.DataFrame({
        'prediction': tau.T[0],
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
    xmin, xmax = m.f_p_x.model.X.min(), m.f_p_x.model.X.max()
    tau_grid = data.tau.values  # make_grid(xmin, xmax, 100)
    unnorm_x = lambda x: normaliser.unnormalise_x(x)*KM_RATIO
    plot_with_combination(
        f_p_x_ax, tau_grid,
        m.f_p_x_1, m.f_p_x_2,
        'f_{p_x}',
        unnorm_x
    )
    #p_x_1, p_x_2 = unnorm_x(m.f_p_x_1.model.Y), unnorm_x(m.f_p_x_2.model.Y)
    #plot_data(f_p_x_ax, m.f_p_x.model.X, unnorm_x(m.f_p_x.model.Y))
    #plot_data(f_p_x_ax, m.f_p_x_2.model.X, p_x_2)
    #m.f_p_x.model.plot_inducing(ax=f_p_x_ax)

    plot_data(f_p_x_ax, data.tau, unnorm_x(data.x))
    f_p_x_ax.set_title(r'Model function $f_{p_x}: \tau \mapsto p_x$')
    f_p_x_ax.set_xlabel(r'$\tau$')
    f_p_x_ax.set_ylabel(r'$p_x$')
    f_p_x_ax.set_xlim(0, 1)
    f_p_x_ax.legend(loc=2)

    unnorm_y = lambda y: normaliser.unnormalise_y(y)*KM_RATIO
    plot_with_combination(
        f_p_y_ax, tau_grid,
        m.f_p_y_1, m.f_p_y_2,
        'f_{p_y}',
        unnorm_y
    )
    #p_y_1, p_y_2 = unnorm_y(m.f_p_y_1.model.Y), unnorm_y(m.f_p_y_2.model.Y)
    #plot_data(f_p_y_ax, m.f_p_y_1.model.X, p_y_1, label=None)
    #plot_data(f_p_y_ax, m.f_p_y_2.model.X, p_y_2)
    #m.f_p_y.model.plot_inducing(ax=f_p_y_ax)
    plot_data(f_p_y_ax, data.tau, unnorm_y(data.y))
    f_p_y_ax.set_title(r'Model function $f_{p_y}: \tau \mapsto p_y$')
    f_p_y_ax.set_xlabel(r'$\tau$')
    f_p_y_ax.set_ylabel(r'$p_y$')
    f_p_y_ax.set_xlim(0, 1)
    f_p_y_ax.legend(loc=3)

    unnorm_dx = lambda dx: normaliser.unnormalise_dx(dx)*KM_H_RATIO
    plot_with_combination(
        f_v_x_ax, tau_grid,
        m.f_v_x_1, m.f_v_x_2,
        'f_{v_x}',
        unnorm_dx
    )
    v_x_1, v_x_2 = unnorm_dx(m.f_v_x_1.model.Y), unnorm_dx(m.f_v_x_2.model.Y)
    #plot_data(f_v_x_ax, m.f_v_x_1.model.X, v_x_1, label=None)
    #plot_data(f_v_x_ax, m.f_v_x_2.model.X, v_x_2)
    #m.f_v_x.model.plot_inducing(ax=f_v_x_ax)
    plot_data(f_v_x_ax, data.tau, unnorm_dx(data.dx))
    f_v_x_ax.set_title(r'Model function $f_{v_x}: \tau \mapsto v_x$')
    f_v_x_ax.set_xlabel(r'$\tau$')
    f_v_x_ax.set_ylabel(r'$v_x$')
    f_v_x_ax.set_xlim(0, 1)
    f_v_x_ax.legend()

    unnorm_dy = lambda dy: normaliser.unnormalise_dy(dy)*KM_H_RATIO
    plot_with_combination(
        f_v_y_ax, tau_grid,
        m.f_v_y_1, m.f_v_y_2,
        'f_{v_y}',
        unnorm_dy
    )

    #v_y_1, v_y_2 = unnorm_dy(m.f_v_y_1.model.Y), unnorm_dy(m.f_v_y_2.model.Y)
    #plot_data(f_v_y_ax, m.f_v_y_1.model.X, v_y_1, label=None)
    #plot_data(f_v_y_ax, m.f_v_y_2.model.X, v_y_2)
    plot_data(f_v_y_ax, data.tau, unnorm_dy(data.dy))
    #m.f_v_y.model.plot_inducing(ax=f_v_y_ax)
    f_v_y_ax.set_title(r'Model function $f_{v_y}: \tau \mapsto v_y$')
    f_v_y_ax.set_xlabel(r'$\tau$')
    f_v_y_ax.set_ylabel(r'$v_y$')
    f_v_y_ax.set_xlim(0, 1)
    f_v_y_ax.legend()

    # h
    t_mu, t_var = predict(m.h, tau_grid.reshape(-1, 1))
    plot_posterior(
        h_ax, tau_grid, t_mu, np.sqrt(t_var),
        color=default_color(0), label=r'$h$'
    )
    plot_data(h_ax, data.tau, data.time_left)
    #m.h.model.plot_inducing(ax=h_ax)
    h_ax.set_title(r'Arrival time function $h: \tau \mapsto y$')
    h_ax.set_xlabel(r'$\tau$')
    h_ax.set_ylabel(r'$t$')
    h_ax.set_xlim(0, 1)
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


def to_probabilities(likelihoods):
    def to_prob(x):
        x = np.exp(x - x.max())
        return x / x.sum()

    cum_likelihoods = np.apply_along_axis(np.cumsum, 2, likelihoods)
    return np.apply_along_axis(to_prob, 1, cum_likelihoods)





def plot_model_probabilities(
        tau_grid,
        model_ixs,
        cum_model_probs,
        linestyle):
    _, axs = plot_grid(1, 1)
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
        t_grid: ndarray,
        model_ixs: ndarray,
        t_mus: ndarray,
        t_vars: ndarray,
        weights: ndarray,
        time_left: ndarray,
        linestyle):
    _, axs = plot_grid(2, 2)
    prior_t_components_ax = axs[0][0]
    posterior_t_components_ax = axs[0][1]
    prior_t_ax = axs[1][0]
    posterior_t_ax = axs[1][1]
    label = lambda i: r'$\mathcal{M}_{' + str(i) + '}$'
    uniform_weights = np.repeat(1, len(t_mus))
    t_sigmas = np.sqrt(t_vars)

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


def plot_mixture_density(
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

    def add_true_model_legend(ax):
        ax.plot([], [], label='True model', color="#202020")
        ax.plot([], [], linestyle='--', label='Other models',
                      color="#202020")

    prior_components = [
        norm.pdf(t_grid, mu, sigma)
        for mu, sigma in
        zip(mus, sigmas)
    ]

    posterior_components = [
        w * norm.pdf(t_grid, mu, sigma)
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
    add_true_model_legend(prior_component_ax)
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
    add_true_model_legend(posterior_component_ax)
    posterior_component_ax.set_title(r'$\mu_t$ Posterior Components')
    posterior_component_ax.set_xlabel('Seconds')
    posterior_component_ax.set_ylabel('Density')
    # posterior_component_ax.legend()