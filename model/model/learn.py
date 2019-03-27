"""Stuff for learning the model."""

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from GPy.models import SparseGPRegression, GPRegression
from trajectory_model import FunctionModel, TrajectoryModel
from predict import predict
import GPy


def learn_function(
        X: ndarray, Y: ndarray, applyPriors,
        f_type: str, route: int, segment: int, kernel=None,
        z=None, n_restarts=3) -> FunctionModel:
    # print(kernel)
    model = GPRegression(
        X, Y, kernel,  # Z=z,
        normalizer=False
    )
    applyPriors(model)
    # print(model)
    model.optimize_restarts(n_restarts)
    return FunctionModel(
        f_type, model
    )


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
        1 / delta
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


def gamma_prior(mean, var):
    return GPy.priors.Gamma.from_EV(mean, var)


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
