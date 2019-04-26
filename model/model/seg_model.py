"""The entire model, consisting of several trajectory models
"""
from numpy import ndarray, math
import numpy as np
from typing import List, Callable, Tuple

from pandas import DataFrame
from scipy.stats import norm

from .storage import load_models
from .function_model import predict, plot_with_credible_bands
from .plotting import plot_data, default_color, plot_grid, grid_for
from .segment_normaliser import SegmentNormaliser
from .trajectory_model import TrajectoryModel, loglik, combined_prediction, \
    model_label, combine, KM_H_RATIO, KM_RATIO, arrival_time_prediction, \
    INTEGRATION_DELTA, model_uncertainty, F_CODOMAIN, \
    trajectory_arrival_time_prior


class SegmentModel:
    route_n: int
    seg_n: int
    trajectory_models: List[TrajectoryModel]

    def __init__(
            self, route_n: int, seg_n: int,
            trajectory_models: List[TrajectoryModel]) -> None:
        super().__init__()
        self.route_n = route_n
        self.seg_n = seg_n
        self.trajectory_models = trajectory_models

    def __len__(self):
        return len(self.trajectory_models)

    def add_model(self, model: TrajectoryModel) -> None:
        self.trajectory_models.append(model)

    def predict_arrival_time(self, traj: DataFrame) -> float:
        return mixture_of_gps_predictor(self, traj[F_CODOMAIN].values)

    def arrival_time_prior(self) -> ndarray:
        return np.hstack([
            trajectory_arrival_time_prior(m).reshape(1, 2)
            for m in self.trajectory_models
        ]).reshape(-1, 2)

    def arrival_time_posterior(self, obs: ndarray) -> Tuple[ndarray, ndarray]:
        if obs.shape == (0, 0):
            prior_params = self.arrival_time_prior()
            return prior_params[:, 0], prior_params[:, 1]

        pos = obs[:, :2]
        mus, vars = np.ndarray((len(self),)), np.ndarray((len(self),))
        for i, m in enumerate(self.trajectory_models):
            tau, _ = predict(m.g, pos)
            mu, var = predict(m.h, tau.reshape(-1, 1))
            var += model_uncertainty(m, tau[-1], INTEGRATION_DELTA)
            mus[i] = mu.reshape(-1)
            vars[i] = var.reshape(-1)

        return mus, vars


def load_seg_model(
        route_n: int, seg_n: int, limit: int = 10000) -> SegmentModel:
    models = load_models(route_n, seg_n, limit)
    return SegmentModel(route_n, seg_n, models)


def prior_density_from_params(
        x: ndarray,
        model_prior_params: ndarray) -> ndarray:
    assert model_prior_params.shape[1] == 2

    unnormed_density = np.vstack([
        norm.pdf(x, mu, np.sqrt(var))
        for mu, var in model_prior_params
    ]).sum(axis=0).reshape(-1)
    return unnormed_density / unnormed_density.sum()


def gp_logliks(
    model: SegmentModel,
    X_obs: ndarray) -> ndarray:
    n_models = 4
    models = model.trajectory_models
    logliks = np.empty((n_models, len(models), X_obs.shape[0]))
    for i, m in enumerate(models):
        for j, obs in enumerate(np.rollaxis(X_obs, axis=0)):
            pos, vel = obs[0:2], obs[2:4]
            tau, _ = predict(m.g, pos.reshape(-1, 2))
            p_x, p_y = pos[0].reshape(-1, 1), pos[1].reshape(-1, 1)
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


def to_model_probabilities(gp_loglikelihoods: ndarray):
    model_likelihoods = gp_loglikelihoods.sum(axis=0)
    cum_model_likelihoods = model_likelihoods.cumsum(axis=1)
    cum_model_likelihoods -= cum_model_likelihoods.max(axis=0)
    cum_model_likelihoods = np.exp(cum_model_likelihoods)
    return np.apply_along_axis(
        lambda x: x / x.sum(), 0, cum_model_likelihoods
    )


def most_probable_models(X_obs, models):
    """Point-wise most probable model"""
    assert X_obs.shape[0] > 0
    gp_likelihoods = gp_logliks(models, X_obs)
    probs = to_model_probabilities(gp_likelihoods)
    return probs.argmax(axis=0)


def model_probabilities(model: SegmentModel, X_obs: ndarray) -> ndarray:
    gp_likelihoods = gp_logliks(model, X_obs)
    model_probs = to_model_probabilities(gp_likelihoods)
    probabilities = model_probs[:, -1]
    return probabilities

def most_probable_model_predictor(
        model: SegmentModel,
        X_obs: ndarray) -> float: # Tuple[float, float, int]:
    likelihoods = model_probabilities(model, X_obs)
    most_probable_model = model.trajectory_models[likelihoods.argmax()]
    latest_pos = X_obs[-1, 0:2].reshape(-1, 2)
    t_mu, t_var = arrival_time_prediction(
        most_probable_model, latest_pos
    )
    return t_mu #, t_var, most_probable_model.traj


def mixture_of_gps_predictor(
        model: SegmentModel,
        X_obs: ndarray) -> float:
    likelihoods = model_probabilities(model, X_obs)
    last_obs = X_obs[-1, :].reshape(1, -1)
    mus, vars = model.arrival_time_posterior(last_obs)
    sigmas = np.sqrt(vars)
    xx = grid_for(mus.reshape(-1), sigmas.reshape(-1))
    unnormed_density = np.vstack([
        w*norm.pdf(xx, mu, sigma)
        for w, mu, sigma in zip(likelihoods, mus, sigmas)
    ]).sum(axis=0).reshape(-1)
    mean_pred = float(np.sum(unnormed_density / unnormed_density.sum() * xx))
    return mean_pred


def plot_most_probable_models(
        tau_grid: ndarray,
        model: SegmentModel,
        model_weights: ndarray,
        normaliser: SegmentNormaliser,
        data: DataFrame,
        linestyle: Callable[[int], str],
        n_models=3):

    models = model.trajectory_models
    model_indicies = range(len(models))
    most_probable_models = sorted(
        zip(model_weights, model_indicies, models),
        key=lambda x: x[0],
        reverse=True
    )

    _, f_ax = plot_grid(2, 2)
    _, ax2 = plot_grid(1, 2)
    xmin, xmax = 0, data.iloc[-1].tau
    #tau_grid = make_grid(xmin, xmax, 100, padding=0)
    p_x_ax = f_ax[0][0]
    p_y_ax = f_ax[0][1]
    v_x_ax = f_ax[1][0]
    v_y_ax = f_ax[1][1]
    t_ax = ax2[0]
    probs_ax = ax2[1]

    unnorm_x = lambda x: normaliser.unnormalise_x(x)*KM_RATIO
    unnorm_y = lambda y: normaliser.unnormalise_y(y)*KM_RATIO
    unnorm_dx = lambda dx: normaliser.unnormalise_dx(dx)*KM_H_RATIO
    unnorm_dy = lambda dy: normaliser.unnormalise_dy(dy)*KM_H_RATIO
    models_to_plot = most_probable_models[:n_models]
    pos = data[['x', 'y']].values
    for w, i, m in models_to_plot:
        tau, _ = predict(m.g, pos)
        mu_p_x_1, var_p_x_1 = predict(m.f_p_x_1, tau)
        mu_p_x_2, var_p_x_2 = predict(m.f_p_x_2, tau)
        mu_p_x, var_p_x = combine(
            np.hstack((unnorm_x(mu_p_x_1), unnorm_x(mu_p_x_2))),
            np.hstack((var_p_x_1, var_p_x_2))
        )
        plot_with_credible_bands(
            p_x_ax, tau_grid, mu_p_x, var_p_x,
            model_label(i), default_color(i), linestyle(i)
        )

        mu_p_y, var_p_y = combined_prediction(
            tau, m.f_p_y_1, m.f_p_y_2
        )
        plot_with_credible_bands(
            p_y_ax, tau_grid, unnorm_y(mu_p_y), var_p_y,
            model_label(i), default_color(i), linestyle(i)
        )

        mu_v_x_1, var_v_x_1 = predict(m.f_v_x_1, tau)
        mu_v_x_2, var_v_x_2 = predict(m.f_v_x_2, tau)
        mu_v_x, var_v_x = combine(
            np.hstack((unnorm_dx(mu_v_x_1), unnorm_dx(mu_v_x_2))),
            np.hstack((var_v_x_1, var_v_x_2))
        )
        plot_with_credible_bands(
            v_x_ax, tau_grid, mu_v_x, var_v_x,
            model_label(i), default_color(i), linestyle(i)
        )

        mu_v_y_1, var_v_y_1 = predict(m.f_v_y_1, tau)
        mu_v_y_2, var_v_y_2 = predict(m.f_v_y_2, tau)
        mu_v_y, var_v_y = combine(
            np.hstack((unnorm_dy(mu_v_y_1), unnorm_dy(mu_v_y_2))),
            np.hstack((var_v_y_1, var_v_y_2))
        )
        plot_with_credible_bands(
            v_y_ax, tau_grid, mu_v_y, var_v_y,
            model_label(i), default_color(i), linestyle(i)
        )

        mu_h, var_h = predict(m.h, tau)
        plot_with_credible_bands(
            t_ax, tau_grid, mu_h, var_h,
            model_label(i), default_color(i), linestyle(i)
        )

    suffix = lambda n: ' for {} most probable models'.format(n)
    plot_data(p_x_ax, data.tau, unnorm_x(data.x))
    p_x_ax.set_title(r'Predicted $x$-position' + suffix(n_models))
    p_x_ax.set_xlabel(r'$\tau$')
    p_x_ax.set_ylabel(r'$p(p_x | \mathcal{M}_k, X_{obs})$')
    p_x_ax.legend()

    plot_data(p_y_ax, data.tau, unnorm_y(data.y))
    p_y_ax.set_title(r'Predicted $y$-position' + suffix(n_models))
    p_y_ax.set_xlabel(r'$\tau$')
    p_y_ax.set_ylabel(r'$p(p_y | \mathcal{M}_k, X_{obs})$')
    p_y_ax.legend()

    plot_data(v_x_ax, data.tau, unnorm_dx(data.dx))
    v_x_ax.set_title(r'Predicted $x$-velocity' + suffix(n_models))
    v_x_ax.set_xlabel(r'$\tau$')
    v_x_ax.set_ylabel(r'$p(v_x | \mathcal{M}_k, X_{obs})$')
    v_x_ax.legend()

    plot_data(v_y_ax, data.tau, unnorm_dy(data.dy))
    v_y_ax.set_title(r'Predicted $y$-velocity' + suffix(n_models))
    v_y_ax.set_xlabel(r'$\tau$')
    v_y_ax.set_ylabel(r'$p(v_y | \mathcal{M}_k, X_{obs})$')
    v_y_ax.legend()

    plot_data(t_ax, data.tau, data.time_left)
    t_ax.set_title(r'Predicted $\mu_t$' + suffix(n_models))
    t_ax.set_xlabel(r'$\tau$')
    t_ax.set_ylabel(r'$p(\mu_t | \mathcal{M}_k, X_{obs})$')
    t_ax.legend()

    probs_ax.set_title(r'Pointwise model log likelihoods' +
                       suffix(n_models * 2))
    probs_ax.set_ylabel(r'$p(\mathcal{M}_k \vert X_{1:n})$')
    probs_ax.set_xlabel(r'$\tau$')
    probs_ax.axis('off')