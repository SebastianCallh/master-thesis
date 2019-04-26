from typing import NamedTuple, Tuple
from numpy import ndarray
import numpy as np
import GPy

GP = GPy.models.SparseGPRegression
GPR = 'gpr'
SGPR = 'sgpr'


class FunctionModelPriors(NamedTuple):
    kern_lengthscale: GPy.priors.Gamma
    kern_variance: GPy.priors.Gamma
    likelihood_noise: GPy.priors.Gamma


def gamma_prior(mean, var):
    return GPy.priors.Gamma(mean, var)


class FunctionModel(NamedTuple):
    f_type: str
    model: GP


def predict(func: FunctionModel, X: ndarray) -> Tuple[ndarray, ndarray]:
    return func.model.predict(X)


def learn_sparse(
        X: ndarray, Y:ndarray, kernel,
        num_inducing: int,
        sigma_n: float,
        n_restarts) -> FunctionModel:

    m = GPy.models.SparseGPRegression(
        X, Y, kernel,
        num_inducing=num_inducing,
        normalizer=False
    )

    m.likelihood.variance = sigma_n
    m.likelihood.variance.fix()
    m.optimize_restarts(n_restarts)
    return FunctionModel(SGPR, m)


def learn_function(
        X: ndarray, Y:ndarray, kernel,
        sigma_n: float,
        n_restarts) -> FunctionModel:

    m = GPy.models.GPRegression(
        X, Y, kernel,
        normalizer=False
    )

    m.likelihood.variance = sigma_n
    m.likelihood.variance.fix()
    m.optimize_restarts(n_restarts)
    return FunctionModel(GPR, m)


def learn_function_raw(
        X: ndarray, Y: ndarray, applyPriors,
        f_type: str,  kernel=None,
        num_inducing=None, n_restarts=3) -> FunctionModel:
    model = GP(
        X, Y, kernel, num_inducing=num_inducing,
        normalizer=False
    )
    applyPriors(model)
    model.optimize_restarts(n_restarts)
    return FunctionModel(
        f_type, model
    )


def plot_function(f: FunctionModel, ax=None):
    if ax:
        f.model.plot(ax=ax)
    else:
        f.model.plot()


def plot_with_credible_bands(
        ax, grid, mu, var,
        label, color, linestyle):
    N = len(grid)
    sigma = np.sqrt(var)
    upper_env = mu + sigma * 2
    lower_env = mu - sigma * 2
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


def plot_posterior(
        ax, grid: ndarray,
        mu: ndarray, sigma: ndarray,
        label,
        color: str,
        linestyle: str = '-'):
    upper_env = (mu + sigma * 2).reshape(-1)
    lower_env = (mu - sigma * 2).reshape(-1)
    ax.plot(
        grid, mu,
        label=label,
        color=color,
        linestyle=linestyle
    )

    fill_label = 'Probability bands' if label is not None else None
    ax.fill_between(
        grid,
        upper_env,
        lower_env,
        color=color,
        alpha=.3,
        label=fill_label
    )