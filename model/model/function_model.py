from typing import NamedTuple

import GPy

# TYPE ALIAS
GP = GPy.models.SparseGPRegression


class FunctionModelPriors(NamedTuple):
    kern_lengthscale: GPy.priors.Gamma
    kern_variance: GPy.priors.Gamma
    likelihood_noise: GPy.priors.Gamma


def gamma_prior(mean, var):
    return GPy.priors.Gamma.from_EV(mean, var)


class FunctionModel(NamedTuple):
    f_type: str
    model: GP

def plot_function(func: FunctionModel, ax=None) -> ():
    if ax:
        func.model.plot(ax=ax)
    else:
        func.model.plot()