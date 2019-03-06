from numpy import ndarray
from scipy.stats import norm
import numpy as np
import math
from functools import reduce
from numpy.linalg import inv, det
from model import TrajectoryModel, FunctionModel
from plotting import plot_grid


def predict(func: FunctionModel, X: np.ndarray) -> np.ndarray:
    return func.model.predict(X)

def model_cum_loglik(model: TrajectoryModel, X_obs: ndarray, tau: ndarray):
    pos = X_obs[:,0:2]
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

def model_data_loglik(model: TrajectoryModel, X_obs: ndarray, tau: ndarray) -> ndarray:
    """"""

    def loglik(x: ndarray, mu: ndarray, sigma: ndarray):
        """Assumes sigma is the diagonal of the covariance matrix."""
        #return np.sum(-0.5*((x-mu))*(1/sigma)*(x-mu)) \
        #    -0.5*np.log(np.prod(sigma))

        return -0.5*((x-mu).T).dot(inv(sigma)).dot(x-mu) \
            -0.5*np.log(det(sigma))

    X_k_p_x, sigma_p_x = predict(model.f_p_x, tau)
    X_k_p_y, sigma_p_y = predict(model.f_p_y, tau)
    X_k_v_x, sigma_v_x = predict(model.f_v_x, tau)
    X_k_v_y, sigma_v_y = predict(model.f_v_y, tau)
    #, X_k_p_x, np.diag(sigma_p_x.T[0])))
    #print(loglik_pos_x)
    pos = X_obs[:,0:2]
    vel = X_obs[:,2:4]
    #loglik_pos_x = loglik(pos[:,0].reshape(-1, 1), X_k_p_x, sigma_p_x)
    #loglik_pos_y = loglik(pos[:,1].reshape(-1, 1), X_k_p_y, sigma_p_y)
    #loglik_vel_x = loglik(vel[:,0].reshape(-1, 1), X_k_v_x, sigma_v_x)
    #loglik_vel_y = loglik(vel[:,1].reshape(-1, 1), X_k_v_y, sigma_v_y)

    loglik_pos_x = loglik(pos[:,0].reshape(-1, 1), X_k_p_x, np.diag(sigma_p_x.T[0]))
    loglik_pos_y = loglik(pos[:,1].reshape(-1, 1), X_k_p_y, np.diag(sigma_p_y.T[0]))
    loglik_vel_x = loglik(vel[:,0].reshape(-1, 1), X_k_v_x, np.diag(sigma_v_x.T[0]))
    loglik_vel_y = loglik(vel[:,1].reshape(-1, 1), X_k_v_y, np.diag(sigma_v_y.T[0]))

    # loglik_pos_x = model.f_p_x.model.log_predictive_density(tau, pos[:,0])
    # loglik_pos_y = model.f_p_y.model.log_predictive_density(tau, pos[:,1])
    # loglik_vel_x = model.f_v_x.model.log_predictive_density(tau, vel[:,0])
    # loglik_vel_y = model.f_v_y.model.log_predictive_density(tau, vel[:,1])
    #print(loglik_pos_y)
    loglik_sum = \
        loglik_pos_x + \
        loglik_pos_y + \
        loglik_vel_x + \
        loglik_vel_y

    if math.isinf(loglik_sum) or math.isnan(loglik_sum):
        print('inf loglik', loglik_pos_x, loglik_pos_y, loglik_vel_x, loglik_vel_y)
        # mu = X_k_v_x
        # x = vel[:,0].reshape(-1, 1)
        # sigma = np.diag(sigma_v_x.T[0])
        # #print(np.hstack([x, mu]))
        # term1 = -0.5*((x-mu).T).dot(inv(sigma)).dot(x-mu)
        # term2 = - 0.5*np.log(det(sigma))
        # print('term 1', term1)
        # print('term 2', term2, 'det', np.diag(sigma).T)
        # print('tota',  term1 + term2)
        # print('velocities', loglik_vel_x, loglik_vel_y)
        # _, ax = plot_grid(1, 1)
        # plot_function(model.f_v_x, ax=ax)
        # ax.plot(tau, vel[:,0])
    return loglik_sum

def prediction_distribution(grid, mu_and_sigma):
    mu = mu_and_sigma[0]
    sigma = mu_and_sigma[1]
    return norm.pdf(grid, mu, np.sqrt(sigma))

def mixture_distributions(
        mus: ndarray,
        sigmas: ndarray,
        weights: ndarray,
        grid_res=100,
        grid_pad=3) -> ndarray:

    # Create grid that covers all components
    asc_preds = sorted(zip(mus, sigmas))
    smallest_pred_mean = float(asc_preds[0][0])
    smallest_pred_var = float(asc_preds[0][1])
    xmin = max(0, np.floor(smallest_pred_mean - smallest_pred_var*grid_pad))
    biggest_pred_mean = float(asc_preds[-1][0])
    biggest_pred_var = float(asc_preds[-1][1])
    xmax = np.ceil(biggest_pred_mean + biggest_pred_var*grid_pad)

    t_grid = np.linspace(xmin, xmax, (xmax-xmin)*grid_res)
    distributions = [
        w*prediction_distribution(t_grid, (mu, sigma))
        for mu, sigma, w in zip(mus, sigmas, weights)
    ]

    return distributions, t_grid

def normalise_logliks(logliks):
    """Normalise model logliks for an observation."""
    # Scale to avoid numerical errors due to small numbers
    c = 1/max(logliks)
    loglik_sum = np.sum(logliks)
    f = np.vectorize(lambda l: l - loglik_sum)
    return f(logliks)
