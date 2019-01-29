"""This module defines the "inverse-Gaussian Process" trajectory model.

The model consist of three Gaussian Processes (GPs).

In addition, this module contains an interface for saving and loading
modules into a Postgres database.

"""
from typing import List, NamedTuple
import GPy
from GPy.models import GPRegression
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
import numpy as np
from dateutil.parser import parse

# DB CONSTANTS
DB_NAME = 'gp'
DB_USER = 'gp_user'
DB_PW = 'gp_pw'

# TYPE ALIAS
GP = GPRegression
Domain = List[str]
Codomain = List[str]


# TRAJECTORY MODEL


class TrajectoryModel(NamedTuple):
    """A model of a trajectory consisting of three GPs
    modeling the functions
    f : tau      -> state
    g : position -> tau
    h : tau      -> arrival time.
    """
    f: GP
    g: GP
    h: GP


def learn_trajectory_model(
        data: DataFrame,
        codomain_f: Domain,
        domain_h: Domain) -> TrajectoryModel:

    # Create tau
    tau = [(x + 1) / len(data) for x in range(0, len(data))]
    sorted_data = data.sort_values('timestamp')
    sorted_data['tau'] = tau

    # Compute time left
    arrival_time = parse(data.iloc[-1].timestamp)
    time_left = [(arrival_time - parse(t)).seconds
                 for t in data.timestamp]
    sorted_data['time_left'] = time_left

    # Learn all GPs
    f = learn_function(sorted_data, ['tau'], codomain_f)
    g = learn_function(sorted_data, domain_h, ['tau'])
    h = learn_function(sorted_data, ['tau'], ['time_left'])

    return TrajectoryModel(f, g, h)


# FUNCTION MODEL


class FunctionModel(NamedTuple):
    """A GP model of a function, which is
    scaled to have zero mean and unit variance.
    """
    gp: GP
    x_scaler: StandardScaler
    y_scaler: StandardScaler


def loglik(model: FunctionModel) -> float:
    return model.gp.log_likelihood()


def predict(gp: FunctionModel, X: np.ndarray) -> np.ndarray:
    X_scaled = gp.X_scaler.transform(X)
    Y_scaled, var = gp.model.predict(X_scaled)
    return (gp.Y_scaler.inverse_transform(Y_scaled), var)


def plot_function(model: FunctionModel) -> ():
    model.gp.plot()


def learn_function(
        data: DataFrame,
        domain: Domain,
        codomain: Codomain,
        n_restarts=3,
        messages=False) -> FunctionModel:
    x = data[domain]
    y = data[codomain]
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(x)
    y_scaler.fit(y)
    gp = GPRegression(
        x_scaler.transform(x),
        y_scaler.transform(y),
        GPy.kern.RBF(input_dim=x.shape[1],
                     ARD=False))

    gp.optimize_restarts(n_restarts, messages)
    return FunctionModel(
        gp, x_scaler, y_scaler)


# AD HOC


def learn_progress_function(
        data: DataFrame,
        codomain: Codomain) -> FunctionModel:
    """Learns a function from progress = [0, 1] to provided codomain.
    The codomain should be the state space of a time series with
    a 'timestamp' dimension.
    """
    progress = [(x + 1) / len(data) for x in range(0, len(data))]
    sorted_data = data.sort_values('timestamp')
    sorted_data['progress'] = progress
    return learn_function(sorted_data, ['progress'], codomain)
