


"""This module defines the "inverse-Gaussian Process" trajectory model.

The model consist of three Gaussian Processes (GPs).
f: tau        -> state
g: (lat, lon) -> tau
h: tau        -> time 
In addition, this module contains an interface for saving and loading
modules into a Postgres database.
"""
from math import radians, cos, sin, asin, sqrt, atan2

import json
import pickle
import psycopg2 as pg
from psycopg2.extras import DictCursor
from typing import List, NamedTuple
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
import GPy
from GPy.models import GPRegression
from GPy.kern.src.kern import Kern

# DB CONSTANTS
DB_NAME = 'msc'
DB_USER = 'gp_user'
DB_PW = 'gp_pw'


# TYPE ALIAS
GP = GPRegression
Domain = List[str]
Codomain = List[str]


# FUNCTION MODEL

class FunctionModelPriors(NamedTuple):
    kern_lengthscale: GPy.priors.Gamma
    kern_variance: GPy.priors.Gamma
    likelihood_noise: GPy.priors.Gamma


def gamma_prior(mean, var):
    return GPy.priors.Gamma.from_EV(mean, var)


class FunctionModel(NamedTuple):
    f_type: str
    model: GP


def data_loglik(
        model: FunctionModel, 
        X: ndarray, 
        Y: ndarray) -> float:

    def loglik(x: ndarray, y: ndarray):
        mu, sigma = predict(model, x.reshape(1, 1))
        return -0.5*(y-mu)*np.linalg.inv(sigma)*(y-mu).T \
                -0.5*np.log(np.abs(sigma))

    return np.sum([loglik(x, y) for x, y in zip(X, Y)])


def loglik(func: FunctionModel) -> float:
    return func.model.log_likelihood()


def normalise(data: DataFrame) -> DataFrame:
    mean = data.mean()
    for c in data.columns:
        data[c] = data[c] - mean[c]

    return data


def predict(func: FunctionModel, X: np.ndarray) -> np.ndarray:
    return func.model.predict(X)



def plot_function(func: FunctionModel, ax=None) -> ():
    if ax:
        func.model.plot(ax=ax)
    else:
        func.model.plot()


def learn_function(
        data: DataFrame,
        domain: Domain,
        codomain: Codomain,
        kernel: Kern,
        f_type: str,
        priors: FunctionModelPriors=None,
        fixed_likelihood: int=None,
        n_restarts=3,
        verbose=False) -> FunctionModel:
    """Fits a gp to a function from provided domain -> codomain.
    The data provided is assumed to have fields for both domain and codomain.
    """

    x = data[domain]
    y = data[codomain]

    model = GPRegression(
        x.values,
        y.values,
        kernel,
        normalizer=False
    )

    if priors:
        if priors.kern_lengthscale:
            model.kern.lengthscale.set_prior(priors.kern_lengthscale)
        if priors.kern_variance:
            model.kern.variance.set_prior(priors.kern_variance)
        if priors.likelihood:
            model.likelihood.set_prior(priors.likelihood)
            
    if fixed_likelihood:
        model.likelihood.variance = fixed_likelihood 
        model.likelihood.variance.fix()

    model.optimize_restarts(n_restarts, messages=verbose)
    return FunctionModel(
        f_type, model
    )


# TRAJECTORY MODEL


class TrajectoryModel(NamedTuple):
    """A model of a trajectory consisting of three GPs
    modeling the functions
    f : tau      -> state
    g : position -> tau
    h : tau      -> arrival time.
    """
    route: int
    segment: int
    f_p_x: FunctionModel
    f_p_y: FunctionModel
    f_v_x: FunctionModel
    f_v_y: FunctionModel
    g: FunctionModel
    h: FunctionModel

# def predict_arrival_time(model: TrajectoryModel, X: ndarray) -> ndarray:


def stop_compress(data: DataFrame, delta: float) -> DataFrame:
    """ Downsamples the data such that each consecutive data point have a least
    delta distance between each other. Data that is very dense will skew the
    result since the aggreageted mean will be strongly defined by tightly clustered data.
    """

    def distance(x1, y1, x2, y2):
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return sqrt(dx**2 + dy**2)

    def mean_timestamp(timestamps):
        return pd.to_datetime(timestamps.values.astype(np.int64).mean())
 
    def compress(data: DataFrame) -> DataFrame:
        if data.shape[0] == 1:
            return data
        
        #data.speed = np.max(data.speed, 0) # data contains -1 sentinel values for missing speed

        # def contains_entered_event(df):
        #    return df.event.transform(lambda e: e == 'EnteredEvent').any()
        
        special_treatment_fields = ['timestamp', 'seg', 'traj']
        compressed_data = data.drop(special_treatment_fields, axis=1).apply(np.mean, axis=0)

        # Since python is a shit language it will wrongly cast stuff
        # unless explicitly provided a pd.Timestamp first
        compressed_data['timestamp'] = pd.Timestamp(2017, 1, 1, 12)
        compressed_data.timestamp = mean_timestamp(data.timestamp)
        compressed_data['seg'] = data.seg.min()
        compressed_data['traj'] = data.iloc[0].traj        

        #compressed_data['event'] = \
        #    'EnteredEvent' if contains_entered_event(data) \
        #    else 'ObservedPositionEvent'

        #compressed_data['station'] = \
        #    data[data.event == 'EnteredEvent'].station \
        #if contains_entered_event(data) else 'NaN'

        # In the case of overlapping segments we let the data belong to the first

        #compressed_data['line'] = data.iloc[0].line

        return compressed_data
   
    output = pd.DataFrame(columns=data.columns)
    data_buffer: List[Series] = [data.iloc[0]]
    for _, current in data.iterrows():
        dist = distance(
            current.x,
            current.y,
            np.mean([x.x for x in data_buffer]),
            np.mean([x.y for x in data_buffer])
        )
   
        if dist > delta:
            output = output.append(compress(pd.DataFrame(data_buffer)), ignore_index=True)
            data_buffer.clear()
        
        data_buffer.append(current)

    output.append(compress(pd.DataFrame(data_buffer)), ignore_index=True)
    return output


def delta_vector(a, b):
    """Returns the vector from a->b."""
    dx = b.x - a.x
    dy = b.y - a.y
    return np.array([dx, dy])


def obs_vector(obs):
    """Returns the position vector of the provided observations."""
    return np.array([obs.x, obs.y])


def create_support_data(
        data: DataFrame,
        f_p: FunctionModel,
        f_v: FunctionModel,
        f_p_codomain: List[str],
        f_v_codomain: List[str],
        n_samples: int,
        delta: float,
        sigma: float) -> DataFrame:

    def orth_comp(v):
        return np.array([-v[1], v[0]])

    tau_grid = np.linspace(
        np.min(data.tau.min()),
        np.min(data.tau.max()),
        1/delta
    )

    tau = tau_grid.reshape(len(tau_grid), 1)
    acc = []
    pos, _ = predict(f_p, tau)
    vel, _ = predict(f_v, tau)
    
    for n in range(len(tau_grid)-1):
        cur_vel = vel[n]
        cur_pos = pos[n]
        nxt_pos = pos[n+1]
        orth_delta = orth_comp(nxt_pos - cur_pos)
        orth_delta = orth_delta / np.linalg.norm(orth_delta)
        
        acc.extend([
            {f_p_codomain[0]: x[0],
             f_p_codomain[1]: x[1],
             f_v_codomain[0]: cur_vel[0],
             f_v_codomain[1]: cur_vel[1],
             'tau': tau_grid[n]}
            for x in [
                cur_pos + orth_delta * x
                for x in np.random.normal(0, sigma, n_samples)
            ]
        ])

    return pd.DataFrame(acc)


def compute_tau(data: DataFrame) -> [float]:
    N = data.shape[0]
    return [(x + 1) / N for x in range(N)]


def learn_trajectory_model(
        data: DataFrame,
        route: int,
        segment: int,
        f_p_codomain: Domain,
        f_v_codomain: Domain,
        g_domain: Domain,
        f_p_priors: FunctionModelPriors=None,
        f_v_priors: FunctionModelPriors=None,
        g_priors: FunctionModelPriors=None,
        h_priors: FunctionModelPriors=None,
        fix_f_p_likelihood: float=None,
        fix_f_v_likelihood: float=None,
        n_restarts=3,
        n_augment_samples=5,
        augment_sigma=.8,
        augment_delta=.1,
        verbose=True) -> TrajectoryModel:

    # Stop compress
    #compressed_data = stop_compress(data, stop_compress_delta)

    # Create tau
    #sorted_data = data.sort_values('timestamp')
    #sorted_data['tau'] = compute_tau(sorted_data)

    # Compute time left
    #arrival_time = sorted_data.iloc[-1].timestamp
    #time_left = [(arrival_time - t).seconds
    #             for t in sorted_data.timestamp]
    #sorted_data['time_left'] = time_left

    f_domain = ['tau']

    # Learn f_p
    f_p_kernel = GPy.kern.RBF(
        input_dim=len(f_domain),
        ARD=True
    ) + GPy.kern.Linear(
        input_dim=len(f_domain),
        ARD=True
    )
    f_p = learn_function(
        data, f_domain,
        f_p_codomain, f_p_kernel, 'f_p',
        priors=f_p_priors,
        n_restarts=n_restarts,
        fixed_likelihood=fix_f_p_likelihood,
        verbose=verbose
    )

    # Learn f_v
    f_v_kernel = GPy.kern.RBF(
        input_dim=len(f_domain),
        ARD=True
    )
    f_v = learn_function(
        data, f_domain,
        f_v_codomain, f_v_kernel, 'f_v',
        priors=f_p_priors,
        n_restarts=n_restarts,
        fixed_likelihood=fix_f_v_likelihood,
        verbose=verbose
    )

    # Data augmentation for g
    #data0 = sorted_data.iloc[0]
    #v = obs_vector(data0)
    #u = delta_vector(data0, sorted_data.iloc[1])
    support_data = create_support_data(
        data, f_p, f_v, 
        f_p_codomain, f_v_codomain,
        n_augment_samples,
        augment_delta, augment_sigma
    )
    augmented_data = \
        data[g_domain + f_domain] \
        .append(support_data)

    # Learn g
    g_kernel = GPy.kern.RBF(
        input_dim=len(g_domain),
        ARD=False
    ) + GPy.kern.Linear(
        input_dim=len(g_domain),
        ARD=False
    )

    g_model = GPRegression(
        augmented_data[g_domain],
        augmented_data[f_domain],
        g_kernel,
        normalizer=False
    )
    g_model.kern.lengthscale.set_prior(g_priors.kern_lengthscale)
    g_model.kern.variance.set_prior(g_priors.kern_variance)
    g_model.likelihood.set_prior(g_priors.likelihood)
    model.optimize_restarts(n_restarts, messages=verbose)
    return FunctionModel(
        f_type, model
    )

    g = learn_function(
        augmented_data, g_domain,
        f_domain, g_kernel, 'g',
        priors=g_priors,
        n_restarts=n_restarts,
        verbose=verbose
    )

    h_domain = ['tau']
    h_kernel = GPy.kern.RBF(
        input_dim=len(h_domain),
        ARD=False
    ) + GPy.kern.Linear(
        input_dim=len(h_domain),
        ARD=False
    )

    h = learn_function(
        data, h_domain,
        ['time_left'], h_kernel, 'h',
        priors=h_priors,
        n_restarts=n_restarts,
        verbose=verbose)

    return TrajectoryModel(
        route, segment, f_p, f_v, g, h
    )


# STORAGE


def acquire_db_conn():
    return pg.connect('dbname={} user={} password={}' \
                      .format(DB_NAME, DB_USER, DB_PW))


def save_model(model: TrajectoryModel, conn) -> int:
    f_p_x_id = save_function(model.f_p_x, conn)
    f_p_y_id = save_function(model.f_p_y, conn)
    f_v_x_id = save_function(model.f_v_x, conn)
    f_v_y_id = save_function(model.f_v_y, conn)
    g_id = save_function(model.g, conn)
    h_id = save_function(model.h, conn)

    with conn.cursor() as cur:
        cur.execute(
            '''
            INSERT INTO model (route, segment, fpxid, fpyid, fvxid, fvyid, gid, hid)
            VALUES (%(route)s, %(segment)s, %(fpxid)s, %(fpyid)s, %(fvxid)s, %(fvyid)s, %(gid)s, %(hid)s)
            RETURNING id
            ''', {
                'route': model.route,
                'segment': model.segment,
                'fpxid': f_p_x_id,
                'fpyid': f_p_y_id,
                'fvxid': f_v_x_id,
                'fvyid': f_v_y_id,
                'gid': g_id,
                'hid': h_id
            })
        model_id = cur.fetchone()[0]
        conn.commit()

    return model_id


def model_from_db(res, conn):
    f_p_x = load_function(res['fpxid'], conn)
    f_p_y = load_function(res['fpyid'], conn)
    f_v_x = load_function(res['fvxid'], conn)
    f_v_y = load_function(res['fvyid'], conn)
    g = load_function(res['gid'], conn)
    h = load_function(res['hid'], conn)
    return TrajectoryModel(
        res['route'], res['segment'], 
        f_p_x, f_p_y, 
        f_v_x, f_v_y, 
        g, h
    )


def load_models(route: int, segment: int, conn) -> [TrajectoryModel]:
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            '''
            SELECT route, segment, fpxid, fpyid, fvxid, fvyid, gid, hid
            FROM model
            WHERE route = %s
            AND segment = %s;
            ''',
            (route, segment))
        res = cur.fetchall()

    return [model_from_db(x, conn) for x in res]


def save_function(func: FunctionModel, conn) -> int:
    """
    Saves the GP to a database using the provided connection.
    """
    with conn.cursor() as cur:
        cur.execute(
            '''
            INSERT INTO function (type, model)
            VALUES (%(type)s, %(model)s)
            RETURNING id
            ''', {
                'type': func.f_type,
                'model': json.dumps(func.model.to_dict())
            })
        func_id = cur.fetchone()[0]
        conn.commit()

    return func_id

def load_function(func_id: int, conn) -> GP:
    """
    Loads the model for function_id using the provided connection.
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            '''
            SELECT type, model
            FROM function
            WHERE id = %s;
            ''', (func_id,))
        res = cur.fetchone()

    f_type = res['type']
    model = GPRegression.from_dict(dict(res['model']))
    return FunctionModel(
        f_type, model
    )
