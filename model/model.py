"""This module defines the "inverse-Gaussian Process" trajectory model.

The model consist of three Gaussian Processes (GPs).
f: tau        -> state
g: (lat, lon) -> tau
h: tau        -> time 
In addition, this module contains an interface for saving and loading
modules into a Postgres database.
"""
from math import radians, cos, sin, asin, sqrt 
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
from sklearn.preprocessing import StandardScaler

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
    likelihood: GPy.priors.Gamma


def gamma_prior(mean, var):
    return GPy.priors.Gamma.from_EV(mean, var)


class FunctionModel(NamedTuple):
    """A GP model of a function, which is
    scaled to have zero mean and unit variance.
    """
    f_type: str
    model: GP
    x_scaler: StandardScaler
    y_scaler: StandardScaler


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
    x_scaled = func.x_scaler.transform(X)
    y_scaled, var = func.model.predict(x_scaled)
    return func.y_scaler.inverse_transform(y_scaled), var


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

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(x)
    y_scaler.fit(y)

    model = GPRegression(
        x_scaler.transform(x),
        y_scaler.transform(y),
        kernel)


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
        f_type, model, x_scaler, y_scaler
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
    f: FunctionModel
    g: FunctionModel
    h: FunctionModel

# def predict_arrival_time(model: TrajectoryModel, X: ndarray) -> ndarray:
    

def stop_compress(data: DataFrame, delta: float) -> DataFrame:
    """ Downsamples the data such that each consecutive data point have a least
    delta distance between each other. Data that is very dense will skew the
    result since the aggreageted mean will be strongly defined by tightly clustered data.
    """


    def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371 # Radius of earth in kilometers
        return c * r

    def mean_timestamp(timestamps):
        return pd.to_datetime(timestamps.values.astype(np.int64).mean())
 
    def compress(data: DataFrame) -> DataFrame:
        if data.shape[0] == 1:
            return data
        
        data.speed = np.max(data.speed, 0) # data contains -1 sentinel values for missing speed

        def contains_entered_event(df):
            return df.event.transform(lambda e: e == 'EnteredEvent').any()
        
        special_treatment_fields = ['timestamp', 'event', 'seg', 'station', 'line', 'traj']
        compressed_data = data.drop(special_treatment_fields, axis=1).apply(np.mean, axis=0)
        
        # Since python is a shit language it will wrongly cast stuff
        # unless explicitly provided a pd.Timestamp first
        compressed_data['timestamp'] = pd.Timestamp(2017, 1, 1, 12)
        compressed_data.timestamp = mean_timestamp(data.timestamp)
        compressed_data['event'] = \
            'EnteredEvent' if contains_entered_event(data) \
            else 'ObservedPositionEvent'

        compressed_data['station'] = \
            data[data.event == 'EnteredEvent'].station \
        if contains_entered_event(data) else 'NaN'

        # In the case of overlapping segments we let the data belong to the first
        compressed_data['seg'] = data.seg.min()
        compressed_data['line'] = data.iloc[0].line
        compressed_data['traj'] = data.iloc[0].traj
        return compressed_data
   
    output = pd.DataFrame(columns=data.columns)
    data_buffer: List[Series] = [data.iloc[0]]
    for _, current in data.iterrows():
        distance = haversine(
            current.lat, 
            current.lon, 
            np.mean([x.lat for x in data_buffer]),
            np.mean([x.lon for x in data_buffer]))
   
        if distance > delta:
            output = output.append(compress(pd.DataFrame(data_buffer)), ignore_index=True)
            data_buffer.clear()
        
        data_buffer.append(current)

    output.append(compress(pd.DataFrame(data_buffer)), ignore_index=True)
    return output


def delta_vector(a, b):
    """Returns the vector from a->b."""
    d_lat = b.lat - a.lat
    d_lon = b.lon - a.lon
    return np.array([d_lat, d_lon])


def obs_vector(obs):
    """Returns the position vector of the provided observations."""
    return np.array([obs.lat, obs.lon])


def move_to(data, vec):
    data.lat = vec[0]
    data.lon = vec[1]
    return data


def create_support_data(
        data: DataFrame,
        f: FunctionModel,
        f_codomain: List[str],
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
    
    acc = []
    for n in range(len(tau_grid)-1):
        cur_tau = np.array(tau_grid[n]).reshape(1, 1)
        nxt_tau = np.array(tau_grid[n+1]).reshape(1, 1)
        cur, _ = predict(f, cur_tau)
        nxt, _ = predict(f, nxt_tau)
        orth = orth_comp(nxt[0]-cur[0])
        orth = orth / np.linalg.norm(orth)

        samples = [
            cur + orth * x
            for x in np.random.normal(0, sigma, n_samples)
        ]

        acc.extend([
            {f_codomain[0]: x[0][0],
             f_codomain[1]: x[0][1],
             'tau': tau_grid[n]}
            for x in samples
        ])

    return pd.DataFrame(acc)


def compute_tau(data: DataFrame) -> [float]:
    N = data.shape[0]
    return [(x + 1) / N for x in range(N)]


def learn_trajectory_model(
        data: DataFrame,
        route: int,
        trajectory: int,
        f_codomain: Domain,
        g_domain: Domain,
        f_priors: FunctionModelPriors=None,
        g_priors: FunctionModelPriors=None,
        h_priors: FunctionModelPriors=None,
        fix_f_likelihood: float=None,
        n_restarts=3,
        stop_compress_delta=.0004,
        n_augment_samples=5,
        augment_sigma=.8,
        augment_delta=.1,
        verbose=True) -> TrajectoryModel:

    # Stop compress
    compressed_data = stop_compress(data, stop_compress_delta)

    # Create tau
    sorted_data = compressed_data.sort_values('timestamp')
    sorted_data['tau'] = compute_tau(sorted_data)

    # Compute time left
    #arrival_time = sorted_data.iloc[-1].timestamp
    #time_left = [(arrival_time - t).seconds
    #             for t in sorted_data.timestamp]
    #sorted_data['time_left'] = time_left

    # Learn f
    f_domain = ['tau']
    f_kernel = GPy.kern.RBF(
        input_dim=len(f_domain),
        ARD=False
    )
    f = learn_function(
        sorted_data, f_domain,
        f_codomain, f_kernel, 'f',
        priors=f_priors,
        n_restarts=n_restarts,
        fixed_likelihood=fix_f_likelihood,
        verbose=verbose
    )

    # Data augmentation for g
    data0 = sorted_data.iloc[0]
    v = obs_vector(data0)
    u = delta_vector(data0, sorted_data.iloc[1])
    support_data = create_support_data(
        sorted_data, f, f_codomain, n_augment_samples,
        augment_delta, augment_sigma
    )
    augmented_data = \
        sorted_data[g_domain + f_domain] \
        .append(support_data)

    # Learn g
    g_kernel = GPy.kern.RBF(
        input_dim=len(g_domain),
        ARD=False
    ) + GPy.kern.Linear(
        input_dim=len(g_domain),
        ARD=False
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
        sorted_data, h_domain,
        ['time_left'], h_kernel, 'h',
        priors=h_priors,
        n_restarts=n_restarts,
        verbose=verbose)

    return TrajectoryModel(
        route, trajectory, f, g, h
    )


# STORAGE


def acquire_db_conn():
    return pg.connect('dbname={} user={} password={}' \
                      .format(DB_NAME, DB_USER, DB_PW))


def save_model(model: TrajectoryModel, conn) -> int:
    f_id = save_function(model.f, conn)
    g_id = save_function(model.g, conn)
    h_id = save_function(model.h, conn)

    with conn.cursor() as cur:
        cur.execute(
            '''
            INSERT INTO model (route, segment, fid, gid, hid)
            VALUES (%(route)s, %(segment)s, %(fid)s, %(gid)s, %(hid)s)
            RETURNING id
            ''', {
                'route': model.route,
                'segment': model.segment,
                'fid': f_id,
                'gid': g_id,
                'hid': h_id
            })
        model_id = cur.fetchone()[0]
        conn.commit()

    return model_id


def model_from_db(res, conn):
    f = load_function(res['fid'], conn)
    g = load_function(res['gid'], conn)
    h = load_function(res['hid'], conn)
    return TrajectoryModel(
        res['route'], res['segment'], f, g, h
    )


def load_models(route: int, segment: int, conn) -> [TrajectoryModel]:
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            '''
            SELECT route, segment, fid, gid, hid
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
            INSERT INTO function (type, model, featurescaler, targetscaler)
            VALUES (%(type)s, %(model)s, %(x_scaler)s, %(y_scaler)s)
            RETURNING id
            ''', {
                'type': func.f_type,
                'model': json.dumps(func.model.to_dict()),
                'x_scaler': pickle.dumps(func.x_scaler),
                'y_scaler': pickle.dumps(func.y_scaler)
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
            SELECT type, model, featurescaler, targetscaler
            FROM function
            WHERE id = %s;
            ''', (func_id,))
        res = cur.fetchone()

    f_type = res['type']
    model = GPRegression.from_dict(dict(res['model']))
    x_scaler = pickle.loads(res['featurescaler'])
    y_scaler = pickle.loads(res['targetscaler'])
    return FunctionModel(
        f_type, model, x_scaler, y_scaler
    )
