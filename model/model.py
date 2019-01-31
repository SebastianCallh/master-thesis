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
import pandas as pd
from pandas import DataFrame, Series
import GPy
from GPy.models import GPRegression
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


def loglik(func: FunctionModel) -> float:
    return func.model.log_likelihood()


def predict(func: FunctionModel, X: np.ndarray) -> np.ndarray:
    x_scaled = func.x_scaler.transform(X)
    y_scaled, var = func.model.predict(x_scaled)
    return (func.y_scaler.inverse_transform(y_scaled), var)


def plot_function(func: FunctionModel, ax=None) -> ():
    if ax:
        func.model.plot(ax=ax)
    else:
        func.model.plot()

def learn_function(
        data: DataFrame,
        domain: Domain,
        codomain: Codomain,
        f_type: str,
        priors: FunctionModelPriors=None,
        fixed_likelihood: int=None,
        n_restarts=3,
        messages=False) -> FunctionModel:
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
        GPy.kern.RBF(input_dim=x.shape[1],
                     ARD=False))

    if priors.kern_lengthscale:
        model.kern.lengthscale.set_prior(priors.kern_lengthscale)
    if priors.kern_variance:
        model.kern.variance.set_prior(priors.kern_variance)
    if priors.likelihood:
        model.likelihood.set_prior(priors.likelihood)
    if fixed_likelihood:
        model.likelihood.variance = fixed_likelihood
        model.likelihood.variance.fix()

    model.optimize_restarts(n_restarts, messages)
    return FunctionModel(
        model, f_type, x_scaler, y_scaler)


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
 
def stop_compress(data: DataFrame, delta: float) -> DataFrame:
    """ Downsamples the data such that each consecutive data point have a least
    delta distance between each other. Data that is very dense will skew the
    result since the aggreageted mean will be strongly defined by tightly clustered data.
    """

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


def create_support_data(data: DataFrame, n_samples: int, sigma: float) -> DataFrame:
    """Iterates over provided data frame with lat/lon-data 
    and places a Gaussian distribution with provided variance on each observation.
    From this distribution n_samples samples are drawn, which are aggregated together into
    the returned support_data DataFrame.
    """

    def orth_comp(a):
        return np.array([-a[1], a[0]])

    acc = []
    for n in range(data.shape[0]-1):
        cur = data.iloc[n]
        nxt = data.iloc[n+1]
        orth_vec = orth_comp(delta_vector(cur, nxt))
        cur_vec = obs_vector(cur)

        samples = np.random.normal(0, sigma, n_samples)
        support_latlon = [cur_vec + orth_vec*x for x in samples]
        acc.extend([move_to(cur.copy(), vec) 
                     for vec in support_latlon])
    
    return pd.DataFrame(acc)
    

def learn_trajectory_model(
        data: DataFrame,
        route: int,
        trajectory: int,
        codomain_f: Domain,
        domain_h: Domain,
        f_priors: FunctionModelPriors=None,
        g_priors: FunctionModelPriors=None,
        h_priors: FunctionModelPriors=None,
        fix_f_likelihood: float=None,
        n_restarts=3,
        stop_compress_delta=.0004,
        n_augment_samples=5,
        augment_sigma=.8) -> TrajectoryModel:

    # Stop compress
    compressed_data = stop_compress(data, stop_compress_delta)

    # Create tau
    N = compressed_data.shape[0]
    tau = [(x + 1) / N for x in range(N)]
    sorted_data = compressed_data.sort_values('timestamp')
    sorted_data['tau'] = tau

    # Compute time left
    arrival_time = sorted_data.iloc[-1].timestamp
    time_left = [(arrival_time - t).seconds
                 for t in sorted_data.timestamp]
    sorted_data['time_left'] = time_left

    # Data augmentation
    data0 = sorted_data.iloc[0]
    v = obs_vector(data0)
    u = delta_vector(data0, sorted_data.iloc[1])
    start_stretch_len = 10
    start_stretch = [move_to(data0.copy(), v-u*x)
                     for x in range(start_stretch_len)]
    
    dataN = sorted_data.iloc[-1]
    w = obs_vector(dataN)
    z = delta_vector(dataN, sorted_data.iloc[-2])
    stop_stretch_len = 10
    stop_stretch = [move_to(dataN.copy(), w-z*x)
                     for x in range(stop_stretch_len)]
    
    support_data = create_support_data(sorted_data, n_augment_samples, augment_sigma)
    augmented_data = sorted_data.append(support_data)#.append(start_stretch).append(stop_stretch)

    # Learn all GPs
    f = learn_function(
        sorted_data, ['tau'], codomain_f, 'f',
        priors=f_priors, 
        n_restarts=n_restarts)

    g = learn_function(
        augmented_data, domain_h, ['tau'], 'g',
        priors=g_priors, 
        n_restarts=n_restarts, 
        fixed_likelihood=fix_f_likelihood)

    h = learn_function(
        sorted_data, ['tau'], ['time_left'], 'h',
        priors=h_priors, 
        n_restarts=n_restarts)

    return TrajectoryModel(route, trajectory, f, g, h)


# STORAGE


def acquire_db_conn():
    return pg.connect('dbname={} user={} password={}' \
                      .format(DB_NAME, DB_USER, DB_PW))

def save_model(model: TrajectoryModel, conn) -> int:
    f_id = save_function(model.f, conn)
    g_id = save_function(model.g, conn)
    h_id = save_function(model.h, conn)
    print(f_id, g_id, h_id)
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
        res = cur.fetchone()
        f = load_function(res['fid'], conn)
        g = load_function(res['gid'], conn)
        h = load_function(res['hid'], conn)
        return TrajectoryModel(
            res['route'], res['segment'], f, g, h
        )

def save_function(func: FunctionModel , conn) -> int:
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
        print(res)

    f_type = res['type']
    model = GPRegression.from_dict(dict(res['model']))
    x_scaler = pickle.loads(res['featurescaler'])
    y_scaler = pickle.loads(res['targetscaler'])
    return FunctionModel(
        f_type, model, x_scaler, y_scaler
    )
