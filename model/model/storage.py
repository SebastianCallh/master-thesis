"""This module contains an interface for saving and loading
modules into a database. Currently uses a Postgres implementation.
"""

import json
import psycopg2 as pg
from psycopg2.extras import DictCursor
from GPy.models import GPRegression
from itertools import chain
from .trajectory_model import TrajectoryModel, FunctionModel
import pickle
from datetime import datetime

# DB CONSTANTS
DB_NAME = 'msc'
DB_USER = 'gp_user'
DB_PW = 'gp_pw'

VERSION = 0  # added to route

PICKLE_DIR = './pickles'

def pickle_path(name: str):
    return'{}/{}.pkl'.format(PICKLE_DIR, name)


def save_pickle(name: str, data):
    with open(pickle_path(name), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name: str):
    with open(pickle_path(name), 'rb') as handle:
        return pickle.load(handle)


def acquire_db_conn():
    return pg.connect(
        'dbname={} user={} password={}'
        .format(DB_NAME, DB_USER, DB_PW)
    )


def save_model(model: TrajectoryModel, conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            '''
            INSERT INTO model (
            route, segment, traj, 
            fpx, fpx1, fpx2,
            fpy, fpy1, fpy2, 
            fvx, fvx1, fvx2,
            fvy, fvy1, fvy2,
            g, h
            ) VALUES (
            %(route)s, %(segment)s, %(traj)s, 
            %(fpx)s, %(fpx1)s, %(fpx2)s,
            %(fpy)s, %(fpy1)s, %(fpy2)s,
            %(fvx)s, %(fvx1)s, %(fvx2)s,
            %(fvy)s, %(fvy1)s, %(fvy2)s,
            %(g)s, %(h)s)
            RETURNING id
            ''', {
                'route': int(model.route) + VERSION,
                'segment': int(model.segment),
                'traj': int(model.traj),
                'fpx': pickle.dumps(model.f_p_x),
                'fpx1': pickle.dumps(model.f_p_x_1),
                'fpx2': pickle.dumps(model.f_p_x_2),
                'fpy': pickle.dumps(model.f_p_y),
                'fpy1': pickle.dumps(model.f_p_y_1),
                'fpy2': pickle.dumps(model.f_p_y_2),
                'fvx': pickle.dumps(model.f_v_x),
                'fvx1': pickle.dumps(model.f_v_x_1),
                'fvx2': pickle.dumps(model.f_v_x_2),
                'fvy': pickle.dumps(model.f_v_y),
                'fvy1': pickle.dumps(model.f_v_y_1),
                'fvy2': pickle.dumps(model.f_v_y_2),
                'g': pickle.dumps(model.g),
                'h': pickle.dumps(model.h)
            })
        model_id = cur.fetchone()[0]
        conn.commit()

    return model_id


def model_from_db(res):
    return TrajectoryModel(
        route=res['route'],
        segment=res['segment'],
        traj=res['traj'],
        f_p_x=pickle.loads(res['fpx']),
        f_p_x_1=pickle.loads(res['fpx1']),
        f_p_x_2=pickle.loads(res['fpx2']),
        f_p_y=pickle.loads(res['fpy']),
        f_p_y_1=pickle.loads(res['fpy1']),
        f_p_y_2=pickle.loads(res['fpy2']),
        f_v_x=pickle.loads(res['fvx']),
        f_v_x_1=pickle.loads(res['fvx1']),
        f_v_x_2=pickle.loads(res['fvx2']),
        f_v_y=pickle.loads(res['fvy']),
        f_v_y_1=pickle.loads(res['fvy1']),
        f_v_y_2=pickle.loads(res['fvy2']),
        g=pickle.loads(res['g']),
        h=pickle.loads(res['h'])
    )


def model_ids(route: int, segment: int, conn) -> [int]:
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            '''SELECT id FROM model where route = %s AND segment = %s''',
            (route, segment)
        )
        res = cur.fetchall()
    return list(chain.from_iterable(res))


def load_models(route_n, seg_n, limit):
    with acquire_db_conn() as conn:
        models = sorted(load_models_with_conn(
            int(route_n),
            int(seg_n),
            limit, conn
        ), key=lambda m: m.traj)

    print('loaded {} models: {}'.format(len(models), [m.traj for m in models]))
    return models


def load_models_with_conn(
        route: int, segment: int,
        limit: int, conn) -> [TrajectoryModel]:

    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            '''
            SELECT 
            route, segment, traj, 
            fpx, fpx1, fpx2,
            fpy, fpy1, fpy2,
            fvx, fvx1, fvx2,
            fvy, fvy1, fvy2,
            g, h
            FROM model
            WHERE route = %s
            AND segment = %s
            ORDER BY traj
            LIMIT %s;
            ''',
            (route + VERSION, segment, limit))
        res = cur.fetchall()

    return [model_from_db(x) for x in res]


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


def load_function(func_id: int, conn) -> FunctionModel:
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
