"""This module contains an interface for saving and loading
modules into a database. Currently uses a Postgres implementation.
"""

import json
import psycopg2 as pg
from psycopg2.extras import DictCursor
from GPy.models import GPRegression
from itertools import chain
from .trajectory_model import TrajectoryModel, FunctionModel

# DB CONSTANTS
DB_NAME = 'msc'
DB_USER = 'gp_user'
DB_PW = 'gp_pw'


def acquire_db_conn():
    return pg.connect(
        'dbname={} user={} password={}'
        .format(DB_NAME, DB_USER, DB_PW)
    )


def save_model(model: TrajectoryModel, conn) -> int:
    f_p_x_id_1 = save_function(model.f_p_x_1, conn)
    f_p_y_id_1 = save_function(model.f_p_y_1, conn)
    f_p_x_id_2 = save_function(model.f_p_x_2, conn)
    f_p_y_id_2 = save_function(model.f_p_y_2, conn)
    f_v_x_id_1 = save_function(model.f_v_x_1, conn)
    f_v_y_id_1 = save_function(model.f_v_y_1, conn)
    f_v_x_id_2 = save_function(model.f_v_x_2, conn)
    f_v_y_id_2 = save_function(model.f_v_y_2, conn)
    g_id = save_function(model.g, conn)
    h_id = save_function(model.h, conn)
    with conn.cursor() as cur:
        cur.execute(
            '''
            INSERT INTO model (
            route, segment, traj, 
            fpxid1, fpyid1,
            fpxid2, fpyid2, 
            fvxid1, fvyid1,
            fvxid2, fvyid2,
            gid, hid
            ) VALUES (
            %(route)s, %(segment)s, %(traj)s, 
            %(fpxid1)s, %(fpyid1)s,
            %(fpxid2)s, %(fpyid2)s, 
            %(fvxid1)s, %(fvyid1)s,
            %(fvxid2)s, %(fvyid2)s, 
            %(gid)s, %(hid)s)
            RETURNING id
            ''', {
                'route': int(model.route),
                'segment': int(model.segment),
                'traj': int(model.traj),
                'fpxid1': f_p_x_id_1,
                'fpyid1': f_p_y_id_1,
                'fpxid2': f_p_x_id_2,
                'fpyid2': f_p_y_id_2,
                'fvxid1': f_v_x_id_1,
                'fvyid1': f_v_y_id_1,
                'fvxid2': f_v_x_id_2,
                'fvyid2': f_v_y_id_2,
                'gid': g_id,
                'hid': h_id
            })
        model_id = cur.fetchone()[0]
        conn.commit()

    return model_id


def model_from_db(res, conn):
    f_p_x_1 = load_function(res['fpxid1'], conn)
    f_p_y_1 = load_function(res['fpyid1'], conn)
    f_p_x_2 = load_function(res['fpxid2'], conn)
    f_p_y_2 = load_function(res['fpyid2'], conn)
    f_v_x_1 = load_function(res['fvxid1'], conn)
    f_v_y_1 = load_function(res['fvyid1'], conn)
    f_v_x_2 = load_function(res['fvxid2'], conn)
    f_v_y_2 = load_function(res['fvyid2'], conn)
    g = load_function(res['gid'], conn)
    h = load_function(res['hid'], conn)
    return TrajectoryModel(
        res['route'],
        res['segment'],
        res['traj'],
        None, None, None, None,
        f_p_x_1, f_p_x_2,
        f_p_y_1, f_p_y_2,
        f_v_x_1, f_v_x_2,
        f_v_y_1, f_v_y_2,
        g, h
    )


def model_ids(route: int, segment: int, conn) -> [int]:
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            '''SELECT id FROM model where route = %s AND segment = %s''',
            (route, segment)
        )
        res = cur.fetchall()
    return list(chain.from_iterable(res))


def load_models(
        route: int, segment: int,
        limit: int, conn) -> [TrajectoryModel]:

    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            '''
            SELECT 
            route, segment, traj, 
            fpxid1, fpxid2,
            fpyid1, fpyid2, 
            fvxid1, fvxid2,
            fvyid1, fvyid2,
            gid, hid
            FROM model
            WHERE route = %s
            AND segment = %s
            LIMIT %s;
            ''',
            (route, segment, limit))
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
