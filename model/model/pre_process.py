from itertools import tee
from math import sqrt
from typing import List, Tuple, Callable, NamedTuple
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series


# https://stackoverflow.com/questions/52907328/pandas-delete-first-n-rows-until-condition-on-columns-is-fulfilled
def drop_start_dwell(data: DataFrame) -> DataFrame:
    return data.loc[((data.dx > 0) & (data.dy > 0)).idxmax():]


def take_fraction(data: DataFrame, r: float) -> DataFrame:
    return data[data.tau <= r]
    # n = int(np.round(r*data.shape[0]))
    # return data.iloc[:n]


def compute_tau(data: DataFrame) -> ndarray:
    return np.linspace(0, 1, data.shape[0])


def compute_time_left(data: DataFrame) -> DataFrame:
    sorted_data = data.sort_values('timestamp')
    arrival_time = sorted_data.iloc[-1].timestamp
    sorted_data['time_left'] = [
        (arrival_time - t).seconds
        for t in sorted_data.timestamp
    ]
    return sorted_data


def stop_compress(data: DataFrame, delta: float) -> DataFrame:
    """ Downsamples the data such that each consecutive data point have a least
    delta distance between each other. Data that is very dense will skew the
    result since the aggreageted mean will be strongly defined by tightly clustered data.
    """

    def distance(x1, y1, x2, y2):
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return sqrt(dx ** 2 + dy ** 2)

    def mean_timestamp(timestamps):
        return pd.to_datetime(timestamps.values.astype(np.int64).mean())

    def compress(data: DataFrame) -> DataFrame:
        if data.shape[0] == 1:
            return data

        # data.speed = np.max(data.speed, 0) # data contains -1 sentinel values for missing speed

        # def contains_entered_event(df):
        #    return df.event.transform(lambda e: e == 'EnteredEvent').any()

        special_treatment_fields = ['timestamp', 'seg', 'traj']
        compressed_data = data.drop(special_treatment_fields, axis=1).apply(
            np.mean, axis=0)

        # Since python is a shit language it will wrongly cast stuff
        # unless explicitly provided a pd.Timestamp first
        compressed_data['timestamp'] = pd.Timestamp(2017, 1, 1, 12)
        compressed_data.timestamp = mean_timestamp(data.timestamp)
        compressed_data['seg'] = data.seg.min()
        compressed_data['traj'] = data.iloc[0].traj

        # compressed_data['event'] = \
        #    'EnteredEvent' if contains_entered_event(data) \
        #    else 'ObservedPositionEvent'

        # compressed_data['station'] = \
        #    data[data.event == 'EnteredEvent'].station \
        # if contains_entered_event(data) else 'NaN'

        # In the case of overlapping segments we let the data belong to the first

        # compressed_data['line'] = data.iloc[0].line

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
            output = output.append(compress(pd.DataFrame(data_buffer)),
                                   ignore_index=True)
            data_buffer.clear()

        data_buffer.append(current)

    output.append(compress(pd.DataFrame(data_buffer)), ignore_index=True)
    return output


def sliding_diff(data: ndarray):
    """Last item is duplicated"""
    D = data.shape[0]
    M = -1*np.eye(D) + np.eye(D, k=1)
    v= M @ data
    v[-1] = v[-2]
    return v


def compute_velocity_from_pos(
        data: DataFrame) -> Tuple[ndarray, ndarray]:
    """Assume one data point per second."""
    dx = sliding_diff(data.x.values)
    dy = sliding_diff(data.y.values)
    return dx, dy


def duplicate(x: ndarray, y: ndarray, delta: float):
    """Duplicates data spatially, orthogonal to progression.
    Steps equidistantly in tau.
    """

    pos = np.hstack([
        x.reshape(-1, 1),
        y.reshape(-1, 1)
    ])

    dx, dy = sliding_diff(x), sliding_diff(y)
    delta_pos = np.hstack([
        dy.reshape(-1, 1),
        -dx.reshape(-1, 1)
    ])

    delta_pos = np.apply_along_axis(
        lambda z: z / np.linalg.norm(z) * delta,
        1, delta_pos
    )

    dupe1 = pos + delta_pos
    dupe2 = pos - delta_pos
    return dupe1, dupe2


def pre_process(
        data: DataFrame,
        stop_compress_delta: float,
        normaliser: SegmentNormaliser,
        fraction_observed=1) -> DataFrame:

    data = compute_time_left(data)
    data = stop_compress(data, stop_compress_delta)
    data['tau'] = compute_tau(data)
    data = take_fraction(data, fraction_observed)
    dx, dy = compute_velocity_from_pos(data)
    data['dx'] = dx
    data['dy'] = dy
    data = normaliser.normalise(data)
    return data
