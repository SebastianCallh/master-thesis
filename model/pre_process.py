from numpy import ndarray
from pandas import DataFrame
from model import stop_compress
import numpy as np
import pandas as pd
from segment_normaliser import SegmentNormaliser

# https://stackoverflow.com/questions/52907328/pandas-delete-first-n-rows-until-condition-on-columns-is-fulfilled
def drop_start_dwell(data: DataFrame):
    return data.loc[((data.dx > 0) & (data.dy > 0)).idxmax():]


def take_fraction(data, r):
    return data[data.tau <= r]
    #n = int(np.round(r*data.shape[0]))
    #return data.iloc[:n]


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


def compute_velocity_from_pos(data: DataFrame):
    """Assume one data point per second."""
    def velocity(cur, nxt):
        #dt = nxt.timestamp.second - cur.timestamp.second
        dxx = float(nxt.x - cur.x) #/ dt
        dyy = float(nxt.y - cur.y) #/ dt
       # print(nxt.y, cur.y)
        return dxx, dyy #if dt > 0 else None

    dx = []
    dy = []
    for n in range(0, data.shape[0]-1):
        cur = data.iloc[n]
        nxt = data.iloc[n+1]
        dv = velocity(cur, nxt)
        
        # If two observations are sent the same 
        # seconds, everything breaks
        #if dv is None:
        #    nxt2 = data.iloc[n+2]
        #    dv = velocity(cur, nxt2)
        

        dx.append(dv[0])
        dy.append(dv[1])
            
    dx.append(dx[-1])
    dy.append(dy[-1])
    return dx, dy


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
