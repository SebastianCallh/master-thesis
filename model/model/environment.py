
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from pandas import DataFrame
from .storage import load_pickle
from .pre_process import pre_process
from .segment_normaliser import normaliser_for_seg

set_matplotlib_formats('pdf', quality=90)

n_hyperparam_trajs = 100
n_train_trajs = 5000

plt.style.use('seaborn')
np.random.seed(1)
trajs_pickle_path = '../data/train3.pkl'
data = pd.read_pickle(trajs_pickle_path)
n_trajs = list(data.traj.unique())
np.random.shuffle(n_trajs)
train_traj_ids = frozenset(n_trajs[:n_train_trajs])
hyperparam_traj_ids = frozenset(
    n_trajs[n_train_trajs:(n_train_trajs + n_hyperparam_trajs)])
hyperparam_trajs = data[data.traj.transform(lambda j: j in hyperparam_traj_ids)]
train_trajs = data[data.traj.transform(lambda j: j in train_traj_ids)]
train_traj_ids = train_trajs.traj.unique()
train_seg_ids = list(train_trajs.seg.unique())
seg_ns = train_seg_ids

# REMOVE ME
# train_trajs = hyperparam_trajs
# train_traj_ids = hyperparam_trajs.traj.unique()
# train_seg_ids = hyperparam_trajs.seg.unique()

# Segment normaliser
seg_dict = dict(tuple(data.groupby('seg')))
seg_normalisers = {
    seg_n: normaliser_for_seg(seg_dict[seg_n])
    for seg_n in seg_dict
}


def load_seg(df: DataFrame, seg_n: int, traj_id: int, frac_observed=1):
    # print('loading segment', seg_n, traj_id, frac_observed)
    return pre_process(
        df[(df.traj == traj_id) & (df.seg == seg_n)],
        stop_compress_delta,
        seg_normalisers[seg_n],
        frac_observed
    )


def from_km(km):
    return km * 1000


def from_km_h(km_h):
    """To m/s"""
    return km_h / 3.6

hyperparams = load_pickle('hyperparams')

stop_compress_delta = 4  # meters
f_p_codomain = ['x', 'y']
f_v_codomain = ['dx', 'dy']
f_p_sigma_n = .0001  # meters
f_v_sigma_n = from_km_h(.000001)  # m/s
g_sigma_n = .0001  # tau is deterministic
h_sigma_n = 1  # seconds
delta_xy = 4  # metres, spatial cluster width
delta_p = 4  # metres, p cluster width
delta_v = from_km_h(1)  # metres/second, v cluster width
route_n = 3