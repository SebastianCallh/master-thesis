
from typing import Tuple
from tqdm import tqdm
import numpy as np
from numpy import ndarray
from .plotting import plot_grid
from .trajectory_model import F_CODOMAIN
from .plotting import default_color
from .storage import acquire_db_conn
from .metric import time


def compute_errors(pred, truth):
    abs_diff = np.abs(pred - truth)
    mae = abs_diff
    mape = (100 * abs_diff) / np.sum(truth)
    return mae, mape


def evaluate(
    route_n, tau_grid, predict,
    load_models, n_models,
    seg_loader, seg_ns, traj_ns) -> Tuple[ndarray, ndarray, ndarray]:

    #seg_ns = data.seg.unique()
    #traj_ns = data.traj.unique()
    shape = (len(seg_ns), len(tau_grid), len(traj_ns))
    maes, mapes, misclass = np.empty(shape), np.empty(shape), np.empty(shape)
    #print('unique trajs', traj_ns)
    for i, seg_n in enumerate(tqdm(seg_ns)):
        #seg = data[data.seg == seg_n]
        models = time(lambda: load_models(
            route_n, seg_n, n_models
        ), 'load models')

        #print('seg n', seg_n)
        for j, tau in enumerate(tau_grid):

            #traj = seg[seg.traj == traj_n]
            #print('trajectory n', traj_n)
            #traj = time(lambda: pre_process(
            #    traj, seg_n, 1), 'pre-process')

            for k, traj_n in enumerate(tqdm(traj_ns)):
                observed = seg_loader(seg_n, traj_n, tau)
                #traj[traj.tau <= tau] pre_process
                X_obs = observed[F_CODOMAIN].values
                pred, pred_model_n = predict(models, X_obs)
                truth = observed.iloc[-1].time_left
                mae, mape = compute_errors(pred, truth)
                print('thes guys', pred_model_n, traj_n)
                misclass[i, j, k] = 1 if pred_model_n != traj_n else 0
                maes[i, j, k] = mae
                mapes[i, j, k] = mape

    return maes, mapes, misclass


def plot_route_performance(
        route_n, n_models, seg_ns,
        tau_grid, maes, mapes, misclass):

    # print(maes.shape, mapes.shape)
    fix, axs = plot_grid(1, 3)
    mae_ax = axs[0]
    mape_ax = axs[1]
    misclass_ax = axs[2]
    n_trajs = maes.shape[2]
    print(maes[1].shape)
    for i, seg_n in enumerate(seg_ns):

        tau_mean_maes = np.apply_along_axis(np.mean, 1, maes[i])
        tau_std_maes = np.apply_along_axis(np.std, 1, maes[i])
        mae_ax.plot(
            tau_grid, tau_mean_maes,
            marker='o',
            label='Segment {}'.format(seg_n),
            color=default_color(i)
        )

        alpha = .3
        mae_ax.fill_between(
            tau_grid,
            (tau_mean_maes + tau_std_maes),
            (tau_mean_maes - tau_std_maes),
            color=default_color(i), alpha=alpha
        )

        tau_mean_mapes = np.apply_along_axis(np.mean, 1, mapes[i])
        tau_std_mapes = np.apply_along_axis(np.std, 1, mapes[i])
        mape_ax.plot(
            tau_grid, tau_mean_mapes,
            marker='o',
            label='Segment {}'.format(seg_n),
            color=default_color(i)
        )

        mape_ax.fill_between(
            tau_grid,
            (tau_mean_mapes + tau_std_mapes),
            (tau_mean_mapes - tau_std_mapes),
            color=default_color(i), alpha=alpha
        )

        tau_misclass = np.apply_along_axis(np.mean, 1, misclass[i])
        misclass_ax.plot(
            tau_grid, tau_misclass,
            marker='o',
            label='Segment {}'.format(seg_n),
            color=default_color(i)
        )

        # mape_ax.plot(
        #     tau_grid, (tau_mean_mapes - tau_std_mapes),
        #     color=default_color(i), alpha=alpha
        # )

    title = 'Average {} of route {} segments using {} models and {} ' \
            'trajectories'
    mae_ax.set_title(title.format('MAE', route_n, n_models, n_trajs))
    mae_ax.set_ylabel('MAE (s)')
    mae_ax.set_xlabel(r'Observed $\tau$')
    mae_ax.legend()

    mape_ax.set_title(title.format('MAPE', route_n, n_models, n_trajs))
    mape_ax.set_ylabel('MAPE (%)')
    mape_ax.set_xlabel(r'Observed $\tau$')
    mape_ax.legend()

    misclass_ax.set_title('Misclassification rate of segments')
    misclass_ax.set_ylabel('Misclassification rate')
    misclass_ax.set_xlabel(r'Observed $\tau$')
    misclass_ax.legend()