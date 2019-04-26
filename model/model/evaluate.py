
from typing import Tuple, Callable, List

from pandas import DataFrame
from scipy.integrate import quad
from tqdm import tqdm
import numpy as np
from numpy import ndarray
from .plotting import plot_grid
from .trajectory_model import F_CODOMAIN
from .seg_model import gp_logliks, to_model_probabilities, \
    arrival_time_prediction, \
    most_probable_models, mixture_of_gps_predictor, SegmentModel
from .plotting import default_color
from scipy.stats import entropy
from scipy.spatial.distance import euclidean

SQRT2 = np.sqrt(2)

def compute_errors(pred, truth) -> Tuple[ndarray, ndarray]:
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
        print('loading for', route_n, seg_n, n_models)
        models = load_models(route_n, seg_n, n_models)

        #print('seg n', seg_n)
        for j, tau in enumerate(tau_grid):

            #traj = seg[seg.traj == traj_n]
            #print('trajectory n', traj_n)
            #traj = time(lambda: pre_process(
            #    traj, seg_n, 1), 'pre-process')

            for k, traj_n in enumerate(tqdm(traj_ns)):
                print('loading', seg_n, traj_n, tau)
                observed = seg_loader(seg_n, traj_n, tau)
                #traj[traj.tau <= tau] pre_process
                X_obs = observed[F_CODOMAIN].values
                pred, pred_model_n = predict(models, X_obs)
                truth = observed.iloc[-1].time_left
                mae, mape = compute_errors(pred, truth)
                print('pred model n', pred_model_n, 'traj n', traj_n)
                misclass[i, j, k] = 1 if pred_model_n != traj_n else 0
                maes[i, j, k] = mae
                mapes[i, j, k] = mape

    return maes, mapes, misclass


def evaluate2(
        route_n: int,
        seg_ns: ndarray,
        traj_ids: ndarray,
        tau_grid: ndarray,
        load_model: Callable[[int, int], SegmentModel],
        load_seg: Callable[[int, int], DataFrame]
    ) -> Tuple[ndarray, ndarray, ndarray]:

    def model_cum_likelihood(model: SegmentModel, X_obs: ndarray):
        gp_likelihoods = gp_logliks(model, X_obs)
        return to_model_probabilities(gp_likelihoods)

    shape = (len(seg_ns), len(tau_grid), len(traj_ids))
    maes, mapes, misclass = np.empty(shape), np.empty(shape), np.empty(shape)
    for i, seg_n in enumerate(tqdm(seg_ns)):
        models = load_model(route_n, seg_n)

        for j, traj_id in enumerate(traj_ids):
            #print('loading for', route_n, seg_n)
            seg = load_seg(seg_n, traj_id)
            X_obs = seg[F_CODOMAIN].values
            lls = model_cum_likelihood(models, X_obs)
            map_models = [models[i] for i in lls.argmax(axis=0)]
            breakpoints = [
                seg[seg.tau >= tau].index[0]
                for tau in tau_grid
            ]
            for k, bp in enumerate(breakpoints):
                subseg = seg.iloc[:bp]
                model = map_models[bp]
                X_obs = subseg[F_CODOMAIN].values
                pred, _ = arrival_time_prediction(model, X_obs)
                truth = subseg.iloc[-1].time_left
                mae, mape = compute_errors(pred, truth)
                misclass[i, k, j] = 1 if model.traj != traj_id else 0
                maes[i, k, j] = mae
                mapes[i, k, j] = mape

    return maes, mapes, misclass


def evaluate_mixture_prediction(
        route_n: int,
        seg_ns: ndarray,
        traj_ids: ndarray,
        tau_grid: ndarray,
        load_model: Callable[[int, int], SegmentModel],
        load_seg: Callable[[int, int], DataFrame]
    ) -> Tuple[ndarray, ndarray]:

    shape = (len(seg_ns), len(tau_grid), len(traj_ids))
    maes, mapes, misclass = np.empty(shape), np.empty(shape), np.empty(shape)
    for i, seg_n in enumerate(tqdm(seg_ns)):
        model = load_model(route_n, seg_n)

        for j, traj_id in enumerate(traj_ids):
            #print('loading for', route_n, seg_n)
            seg = load_seg(seg_n, traj_id)
            X_obs = seg[F_CODOMAIN].values
            breakpoints = [
                seg[seg.tau >= tau].index[0]
                for tau in tau_grid
            ]
            for k, bp in enumerate(breakpoints):
                subseg = seg.iloc[:bp]
                X_obs = subseg[F_CODOMAIN].values
                pred = mixture_of_gps_predictor(model, X_obs)
                truth = subseg.iloc[-1].time_left
                mae, mape = compute_errors(pred, truth)
                #misclass[i, k, j] = 1 if model.traj != traj_id else 0
                maes[i, k, j] = mae
                mapes[i, k, j] = mape

    return maes, mapes #, misclass


def hellinger2_quad(
        p: Callable[[float], float],
        q: Callable[[float], float],
        xmin: float, xmax: float) -> float:
    """Hellinger^2 for continuous functions"""
    hellinger_integrand = lambda p, q: lambda x: np.sqrt(p(x) * q(x))
    dist, err = quad(hellinger_integrand(p, q), xmin, xmax)
    if err > 1e-10:
        print('error was large', err)

    return dist


def normalise_args(p, q):
    return p / p.sum(), q / q.sum()


def hellinger2(p: ndarray, q: ndarray) -> float:
    """Hellinger^2 for discrete samples"""
    p, q = normalise_args(p, q)
    return euclidean(np.sqrt(p), np.sqrt(q))**2 / 2


def kl_divergence(p: ndarray, q: ndarray):
    p, q = normalise_args(p, q)
    return entropy(p, q)


def js_divergence(p: ndarray, q: ndarray):
    p, q = normalise_args(p, q)
    m = .5*(p + q)
    return .5*(kl_divergence(p, m) + kl_divergence(q, m))


def is_within_band(truth, X_obs, map_model):
    assert X_obs.shape[0] > 0
    mus_t, vars_t = arrival_time_prediction(map_model, X_obs[:, :2])
    sigmas_t = np.sqrt(vars_t)
    upper = (mus_t + 2 * sigmas_t).reshape(-1)
    lower = (mus_t - 2 * sigmas_t).reshape(-1)
    within_bands = int(((lower < truth) & (truth < upper)) + 0)
    return within_bands, lower, upper


def within_bands_vector(seg, breakpoints, models):
    """Returns a vector with either 0 or 1
    depending on if observation is inside probability bands"""
    assert breakpoints[0] > 0

    X_obs = seg[F_CODOMAIN].values
    most_prob_models = [models[n] for n in most_probable_models(X_obs, models)]
    return np.array([
        is_within_band(
            seg.iloc[bp].time_left,
            X_obs[bp, :].reshape(1, 4),
            most_prob_models[bp]
        )[0] for bp in breakpoints
    ])


def plot_route_performance(
        route_n, n_models, seg_ns,
        tau_grid, maes, mapes, misclass):

    # print(maes.shape, mapes.shape)
    fix, axs = plot_grid(1, 2)
    mae_ax = axs[0]
    mape_ax = axs[1]
    #misclass_ax = axs[2]
    n_trajs = maes.shape[2]
    #print(maes[1].shape)
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

        # tau_misclass = np.apply_along_axis(np.mean, 1, misclass[i])
        # misclass_ax.plot(
        #     tau_grid, tau_misclass,
        #     marker='o',
        #     label='Segment {}'.format(seg_n),
        #     color=default_color(i)
        # )

        # mape_ax.plot(
        #     tau_grid, (tau_mean_mapes - tau_std_mapes),
        #     color=default_color(i), alpha=alpha
        # )

    title = 'Average {} of route {} segments using {} models and {} ' \
            'trajectories'
    mae_ax.set_title(title.format('MAE', route_n, n_models, n_trajs))
    mae_ax.set_ylabel('MAE (s)')
    mae_ax.set_xlabel(r'Observed $\tau$')
    mae_ax.set_ylim(0, 50)
    mae_ax.legend()

    mape_ax.set_title(title.format('MAPE', route_n, n_models, n_trajs))
    mape_ax.set_ylabel('MAPE (%)')
    mape_ax.set_xlabel(r'Observed $\tau$')
    mape_ax.set_ylim(0, 100)
    mape_ax.legend()

    # misclass_ax.set_title('Misclassification rate of segments')
    # misclass_ax.set_ylabel('Misclassification rate')
    # misclass_ax.set_xlabel(r'Observed $\tau$')
    # misclass_ax.legend()