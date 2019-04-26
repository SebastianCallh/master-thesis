from pandas import DataFrame
from typing import Tuple, Callable

from .route_model import RouteModel
from .plotting import plot_grid
from .seg_model import mixture_of_gps_predictor, gp_logliks, SegmentModel, \
    prior_density_from_params
from .trajectory_model import F_CODOMAIN, TrajectoryModel, learn_model

from .evaluate import compute_errors
from .evaluate import hellinger2, js_divergence
from .environment import train_trajs, load_seg, hyperparams, seg_normalisers
from .environment import f_p_sigma_n, f_v_sigma_n, delta_xy, delta_p, \
    delta_v

from scipy.stats import gaussian_kde
from tqdm import tqdm
import numpy as np

# predict_arrival_time = lambda models, x: most_probable_model_predictor(models, x)[0]
predict_arrival_time = mixture_of_gps_predictor


def time_left(traj):
    return (traj.iloc[-1].timestamp - traj.iloc[0].timestamp).seconds


def seg_travel_times(seg):
    seg_traj_ids = seg.traj.unique()
    return np.array(
        [time_left(seg[seg.traj == traj_id]) for traj_id in seg_traj_ids])


def grid_for(data, grid_resolution) -> np.ndarray:
    sd = data.std()
    return np.linspace(data.min() - sd, data.max() + sd, grid_resolution)


def predict(models, seg) -> float:
    X_obs = seg[F_CODOMAIN].values
    return predict_arrival_time(models, X_obs)


def traj_mae(model: SegmentModel, seg: DataFrame) -> np.ndarray:
    pred = model.predict_arrival_time(seg)
    truth = seg.iloc[-1].time_left
    mae, _ = compute_errors(pred, truth)
    return mae


def traj_likelihood(model: SegmentModel, seg: DataFrame) -> np.ndarray:
    X_obs = seg[F_CODOMAIN].values
    # Count only velocity models
    return gp_logliks(model, X_obs)[2:4, :].sum(axis=0).sum(axis=1)


def learn_model_with_defaults(route_n: int, seg_n: int, train_traj_id: int) \
        -> TrajectoryModel:
    normaliser = seg_normalisers[seg_n]
    scaled_f_p_sigma_n = f_p_sigma_n / normaliser.p_scale  # meters
    scaled_f_v_sigma_n = f_v_sigma_n / normaliser.v_scale  # m/s
    g_sigma_n = .0001  # tau is deterministic
    h_sigma_n = 1  # seconds
    scaled_delta_xy = delta_xy / normaliser.p_scale  # metres, spatial cluster width
    scaled_delta_p = delta_p / normaliser.p_scale  # metres, p cluster width
    scaled_delta_v = delta_v / normaliser.v_scale  # metres/second, v cluster widt

    return learn_model(
        load_seg(train_trajs, seg_n, train_traj_id),
        route_n, seg_n, train_traj_id,
        scaled_f_p_sigma_n, scaled_f_v_sigma_n,
        g_sigma_n, h_sigma_n,
        scaled_delta_xy, scaled_delta_p, scaled_delta_v,
        hyperparams[seg_n],
        n_restarts=3
    )


def learn_trajectories_by_strategy(
        seg_n: int,
        n_models_to_train: int,
        n_trajs_to_use: int,
        strategy: Callable[[np.ndarray, np.ndarray], int],
        fraction_observed: float
    ) -> Tuple[SegmentModel,
               np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:

    route_n = 3
    seg = train_trajs[train_trajs.seg == seg_n]
    traj_ids = seg.traj.unique()[:n_trajs_to_use]
    time_lefts = seg_travel_times(seg)
    xx = grid_for(time_lefts, 10000)
    empirical_distribution = gaussian_kde(time_lefts)(xx)
    empirical_distribution = empirical_distribution / empirical_distribution.sum()

    # Learn initial model
    initial_model = [learn_model_with_defaults(route_n, seg_n, traj_ids[0])]
    model = SegmentModel(route_n, seg_n, initial_model)
    n_models_to_train -= 1  # Already trained an initial model

    trajs = [
        load_seg(train_trajs, seg_n, traj_id, frac_observed=fraction_observed)
        for traj_id in traj_ids
    ]

    hel_dists = np.ndarray((n_models_to_train,))
    js_divs = np.ndarray((n_models_to_train,))
    likelihoods = np.ndarray((n_models_to_train, len(trajs)))
    maes = np.ndarray((n_models_to_train, len(trajs)))

    # If early stop, this is updated
    stopped_at = n_models_to_train
    already_learned = set()
    for i in tqdm(range(n_models_to_train)):
        # if things bugs out, check these lines
        model_maes = np.ndarray((len(trajs),))
        model_likelihoods = np.ndarray((len(trajs),))
        for j, traj in enumerate(trajs):
            model_maes[j] = traj_mae(model, traj)
            model_likelihoods[j] = traj_likelihood(model, traj).max()

        model_prior_params = model.arrival_time_prior()
        prior_density = prior_density_from_params(xx, model_prior_params)
        p, q = prior_density, empirical_distribution
        hel_dists[i] = hellinger2(p, q)
        js_divs[i] = js_divergence(p, q)
        likelihoods[i, :] = model_likelihoods
        maes[i, :] = model_maes

        traj_to_learn_n = strategy(model_maes, model_likelihoods)
        tran_to_learn_id = traj_ids[traj_to_learn_n]

        # After learned, a trajectory should never be least explained
        if tran_to_learn_id in already_learned:
            print('learned same traj again', tran_to_learn_id)
            stopped_at = i
            break

        already_learned.add(tran_to_learn_id)

        print('worst traj', tran_to_learn_id,
              'max mae', model_maes.max(),
              'min likelihood', model_likelihoods.min())

        model.add_model(learn_model_with_defaults(
            route_n, seg_n, tran_to_learn_id))

    _, ax = plot_grid(1, 1)
    ax.plot(xx, empirical_distribution, label='Empirical distribution')
    ax.plot(xx, prior_density, label='Model prior')
    ax.legend()

    j = stopped_at
    return model, hel_dists[:j], js_divs[:j], maes[:j], likelihoods[:j]


def posterior_predictive_over_entire_route(model: RouteModel):
    model.predict(from_seg=1, forward_segs=4, )