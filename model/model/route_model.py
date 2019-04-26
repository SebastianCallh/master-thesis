'Model of several igp models in sequence'
from typing import List
from numpy import ndarray
from pandas import DataFrame
import numpy as np

from .seg_model import SegmentModel, load_seg_model
from .trajectory_model import F_CODOMAIN

FAIRLY_LARGE_LIMIT = 10000

class RouteModel:
    route_n: int
    seg_models: List[SegmentModel]

    def __init__(self, route_n: int, seg_models: List[SegmentModel]) -> None:
        super().__init__()
        self.route_n = route_n
        self.seg_models = seg_models

    def __len__(self):
        return len(self.seg_models)

    def predict(self, from_seg: int, forward_segs: int,
                observations: List[DataFrame]) -> ndarray:

        models = self.seg_models[from_seg: from_seg + forward_segs]

        # The length should be uniform
        assert len(set([len(m) for m in models])) == 1
        n_traj_models = len(models[0])

        posteriors = np.zeros((len(self), 2, n_traj_models))
        for i, (m, obs) in enumerate(zip(models, observations)):
            x_obs = obs[F_CODOMAIN].values
            prior_vars = posteriors[i-1, 1, :]
            mus, vars = m.arrival_time_posterior(x_obs)
            posteriors[i] = np.array((mus, vars + prior_vars))

        return posteriors


def load(route_n: int, seg_ns: List[int], trajectory_limit=FAIRLY_LARGE_LIMIT):
    models = [load_seg_model(route_n, seg_n, limit=trajectory_limit) for seg_n in seg_ns]
    return RouteModel(route_n, models)
