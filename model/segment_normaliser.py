"""Normalisation per segment."""

from math import sqrt
from pandas import DataFrame
from numpy import ndarray
from typing import NamedTuple, Callable
import numpy as np

class SegmentNormaliser(NamedTuple):
    p_translation: ndarray
    p_scale: float
    v_translation: ndarray
    v_scale: float
    normalise: Callable[[DataFrame], DataFrame]

def normaliser_for_seg(seg: DataFrame) -> SegmentNormaliser:
    d_px = seg.x.mean()
    d_py = seg.y.mean()
    sd_p = sqrt(max(seg.x.var(), seg.y.var()))

    d_vx = seg.dx.mean()
    d_vy = seg.dy.mean()
    sd_v = sqrt(max(seg.dx.var(), seg.dy.var()))

    def do_normalisation(data: DataFrame) -> DataFrame:
        data2 = data.copy()
        data2.x = (data.x - d_px)/sd_p
        data2.y = (data.y - d_py)/sd_p
        data2.dx = (data.dx - d_vx)/sd_v 
        data2.dy = (data.dy - d_vy)/sd_v
        return data2

    return SegmentNormaliser(
        p_translation=np.array([d_px, d_py]),
        p_scale=sd_p,
        v_translation=np.array([d_vx, d_vy]),
        v_scale=sd_v,
        normalise=do_normalisation
    )
