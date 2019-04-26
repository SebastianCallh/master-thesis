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
    normalise_x: Callable[[ndarray], ndarray]
    normalise_y: Callable[[ndarray], ndarray]
    normalise_dx: Callable[[ndarray], ndarray]
    normalise_dy: Callable[[ndarray], ndarray]
    unnormalise_x: Callable[[ndarray], ndarray]
    unnormalise_y: Callable[[ndarray], ndarray]
    unnormalise_dx: Callable[[ndarray], ndarray]
    unnormalise_dy: Callable[[ndarray], ndarray]


def normaliser_for_seg(seg: DataFrame) -> SegmentNormaliser:
    d_px, d_py = seg.x.mean(), seg.y.mean()
    sd_p = max(seg.x.std(), seg.y.std())

    d_vx, d_vy = seg.dx.mean(), seg.dy.mean()
    sd_v = max(seg.dx.std(), seg.dy.std())

    normalise_x = lambda x: (x - d_px) / sd_p
    normalise_y = lambda y: (y - d_py) / sd_p
    normalise_dx = lambda dx: (dx - d_vx) / sd_v
    normalise_dy = lambda dy: (dy - d_vy) / sd_v

    unnormalise_x = lambda x: sd_p * x + d_px
    unnormalise_y = lambda y: sd_p * y + d_py
    unnormalise_dx = lambda dx: sd_v * dx + d_vx
    unnormalise_dy = lambda dy: sd_v * dy + d_vy

    return SegmentNormaliser(
        p_translation=np.array([d_px, d_py]),
        p_scale=sd_p,
        v_translation=np.array([d_vx, d_vy]),
        v_scale=sd_v,
        normalise_x=normalise_x,
        normalise_y=normalise_y,
        normalise_dx=normalise_dx,
        normalise_dy=normalise_dy,
        unnormalise_x=unnormalise_x,
        unnormalise_y=unnormalise_y,
        unnormalise_dx=unnormalise_dx,
        unnormalise_dy=unnormalise_dy
    )
