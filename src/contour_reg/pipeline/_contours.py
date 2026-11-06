"""Shared per-view target-contour construction.

Single source of truth for turning per-view masks/images into target contour
point arrays for a given :class:`~contour_reg.config.CannyConfig`.  Used by both
:mod:`contour_reg.pipeline.sweep` and :mod:`contour_reg.pipeline.run` so the
coarse (sweep) pose and the refined pose are optimised against *identical*
contours.

Duck-typing rules (mask-only path):
  - ``(N, 2)`` non-uint8 array → already-extracted contour points, used as-is.
  - 2-D uint8 / mask image     → boundary extracted via ``mask_boundary_contour``.

When *images* is provided, Canny edges are extracted per view+config instead.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from contour_reg.config import CannyConfig
from contour_reg.contours.canny import (
    contour_image_to_points,
    extract_canny_contours,
    mask_boundary_contour,
)


def _is_contour_array(arr: np.ndarray) -> bool:
    """Return True if *arr* is already an ``(N, 2)`` contour point array.

    Duck-typing: a 2-column non-uint8 numeric array is treated as
    pre-extracted contours; any other 2-D array is treated as a mask image.
    """
    return (
        isinstance(arr, np.ndarray)
        and arr.ndim == 2
        and arr.shape[1] == 2
        and arr.dtype != np.uint8  # mask images are uint8
    )


def build_target_contours(
    cfg: CannyConfig,
    masks: List[np.ndarray],
    images: Optional[List[np.ndarray]],
) -> List[np.ndarray]:
    """Build per-view target contour point arrays for *cfg*.

    Args:
        cfg: Canny configuration controlling thresholds / margin / kernel.
        masks: One entry per view: either a ``(N, 2)`` contour array or a
            binary mask image.  Ignored when *images* is provided.
        images: Optional list of grayscale images (one per view).  When
            provided, Canny edges are extracted per image+mask and converted
            to points.

    Returns:
        List of ``(N, 2)`` contour point arrays, one per view.
    """
    target_contours: List[np.ndarray] = []

    for i, mask in enumerate(masks):
        if images is not None:
            edge_img = extract_canny_contours(
                images[i],
                mask,
                margin_size=cfg.margin_size,
                canny_low=cfg.canny_low,
                canny_high=cfg.canny_high,
                sobel_kernel=cfg.sobel_kernel,
            )
            pts = contour_image_to_points(edge_img)
        else:
            if _is_contour_array(mask):
                pts = mask
            else:
                pts = mask_boundary_contour(mask, include_inner=cfg.include_inner_contours)
        target_contours.append(pts)

    return target_contours
