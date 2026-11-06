"""Leave-one-view-out (LOO) GT-free registration evaluation.

Assesses registration quality WITHOUT ground-truth pose by holding out one view
at a time, registering on the remaining N-1 views, and measuring reprojection
error on the held-out view.  This is a self-consistency / generalization metric:
low LOO error means the pose estimated from any subset of views predicts the
appearance in unseen views well.

Standalone subcommand, not part of the main validation pipeline.

Usage::

    from contour_reg.pipeline.loo import run_loo
    result = run_loo(mesh, cameras, target_contours, init_pose, icp_config)
    # result == {'per_view': [{'view': i, 'held_out_error_px': float}, ...],
    #            'median': float, 'iqr': float}
"""
from __future__ import annotations

from typing import List

import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors

from contour_reg.config import ICPConfig
from contour_reg.io.camera import Camera
from contour_reg.io.mesh import apply_transform_to_mesh_with_scale
from contour_reg.registration.correspondences import extract_silhouette_edges_fast
from contour_reg.registration.icp import contour_based_icp


def _held_out_stats(mesh, camera, contour, pose, scale) -> dict | None:
    """Held-out contour->silhouette distance stats (scale-aware, reversed dir)."""
    mesh_t = apply_transform_to_mesh_with_scale(mesh, pose[:3], pose[3:6], scale)
    silhouette, _ = extract_silhouette_edges_fast(mesh_t, camera)
    if len(silhouette) == 0 or len(contour) == 0:
        return None
    d = NearestNeighbors(n_neighbors=1).fit(silhouette).kneighbors(contour)[0].ravel()
    return {
        "held_out_error_px": float(d.mean()),
        "median_px": float(np.median(d)),
        "pct_lt_1px": float((d <= 1.0).mean() * 100),
        "pct_lt_2px": float((d <= 2.0).mean() * 100),
        "pct_lt_5px": float((d <= 5.0).mean() * 100),
    }


def run_loo(
    mesh: trimesh.Trimesh,
    cameras: List[Camera],
    target_contours: List[np.ndarray],
    init_pose: np.ndarray,
    icp_config: ICPConfig,
    *,
    min_canny_points: int = 0,
    max_correspondence_distance: float = 0.0,
    optimize_scale: bool = False,
    initial_scale: float = 1.0,
) -> dict:
    """Leave-one-view-out GT-free evaluation.

    For each view index *i*:

    1. Build the **training set** = all views *j != i* for which
       ``len(target_contours[j]) >= min_canny_points`` (when
       ``min_canny_points == 0`` no training view is skipped on point count).
    2. Register on the training subset via :func:`contour_based_icp`.
    3. Measure the held-out reprojection error on view *i*:

       - If view *i* has 0 target points, the error is recorded as ``inf``
         (the camera is present but has no observable contour, so evaluation
         is undefined; documented as ``inf`` rather than silently skipped so
         the caller can decide how to handle it).
       - If view *i* has between 1 and ``min_canny_points - 1`` points, it is
         still evaluated (the threshold only gates *training* views, not the
         test view).

    This method requires **no ground-truth pose**: it measures whether the
    registration generalises across views.

    Args:
        mesh: Source mesh (untransformed).
        cameras: List of :class:`~contour_reg.io.camera.Camera` objects.
        target_contours: One ``(N_i, 2)`` contour point array per camera.
        init_pose: Initial pose ``[rx, ry, rz, tx, ty, tz]``.
        icp_config: ICP hyper-parameters (``max_iterations``,
            ``error_percentile``, ``include_inner_contours`` are forwarded;
            ``max_correspondence_distance`` on :class:`ICPConfig` is ignored , 
            use the keyword argument instead to make the dependency explicit).
        min_canny_points: Minimum number of contour points a *training* view
            must have to be included in the training set.  ``0`` disables
            filtering (default).
        max_correspondence_distance: Correspondence distance cutoff forwarded
            to ICP (pixels).  ``0`` disables filtering (default).

    Returns:
        Dict with keys:

        - ``'per_view'``: list of ``{'view': int, 'held_out_error_px': float}``,
          one entry per view (including those with ``inf`` error).
        - ``'median'``: median of the **finite** held-out errors.
        - ``'iqr'``: interquartile range (75th − 25th percentile) of finite
          held-out errors.

    Raises:
        ValueError: If fewer than 2 views are provided (need at least one
            training view and one held-out view).
    """
    n_views = len(cameras)
    if n_views < 2:
        raise ValueError(
            f"run_loo requires at least 2 views, got {n_views}."
        )

    per_view_results = []

    for i in range(n_views):
        # Build training set: all views except i, filtered by min_canny_points.
        train_cameras = []
        train_contours = []
        for j in range(n_views):
            if j == i:
                continue
            if len(target_contours[j]) < min_canny_points:
                continue
            train_cameras.append(cameras[j])
            train_contours.append(target_contours[j])

        # Held-out view has 0 contour points → evaluation undefined.
        held_out_contour = target_contours[i]
        if len(held_out_contour) == 0:
            per_view_results.append({'view': i, 'held_out_error_px': float('inf')})
            continue

        # If no training views remain (all filtered out), skip gracefully.
        if not train_cameras:
            per_view_results.append({'view': i, 'held_out_error_px': float('inf')})
            continue

        # Register on training subset.
        pose_i, _, scale_i = contour_based_icp(
            mesh,
            init_pose,
            train_cameras,
            train_contours,
            max_iterations=icp_config.max_iterations,
            error_threshold_percentile=icp_config.error_percentile,
            max_correspondence_distance=max_correspondence_distance,
            include_inner_contours=icp_config.include_inner_contours,
            optimize_scale=optimize_scale,
            initial_scale=initial_scale,
            verbose=False,
        )

        # Measure held-out error on view i (scale-aware, contour->silhouette).
        stats = _held_out_stats(mesh, cameras[i], held_out_contour, pose_i, scale_i)
        if stats is None:
            per_view_results.append({'view': i, 'held_out_error_px': float('inf')})
        else:
            per_view_results.append({'view': i, **stats})

    # Aggregate over finite errors.
    finite_errors = np.array(
        [r['held_out_error_px'] for r in per_view_results
         if np.isfinite(r['held_out_error_px'])],
        dtype=float,
    )

    if len(finite_errors) == 0:
        median_err = float('inf')
        iqr_err = float('nan')
    else:
        median_err = float(np.median(finite_errors))
        iqr_err = float(
            np.percentile(finite_errors, 75) - np.percentile(finite_errors, 25)
        )

    return {
        'per_view': per_view_results,
        'median': median_err,
        'iqr': iqr_err,
    }
