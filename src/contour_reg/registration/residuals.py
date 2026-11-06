"""Contour-ICP residual functions.

The residual for a view is the pixel distance from each 2D contour point to the
nearest projected mesh-silhouette point; residuals are stacked over all views.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors

from contour_reg.io.camera import Camera


def _view_residuals(projected_2d: np.ndarray, contour_2d: np.ndarray) -> np.ndarray:
    """Distance from each contour point to its nearest projected silhouette point."""
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(projected_2d)
    distances, _ = nn.kneighbors(contour_2d)
    return distances.flatten()


def residual_function_optimized(
    params: np.ndarray,
    mesh_vertices: np.ndarray,
    cameras: List[Camera],
    fixed_correspondences: Dict,
) -> np.ndarray:
    """Residuals for a rigid pose ``[rx, ry, rz, tx, ty, tz]`` (pixels)."""
    euler_angles = params[:3]
    translation = params[3:6]

    rot_matrix = R.from_euler('xyz', euler_angles).as_matrix()
    vertices_transformed = (rot_matrix @ mesh_vertices.T).T + translation

    all_residuals: List[float] = []
    for cam_idx, camera in enumerate(cameras):
        corr = fixed_correspondences.get(cam_idx)
        if corr is None:
            continue
        indices_3d = corr['indices_3d']
        contour_2d = corr['points_2d']
        if len(indices_3d) == 0 or len(contour_2d) == 0:
            continue

        projected_2d = camera.project_points(vertices_transformed[indices_3d])
        all_residuals.extend(_view_residuals(projected_2d, contour_2d))

    if len(all_residuals) == 0:
        return np.array([1e6])
    return np.array(all_residuals)


def residual_function_with_scale(
    params: np.ndarray,
    mesh_vertices: np.ndarray,
    cameras: List[Camera],
    fixed_correspondences: Dict,
) -> np.ndarray:
    """Residuals for a similarity ``[rx, ry, rz, tx, ty, tz, scale]``.

    Transformation: ``T(v) = scale * R * v + t``.
    """
    euler_angles = params[:3]
    translation = params[3:6]
    scale = params[6]

    rot_matrix = R.from_euler('xyz', euler_angles).as_matrix()
    vertices_transformed = scale * (rot_matrix @ mesh_vertices.T).T + translation

    all_residuals: List[float] = []
    for cam_idx, camera in enumerate(cameras):
        corr = fixed_correspondences.get(cam_idx)
        if corr is None:
            continue
        indices_3d = corr['indices_3d']
        contour_2d = corr['points_2d']
        if len(indices_3d) == 0 or len(contour_2d) == 0:
            continue

        projected_2d = camera.project_points(vertices_transformed[indices_3d])
        all_residuals.extend(_view_residuals(projected_2d, contour_2d))

    if len(all_residuals) == 0:
        return np.array([1e6])
    return np.array(all_residuals)
