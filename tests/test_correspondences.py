"""Tests for contour_reg.registration.correspondences (Task 3.2)."""
import numpy as np
import trimesh

from contour_reg.io.camera import Camera
from contour_reg.registration.correspondences import (
    compute_correspondences_parallel,
    compute_error_parallel,
    extract_silhouette_edges_fast,
)


def _cam():
    R = np.eye(3)
    return Camera(
        position=np.array([0, 0, -3.0]),
        rotation_world2cam=R,
        focal_length=800.0,
        principal_point=np.array([320.0, 240.0]),
        width=640,
        height=480,
    )


def test_silhouette_nonempty():
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    pts, idx = extract_silhouette_edges_fast(m, _cam())
    assert len(pts) > 0 and len(pts) == len(idx)


def test_error_zero_when_target_is_self_silhouette():
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    cam = _cam()
    pts, _ = extract_silhouette_edges_fast(m, cam)
    corr = compute_correspondences_parallel(m, [cam], [pts], n_jobs=1)
    mean, per_view = compute_error_parallel(m.vertices, [cam], corr, n_jobs=1)
    assert mean < 2.0  # projected silhouette already matches target
