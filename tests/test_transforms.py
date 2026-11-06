import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from contour_reg.registration.transforms import (
    matrix_to_pose,
    pose_to_matrix,
    similarity_to_pose_scale,
)


def test_roundtrip():
    pose = np.array([0.1, -0.2, 0.3, 1.0, 2.0, 3.0])
    T = pose_to_matrix(pose)
    assert np.allclose(matrix_to_pose(T), pose, atol=1e-8)


def _make_similarity(euler, t, s):
    R = Rotation.from_euler("xyz", euler).as_matrix()
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def test_similarity_to_pose_scale_roundtrip():
    euler = np.array([0.1, -0.2, 0.3])
    t = np.array([-0.07, 0.11, 0.05])
    s = 0.0019
    T = _make_similarity(euler, t, s)
    pose, scale = similarity_to_pose_scale(T)

    assert scale == pytest.approx(s, rel=1e-9)
    assert np.allclose(pose[3:6], t, atol=1e-9)
    # The decomposed (pose, scale) must reproduce T's action s*R@v + t.
    R = Rotation.from_euler("xyz", pose[:3]).as_matrix()
    v = np.random.default_rng(0).normal(size=(8, 3))
    lhs = (T[:3, :3] @ v.T).T + T[:3, 3]
    rhs = scale * (R @ v.T).T + pose[3:6]
    assert np.allclose(lhs, rhs, atol=1e-9)


def test_similarity_to_pose_scale_rejects_reflection():
    T = _make_similarity(np.zeros(3), np.zeros(3), 1.0)
    T[:3, 0] *= -1.0  # improper (determinant < 0)
    with pytest.raises(ValueError):
        similarity_to_pose_scale(T)


def test_similarity_to_pose_scale_rejects_wrong_shape():
    with pytest.raises(ValueError):
        similarity_to_pose_scale(np.eye(3))
