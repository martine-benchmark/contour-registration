from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert 6D pose [rx, ry, rz, tx, ty, tz] to 4x4 homogeneous transform matrix.

    Args:
        pose: Array of shape (6,) with Euler angles (xyz convention) and translation.

    Returns:
        T: 4x4 homogeneous transform matrix.
    """
    R = Rotation.from_euler('xyz', pose[:3]).as_matrix()
    t = pose[3:6]

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 homogeneous transform matrix to 6D pose [rx, ry, rz, tx, ty, tz].

    Args:
        T: 4x4 homogeneous transform matrix.

    Returns:
        pose: Array of shape (6,) with Euler angles (xyz convention) and translation.
    """
    euler_xyz = Rotation.from_matrix(T[:3, :3]).as_euler('xyz')
    translation = T[:3, 3]

    pose = np.concatenate([euler_xyz, translation])

    return pose


def similarity_to_pose_scale(T: np.ndarray) -> Tuple[np.ndarray, float]:
    """Decompose a 4x4 similarity ``v -> s * R @ v + t`` into a 6D pose and scale.

    The transform must have a linear block ``A = s * R`` with ``R`` a proper
    rotation and ``s > 0`` a single uniform scale (as produced e.g. by a MeshLab
    alignment ``mlp_transform.npy``). This matches the convention applied by
    :func:`contour_reg.io.mesh.apply_transform_to_mesh_with_scale`
    (``scale * R @ v + t``), so the returned ``(pose, scale)`` can seed
    :func:`contour_reg.registration.icp.contour_based_icp` directly.

    Args:
        T: 4x4 homogeneous similarity transform.

    Returns:
        pose: shape (6,) ``[rx, ry, rz, tx, ty, tz]`` (Euler xyz for ``R``).
        scale: the uniform scale ``s``.

    Raises:
        ValueError: if ``T`` is not 4x4 or has a non-positive determinant
            (i.e. not a proper similarity).
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"similarity_to_pose_scale expects a 4x4 matrix, got {T.shape}")
    A = T[:3, :3]
    det = np.linalg.det(A)
    if det <= 0.0:
        raise ValueError(
            "similarity_to_pose_scale: non-positive determinant "
            f"({det:.3e}); not a proper similarity (reflection/degenerate)."
        )
    scale = float(det ** (1.0 / 3.0))
    R = A / scale
    euler_xyz = Rotation.from_matrix(R).as_euler('xyz')
    pose = np.concatenate([euler_xyz, T[:3, 3]])
    return pose, scale
