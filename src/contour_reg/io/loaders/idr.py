"""IDR / NeuS cameras.npz loader.

Reads a cameras.npz file with keys ``world_mat_{i}`` (3×4 or 4×4 projection
matrix P = K @ [R|t]) and optional ``scale_mat_{i}`` (4×4 normalization
transform).  Returns a list of :class:`~contour_reg.io.camera.Camera` objects.

Ported from MVSCPS/dataloader/load_fn_idr.py (``load_K_Rt_from_P``,
``load_cameras_idr``).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from contour_reg.io.camera import Camera

logger = logging.getLogger(__name__)


def load_K_Rt_from_P(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose a projection matrix into intrinsics K and camera-to-world C2W.

    Args:
        P: (3, 4) or (4, 4) projection matrix (K @ [R | t]).

    Returns:
        K:   (3, 3) intrinsics matrix (normalised so K[2,2] == 1).
        C2W: (4, 4) camera-to-world pose matrix.
    """
    # Accept 4×4: slice to 3×4 as expected by cv2
    if P.shape[0] == 4:
        P = P[:3, :4]

    K_3x3, R, t, *_ = cv2.decomposeProjectionMatrix(P.astype(np.float64))
    K_3x3 = K_3x3 / K_3x3[2, 2]

    # cv2 may produce improper rotations (det == -1); fix sign
    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    # Camera centre in world coordinates: t is homogeneous
    C = (t[:3] / t[3])[:, 0]

    C2W = np.eye(4, dtype=np.float64)
    C2W[:3, :3] = R.T
    C2W[:3, 3] = C

    # Validation: recompose and warn on mismatch
    P_recomp = K_3x3 @ np.hstack([R, -R @ C.reshape(3, 1)])
    P_norm = P / np.linalg.norm(P)
    P_recomp_norm = P_recomp / np.linalg.norm(P_recomp)
    max_diff = np.max(np.abs(P_norm - P_recomp_norm))
    if max_diff > 1e-4:
        logger.warning("P recomposition mismatch: max_diff=%.6f", max_diff)

    return K_3x3, C2W


def load_idr(
    npz_path,
    image_size: Optional[Tuple[int, int]] = None,
) -> List[Camera]:
    """Load cameras from an IDR / NeuS ``cameras.npz`` file.

    Args:
        npz_path: Path to the ``cameras.npz`` file.
        image_size: Optional ``(width, height)`` override.  When *None* the
            image size is estimated as ``(round(2*cx), round(2*cy))``, a
            reasonable approximation when the principal point is close to the
            image centre (assumption documented here).

    Returns:
        List of :class:`~contour_reg.io.camera.Camera`, one per view.
    """
    data = np.load(str(npz_path))

    # Collect the actual world_mat_* indices and sort numerically. Iterating
    # the matched keys (rather than range(n_views)) makes the loader robust to
    # gapped / non-sequential indices while preserving ascending order.
    indices = sorted(
        int(k[len("world_mat_"):])
        for k in data.keys()
        if k.startswith("world_mat_")
    )
    logger.info("IDR cameras: %d views found in %s", len(indices), npz_path)

    cameras: List[Camera] = []

    for i in indices:
        world_mat = np.array(data[f"world_mat_{i}"], dtype=np.float64)

        # Compose with scale_mat if present: P = world_mat @ scale_mat maps
        # normalised/object space into the camera (matches NeuS convention).
        scale_key = f"scale_mat_{i}"
        if scale_key in data:
            scale_mat = np.array(data[scale_key], dtype=np.float64)
            # Ensure world_mat is 4×4 for the matrix multiply
            if world_mat.shape[0] == 3:
                world_mat_4x4 = np.eye(4, dtype=np.float64)
                world_mat_4x4[:3, :] = world_mat
                world_mat = world_mat_4x4
            P = world_mat @ scale_mat
        else:
            P = world_mat

        K, C2W = load_K_Rt_from_P(P)

        # --- focal length: average fx / fy if they differ ---
        fx, fy = K[0, 0], K[1, 1]
        focal_length = float(fx) if np.isclose(fx, fy, rtol=1e-4) else float((fx + fy) / 2.0)

        principal_point = np.array([K[0, 2], K[1, 2]])

        # --- image size ---
        if image_size is not None:
            width, height = int(image_size[0]), int(image_size[1])
        else:
            # Assumption: principal point ≈ image centre → w ≈ 2*cx, h ≈ 2*cy.
            # This is approximate; pass image_size for exact dimensions.
            width = int(round(2.0 * float(K[0, 2])))
            height = int(round(2.0 * float(K[1, 2])))

        cam = Camera(
            position=C2W[:3, 3],
            rotation_world2cam=C2W[:3, :3].T,
            focal_length=focal_length,
            principal_point=principal_point,
            width=width,
            height=height,
        )
        cameras.append(cam)

    return cameras
