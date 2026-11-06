"""COLMAP text and IDR npz writers.

Exposes:
    write_colmap(cameras, out_dir, image_names=None)
    write_idr(cameras, npz_path)

Both functions serialize a ``List[Camera]`` so the same scene can be reloaded
identically via ``load_colmap`` / ``load_idr``; they are primarily used to
build cross-format test fixtures guarded by round-trip projection-equivalence
tests.

Conventions
-----------
- ``tvec = -R @ position``  (COLMAP world-to-camera translation)
- Quaternion order: scipy returns ``[qx, qy, qz, qw]``; COLMAP expects
  ``[qw, qx, qy, qz]``; elements are reordered here.
- IDR ``world_mat_i = K @ [R | t]`` packed as a 4×4 with bottom row
  ``[0, 0, 0, 1]``; ``scale_mat_i = eye(4)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from scipy.spatial.transform import Rotation

from contour_reg.io.camera import Camera


def write_colmap(
    cameras: List[Camera],
    out_dir: Union[str, Path],
    image_names: Optional[Sequence[str]] = None,
) -> None:
    """Write a COLMAP text model (cameras.txt + images.txt + points3D.txt).

    Produces a standards-compliant COLMAP text model that real COLMAP tooling
    can read.  One PINHOLE entry is created in cameras.txt per camera
    (``CAMERA_ID = image index + 1``).  The images.txt entries are written in
    list order, and an empty points3D.txt (header only) is also emitted.

    Args:
        cameras: List of :class:`~contour_reg.io.camera.Camera` objects.
        out_dir: Output directory.  Will be created if it does not exist.
        image_names: Optional per-image file names.  Defaults to
            ``["view0.png", "view1.png", ...]`` when *None*.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if image_names is None:
        image_names = [f"view{i}.png" for i in range(len(cameras))]

    n = len(cameras)

    # ---- cameras.txt -------------------------------------------------------
    cam_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {n}",
    ]
    for idx, cam in enumerate(cameras):
        cam_id = idx + 1
        line = (
            f"{cam_id} PINHOLE {cam.width} {cam.height} "
            f"{cam.focal_length} {cam.focal_length} {cam.cx} {cam.cy}"
        )
        cam_lines.append(line)

    (out_dir / "cameras.txt").write_text("\n".join(cam_lines) + "\n")

    # ---- images.txt --------------------------------------------------------
    # Two lines per image:
    #   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    #   <empty points2D line>
    img_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {n}, mean observations per image: 0",
    ]
    for idx, (cam, name) in enumerate(zip(cameras, image_names)):
        image_id = idx + 1
        cam_id = idx + 1

        R = cam.rotation_world2cam
        tvec = -R @ cam.position  # world-to-camera translation

        # scipy as_quat() → [qx, qy, qz, qw]; COLMAP wants [qw, qx, qy, qz]
        q_xyzw = Rotation.from_matrix(R).as_quat()
        qw, qx, qy, qz = q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]

        img_lines.append(
            f"{image_id} {qw} {qx} {qy} {qz} "
            f"{tvec[0]} {tvec[1]} {tvec[2]} {cam_id} {name}"
        )
        img_lines.append("")  # empty points2D line

    (out_dir / "images.txt").write_text("\n".join(img_lines) + "\n")

    # ---- points3D.txt (empty model: header only) ---------------------------
    pts_lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        "# Number of points: 0, mean track length: 0",
    ]
    (out_dir / "points3D.txt").write_text("\n".join(pts_lines) + "\n")


def write_idr(cameras: List[Camera], npz_path: Union[str, Path]) -> None:
    """Write an IDR / NeuS ``cameras.npz`` file from a list of cameras.

    For each camera ``i`` the file contains:

    - ``world_mat_{i}``: 4×4 projection matrix ``K @ [R | t]`` with bottom
      row ``[0, 0, 0, 1]``, where ``t = -R @ position``.
    - ``scale_mat_{i}``: 4×4 identity (no normalisation applied).

    Args:
        cameras: List of :class:`~contour_reg.io.camera.Camera` objects.
        npz_path: Destination ``.npz`` file path (parent directories are
            created as needed).
    """
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict = {}
    for i, cam in enumerate(cameras):
        f = cam.focal_length
        K = np.array([
            [f,   0.0, cam.cx],
            [0.0, f,   cam.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        R = cam.rotation_world2cam
        t = -R @ cam.position  # world-to-camera translation (3,)

        # Assemble 4×4: top 3 rows = K @ [R | t], bottom row = [0,0,0,1]
        world_mat = np.zeros((4, 4), dtype=np.float64)
        world_mat[:3, :3] = K @ R
        world_mat[:3, 3] = K @ t
        world_mat[3, 3] = 1.0

        arrays[f"world_mat_{i}"] = world_mat
        arrays[f"scale_mat_{i}"] = np.eye(4, dtype=np.float64)

    np.savez(str(npz_path), **arrays)
