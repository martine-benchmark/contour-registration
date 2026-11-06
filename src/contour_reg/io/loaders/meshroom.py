"""Meshroom / AliceVision SfM JSON loader.

This is the ONLY place allowed to apply the Blender/Meshroom diag(1,-1,-1)
world-correction, and it is gated behind the ``world_flip`` parameter.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from contour_reg.io.camera import Camera
from contour_reg.io.matching import view_name


def load_meshroom(json_path, world_flip: bool = True) -> List[Camera]:
    """Load cameras from an AliceVision/Meshroom SfM JSON file.

    Args:
        json_path: Path to the SfM JSON file.
        world_flip: When True (default), apply the world-correction matrix
            ``diag(1, -1, -1)`` to both the rotation and the camera centre.
            This corrects the Blender/Meshroom axis convention mismatch.

    Returns:
        List of :class:`~contour_reg.io.camera.Camera` objects ordered by
        ascending ``viewId``.
    """
    json_path = Path(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    # --- Intrinsics map (intrinsicId -> intrinsic) ----------------------------
    # Real AliceVision exports can declare multiple intrinsics; each view
    # references one via its ``intrinsicId``. Build a lookup so each view uses
    # its own intrinsic. Single-intrinsic files fall back to the first one.
    intrinsic_map = {
        str(intr["intrinsicId"]): intr
        for intr in data["intrinsics"]
        if "intrinsicId" in intr
    }

    def _parse_intrinsic(intr):
        width = int(intr["width"])
        height = int(intr["height"])
        focal_mm = float(intr["focalLength"])
        sensor_width = float(intr["sensorWidth"])
        focal_pixels = focal_mm * width / sensor_width

        pp_offset = np.array(
            [float(intr["principalPoint"][0]), float(intr["principalPoint"][1])]
        )
        principal_point = pp_offset + np.array([width / 2.0, height / 2.0])

        dist_params_raw = intr.get("distortionParams", [])
        if dist_params_raw and len(dist_params_raw) > 0:
            distortion_coeffs = np.array([float(p) for p in dist_params_raw[:3]])
        else:
            distortion_coeffs = None
        dist_type = intr.get("distortionType", "none")
        if dist_type == "none" or dist_type is None:
            distortion_coeffs = None

        return width, height, focal_pixels, principal_point, distortion_coeffs

    # --- World-correction matrix (applied only when world_flip=True) ----------
    world_correction = np.diag([1.0, -1.0, -1.0])

    # --- Poses ----------------------------------------------------------------
    pose_map: dict = {}
    for pose_data in data["poses"]:
        pose_id = pose_data["poseId"]
        transform = pose_data["pose"]["transform"]

        # Meshroom stores cam2world; transpose to get world2cam.
        rotation_cam2world = np.array(
            [float(x) for x in transform["rotation"]]
        ).reshape(3, 3)
        rotation_world2cam = rotation_cam2world.T

        center = np.array([float(x) for x in transform["center"]])

        if world_flip:
            rotation_world2cam = rotation_world2cam @ world_correction
            center = world_correction @ center

        pose_map[pose_id] = (rotation_world2cam, center)

    # --- Views (ordered by viewId) --------------------------------------------
    cameras: List[Camera] = []
    views = sorted(data["views"], key=lambda v: int(v["viewId"]))

    for view in views:
        pose_id = view["poseId"]
        if pose_id not in pose_map:
            continue
        rotation_world2cam, center = pose_map[pose_id]

        # Select this view's intrinsic (fallback to first for single-intrinsic
        # files or when intrinsicId is missing / not found).
        intr = intrinsic_map.get(str(view.get("intrinsicId")), data["intrinsics"][0])
        width, height, focal_pixels, principal_point, distortion_coeffs = (
            _parse_intrinsic(intr)
        )

        camera = Camera(
            position=center,
            rotation_world2cam=rotation_world2cam,
            focal_length=focal_pixels,
            principal_point=principal_point,
            width=width,
            height=height,
            distortion_coeffs=distortion_coeffs,
        )
        camera.name = view_name(view["path"]) if view.get("path") else None
        cameras.append(camera)

    return cameras
