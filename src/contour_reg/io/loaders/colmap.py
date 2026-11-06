"""COLMAP model loader (text and binary formats).

Exposes:
    load_colmap(path) -> List[Camera]
    qvec2rotmat(qvec) -> np.ndarray (3, 3)

Supports camera models: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV.
Text format (cameras.txt + images.txt) is preferred; binary is used as fallback.
Images are returned sorted by image name for deterministic ordering.
"""
from __future__ import annotations

import struct
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from contour_reg.io.camera import Camera
from contour_reg.io.matching import view_name

# ---------------------------------------------------------------------------
# Camera model parameter counts (for binary reading)
# ---------------------------------------------------------------------------
_MODEL_NUM_PARAMS: Dict[int, Tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}


# ---------------------------------------------------------------------------
# Quaternion → rotation matrix
# ---------------------------------------------------------------------------

def qvec2rotmat(qvec) -> np.ndarray:
    """Convert a quaternion [qw, qx, qy, qz] to a (3, 3) rotation matrix."""
    qvec = np.asarray(qvec, dtype=np.float64)
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2,
         2 * qx * qy - 2 * qw * qz,
         2 * qz * qx + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz,
         1 - 2 * qx**2 - 2 * qz**2,
         2 * qy * qz - 2 * qw * qx],
        [2 * qz * qx - 2 * qw * qy,
         2 * qy * qz + 2 * qw * qx,
         1 - 2 * qx**2 - 2 * qy**2],
    ])


# ---------------------------------------------------------------------------
# Low-level binary helpers
# ---------------------------------------------------------------------------

def _read_next_bytes(fid, num_bytes: int, fmt: str, endian: str = "<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


# ---------------------------------------------------------------------------
# Text format readers
# ---------------------------------------------------------------------------

def _read_cameras_text(path: Path) -> dict:
    """Return {camera_id: (model, width, height, params)}."""
    cameras: dict = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            cam_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(list(map(float, elems[4:])))
            cameras[cam_id] = (model, width, height, params)
    return cameras


def _read_images_text(path: Path) -> list:
    """Return list of (qvec, tvec, camera_id, image_name) from images.txt.

    Each image occupies two lines; the second line (2D-point list) is consumed
    but discarded.
    """
    images: list = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            qvec = np.array(list(map(float, elems[1:5])))
            tvec = np.array(list(map(float, elems[5:8])))
            camera_id = int(elems[8])
            image_name = elems[9]
            # Consume (and discard) the 2D-point line
            fid.readline()
            images.append((qvec, tvec, camera_id, image_name))
    return images


# ---------------------------------------------------------------------------
# Binary format readers
# ---------------------------------------------------------------------------

def _read_cameras_binary(path: Path) -> dict:
    """Return {camera_id: (model, width, height, params)} from cameras.bin."""
    cameras: dict = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            cam_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ")
            if model_id not in _MODEL_NUM_PARAMS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id}")
            model_name, num_params = _MODEL_NUM_PARAMS[model_id]
            params = np.array(
                _read_next_bytes(fid, 8 * num_params, "d" * num_params)
            )
            cameras[cam_id] = (model_name, width, height, params)
    return cameras


def _read_images_binary(path: Path) -> list:
    """Return list of (qvec, tvec, camera_id, image_name) from images.bin."""
    images: list = []
    with open(path, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = _read_next_bytes(fid, 64, "idddddddi")
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            # Read null-terminated image name
            name_chars: list = []
            while True:
                ch = _read_next_bytes(fid, 1, "c")[0]
                if ch == b"\x00":
                    break
                name_chars.append(ch.decode("utf-8"))
            image_name = "".join(name_chars)
            # Skip 2D-point array
            num_points2d = _read_next_bytes(fid, 8, "Q")[0]
            if num_points2d > 0:
                _read_next_bytes(fid, 24 * num_points2d, "ddq" * num_points2d)
            images.append((qvec, tvec, camera_id, image_name))
    return images


# ---------------------------------------------------------------------------
# Intrinsic extraction helper
# ---------------------------------------------------------------------------

def _extract_intrinsics(model: str, params: np.ndarray, width: int, height: int):
    """Return (focal_length, cx, cy, distortion_coeffs | None)."""
    model = model.upper()
    if model == "SIMPLE_PINHOLE":
        # params: f, cx, cy
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        distortion = None

    elif model == "PINHOLE":
        # params: fx, fy, cx, cy
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
        distortion = None

    elif model == "SIMPLE_RADIAL":
        # params: f, cx, cy, k1
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        k1 = params[3]
        distortion = np.array([k1, 0.0])

    elif model == "RADIAL":
        # params: f, cx, cy, k1, k2
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        k1, k2 = params[3], params[4]
        distortion = np.array([k1, k2])

    elif model == "OPENCV":
        # params: fx, fy, cx, cy, k1, k2, p1, p2
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
        k1, k2 = params[4], params[5]
        p1, p2 = params[6], params[7]
        if p1 != 0.0 or p2 != 0.0:
            warnings.warn(
                f"OPENCV camera model has non-zero tangential distortion "
                f"(p1={p1}, p2={p2}); the Camera model only supports radial "
                f"distortion: tangential coefficients are dropped.",
                UserWarning,
                stacklevel=2,
            )
        distortion = np.array([k1, k2])

    else:
        raise ValueError(
            f"Unsupported COLMAP camera model '{model}'. "
            f"Supported: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV."
        )

    if not np.isclose(fx, fy):
        warnings.warn(
            f"Camera model '{model}' has fx={fx} != fy={fy}; "
            f"using their mean as focal_length.",
            UserWarning,
            stacklevel=2,
        )
    focal_length = float((fx + fy) / 2.0)
    return focal_length, float(cx), float(cy), distortion


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_colmap(path) -> List[Camera]:
    """Load a COLMAP model directory and return a list of :class:`Camera`.

    Args:
        path: Path to the COLMAP model directory.  Must contain either
            ``cameras.txt`` + ``images.txt`` (preferred) or
            ``cameras.bin`` + ``images.bin``.

    Returns:
        List of :class:`~contour_reg.io.camera.Camera` objects sorted by
        image name for deterministic ordering.

    Raises:
        FileNotFoundError: If neither text nor binary model files are found.
        ValueError: If an unsupported camera model is encountered.
    """
    path = Path(path)

    cameras_txt = path / "cameras.txt"
    images_txt = path / "images.txt"
    cameras_bin = path / "cameras.bin"
    images_bin = path / "images.bin"

    if cameras_txt.exists() and images_txt.exists():
        raw_cameras = _read_cameras_text(cameras_txt)
        raw_images = _read_images_text(images_txt)
    elif cameras_bin.exists() and images_bin.exists():
        raw_cameras = _read_cameras_binary(cameras_bin)
        raw_images = _read_images_binary(images_bin)
    else:
        raise FileNotFoundError(
            f"No COLMAP model files found in '{path}'. "
            f"Expected cameras.txt+images.txt or cameras.bin+images.bin."
        )

    # Sort images by name for deterministic ordering
    raw_images_sorted = sorted(raw_images, key=lambda entry: entry[3])

    result: List[Camera] = []
    for qvec, tvec, camera_id, image_name in raw_images_sorted:
        model, width, height, params = raw_cameras[camera_id]
        focal_length, cx, cy, distortion = _extract_intrinsics(
            model, params, width, height
        )

        R = qvec2rotmat(qvec)           # world-to-camera rotation
        position = -R.T @ tvec          # camera centre in world coords

        cam = Camera(
            position=position,
            rotation_world2cam=R,
            focal_length=focal_length,
            principal_point=np.array([cx, cy]),
            width=width,
            height=height,
            distortion_coeffs=distortion,
        )
        cam.name = view_name(image_name)
        result.append(cam)

    return result
