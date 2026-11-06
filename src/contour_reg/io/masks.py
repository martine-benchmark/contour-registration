"""Masks IO utilities: load contour points from .npy or image files."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_contour_points(path: str | Path, data_dir: str | Path | None = None) -> np.ndarray:
    """Load contour points from *path*, returning an (N, 2) float array of [x, y] coords.

    Behaviour:
    - ``.npy`` → ``np.load`` returning the stored float (N, 2) array directly.
    - image (``.png``, ``.jpg``, ``.bmp``, …) → read grayscale, nonzero pixels become
      ``[col, row]`` = ``[x, y]``.

    Raises ``ValueError`` (naming the path and expected formats) if the file cannot be
    decoded or the suffix is clearly unsupported.

    Ported from ``utils.py::load_contour_points``.
    """
    _SUPPORTED = ".npy, .png, .jpg, .jpeg, .bmp, .tif, .tiff"
    _IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    file_path = Path(path)

    if file_path.is_absolute():
        full_path = file_path
    elif data_dir is not None:
        full_path = Path(data_dir) / file_path
    else:
        full_path = file_path

    if not full_path.exists():
        raise FileNotFoundError(f"Contour file not found: {full_path}")

    suffix = full_path.suffix.lower()

    if suffix == '.npy':
        return np.load(full_path)

    if suffix not in _IMAGE_SUFFIXES:
        raise ValueError(
            f"Unsupported contour file '{full_path}': suffix '{full_path.suffix}'. "
            f"Expected one of: {_SUPPORTED}"
        )

    # Image branch
    img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(
            f"Could not decode contour image '{full_path}'. "
            f"Expected a readable image ({_SUPPORTED})."
        )

    ys, xs = np.where(img > 0)
    return np.column_stack([xs, ys]).astype(np.float32)
