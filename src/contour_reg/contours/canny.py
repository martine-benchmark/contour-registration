"""Pure-OpenCV Canny contour extraction.

Ported from contours.py; no runtime dependency on pyalicevisionlib.
Provides `extract_canny_contours`, `filter_connected_components`, and
`_canny_large_kernel`, plus `mask_boundary_contour` and `contour_image_to_points`.
"""

import cv2
import numpy as np


def filter_connected_components(mask: np.ndarray, mode: str = "hybrid") -> np.ndarray:
    """Filter connected components and keep only one.

    Modes:
    - ``center_point``: component containing image center.
    - ``smallest_area``: smallest component.
    - ``hybrid``: ``center_point`` with ``smallest_area`` fallback when
      centre pixel belongs to background.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 2:  # Background + 0 or 1 component
        return mask

    h, w = mask.shape
    center_x, center_y = w // 2, h // 2

    if mode == "center_point" or mode == "hybrid":
        center_label = labels[center_y, center_x]
        if center_label > 0:
            filtered = np.zeros_like(mask)
            filtered[labels == center_label] = 255
            return filtered

    if mode == "smallest_area" or mode == "hybrid":
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        smallest_idx = int(np.argmin(areas)) + 1
        filtered = np.zeros_like(mask)
        filtered[labels == smallest_idx] = 255
        return filtered

    return mask


def _canny_large_kernel(
    image: np.ndarray,
    low: float,
    high: float,
    ksize: int,
) -> np.ndarray:
    """Manual Canny with arbitrary Sobel kernel size (supports ksize > 7).

    Steps: Sobel gradients → magnitude + angle → non-max suppression → hysteresis.
    """
    # Sobel gradients (cv2.Sobel supports ksize up to 31)
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx * gx + gy * gy)
    angle = np.arctan2(gy, gx) * 180.0 / np.pi

    # Normalize magnitude to 0-255 for thresholding
    mag_max = mag.max()
    if mag_max > 0:
        mag_norm = (mag / mag_max * 255).astype(np.uint8)
    else:
        return np.zeros_like(image, dtype=np.uint8)

    # Non-maximum suppression (vectorized)
    angle_q = ((angle % 180) / 45).astype(int) % 4  # 0,1,2,3

    # Pad to handle borders
    pad = np.pad(mag_norm, 1, mode="constant")
    n1 = np.zeros_like(mag_norm, dtype=np.uint8)
    n2 = np.zeros_like(mag_norm, dtype=np.uint8)

    m0 = angle_q == 0  # horizontal
    m1 = angle_q == 1  # diagonal /
    m2 = angle_q == 2  # vertical
    m3 = angle_q == 3  # diagonal \

    n1[m0] = pad[1:-1, :-2][m0]
    n2[m0] = pad[1:-1, 2:][m0]
    n1[m1] = pad[:-2, 2:][m1]
    n2[m1] = pad[2:, :-2][m1]
    n1[m2] = pad[:-2, 1:-1][m2]
    n2[m2] = pad[2:, 1:-1][m2]
    n1[m3] = pad[:-2, :-2][m3]
    n2[m3] = pad[2:, 2:][m3]

    nms = np.where((mag_norm >= n1) & (mag_norm >= n2), mag_norm, 0).astype(np.uint8)

    # Hysteresis thresholding
    strong = nms >= high
    weak = (nms >= low) & (~strong)

    edges = np.zeros_like(nms)
    edges[strong] = 255

    # Connect weak edges adjacent to strong
    for _ in range(3):
        dilated_strong = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        new_edges = weak & (dilated_strong > 0)
        if not new_edges.any():
            break
        edges[new_edges] = 255
        weak[new_edges] = False

    return edges


def extract_canny_contours(
    image_gray: np.ndarray,
    mask: np.ndarray,
    margin_size: int = 20,
    canny_low: int = 50,
    canny_high: int = 150,
    sobel_kernel: int = 3,
) -> np.ndarray:
    """Extract Canny edges within a dilate-erode margin band around *mask*.

    Args:
        image_gray: Grayscale uint8 image.
        mask: Binary mask (object region, uint8).
        margin_size: Half-width in pixels of the margin band.
        canny_low: Canny lower hysteresis threshold.
        canny_high: Canny upper hysteresis threshold.
        sobel_kernel: Sobel aperture size; use an odd value ≥ 3.
            Values > 7 use the internal ``_canny_large_kernel`` path because
            ``cv2.Canny`` only accepts 3, 5, or 7.

    Returns:
        uint8 binary image with Canny edges masked to the margin band.
    """
    # Build margin band: dilate - erode
    kernel = np.ones((margin_size * 2 + 1, margin_size * 2 + 1), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    margin = cv2.subtract(dilated, eroded)

    # Apply Canny (standard path for kernel ≤ 7, manual path for larger kernels)
    if sobel_kernel <= 7:
        edges = cv2.Canny(image_gray, canny_low, canny_high, apertureSize=sobel_kernel)
    else:
        edges = _canny_large_kernel(image_gray, canny_low, canny_high, sobel_kernel)

    # Restrict to margin band
    contours = cv2.bitwise_and(edges, margin)
    return contours


def mask_boundary_contour(mask: np.ndarray, include_inner: bool = False) -> np.ndarray:
    """Extract boundary contour points from a binary mask.

    Uses ``cv2.RETR_TREE`` when *include_inner* is ``True`` (returns both
    outer boundary and inner holes, e.g. for an annulus), otherwise
    ``cv2.RETR_EXTERNAL`` (outer boundary only).

    Args:
        mask: Binary uint8 mask.
        include_inner: Whether to include inner contour loops.

    Returns:
        (N, 2) array of ``[x, y]`` contour points.
    """
    retrieval_mode = cv2.RETR_TREE if include_inner else cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours(mask, retrieval_mode, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return np.empty((0, 2), dtype=np.int32)

    # Each contour has shape (N, 1, 2); squeeze to (N, 2) with [x, y]
    points = np.concatenate([c.reshape(-1, 2) for c in contours], axis=0)
    return points


def contour_image_to_points(edge_img: np.ndarray) -> np.ndarray:
    """Convert a binary edge image to an (N, 2) array of ``[x, y]`` points.

    Nonzero pixels are returned as ``[col, row]`` pairs, consistent with
    the ``[x, y]`` convention used by ``contour_reg.io.masks.load_contour_points``.

    Args:
        edge_img: Binary uint8 edge image (e.g. output of
            ``extract_canny_contours``).

    Returns:
        (N, 2) int array of ``[x, y]`` pixel coordinates.
    """
    rows, cols = np.nonzero(edge_img)
    return np.column_stack((cols, rows)).astype(np.int32) if len(rows) > 0 else np.empty((0, 2), dtype=np.int32)
