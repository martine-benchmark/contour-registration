"""Visualization overlays for contour registration.

Provides resolution-proportional text labels and colored contour/point overlays
so rendered output is legible from 256 px test images up to 9568 px renders.
"""
from __future__ import annotations

import cv2
import numpy as np


def proportional_font_scale(width: int, ref_width: int = 2000, base: float = 1.5) -> float:
    """Return a cv2 fontScale that grows linearly with image width.

    Formula: ``max(0.4, (width / ref_width) * base)``

    At ``ref_width`` the scale equals ``base`` (1.5 by default).
    Very small images are floored at 0.4 so the label stays legible.

    Parameters
    ----------
    width:     Image width in pixels.
    ref_width: Reference width at which ``base`` scale is returned.
    base:      Font scale at ``ref_width``.

    Returns
    -------
    float: cv2-compatible fontScale value.
    """
    return max(0.4, (width / ref_width) * base)


def contour_silhouette_overlay(
    background: np.ndarray,
    contour: np.ndarray | None = None,
    silhouette: np.ndarray | None = None,
    *,
    dim: float = 0.35,
    radius: int = 0,
    crop: bool = False,
    crop_margin: float = 0.08,
    contour_color: tuple = (40, 40, 255),   # BGR red
    silhouette_color: tuple = (0, 255, 0),  # BGR green
) -> np.ndarray:
    """Overlay a target *contour* (red) and a projected *silhouette* (green).

    Draws both 2D point sets on a dimmed copy of *background* (a grayscale or BGR
    image, e.g. the Canny render). ``radius`` is the point half-size in pixels
    (0 = single pixel). When *crop* is set, the result is cropped to the bounding
    box of the drawn points plus a proportional margin. The input is never
    mutated. Returns a BGR uint8 image.
    """
    if background.ndim == 2:
        out = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    else:
        out = background.copy()
    out = (out.astype(np.float32) * dim).clip(0, 255).astype(np.uint8)
    h, w = out.shape[:2]

    def _stamp(pts, color):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        xi = np.round(pts[:, 0]).astype(int)
        yi = np.round(pts[:, 1]).astype(int)
        m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xi, yi = xi[m], yi[m]
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                out[np.clip(yi + dy, 0, h - 1), np.clip(xi + dx, 0, w - 1)] = color

    if silhouette is not None and len(silhouette):
        _stamp(silhouette, silhouette_color)
    if contour is not None and len(contour):
        _stamp(contour, contour_color)

    if crop:
        stacked = [p for p in (contour, silhouette) if p is not None and len(p)]
        if stacked:
            allp = np.vstack([np.asarray(p, float).reshape(-1, 2) for p in stacked])
            x0, y0 = allp.min(0)
            x1, y1 = allp.max(0)
            mx = crop_margin * max(x1 - x0, y1 - y0) + 20
            x0 = int(max(0, x0 - mx))
            y0 = int(max(0, y0 - mx))
            x1 = int(min(w, x1 + mx))
            y1 = int(min(h, y1 + mx))
            if x1 > x0 and y1 > y0:
                out = out[y0:y1, x0:x1]
    return out


def fp_fn_overlay(
    background: np.ndarray,
    contour: np.ndarray,
    silhouette: np.ndarray,
    epsilon: float = 2.0,
    *,
    radius: int = 1,
    dim: float = 0.35,
    crop: bool = False,
    crop_margin: float = 0.08,
    show_fn: bool = True,
    label: str | None = None,
) -> np.ndarray:
    """Spatial precision/recall overlay of a contour against a silhouette.

    Classifies each contour point by its distance to the nearest silhouette
    point at tolerance *epsilon*:

    * **TP** (green): contour point within epsilon of the silhouette,
    * **FP** (red):   contour point farther than epsilon (clutter / off-silhouette).

    When *show_fn* is set, silhouette points farther than epsilon from any
    contour point are drawn in **blue** (FN: silhouette not covered by the
    contour). White would over-clutter, so the silhouette itself is implied by
    the TP/FN split. Returns a BGR uint8 image; the input is never mutated.
    """
    from contour_reg.contour_metrics import nn_distances

    if background.ndim == 2:
        out = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    else:
        out = background.copy()
    out = (out.astype(np.float32) * dim).clip(0, 255).astype(np.uint8)
    h, w = out.shape[:2]
    contour = np.asarray(contour, dtype=float).reshape(-1, 2)
    silhouette = np.asarray(silhouette, dtype=float).reshape(-1, 2)

    def _stamp(pts, color):
        if len(pts) == 0:
            return
        xi = np.round(pts[:, 0]).astype(int)
        yi = np.round(pts[:, 1]).astype(int)
        m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xi, yi = xi[m], yi[m]
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                out[np.clip(yi + dy, 0, h - 1), np.clip(xi + dx, 0, w - 1)] = color

    d_c = nn_distances(contour, silhouette)          # contour -> silhouette
    tp = contour[d_c <= epsilon]
    fp = contour[d_c > epsilon]
    if show_fn and len(silhouette):
        d_s = nn_distances(silhouette, contour)      # silhouette -> contour
        _stamp(silhouette[d_s > epsilon], (255, 60, 0))   # FN blue (BGR)
    _stamp(tp, (0, 255, 0))                          # TP green
    _stamp(fp, (0, 0, 255))                          # FP red

    n_c = len(contour)
    text = label or (f"eps={epsilon:g}px  TP={len(tp)} FP={len(fp)}"
                     f"  P={len(tp) / n_c:.3f}" if n_c else f"eps={epsilon:g}px")
    out = draw_overlay(out, label=text)

    if crop:
        allp = np.vstack([p for p in (contour, silhouette) if len(p)]) \
            if (len(contour) or len(silhouette)) else np.empty((0, 2))
        if len(allp):
            x0, y0 = allp.min(0)
            x1, y1 = allp.max(0)
            mx = crop_margin * max(x1 - x0, y1 - y0) + 20
            x0 = int(max(0, x0 - mx))
            y0 = int(max(0, y0 - mx))
            x1 = int(min(w, x1 + mx))
            y1 = int(min(h, y1 + mx))
            if x1 > x0 and y1 > y0:
                out = out[y0:y1, x0:x1]
    return out


def draw_overlay(
    image: np.ndarray,
    gt_contour: np.ndarray | None = None,
    points: np.ndarray | None = None,
    label: str | None = None,
    ref_width: int = 2000,
    point_radius: int = 2,
    gt_radius: int = 2,
    line_thickness: int = 1,
) -> np.ndarray:
    """Render contour / point overlays on *image* and return a new array.

    The input is never mutated: a copy is always returned.

    Parameters
    ----------
    image:          H×W or H×W×3 uint8 array.
    gt_contour:     (N, 2) array of [x, y] ground-truth contour points drawn in
                    white.
    points:         (N, 2) array of [x, y] projected / FP points drawn in green.
    label:          String to render at the top-left with resolution-proportional
                    font scale and thickness.
    ref_width:      Reference width for :func:`proportional_font_scale`.
    point_radius:   Radius in pixels for projected/FP point circles (default 2).
    gt_radius:      Radius in pixels for ground-truth contour point circles (default 2).
    line_thickness: Line thickness for future polyline use (default 1; not yet
                    applied to circle calls).

    Returns
    -------
    np.ndarray: H×W×3 BGR uint8 image with overlays applied.
    """
    # --- Ensure 3-channel BGR output (never mutate the input) ---
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    H, W = out.shape[:2]

    # --- Ground-truth contour: white ---
    if gt_contour is not None:
        pts = np.asarray(gt_contour, dtype=np.int32)
        for x, y in pts:
            cv2.circle(out, (int(x), int(y)), radius=gt_radius, color=(255, 255, 255), thickness=-1)

    # --- Projected / FP points: green ---
    if points is not None:
        pts = np.asarray(points, dtype=np.int32)
        for x, y in pts:
            cv2.circle(out, (int(x), int(y)), radius=point_radius, color=(0, 255, 0), thickness=-1)

    # --- Top-left text label with proportional sizing + width auto-fit ---
    if label is not None:
        font_scale = proportional_font_scale(W, ref_width)
        thickness = max(1, int(round(font_scale * 2)))
        # Auto-fit: shrink the font until the rendered text fits inside the image
        # (<= 95% of width) so the label never overflows the right edge.
        max_w = int(0.95 * W)
        (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        while text_w > max_w and font_scale > 0.3:
            font_scale -= 0.05
            thickness = max(1, int(round(font_scale * 2)))
            (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # Small proportional margin so text sits inside the image at any size
        org = (int(0.01 * W) + 2, int(0.04 * W) + 2)
        cv2.putText(
            out,
            label,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return out
