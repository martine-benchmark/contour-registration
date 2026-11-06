"""Coarse-then-fine contour ICP refinement.

This module exposes :func:`run_refine`, which wraps :func:`contour_based_icp`
in a two-phase coarse-then-fine strategy:

Phase A (coarse)
    Standard ICP with ``error_threshold_percentile=error_percentile`` (default
    75.0).  This uses the ICP's built-in Phase-2 trimmed step to reject the
    worst correspondences and converge to a good initial estimate.

Phase B (fine)
    A second ICP pass starting from the coarse result, but with a *tighter*
    correspondence filter: ``error_threshold_percentile=keep_pct``.  Setting
    ``keep_pct=90`` keeps the best 90 % of correspondences (measured by
    reprojection-error percentile) and discards the noisiest 10 %, yielding a
    more accurate final alignment.

The ``keep_pct`` parameter therefore maps directly to the ICP's internal
``error_threshold_percentile`` argument in Phase B.  The two error-history
lists are concatenated so callers see the full convergence trajectory.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import trimesh

from contour_reg.io.camera import Camera
from contour_reg.registration.icp import contour_based_icp


def run_refine(
    mesh: trimesh.Trimesh,
    cameras: List[Camera],
    target_contours: List[np.ndarray],
    init_pose: np.ndarray,
    *,
    keep_pct: float = 90.0,
    error_percentile: float = 75.0,
    max_correspondence_distance: float = 0.0,
    max_iterations: int = 50,
    optimize_scale: bool = False,
    initial_scale: float = 1.0,
) -> Tuple[np.ndarray, List[float], float]:
    """Coarse-then-fine contour ICP.

    Args:
        mesh: Source mesh to register.
        cameras: List of :class:`~contour_reg.io.camera.Camera` objects.
        target_contours: List of target contour arrays (one per camera).
        init_pose: Initial pose ``[rx, ry, rz, tx, ty, tz]``.
        keep_pct: Percentile of correspondences to keep in the fine phase.
            Maps directly to ``error_threshold_percentile`` in Phase B of ICP.
            ``keep_pct=90`` → keep the best 90 % by reprojection error,
            discarding the noisiest 10 %.  Higher values are more permissive;
            lower values are stricter.
        error_percentile: Percentile used in the coarse phase (Phase A).
            Passed as ``error_threshold_percentile`` to the first ICP call.
            Default 75.0 follows the ICP default.
        max_correspondence_distance: Per-view residual cutoff (px) applied in
            both phases.  ``0`` disables filtering.
        max_iterations: Maximum ICP iterations per phase.
        optimize_scale: Also optimise a uniform scale (7-DoF similarity); the
            scale found in Phase A is the starting scale of Phase B.
        initial_scale: Starting scale of Phase A (ignored when rigid).

    Returns:
        ``(pose, error_history, scale)`` where *pose* is the final ``[rx,ry,rz,tx,ty,tz]``
        array, *error_history* is the concatenation of Phase A and Phase B
        histories (so ``len(error_history) >= 2`` after both phases run) and
        *scale* is the final uniform scale (``initial_scale`` when rigid).
    """
    # Phase A: coarse ICP from the provided initial pose.
    pose_a, hist_a, scale_a = contour_based_icp(
        mesh,
        init_pose,
        cameras,
        target_contours,
        max_iterations=max_iterations,
        error_threshold_percentile=error_percentile,
        max_correspondence_distance=max_correspondence_distance,
        optimize_scale=optimize_scale,
        initial_scale=initial_scale,
        verbose=False,
    )

    # Phase B: fine ICP from the coarse result with tighter trimming.
    # keep_pct → error_threshold_percentile: keeps the best keep_pct% of
    # correspondences, discarding the noisiest (100 - keep_pct)%.
    pose_b, hist_b, scale_b = contour_based_icp(
        mesh,
        pose_a,
        cameras,
        target_contours,
        max_iterations=max_iterations,
        error_threshold_percentile=keep_pct,
        max_correspondence_distance=max_correspondence_distance,
        optimize_scale=optimize_scale,
        initial_scale=scale_a if optimize_scale else initial_scale,
        verbose=False,
    )

    return pose_b, hist_a + hist_b, float(scale_b if optimize_scale else initial_scale)
