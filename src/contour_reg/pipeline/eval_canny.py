"""Canny-config evaluation: rank sweep configs by contour F-beta.

After the sweep has produced a registered pose per Canny config, this compares, for each config at its own
pose, the projected mesh silhouette against that config's contour, and ranks
configs by :func:`contour_reg.contour_metrics.f_beta` at a chosen epsilon
(default beta=0.5, precision-weighted, at epsilon=2px; F1 is stored alongside). The
selection is an *overlap* criterion, independent of the ICP residual.

The silhouette extractor is injectable so the ranking logic can be unit-tested
without a mesh; production callers use the default.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import trimesh

from contour_reg.contour_metrics import (
    ViewCounts,
    aggregate,
    best_by_score,
    nn_distances,
    view_counts,
)
from contour_reg.io.camera import Camera
from contour_reg.io.mesh import apply_transform_to_mesh_with_scale
from contour_reg.registration.correspondences import (
    clear_mesh_cache,
    extract_silhouette_edges_fast,
)

# A silhouette extractor maps (mesh, camera) -> (projected_2d, vertex_indices).
SilhouetteFn = Callable[[trimesh.Trimesh, Camera], Tuple[np.ndarray, np.ndarray]]


def trim_contour(contour: np.ndarray, silhouette: np.ndarray, percentile: float) -> np.ndarray:
    """Keep the contour points closest to the silhouette (best *percentile*%).

    ``percentile >= 100`` keeps every point (no trimming). Distances are contour
    point -> nearest silhouette point (the registration/metric direction).
    """
    contour = np.asarray(contour, dtype=float).reshape(-1, 2)
    if percentile >= 100 or len(contour) == 0 or len(silhouette) == 0:
        return contour
    dists = nn_distances(contour, silhouette)
    threshold = np.percentile(dists, percentile)
    return contour[dists <= threshold]


def c_trim(contour: np.ndarray, silhouette: np.ndarray, epsilon: float) -> np.ndarray:
    """Keep contour points within *epsilon* px of the silhouette (absolute cutoff).

    This is the "C_trim" used by refinement and leave-one-view-out evaluation:
    raw contour points whose nearest projected-silhouette distance is <= epsilon.
    An empty silhouette yields an empty result (nothing is within epsilon).
    """
    contour = np.asarray(contour, dtype=float).reshape(-1, 2)
    if len(contour) == 0:
        return contour
    return contour[nn_distances(contour, silhouette) <= epsilon]


def config_view_counts(
    mesh: trimesh.Trimesh,
    cameras: Sequence[Camera],
    target_contours: Sequence[np.ndarray],
    pose: np.ndarray,
    scale: float,
    trim_percentile: float = 75.0,
    *,
    silhouette_fn: SilhouetteFn = extract_silhouette_edges_fast,
) -> List[ViewCounts]:
    """Per-view :class:`ViewCounts` (silhouette / contour / trimmed) at *pose*."""
    if len(cameras) != len(target_contours):
        raise ValueError(
            f"cameras ({len(cameras)}) and target_contours "
            f"({len(target_contours)}) must be the same length.")
    pose = np.asarray(pose, dtype=float)
    mesh_t = apply_transform_to_mesh_with_scale(mesh, pose[:3], pose[3:6], scale)
    clear_mesh_cache()
    views: List[ViewCounts] = []
    for cam, contour in zip(cameras, target_contours):
        silhouette, _ = silhouette_fn(mesh_t, cam)
        silhouette = np.asarray(silhouette, dtype=float).reshape(-1, 2)
        trimmed = trim_contour(contour, silhouette, trim_percentile)
        views.append(view_counts(silhouette, contour, trimmed))
    return views


def eval_configs(
    per_config_views: Mapping[str, Sequence[ViewCounts]],
    epsilons: Sequence[float] = (1.0, 2.0, 3.0, 5.0),
    select_epsilon: float = 2.0,
    beta: float = 0.5,
) -> Dict[str, object]:
    """Aggregate contour metrics per config over *epsilons* and pick the best.

    Returns ``{"best": slug|None, "select_epsilon", "beta",
    "configs": {slug: {"eps_<e>": {metrics...}}}}``. The best config maximises
    ``F_beta`` at *select_epsilon* (ties break on slug for determinism).
    """
    # Guarantee the selection epsilon is evaluated (else the F_beta lookup below
    # would KeyError after all the expensive silhouette extraction).
    epsilons = tuple(epsilons)
    if select_epsilon not in epsilons:
        epsilons = epsilons + (select_epsilon,)

    configs: Dict[str, Dict[str, Dict[str, float]]] = {}
    scores: Dict[str, float] = {}
    scores_f1: Dict[str, float] = {}
    for slug, views in per_config_views.items():
        per_eps = {f"eps_{eps:g}": aggregate(views, eps, beta) for eps in epsilons}
        configs[slug] = per_eps
        scores[slug] = per_eps[f"eps_{select_epsilon:g}"]["F_beta"]
        scores_f1[slug] = per_eps[f"eps_{select_epsilon:g}"]["F1"]

    best = best_by_score(scores) or None
    return {
        "best": best,
        "select_epsilon": float(select_epsilon),
        "beta": float(beta),
        "scores": scores,
        "scores_f1": scores_f1,
        "best_f1": best_by_score(scores_f1) or None,
        "configs": configs,
    }
