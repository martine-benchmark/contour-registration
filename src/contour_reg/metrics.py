#!/usr/bin/env python3
"""
Metrics library for pose evaluation in contour-based 2D-3D registration.

Implements standard metrics: ADD, ADD-S, MSSD, MSPD, rotation/translation errors,
reprojection error, AUC, and success rates. Compatible with BOP benchmark standards.

All pose representations use [rx, ry, rz, tx, ty, tz] Euler XYZ (radians + world units).

Ported from validation_metrics.py.
Import note: extract_silhouette_edges_fast and apply_transform_to_mesh are imported
at module level (no circular dependency risk) since neither correspondences.py nor
io/mesh.py imports from this module.
"""


import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

from contour_reg.io.mesh import apply_transform_to_mesh_with_scale
from contour_reg.registration.correspondences import extract_silhouette_edges_fast


def _pose_to_rt(pose: np.ndarray):
    """Extract rotation matrix and translation from [rx, ry, rz, tx, ty, tz]."""
    R = Rotation.from_euler('xyz', pose[:3]).as_matrix()
    t = pose[3:6]
    return R, t


def _transform_vertices(R: np.ndarray, t: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Apply rigid transform: v_new = R @ v + t."""
    return (R @ vertices.T).T + t


def _subsample_vertices(vertices: np.ndarray, max_count: int = 5000, seed: int = 42) -> np.ndarray:
    """Subsample vertices uniformly if count exceeds max_count."""
    if len(vertices) <= max_count:
        return vertices
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(vertices), size=max_count, replace=False)
    return vertices[indices]


def rotation_error_deg(pose_est: np.ndarray, pose_gt: np.ndarray) -> float:
    """Geodesic rotation error in degrees.

    angle = arccos(clip((trace(R_est @ R_gt.T) - 1) / 2, -1, 1))
    """
    R_est, _ = _pose_to_rt(pose_est)
    R_gt, _ = _pose_to_rt(pose_gt)
    R_diff = R_est @ R_gt.T
    cos_angle = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))


def translation_error(pose_est: np.ndarray, pose_gt: np.ndarray) -> float:
    """L2 norm of translation difference: ||t_est - t_gt||."""
    _, t_est = _pose_to_rt(pose_est)
    _, t_gt = _pose_to_rt(pose_gt)
    return float(np.linalg.norm(t_est - t_gt))


def add_metric(pose_est: np.ndarray, pose_gt: np.ndarray,
               mesh_vertices: np.ndarray) -> float:
    """Average Distance of Model Points (ADD).

    ADD = mean(||R_est @ v + t_est - (R_gt @ v + t_gt)||) for all v.
    For large meshes (>5000 vertices), subsamples uniformly to 5000.
    """
    verts = _subsample_vertices(mesh_vertices)
    R_est, t_est = _pose_to_rt(pose_est)
    R_gt, t_gt = _pose_to_rt(pose_gt)
    v_est = _transform_vertices(R_est, t_est, verts)
    v_gt = _transform_vertices(R_gt, t_gt, verts)
    distances = np.linalg.norm(v_est - v_gt, axis=1)
    return float(np.mean(distances))


def add_s_metric(pose_est: np.ndarray, pose_gt: np.ndarray,
                 mesh_vertices: np.ndarray) -> float:
    """Symmetric ADD (ADD-S) for symmetric objects (Bowl, Cup, Die).

    For each vertex transformed by est, find closest vertex transformed by gt.
    ADD-S = mean(min_j(||R_est @ v_i + t_est - (R_gt @ v_j + t_gt)||))
    Uses KD-tree for efficiency.
    """
    verts = _subsample_vertices(mesh_vertices)
    R_est, t_est = _pose_to_rt(pose_est)
    R_gt, t_gt = _pose_to_rt(pose_gt)
    v_est = _transform_vertices(R_est, t_est, verts)
    v_gt = _transform_vertices(R_gt, t_gt, verts)

    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(v_gt)
    distances, _ = nn.kneighbors(v_est)
    return float(np.mean(distances))


def mssd_metric(pose_est: np.ndarray, pose_gt: np.ndarray,
                mesh_vertices: np.ndarray) -> float:
    """Maximum Symmetry-aware Surface Distance (BOP standard).

    BIDIRECTIONAL: compute both directions and return the max.
    d_fwd = max_v min_{v'} ||est(v) - gt(v')||  (est-to-gt)
    d_bwd = max_{v'} min_v ||gt(v') - est(v)||  (gt-to-est)
    MSSD = max(d_fwd, d_bwd)
    Uses KD-tree for both directions.
    """
    verts = _subsample_vertices(mesh_vertices)
    R_est, t_est = _pose_to_rt(pose_est)
    R_gt, t_gt = _pose_to_rt(pose_gt)
    v_est = _transform_vertices(R_est, t_est, verts)
    v_gt = _transform_vertices(R_gt, t_gt, verts)

    # Forward: est -> gt
    nn_gt = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(v_gt)
    dist_fwd, _ = nn_gt.kneighbors(v_est)
    d_fwd = float(np.max(dist_fwd))

    # Backward: gt -> est
    nn_est = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(v_est)
    dist_bwd, _ = nn_est.kneighbors(v_gt)
    d_bwd = float(np.max(dist_bwd))

    return max(d_fwd, d_bwd)


def mspd_metric(pose_est: np.ndarray, pose_gt: np.ndarray,
                mesh_vertices: np.ndarray, cameras) -> float:
    """Maximum Symmetry-aware Projection Distance (BOP standard).

    For each camera:
      Project vertices with both poses.
      For each est-projected point, find closest gt-projected point (KD-tree).
      mspd_view = max over vertices of min 2D distance.
    MSPD = mean over cameras of mspd_view.
    (BOP reports per-view then averages, NOT double-max.)
    """
    verts = _subsample_vertices(mesh_vertices)
    R_est, t_est = _pose_to_rt(pose_est)
    R_gt, t_gt = _pose_to_rt(pose_gt)
    v_est_3d = _transform_vertices(R_est, t_est, verts)
    v_gt_3d = _transform_vertices(R_gt, t_gt, verts)

    mspd_per_view = []
    for cam in cameras:
        proj_est = cam.project_points(v_est_3d)
        proj_gt = cam.project_points(v_gt_3d)

        nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(proj_gt)
        distances, _ = nn.kneighbors(proj_est)
        mspd_view = float(np.max(distances))
        mspd_per_view.append(mspd_view)

    return float(np.mean(mspd_per_view))


def reprojection_error(pose_est: np.ndarray, mesh, cameras,
                       target_contours, scale: float = 1.0) -> dict:
    """Mean reprojection error (pixels) between projected silhouette and target contours.

    Distances are measured from each target contour point to its nearest
    projected silhouette point. ``scale`` applies a uniform similarity scale to
    the mesh before projection (must match the scale the pose was estimated
    with; ``1.0`` for a rigid pose).

    Note: the reference set is the discrete set of projected silhouette
    *vertices*, so the per-point distance has a floor of roughly half the
    projected-vertex spacing: it does not reach 0 at a perfect pose and is not
    directly comparable across mesh resolutions. Use a sufficiently dense
    (e.g. ~200k-vertex) mesh so the floor is sub-pixel.

    Returns {'mean': float, 'per_view': Dict[int, float]}
    per_view is a Dict[int, float] keyed by camera index.
    """
    euler_angles = pose_est[:3]
    translation = pose_est[3:6]
    # Always scale-aware (identical to the rigid transform at scale == 1.0), so a
    # caller that estimated a scale never gets a silently rigid measurement.
    mesh_transformed = apply_transform_to_mesh_with_scale(
        mesh, euler_angles, translation, scale)

    per_view = {}
    all_errors = []

    for cam_idx, (cam, target) in enumerate(zip(cameras, target_contours)):
        if target is None or len(target) == 0:
            continue

        # Project silhouette edges
        try:
            silhouette_2d, _ = extract_silhouette_edges_fast(mesh_transformed, cam)
        except Exception:
            continue

        if len(silhouette_2d) == 0:
            continue

        # Distance from each contour point to its nearest projected silhouette point.
        nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(silhouette_2d)
        distances, _ = nn.kneighbors(target)
        mean_dist = float(np.mean(distances))
        per_view[cam_idx] = mean_dist
        all_errors.append(mean_dist)

    mean_err = float(np.mean(all_errors)) if all_errors else float('inf')
    return {'mean': mean_err, 'per_view': per_view}


def compute_auc(errors: np.ndarray, max_threshold: float,
                num_steps: int = 100) -> float:
    """Area Under the success-threshold Curve.

    For thresholds in linspace(0, max_threshold, num_steps):
        success_rate = fraction of errors < threshold
    AUC = area under this curve, normalized to [0, 1].
    """
    errors = np.asarray(errors)
    if len(errors) == 0:
        return 0.0
    thresholds = np.linspace(0, max_threshold, num_steps)
    success_rates = np.array([np.mean(errors <= t) for t in thresholds])
    _trapz = getattr(np, 'trapezoid', np.trapz)
    auc = float(_trapz(success_rates, thresholds) / max_threshold)
    return auc


def success_rate(errors: np.ndarray, threshold: float) -> float:
    """Fraction of errors below threshold."""
    errors = np.asarray(errors)
    if len(errors) == 0:
        return 0.0
    return float(np.mean(errors <= threshold))


def compute_object_diameter(mesh_vertices: np.ndarray) -> float:
    """Compute object diameter for success thresholds.

    Uses max pairwise distance on a subsample of 1000 vertices.
    """
    verts = mesh_vertices
    if len(verts) > 1000:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(verts), size=1000, replace=False)
        verts = verts[indices]

    # Compute max pairwise distance
    dists = pdist(verts)
    return float(np.max(dists))


def compute_all_metrics(pose_est: np.ndarray, pose_gt: np.ndarray,
                        mesh_vertices: np.ndarray, cameras=None,
                        target_contours=None, mesh=None,
                        is_symmetric: bool = False,
                        object_diameter: float = None,
                        scale: float = 1.0) -> dict:
    """Compute all metrics and return as flat dict.

    Keys: 'rotation_error_deg', 'translation_error', 'add', 'add_s',
          'mssd', 'mspd', 'reprojection_mean_px', 'reprojection_per_view',
          'add_auc', 'success_rate_10pct', 'success_rate_20pct'
    For non-symmetric objects, add_s and mssd still computed but flagged.
    object_diameter: computed from mesh if not provided. Used for
    success_rate threshold (10% of diameter).
    """
    results = {}

    # Rotation and translation errors
    results['rotation_error_deg'] = rotation_error_deg(pose_est, pose_gt)
    results['translation_error'] = translation_error(pose_est, pose_gt)

    # ADD
    results['add'] = add_metric(pose_est, pose_gt, mesh_vertices)

    # ADD-S
    results['add_s'] = add_s_metric(pose_est, pose_gt, mesh_vertices)

    # MSSD
    results['mssd'] = mssd_metric(pose_est, pose_gt, mesh_vertices)

    # MSPD (requires cameras)
    if cameras is not None:
        results['mspd'] = mspd_metric(pose_est, pose_gt, mesh_vertices, cameras)
    else:
        results['mspd'] = None

    # Reprojection error (requires mesh + cameras + contours)
    if mesh is not None and cameras is not None and target_contours is not None:
        reproj = reprojection_error(pose_est, mesh, cameras, target_contours, scale=scale)
        results['reprojection_mean_px'] = reproj['mean']
        results['reprojection_per_view'] = reproj['per_view']
    else:
        results['reprojection_mean_px'] = None
        results['reprojection_per_view'] = None

    # Object diameter for thresholds
    if object_diameter is None:
        object_diameter = compute_object_diameter(mesh_vertices)
    results['object_diameter'] = object_diameter

    # Primary metric for success rate: ADD-S for symmetric, ADD for others
    primary_error = results['add_s'] if is_symmetric else results['add']
    results['is_symmetric'] = is_symmetric
    results['primary_metric'] = 'add_s' if is_symmetric else 'add'
    results['primary_error'] = primary_error

    # Success rates at 10% and 20% of diameter
    threshold_10 = 0.10 * object_diameter
    threshold_20 = 0.20 * object_diameter
    results['success_10pct'] = 1.0 if primary_error <= threshold_10 else 0.0
    results['success_20pct'] = 1.0 if primary_error <= threshold_20 else 0.0
    results['threshold_10pct'] = threshold_10
    results['threshold_20pct'] = threshold_20

    return results
