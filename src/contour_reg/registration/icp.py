"""Contour-based ICP core for 2D-3D mesh registration from silhouettes.

- **Phase 1** (main convergence loop): fix correspondences, compute reprojection
  error, optionally filter contour points by ``max_correspondence_distance``, check
  convergence, then refine the pose with a ``soft_l1`` least-squares step (accept
  only if the error on the same correspondences decreases).  Best pose/scale are
  tracked throughout; with a cutoff, poses are ranked by the truncated error
  ``mean(min(d, cutoff))`` over all contour points, measured at each pose.
- **Phase 2** (percentile refinement / trimmed ICP): runs only when
  ``error_threshold_percentile < 100`` (75 by default).  Recomputes the full
  correspondence set, keeps the best percentile by residual (bidirectional trim),
  and re-optimizes over a few refinement iterations.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from scipy.optimize import least_squares
from sklearn.neighbors import NearestNeighbors

from contour_reg.io.camera import Camera
from contour_reg.io.mesh import (
    apply_transform_to_mesh,
    apply_transform_to_mesh_with_scale,
)
from contour_reg.registration.correspondences import (
    clear_mesh_cache,
    compute_correspondences_parallel,
    compute_error_parallel,
)
from contour_reg.registration.residuals import (
    residual_function_optimized,
    residual_function_with_scale,
)

# Internal default for Phase 2 trimmed-ICP refinement iterations.
_REFINEMENT_ITERATIONS = 3


def _refine_correspondences(
    mesh_transformed: trimesh.Trimesh,
    cameras: List[Camera],
    fixed_correspondences: Dict,
    error_threshold_percentile: float,
) -> Dict:
    """Bidirectional percentile trim of correspondences.

    Ported from ``utils_optimized.py::refine_correspondences_parallel`` (run
    serially here: the per-view work is light and avoids a joblib dependency in
    the hot path).  For each view: keep target points whose NN distance to the
    projected set is within the ``error_threshold_percentile`` percentile, then
    keep projected points (and their ``indices_3d``) within the same percentile of
    distance to the trimmed target set.
    """
    refined: Dict = {}
    for cam_idx in fixed_correspondences:
        corr = fixed_correspondences[cam_idx]
        indices_3d = corr['indices_3d']
        target_points_2d = corr['points_2d']

        if len(indices_3d) == 0 or len(target_points_2d) == 0:
            continue

        vertices_3d = mesh_transformed.vertices[indices_3d]
        projected_2d = cameras[cam_idx].project_points(vertices_3d)

        # Step 1: filter TARGET points far from any projected point.
        nn_proj = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        nn_proj.fit(projected_2d)
        dist_target, _ = nn_proj.kneighbors(target_points_2d)
        dist_target = dist_target.flatten()

        threshold_target = np.percentile(dist_target, error_threshold_percentile)
        target_mask = dist_target <= threshold_target
        filtered_target = target_points_2d[target_mask]

        if len(filtered_target) == 0:
            continue

        # Step 2: filter PROJECTED points (+ indices_3d) far from trimmed target.
        nn_target = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        nn_target.fit(filtered_target)
        dist_proj, _ = nn_target.kneighbors(projected_2d)
        dist_proj = dist_proj.flatten()

        threshold_proj = np.percentile(dist_proj, error_threshold_percentile)
        proj_mask = dist_proj <= threshold_proj
        filtered_indices_3d = indices_3d[proj_mask]

        if len(filtered_indices_3d) == 0:
            continue

        refined[cam_idx] = {
            'indices_3d': filtered_indices_3d,
            'points_2d': filtered_target,
        }

    return refined


def contour_based_icp(
    mesh_source: trimesh.Trimesh,
    initial_pose: np.ndarray,
    cameras: List[Camera],
    target_contours: List[np.ndarray],
    *,
    max_iterations: int = 50,
    convergence_threshold: float = 1e-3,
    error_threshold_percentile: float = 75.0,
    max_correspondence_distance: float = 0.0,
    keep_pct: float = 100.0,
    include_inner_contours: bool = False,
    optimize_scale: bool = False,
    initial_scale: float = 1.0,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    n_jobs: int = -1,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[float], float]:
    """Contour-based ICP: align *mesh_source* to *target_contours* silhouettes.

    Args:
        mesh_source: Source mesh.
        initial_pose: Initial pose ``[rx, ry, rz, tx, ty, tz]``.
        cameras: List of cameras.
        target_contours: List of target contour point arrays (one per camera).
        max_iterations: Maximum Phase-1 ICP iterations.
        convergence_threshold: Phase-1 convergence threshold on error change.
        error_threshold_percentile: Percentile kept during Phase-2 trimming.
            Default 75 (keep the closest 75 %); ``>= 100`` skips Phase 2 entirely.
        max_correspondence_distance: Per-view residual cutoff (px) for Phase-1
            optimization.  ``0`` disables filtering (default).
        keep_pct: After the ``max_correspondence_distance`` cutoff, keep only the
            closest ``keep_pct`` percent of surviving contour points per view
            (``100`` keeps all).  Combined with an absolute cutoff this matches
            the reference "epsilon + top-keep_pct%" refinement.
        include_inner_contours: Include inner (hole) contours in correspondences.
        optimize_scale: Also optimize a uniform scale factor.
        initial_scale: Initial scale (default 1.0).
        n_jobs: Parallel jobs (-1 = all cores).
        verbose: Print progress.

    Returns:
        ``(final_pose, error_history, final_scale)``.
    """
    if len(cameras) != len(target_contours):
        raise ValueError(
            f"cameras ({len(cameras)}) and target_contours "
            f"({len(target_contours)}) must be the same length.")
    current_pose = initial_pose.copy()
    current_scale = initial_scale
    error_history: List[float] = []

    # Best-pose tracking.
    best_pose = initial_pose.copy()
    best_scale = initial_scale
    best_error = float('inf')

    if verbose:
        print("=" * 70)
        print("Contour-based ICP - Starting")
        print("=" * 70)
        print(f"Initial pose: {current_pose}")
        if optimize_scale:
            print(f"Initial scale: {current_scale:.6f}")
        print(f"Number of views: {len(cameras)}")
        print(f"Parallelism: {n_jobs} jobs")
        print(f"Scale optimization: {'ENABLED' if optimize_scale else 'DISABLED'}")

    # Clear caches at startup.
    clear_mesh_cache()
    score_history: List[float] = []      # truncated score per measured pose (cutoff mode)
    last_accepted_unscored = False

    # ------------------------------------------------------------------ Phase 1
    for iteration in range(max_iterations):
        iter_start = time.time()

        if verbose:
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        # Transform mesh with current pose.
        euler_angles = current_pose[:3]
        translation = current_pose[3:6]
        if optimize_scale:
            mesh_current = apply_transform_to_mesh_with_scale(
                mesh_source, euler_angles, translation, current_scale
            )
        else:
            mesh_current = apply_transform_to_mesh(mesh_source, euler_angles, translation)

        # Step 1: fix correspondences (parallel).
        fixed_correspondences = compute_correspondences_parallel(
            mesh_current, cameras, target_contours, n_jobs=n_jobs,
            include_inner_contours=include_inner_contours,
        )

        if not fixed_correspondences:
            if verbose:
                print("  No correspondences found, stopping")
            break

        # Compute mean error (parallel). Need distances if filtering by distance.
        need_distances = max_correspondence_distance > 0
        if need_distances:
            mean_error, per_view_errors, per_view_distances = compute_error_parallel(
                mesh_current.vertices, cameras, fixed_correspondences,
                n_jobs=n_jobs, return_distances=True,
            )
        else:
            mean_error, per_view_errors = compute_error_parallel(
                mesh_current.vertices, cameras, fixed_correspondences, n_jobs=n_jobs,
            )
        error_history.append(mean_error)
        last_accepted_unscored = False
        if need_distances:
            # Truncated error mean(min(d, cutoff)) over ALL contour points: comparable
            # across poses; it scores the pose it is measured at.
            truncated = [np.minimum(d, max_correspondence_distance).mean()
                         for d in per_view_distances.values() if len(d)]
            score = float(np.mean(truncated)) if truncated else float('inf')
            if score < best_error:
                best_pose, best_scale, best_error = current_pose.copy(), current_scale, score

        if verbose:
            print(f"  Correspondences: {len(fixed_correspondences)} views")
            print(f"  Mean error: {mean_error:.6f} px")

        # Filter: keep only contour points within the residual threshold of the mesh.
        correspondences_for_opt = fixed_correspondences
        if max_correspondence_distance > 0 and need_distances:
            correspondences_for_opt = {}
            n_pts_before = sum(len(c['points_2d']) for c in fixed_correspondences.values())
            for cam_idx, corr in fixed_correspondences.items():
                dists = per_view_distances.get(cam_idx, np.array([]))
                if len(dists) == 0:
                    continue
                keep = dists <= max_correspondence_distance
                if keep.sum() == 0:
                    continue  # view entirely rejected
                if keep_pct < 100.0:
                    # Among the points within the cutoff, keep the closest keep_pct%.
                    threshold = np.percentile(dists[keep], keep_pct)
                    keep = keep & (dists <= threshold)
                    if keep.sum() == 0:
                        continue
                correspondences_for_opt[cam_idx] = {
                    'indices_3d': corr['indices_3d'],
                    'points_2d': corr['points_2d'][keep],
                }
            n_pts_after = sum(len(c['points_2d']) for c in correspondences_for_opt.values())
            n_views_dropped = len(fixed_correspondences) - len(correspondences_for_opt)
            if verbose:
                print(f"  Filtering ≤{max_correspondence_distance:.0f}px: "
                      f"{n_pts_before}→{n_pts_after} pts, "
                      f"{n_views_dropped} views entirely rejected, "
                      f"{len(correspondences_for_opt)} views kept")

        # Convergence check (cutoff mode: on the truncated score).
        if need_distances:
            score_history.append(score)
        if iteration > 0:
            h = score_history if need_distances else error_history
            error_change = abs(h[-1] - h[-2])
            if error_change < convergence_threshold:
                if verbose:
                    print(f"\n✓ Converged (change={error_change:.6f})")
                break

        # Step 2: optimize with FILTERED correspondences.
        if verbose:
            print("  Optimizing...")

        if optimize_scale:
            # Params: [rx, ry, rz, tx, ty, tz, scale]
            initial_params = np.concatenate([current_pose, [current_scale]])
            scale_lb = min(0.0001, current_scale * 0.1)
            scale_ub = max(100.0, current_scale * 100.0)
            result = least_squares(
                residual_function_with_scale,
                initial_params,
                args=(mesh_source.vertices, cameras, correspondences_for_opt),
                method='trf',
                loss=loss,
                f_scale=f_scale,
                max_nfev=100,
                ftol=1e-6,
                xtol=1e-6,
                verbose=0,
                bounds=([-np.inf] * 6 + [scale_lb], [np.inf] * 6 + [scale_ub]),
            )
            candidate_pose = result.x[:6]
            candidate_scale = result.x[6]
            mesh_candidate = apply_transform_to_mesh_with_scale(
                mesh_source, candidate_pose[:3], candidate_pose[3:6], candidate_scale
            )
        else:
            result = least_squares(
                residual_function_optimized,
                current_pose,
                args=(mesh_source.vertices, cameras, correspondences_for_opt),
                method='trf',
                loss=loss,
                f_scale=f_scale,
                max_nfev=100,
                ftol=1e-6,
                xtol=1e-6,
                verbose=0,
            )
            candidate_pose = result.x
            candidate_scale = 1.0
            mesh_candidate = apply_transform_to_mesh(
                mesh_source, candidate_pose[:3], candidate_pose[3:6]
            )

        # Error with candidate pose (on the same filtered correspondences), compared with
        # the current pose on that same set.
        candidate_error, _ = compute_error_parallel(
            mesh_candidate.vertices, cameras, correspondences_for_opt, n_jobs=n_jobs,
        )
        reference_error = mean_error
        if need_distances:
            reference_error, _ = compute_error_parallel(
                mesh_current.vertices, cameras, correspondences_for_opt, n_jobs=n_jobs,
            )

        # Accept if error decreases.
        if candidate_error < reference_error:
            current_pose = candidate_pose
            current_scale = candidate_scale
            last_accepted_unscored = need_distances   # scored at the next measurement
            if not need_distances and candidate_error < best_error:
                best_pose = candidate_pose.copy()
                best_scale = candidate_scale
                best_error = candidate_error
            if verbose:
                scale_info = f", scale={candidate_scale:.6f}" if optimize_scale else ""
                print(f"  ✓ Pose accepted (error {reference_error:.4f} -> "
                      f"{candidate_error:.4f} px{scale_info})")
        else:
            if verbose:
                print(f"  ✗ Pose rejected (error increased: {reference_error:.4f} -> "
                      f"{candidate_error:.4f} px)")

        if verbose:
            print(f"  Iteration time: {time.time() - iter_start:.2f}s")

    if last_accepted_unscored:
        # The last accepted pose was never measured (iteration budget exhausted): score it.
        mesh_last = (apply_transform_to_mesh_with_scale(mesh_source, current_pose[:3], current_pose[3:6], current_scale)
                     if optimize_scale else apply_transform_to_mesh(mesh_source, current_pose[:3], current_pose[3:6]))
        corr = compute_correspondences_parallel(mesh_last, cameras, target_contours, n_jobs=n_jobs,
                                                include_inner_contours=include_inner_contours)
        if corr:
            _, _, dists = compute_error_parallel(mesh_last.vertices, cameras, corr, n_jobs=n_jobs,
                                                 return_distances=True)
            truncated = [np.minimum(d, max_correspondence_distance).mean() for d in dists.values() if len(d)]
            if truncated and float(np.mean(truncated)) < best_error:
                best_pose, best_scale, best_error = current_pose.copy(), current_scale, float(np.mean(truncated))

    # ------------------------------------------------------------------ Phase 2
    # Trimmed ICP: filter from full set each iteration. Runs when percentile<100.
    if error_threshold_percentile < 100.0:
        if verbose:
            print("\n" + "=" * 70)
            print("TRIMMED ICP (iterative filtering + optimization)")
            print("=" * 70)

        # Phase 2 scores poses on the trimmed set, which is not comparable with the
        # Phase-1 error: its best pose is tracked on its own and supersedes Phase 1.
        # Each trimmed error scores the pose it is measured at (before optimizing);
        # one extra measurement scores the last optimized pose.
        prev_trimmed_error = float('inf')
        best_error = float('inf')

        for refine_iter in range(_REFINEMENT_ITERATIONS + 1):
            refine_start = time.time()

            if verbose:
                print(f"\n--- Trimmed ICP {refine_iter + 1}/{_REFINEMENT_ITERATIONS} ---")

            # 1. Transform mesh with current pose.
            if optimize_scale:
                mesh_current = apply_transform_to_mesh_with_scale(
                    mesh_source, current_pose[:3], current_pose[3:6], current_scale
                )
            else:
                mesh_current = apply_transform_to_mesh(
                    mesh_source, current_pose[:3], current_pose[3:6]
                )

            # 2. Compute ALL correspondences from scratch (full set).
            full_correspondences = compute_correspondences_parallel(
                mesh_current, cameras, target_contours, n_jobs=n_jobs,
                include_inner_contours=include_inner_contours,
            )

            if not full_correspondences:
                if verbose:
                    print("  No correspondences, stopping")
                break

            # 3. Filter by percentile (keep best h%).
            trimmed_correspondences = _refine_correspondences(
                mesh_current, cameras, full_correspondences,
                error_threshold_percentile=error_threshold_percentile,
            )

            if not trimmed_correspondences:
                if verbose:
                    print("  No correspondences after filtering, stopping")
                break

            # 4. Trimmed error (score on filtered set, before optimization).
            trimmed_error, _ = compute_error_parallel(
                mesh_current.vertices, cameras, trimmed_correspondences, n_jobs=n_jobs,
            )

            if verbose:
                n_full = sum(len(full_correspondences[k]['points_2d'])
                             for k in full_correspondences)
                n_trimmed = sum(len(trimmed_correspondences[k]['points_2d'])
                                for k in trimmed_correspondences)
                print(f"  Correspondences: {n_full} → {n_trimmed} "
                      f"(trimmed {error_threshold_percentile}%)")
                print(f"  Trimmed error: {trimmed_error:.4f} px")

            # Convergence on trimmed error.
            if trimmed_error >= prev_trimmed_error:
                if verbose:
                    print(f"  ✗ Trimmed error did not decrease "
                          f"({prev_trimmed_error:.4f} → {trimmed_error:.4f}), stopping")
                break
            prev_trimmed_error = trimmed_error
            best_pose = current_pose.copy()
            best_scale = current_scale
            best_error = trimmed_error
            error_history.append(trimmed_error)
            if refine_iter == _REFINEMENT_ITERATIONS:
                break

            # 5. Optimize on trimmed correspondences.
            if optimize_scale:
                initial_params = np.concatenate([current_pose, [current_scale]])
                scale_lb = min(0.0001, current_scale * 0.1)
                scale_ub = max(100.0, current_scale * 100.0)
                result = least_squares(
                    residual_function_with_scale,
                    initial_params,
                    args=(mesh_source.vertices, cameras, trimmed_correspondences),
                    method='trf',
                    loss=loss,
                f_scale=f_scale,
                    max_nfev=100,
                    ftol=1e-6,
                    xtol=1e-6,
                    verbose=0,
                    bounds=([-np.inf] * 6 + [scale_lb], [np.inf] * 6 + [scale_ub]),
                )
                refined_pose = result.x[:6]
                refined_scale = result.x[6]
            else:
                result = least_squares(
                    residual_function_optimized,
                    current_pose,
                    args=(mesh_source.vertices, cameras, trimmed_correspondences),
                    method='trf',
                    loss=loss,
                f_scale=f_scale,
                    max_nfev=100,
                    ftol=1e-6,
                    xtol=1e-6,
                    verbose=0,
                )
                refined_pose = result.x
                refined_scale = current_scale

            # Candidate pose, scored at the next iteration.
            current_pose = refined_pose
            current_scale = refined_scale

            if verbose:
                scale_info = f", scale={refined_scale:.6f}" if optimize_scale else ""
                print(f"  ✓ Optimized{scale_info}")
                print(f"  Time: {time.time() - refine_start:.2f}s")

    # Use the best pose.
    final_pose = best_pose if best_error < float('inf') else current_pose
    final_scale = best_scale if best_error < float('inf') else current_scale
    final_error = (best_error if best_error < float('inf')
                   else (error_history[-1] if error_history else float('inf')))

    if verbose:
        print("\n" + "=" * 70)
        print("Optimization finished")
        print("=" * 70)
        print(f"Final pose: {final_pose}")
        if optimize_scale:
            print(f"Final scale: {final_scale:.6f}")
        print(f"Final error: {final_error:.6f} px")

    return final_pose, error_history, final_scale
