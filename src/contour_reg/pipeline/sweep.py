"""Canny parameter sweep for contour-based 2D-3D registration.

Iterates over a ``SweepGrid`` of Canny configurations, runs ICP for each,
ranks results by mean reprojection error, and writes per-config + summary JSON
files with resume support.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
import trimesh

from contour_reg.config import CannyConfig, ICPConfig, SweepGrid
from contour_reg.io.camera import Camera
from contour_reg.metrics import reprojection_error
from contour_reg.pipeline._contours import build_target_contours
from contour_reg.registration.icp import contour_based_icp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_single_config(
    cfg: CannyConfig,
    mesh: trimesh.Trimesh,
    cameras: List[Camera],
    masks: List[np.ndarray],
    images: Optional[List[np.ndarray]],
    icp_config: ICPConfig,
    init_pose: np.ndarray,
) -> dict:
    """Run ICP for a single Canny config and return the result dict."""
    target_contours = build_target_contours(cfg, masks, images)

    final_pose, _error_history, _final_scale = contour_based_icp(
        mesh,
        init_pose,
        cameras,
        target_contours,
        max_iterations=icp_config.max_iterations,
        error_threshold_percentile=icp_config.error_percentile,
        max_correspondence_distance=icp_config.max_correspondence_distance,
        include_inner_contours=cfg.include_inner_contours,
        optimize_scale=icp_config.dof == 7,
        verbose=False,
    )

    reproj = reprojection_error(final_pose, mesh, cameras, target_contours,
                                scale=_final_scale)
    error = reproj["mean"]

    return {
        "pose": final_pose.tolist(),
        "scale": float(_final_scale),
        "error": float(error),
        "config": {
            "canny_low": cfg.canny_low,
            "canny_high": cfg.canny_high,
            "margin_size": cfg.margin_size,
            "sobel_kernel": cfg.sobel_kernel,
            "include_inner_contours": cfg.include_inner_contours,
        },
    }


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write *data* as JSON to *path* atomically (tmp + rename)."""
    dir_ = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp.json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sweep(
    mesh: trimesh.Trimesh,
    cameras: List[Camera],
    masks: List[np.ndarray],
    images: Optional[List[np.ndarray]],
    grid: SweepGrid,
    icp_config: ICPConfig,
    init_pose: np.ndarray,
    out_dir,
    n_workers: int = 1,
    resume: bool = False,
) -> dict:
    """Run the Canny parameter sweep over all configs in *grid*.

    For each :class:`~contour_reg.config.CannyConfig` yielded by
    ``grid.configs()``:

    1. Build per-view target contours (Canny or duck-typed mask passthrough).
    2. Run :func:`~contour_reg.registration.icp.contour_based_icp`.
    3. Rank by mean reprojection error.
    4. Write ``{out_dir}/{slug}.json`` (atomic) and update
       ``{out_dir}/sweep_summary.json``.

    Args:
        mesh: Source mesh.
        cameras: List of cameras (one per view).
        masks: One entry per camera: either a ``(N, 2)`` contour array or a
            binary mask image.  Ignored when *images* is provided.
        images: Optional list of grayscale images (one per camera).  When
            provided, Canny edges are extracted per config.
        grid: Sweep grid defining the Canny parameter combinations to explore.
        icp_config: ICP hyper-parameters shared across all configs.
        init_pose: Initial pose ``[rx, ry, rz, tx, ty, tz]`` (zeros → identity).
        out_dir: Directory for output JSON files.  Created if absent.
        n_workers: Number of parallel workers (``ProcessPoolExecutor``).
            ``1`` → serial execution.
        resume: When ``True``, skip configs whose ``{slug}.json`` already exists
            and load the cached result instead.

    Returns:
        Mapping ``{slug: {'pose': list, 'error': float, 'config': dict}}``
        for every config in *grid*.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = list(grid.configs())
    results: dict = {}

    # Partition configs into cached (resume) and pending.
    pending_configs: List[CannyConfig] = []
    for cfg in configs:
        slug = cfg.slug
        json_path = out_dir / f"{slug}.json"
        if resume and json_path.exists():
            with open(json_path) as fh:
                results[slug] = json.load(fh)
        else:
            pending_configs.append(cfg)

    if n_workers == 1 or len(pending_configs) <= 1:
        # Serial path.
        for cfg in pending_configs:
            result = _run_single_config(cfg, mesh, cameras, masks, images, icp_config, init_pose)
            slug = cfg.slug
            results[slug] = result
            _write_json_atomic(out_dir / f"{slug}.json", result)
            if n_workers == 1:
                logger.info("sweep %s: error=%.2f px", slug, result["error"])
    else:
        # Parallel path via ProcessPoolExecutor.
        future_to_slug = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for cfg in pending_configs:
                future = executor.submit(
                    _run_single_config,
                    cfg, mesh, cameras, masks, images, icp_config, init_pose,
                )
                future_to_slug[future] = cfg.slug

            for future in as_completed(future_to_slug):
                slug = future_to_slug[future]
                result = future.result()
                results[slug] = result
                _write_json_atomic(out_dir / f"{slug}.json", result)

    # Build and write summary (sorted by error, best annotated).
    _write_sweep_summary(results, out_dir)

    return results


def _write_sweep_summary(results: dict, out_dir: Path) -> None:
    """Write ``sweep_summary.json`` sorted by error with best-slug annotation."""
    if not results:
        summary = {"best": None, "configs": {}}
        _write_json_atomic(out_dir / "sweep_summary.json", summary)
        return

    sorted_slugs = sorted(results, key=lambda s: results[s]["error"])
    best_slug = sorted_slugs[0]

    summary = {
        "best": best_slug,
        "configs": {slug: results[slug] for slug in sorted_slugs},
    }
    _write_json_atomic(out_dir / "sweep_summary.json", summary)
