"""Full end-to-end contour registration pipeline.

Orchestration: load poses+mesh → load masks → Canny sweep → pick best config
→ refine → (optional plots) → write result.json.

Does **not** invoke LOO (leave-one-out); LOO is a standalone module.

Mask ↔ camera matching
----------------------
Masks (and images) are matched to the cameras by file name, see
:mod:`contour_reg.io.matching`: ``<stem>.<ext>`` belongs to the camera whose
view image has that stem. Cameras without names (IDR) take the files in
sorted file-name order. Missing or ambiguous matches raise
:class:`~contour_reg.io.matching.MatchingError`.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from contour_reg.config import CannyConfig, ICPConfig, SweepGrid
from contour_reg.io.matching import match_to_cameras
from contour_reg.io.mesh import load_mesh
from contour_reg.io.poses import load_poses
from contour_reg.pipeline._contours import build_target_contours
from contour_reg.pipeline.eval_canny import config_view_counts, eval_configs
from contour_reg.pipeline.refine import run_refine
from contour_reg.pipeline.sweep import run_sweep

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DEFAULT_GRID = SweepGrid(lows=[5, 10, 20, 50], highs=[20, 30, 40, 60, 80, 120])

SELECT_BY = ("fbeta", "error")


class SelectionError(ValueError):
    """No sweep config can be selected (e.g. every config scores F-beta = 0)."""


_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _list_images(directory: Path) -> List[Path]:
    """Return image files in *directory* sorted by name."""
    return sorted(
        p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
    )


def _load_grayscale(path: Path) -> np.ndarray:
    """Load *path* as a uint8 grayscale image; raise :class:`ValueError` on failure."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write *data* as JSON to *path* using a tmp-rename (atomic on POSIX)."""
    dir_ = path.parent
    dir_.mkdir(parents=True, exist_ok=True)
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


def _write_plots(summary: dict, error_history: List[float], plots_dir: Path) -> None:
    """Write sweep_heatmap, convergence, and pareto PNGs.

    Each plot is attempted independently; failures are caught, logged as a
    warning, and do not abort the pipeline.
    """
    # Lazy import: keep top-level imports fast and avoid matplotlib at load time.
    from contour_reg.viz.plots import (  # noqa: PLC0415
        plot_convergence,
        plot_pareto,
        plot_sweep_heatmap,
    )

    plots = [
        ("sweep_heatmap.png", plot_sweep_heatmap, (summary, plots_dir / "sweep_heatmap.png")),
        ("convergence.png", plot_convergence, (error_history, plots_dir / "convergence.png")),
        ("pareto.png", plot_pareto, (summary, plots_dir / "pareto.png")),
    ]
    for name, fn, args in plots:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Plotting {name} failed (non-fatal): {exc}",
                stacklevel=3,
            )
            logger.warning("Plot %s failed: %s", name, exc)


def _canny_config(cfg_data: dict) -> CannyConfig:
    return CannyConfig(
        canny_low=cfg_data["canny_low"],
        canny_high=cfg_data["canny_high"],
        margin_size=cfg_data["margin_size"],
        sobel_kernel=cfg_data["sobel_kernel"],
        include_inner_contours=cfg_data["include_inner_contours"],
    )


def select_by_fbeta(
    summary: dict,
    mesh,
    cameras,
    masks,
    images,
    *,
    beta: float = 0.5,
    select_epsilon: float = 2.0,
    trim_percentile: float = 75.0,
) -> dict:
    """Rank the sweep configs by contour F-beta, as ``contour-reg eval-canny`` does.

    Each config is scored at its own registered pose: its contour is compared with
    the projected mesh silhouette (see :mod:`contour_reg.pipeline.eval_canny`).

    Raises:
        SelectionError: When every config scores F-beta = 0, i.e. no contour point
            lies within *select_epsilon* px of the silhouette in any view.
    """
    per_config_views = {}
    for slug, info in summary["configs"].items():
        contours = build_target_contours(_canny_config(info["config"]), masks, images)
        per_config_views[slug] = config_view_counts(
            mesh, cameras, contours, np.asarray(info["pose"], dtype=float),
            float(info.get("scale", 1.0)), trim_percentile=trim_percentile,
        )
    result = eval_configs(per_config_views, select_epsilon=select_epsilon, beta=beta)
    if result["best"] is None or result["scores"][result["best"]] <= 0.0:
        raise SelectionError(
            f"No sweep config reaches the silhouette within {select_epsilon:g} px "
            f"(F{beta:g} = 0 for all {len(per_config_views)} configs): "
            "check the initial pose or the contours."
        )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    poses_path,
    mesh_path,
    masks_dir,
    out_dir,
    *,
    images_dir=None,
    grid: Optional[SweepGrid] = None,
    icp_config: Optional[ICPConfig] = None,
    init_pose: Optional[np.ndarray] = None,
    plot: bool = True,
    fmt: str = "auto",
    n_workers: int = 1,
    resume: bool = True,
    select_by: str = "fbeta",
) -> dict:
    """One-command end-to-end contour registration pipeline.

    Steps
    -----
    1. Load camera poses (``load_poses``) and mesh (``load_mesh``).
    2. Match one mask per camera from *masks_dir* by file name (see the
       module docstring); optionally the images from *images_dir* the same way.
    3. Run the Canny parameter sweep (``run_sweep``).
    4. Select a config: by default the one with the highest contour F0.5 at
       2 px (written to ``sweep/eval_canny.json``); with ``select_by="error"``,
       the one with the lowest reprojection error (``sweep_summary["best"]``).
       Take its coarse pose and scale.
    5. Rebuild per-view target contours for the best :class:`CannyConfig`
       and run ``run_refine`` (coarse-then-fine ICP).
    6. If *plot* is ``True``, write ``plots/{sweep_heatmap,convergence,pareto}.png``
       under *out_dir*.  Plotting failures are non-fatal (logged as warnings).
    7. Write ``out_dir/result.json`` atomically.
    8. Return a result dict.

    Args:
        poses_path: Path to the SfM file or directory (forwarded to
            :func:`~contour_reg.io.poses.load_poses`).
        mesh_path: Path to the mesh PLY/OBJ file.
        masks_dir: Directory of binary mask images, one per camera, matched
            by file name.
        out_dir: Root output directory; sub-directories (``sweep/``,
            ``plots/``) are created as needed.
        images_dir: Optional directory of grayscale source images used for
            Canny edge extraction instead of mask boundaries.
        grid: :class:`~contour_reg.config.SweepGrid` controlling Canny
            parameter combinations.  Defaults to :data:`DEFAULT_GRID`.
        icp_config: ICP hyper-parameters shared across all sweep configs and
            the final refinement.  Defaults to :class:`ICPConfig` defaults
            (``dof=6``: rigid; ``dof=7``: similarity with a uniform scale).
        init_pose: Starting pose ``[rx, ry, rz, tx, ty, tz]``.
            Defaults to ``np.zeros(6)`` (identity transform).
        plot: Write diagnostic PNGs under ``out_dir/plots/`` when ``True``.
        fmt: Format hint forwarded to :func:`~contour_reg.io.poses.load_poses`
            (``"auto"`` by default).
        n_workers: Number of parallel sweep workers forwarded to ``run_sweep``
            (``1`` → serial).
        resume: When ``True`` (default), ``run_sweep`` skips configs whose
            per-config JSON already exists, enabling checkpoint/resume after a
            wall-time kill on SLURM.  Forwarded to ``run_sweep``.
        select_by: ``"fbeta"`` (default) or ``"error"``; see step 4.

    Returns:
        ``dict`` with at least::

            {
                "best_config":   str,           # slug of the selected Canny config
                "final_pose":    list[float],   # length-6 refined pose
                "scale":         float,         # uniform scale (1.0 when rigid)
                "error_history": list[float],   # concatenated Phase A + B errors
                "result_path":   str,           # absolute path of result.json (JSON-safe)
            }

    Raises:
        MatchingError: When masks (or images) cannot be matched to every
            camera (missing or ambiguous file names).
        SelectionError: With ``select_by="fbeta"``, when no config reaches the
            silhouette (F-beta = 0 for every config).
    """
    if select_by not in SELECT_BY:
        raise ValueError(f"select_by must be one of {SELECT_BY}, got {select_by!r}")
    # Resolve mutable defaults
    if grid is None:
        grid = DEFAULT_GRID
    if icp_config is None:
        icp_config = ICPConfig()
    if init_pose is None:
        init_pose = np.zeros(6)

    poses_path = Path(poses_path)
    mesh_path = Path(mesh_path)
    masks_dir = Path(masks_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: load cameras and mesh
    # ------------------------------------------------------------------
    cameras = load_poses(poses_path, fmt=fmt)
    mesh = load_mesh(mesh_path)
    n_views = len(cameras)
    logger.debug("Loaded %d cameras; mesh has %d vertices.", n_views, len(mesh.vertices))

    # ------------------------------------------------------------------
    # Step 2: load masks (matched to the cameras by view name)
    # ------------------------------------------------------------------
    mask_paths = match_to_cameras(_list_images(masks_dir), cameras, "mask")
    masks: List[np.ndarray] = [_load_grayscale(p) for p in mask_paths]
    logger.info("Loaded %d cameras, mesh (%d vertices), and %d masks.", n_views, len(mesh.vertices), len(masks))

    images: Optional[List[np.ndarray]] = None
    if images_dir is not None:
        image_paths = match_to_cameras(_list_images(Path(images_dir)), cameras, "image")
        images = [_load_grayscale(p) for p in image_paths]

    # ------------------------------------------------------------------
    # Step 3: Canny parameter sweep
    # ------------------------------------------------------------------
    n_configs = len(list(grid.configs()))
    logger.info("Running Canny sweep over %d configs…", n_configs)
    sweep_dir = out_dir / "sweep"
    run_sweep(
        mesh=mesh,
        cameras=cameras,
        masks=masks,
        images=images,
        grid=grid,
        icp_config=icp_config,
        init_pose=init_pose,
        out_dir=sweep_dir,
        n_workers=n_workers,
        resume=resume,
    )

    # ------------------------------------------------------------------
    # Step 4: select a config (contour F-beta by default)
    # ------------------------------------------------------------------
    summary_path = sweep_dir / "sweep_summary.json"
    with open(summary_path) as fh:
        summary = json.load(fh)

    selection_score: Optional[float] = None
    if select_by == "fbeta":
        evaluation = select_by_fbeta(summary, mesh, cameras, masks, images)
        _write_json_atomic(sweep_dir / "eval_canny.json", evaluation)
        best_slug: str = evaluation["best"]
        selection_score = float(evaluation["scores"][best_slug])
    else:
        best_slug = summary["best"]
    best_info: dict = summary["configs"][best_slug]
    coarse_pose = np.array(best_info["pose"], dtype=np.float64)
    coarse_scale = float(best_info.get("scale", 1.0))
    logger.info("Selected config: %s (by %s; error=%.2f px)", best_slug, select_by, best_info["error"])

    best_cfg = _canny_config(best_info["config"])

    # ------------------------------------------------------------------
    # Step 5: rebuild target contours for the selected config and refine
    # ------------------------------------------------------------------
    logger.info("Running coarse-then-fine refinement…")
    target_contours = build_target_contours(best_cfg, masks, images)

    pose, hist, scale = run_refine(
        mesh=mesh,
        cameras=cameras,
        target_contours=target_contours,
        init_pose=coarse_pose,
        keep_pct=90.0,
        error_percentile=icp_config.error_percentile,
        max_correspondence_distance=icp_config.max_correspondence_distance,
        max_iterations=icp_config.max_iterations,
        optimize_scale=icp_config.dof == 7,
        initial_scale=coarse_scale if icp_config.dof == 7 else 1.0,
    )

    # ------------------------------------------------------------------
    # Step 6: optional plots (failures are non-fatal)
    # ------------------------------------------------------------------
    if plot:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        _write_plots(summary, hist, plots_dir)

    # ------------------------------------------------------------------
    # Step 7: write result.json atomically
    # ------------------------------------------------------------------
    result_path = out_dir / "result.json"
    result_data = {
        "best_config": best_slug,
        "final_pose": pose.tolist(),
        "scale": float(scale),
        "dof": int(icp_config.dof),
        "select_by": select_by,
        "selection_score": selection_score,
        "error_history": [float(e) for e in hist],
        "sweep_summary": "sweep/sweep_summary.json",
    }
    _write_json_atomic(result_path, result_data)
    logger.debug("Written result to %s", result_path)
    logger.info("Done: final pose written to %s", result_path)

    return {
        "best_config": best_slug,
        "final_pose": pose.tolist(),
        "scale": float(scale),
        "error_history": [float(e) for e in hist],
        "result_path": str(result_path),
    }
