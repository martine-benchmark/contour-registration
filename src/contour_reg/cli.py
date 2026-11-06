"""contour-reg command-line interface.

Exposes ``main(argv=None) -> int`` as the console entry point for the
``contour-reg`` command declared in pyproject.toml.

Subcommands
-----------
- ``extract-contours``: extract per-mask contours, save as .npy
- ``register``: run contour-based ICP on a set of contours
- ``sweep``: Canny parameter sweep
- ``refine``: coarse-then-fine refinement from a sweep result
- ``eval-loo``: leave-one-out registration evaluation
- ``pipeline``: full end-to-end pipeline (masks are an INPUT; never runs SAM)
- ``plot``: write sweep heatmap / pareto PNGs from a results dir
- ``segment``: text-promptable SAM 3 mask generation (optional extra)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_contour_dir(contours_dir: Path, cameras=None) -> List[np.ndarray]:
    """Load one contour file per camera from *contours_dir*, matched by file name.

    Supports ``.npy`` and image formats understood by
    :func:`~contour_reg.io.masks.load_contour_points`. Without *cameras*, all
    files are returned in sorted file-name order.
    """
    from contour_reg.io.masks import load_contour_points

    _IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    supported = {".npy"} | _IMAGE_SUFFIXES

    paths = sorted(
        p for p in contours_dir.iterdir()
        if p.suffix.lower() in supported
    )
    if cameras is not None:
        from contour_reg.io.matching import match_to_cameras

        paths = match_to_cameras(paths, cameras, "contour")
    return [load_contour_points(p) for p in paths]


def _load_grayscale(path: Path) -> np.ndarray:
    """Read an image as grayscale uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def _list_images(directory: Path) -> List[Path]:
    """Return image files in *directory*, sorted by name."""
    _IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)


def _list_masks(directory: Path) -> List[Path]:
    """Return mask files (.npy or images) from *directory*, sorted by name."""
    _SUFFIXES = frozenset({".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _SUFFIXES)


def _parse_grid(grid_str: str):
    """Parse a grid string like ``"lo=5,10,20 hi=20,30,40,60"`` into a SweepGrid."""
    from contour_reg.config import SweepGrid

    lows: Optional[List[int]] = None
    highs: Optional[List[int]] = None

    for token in grid_str.split():
        if token.startswith("lo="):
            lows = [int(x) for x in token[3:].split(",")]
        elif token.startswith("hi="):
            highs = [int(x) for x in token[3:].split(",")]

    if lows is None or highs is None:
        raise argparse.ArgumentTypeError(
            f"Invalid grid string {grid_str!r}. "
            "Expected format: \"lo=5,10,20 hi=20,30,40,60\""
        )
    return SweepGrid(lows=lows, highs=highs)


def _write_json(path: Path, data: dict) -> None:
    """Write *data* to *path* atomically (tmp + rename)."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_extract_contours(args: argparse.Namespace) -> int:
    """Handler for ``extract-contours``."""
    from contour_reg.contours.canny import (
        contour_image_to_points,
        extract_canny_contours,
        filter_connected_components,
        mask_boundary_contour,
    )

    masks_dir = Path(args.masks)
    out_dir = Path(args.o)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images) if args.images else None

    mask_paths = _list_masks(masks_dir)
    if not mask_paths:
        print(f"No mask files found in {masks_dir}", file=sys.stderr)
        return 1

    image_paths: Optional[List[Path]] = None
    if images_dir is not None:
        from contour_reg.io.matching import match_by_name

        image_paths = match_by_name(_list_images(images_dir), [p.stem for p in mask_paths], "image", "mask")

    # Warn once if Canny was requested but no images were provided.
    if args.mode == "canny" and image_paths is None:
        print(
            "Warning: --mode canny requires --images; no images provided; "
            "falling back to mask-boundary extraction.",
            file=sys.stderr,
        )

    for idx, mp in enumerate(mask_paths):
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Could be .npy contour array: treat as points directly
            pts = np.load(mp)
        else:
            # Keep a single connected component before contour extraction.
            mask = filter_connected_components(mask, mode=args.component_mode)
            if args.mode == "canny" and image_paths is not None:
                image_gray = _load_grayscale(image_paths[idx])
                edge_img = extract_canny_contours(
                    image_gray,
                    mask,
                    margin_size=args.margin,
                    canny_low=args.canny_low,
                    canny_high=args.canny_high,
                    sobel_kernel=args.sobel,
                )
                pts = contour_image_to_points(edge_img)
            else:
                pts = mask_boundary_contour(mask, include_inner=args.include_inner_contours)

        out_path = out_dir / f"{mp.stem}.npy"
        np.save(out_path, pts)

    return 0


def _cmd_register(args: argparse.Namespace) -> int:
    """Handler for ``register``."""
    from contour_reg.io.mesh import load_mesh
    from contour_reg.io.poses import load_poses
    from contour_reg.registration.icp import contour_based_icp

    cameras = load_poses(args.poses, fmt=args.format)
    mesh = load_mesh(args.mesh)

    contours_dir = Path(args.contours)
    target_contours = _load_contour_dir(contours_dir, cameras)

    # View selection: drop views whose contour has too few points.
    if args.min_canny_points > 0:
        kept = [
            (cam, contour)
            for cam, contour in zip(cameras, target_contours)
            if len(contour) >= args.min_canny_points
        ]
        dropped = len(cameras) - len(kept)
        if dropped:
            print(
                f"Dropped {dropped} view(s) with fewer than "
                f"{args.min_canny_points} contour points.",
                file=sys.stderr,
            )
        cameras = [cam for cam, _ in kept]
        target_contours = [contour for _, contour in kept]

    if args.init_transform is not None:
        if args.init_pose is not None:
            print("--init-transform and --init-pose are mutually exclusive.", file=sys.stderr)
            return 2
        from contour_reg.registration.transforms import similarity_to_pose_scale

        init_matrix = np.load(args.init_transform)
        init_pose, initial_scale = similarity_to_pose_scale(init_matrix)
        print(f"Init from {args.init_transform}: scale={initial_scale:.6f}, "
              f"pose={np.array2string(init_pose, precision=5)}")
        # The rigid (dof=6) path applies rotation+translation only and ignores
        # scale, so a non-unit init scale would be silently dropped and diverge.
        if args.dof != 7 and abs(initial_scale - 1.0) > 1e-3:
            print(
                f"ERROR: --init-transform carries scale {initial_scale:.6f} but --dof {args.dof} "
                "ignores scale (rigid). Use --dof 7 to honour the init scale.",
                file=sys.stderr,
            )
            return 2
    else:
        init_pose = np.array(args.init_pose, dtype=float) if args.init_pose else np.zeros(6)
        initial_scale = 1.0
    optimize_scale = (args.dof == 7)

    final_pose, error_history, final_scale = contour_based_icp(
        mesh,
        init_pose,
        cameras,
        target_contours,
        max_iterations=args.max_iterations,
        error_threshold_percentile=args.error_percentile,
        max_correspondence_distance=args.max_correspondence_distance,
        include_inner_contours=args.include_inner_contours,
        optimize_scale=optimize_scale,
        initial_scale=initial_scale,
        loss=args.loss,
        f_scale=args.f_scale,
    )

    result = {
        "pose": final_pose.tolist(),
        "scale": float(final_scale),
        "loss": args.loss,
        "f_scale": args.f_scale,
        "error_history": error_history,
        "dof": args.dof,
        "init_pose": np.asarray(init_pose, dtype=float).tolist(),
        "init_scale": float(initial_scale),
        "init_transform": args.init_transform,
    }

    out_path = Path(args.o)
    _write_json(out_path, result)
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    """Handler for ``sweep``."""
    from contour_reg.config import ICPConfig
    from contour_reg.io.mesh import load_mesh
    from contour_reg.io.poses import load_poses
    from contour_reg.pipeline.sweep import run_sweep
    from contour_reg.viz.plots import plot_pareto, plot_sweep_heatmap

    cameras = load_poses(args.poses, fmt=args.format)
    mesh = load_mesh(args.mesh)

    from contour_reg.io.matching import match_to_cameras

    masks_dir = Path(args.masks)
    mask_paths = match_to_cameras(_list_masks(masks_dir), cameras, "mask")
    masks = []
    for mp in mask_paths:
        if mp.suffix.lower() == ".npy":
            masks.append(np.load(mp))
        else:
            img = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read mask: {mp}")
            masks.append(img)

    images: Optional[List[np.ndarray]] = None
    if args.images:
        images_dir = Path(args.images)
        image_paths = match_to_cameras(_list_images(images_dir), cameras, "image")
        images = [_load_grayscale(p) for p in image_paths]

    grid = _parse_grid(args.grid)
    grid.sobel_kernel = args.sobel
    grid.margin_size = args.margin

    icp_config = ICPConfig(
        error_percentile=args.error_percentile,
        dof=args.dof,
    )

    init_pose = np.zeros(6)

    out_dir = Path(args.o)

    run_sweep(
        mesh=mesh,
        cameras=cameras,
        masks=masks,
        images=images,
        grid=grid,
        icp_config=icp_config,
        init_pose=init_pose,
        out_dir=out_dir,
    )

    if args.plot:
        sweep_summary_path = out_dir / "sweep_summary.json"
        if sweep_summary_path.exists():
            with open(sweep_summary_path) as fh:
                sweep_summary = json.load(fh)
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            try:
                plot_sweep_heatmap(sweep_summary, plots_dir / "sweep_heatmap.png")
            except Exception as exc:
                print(f"Warning: heatmap plot failed: {exc}", file=sys.stderr)
            try:
                plot_pareto(sweep_summary, plots_dir / "pareto.png")
            except Exception as exc:
                print(f"Warning: pareto plot failed: {exc}", file=sys.stderr)

    return 0


def _cmd_refine(args: argparse.Namespace) -> int:
    """Handler for ``refine``."""
    from contour_reg.io.mesh import load_mesh
    from contour_reg.io.poses import load_poses
    from contour_reg.pipeline.refine import run_refine

    cameras = load_poses(args.poses, fmt=args.format)
    mesh = load_mesh(args.mesh)

    contours_dir = Path(args.contours)
    target_contours = _load_contour_dir(contours_dir, cameras)

    # Read best pose from sweep_summary.json
    sweep_dir = Path(args.sweep_dir)
    summary_path = sweep_dir / "sweep_summary.json"
    with open(summary_path) as fh:
        sweep_summary = json.load(fh)

    best_slug = args.sweep_config or sweep_summary["best"]
    if best_slug not in sweep_summary["configs"]:
        print(f"--sweep-config {best_slug!r} is not in {summary_path}.", file=sys.stderr)
        return 2
    best_info = sweep_summary["configs"][best_slug]
    init_pose = np.array(best_info["pose"], dtype=float)
    init_scale = float(best_info.get("scale", 1.0))
    if args.dof != 7 and abs(init_scale - 1.0) > 1e-3:
        print(
            f"ERROR: sweep config {best_slug} carries scale {init_scale:.6f} but --dof {args.dof} "
            "ignores scale (rigid). Use --dof 7 to honour it.",
            file=sys.stderr,
        )
        return 2

    final_pose, error_history, final_scale = run_refine(
        mesh=mesh,
        cameras=cameras,
        target_contours=target_contours,
        init_pose=init_pose,
        keep_pct=args.keep_pct,
        error_percentile=args.error_percentile,
        optimize_scale=args.dof == 7,
        initial_scale=init_scale,
    )

    result = {
        "pose": final_pose.tolist(),
        "scale": float(final_scale),
        "dof": args.dof,
        "error_history": error_history,
        "best_sweep_config": best_slug,
    }

    out_path = Path(args.o)
    _write_json(out_path, result)
    return 0


def _cmd_eval_canny(args: argparse.Namespace) -> int:
    """Handler for ``eval-canny``: rank swept Canny configs by contour F-beta."""
    from contour_reg.config import CannyConfig
    from contour_reg.io.mesh import load_mesh
    from contour_reg.io.poses import load_poses
    from contour_reg.pipeline._contours import build_target_contours
    from contour_reg.pipeline.eval_canny import config_view_counts, eval_configs

    cameras = load_poses(args.poses, fmt=args.format)
    mesh = load_mesh(args.mesh)
    from contour_reg.io.matching import match_to_cameras

    masks = [_load_grayscale(p) for p in match_to_cameras(_list_images(Path(args.masks)), cameras, "mask")]
    images = None
    if args.images:
        images = [_load_grayscale(p) for p in match_to_cameras(_list_images(Path(args.images)), cameras, "image")]

    with open(Path(args.sweep_dir) / "sweep_summary.json") as fh:
        summary = json.load(fh)

    per_config_views = {}
    for slug, info in summary["configs"].items():
        c = info["config"]
        cfg = CannyConfig(
            canny_low=c["canny_low"], canny_high=c["canny_high"],
            margin_size=c["margin_size"], sobel_kernel=c["sobel_kernel"],
            include_inner_contours=c["include_inner_contours"],
        )
        contours = build_target_contours(cfg, masks, images)
        pose = np.array(info["pose"], dtype=float)
        scale = float(info.get("scale", 1.0))
        per_config_views[slug] = config_view_counts(
            mesh, cameras, contours, pose, scale, trim_percentile=args.trim_percentile,
        )

    result = eval_configs(
        per_config_views, epsilons=tuple(args.epsilons),
        select_epsilon=args.select_epsilon, beta=args.beta,
    )
    _write_json(Path(args.o), result)
    if result["best"] is not None:
        print(f"best config: {result['best']} "
              f"(F{args.beta:g}@{args.select_epsilon:g} = {result['scores'][result['best']]:.4f})")
    return 0


def _cmd_eval_loo(args: argparse.Namespace) -> int:
    """Handler for ``eval-loo``."""
    from contour_reg.config import ICPConfig
    from contour_reg.io.mesh import load_mesh
    from contour_reg.io.poses import load_poses
    from contour_reg.pipeline.loo import run_loo

    cameras = load_poses(args.poses, fmt=args.format)
    mesh = load_mesh(args.mesh)

    contours_dir = Path(args.contours)
    target_contours = _load_contour_dir(contours_dir, cameras)

    init_pose = np.array(args.init_pose, dtype=float) if args.init_pose else np.zeros(6)

    # ``run_loo`` reads ``min_canny_points`` from the explicit kwarg, not the
    # ICPConfig: keep the config free of that field to avoid an ambiguous path.
    icp_config = ICPConfig()

    result = run_loo(
        mesh=mesh,
        cameras=cameras,
        target_contours=target_contours,
        init_pose=init_pose,
        icp_config=icp_config,
        min_canny_points=args.min_canny_points,
        max_correspondence_distance=args.max_correspondence_distance,
        optimize_scale=args.optimize_scale,
        initial_scale=args.initial_scale,
    )

    out_path = Path(args.o)
    _write_json(out_path, result)

    if args.plot:
        from contour_reg.viz.plots import plot_loo

        plot_path = out_path.parent / "loo_per_view.png"
        try:
            plot_loo(result, plot_path)
        except Exception as exc:
            print(f"Warning: LOO plot failed: {exc}", file=sys.stderr)

    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    """Handler for ``pipeline``."""
    from contour_reg.config import ICPConfig
    from contour_reg.pipeline.run import DEFAULT_GRID, SelectionError, run_pipeline

    grid = dataclasses.replace(DEFAULT_GRID, sobel_kernel=args.sobel, margin_size=args.margin)
    try:
        run_pipeline(
            poses_path=args.poses,
            mesh_path=args.mesh,
            masks_dir=args.masks,
            out_dir=args.o,
            images_dir=args.images if args.images else None,
            grid=grid,
            icp_config=ICPConfig(dof=args.dof),
            plot=args.plot,
            fmt=args.format,
            select_by=args.select_by,
        )
    except SelectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_segment(args: argparse.Namespace) -> int:
    """Handler for ``segment``: text-promptable SAM 3 mask generation.

    The heavy ``transformers``/``torch`` dependencies are imported lazily here
    (never at module top-level) so the rest of the CLI works without the
    optional ``[segmentation]`` extra installed.
    """
    from contour_reg.segmentation import INSTALL_HINT

    try:
        from contour_reg.segmentation.sam3_segmenter import segment_directory
    except ImportError:
        print(INSTALL_HINT, file=sys.stderr)
        return 1

    try:
        result = segment_directory(
            image_dir=args.images,
            prompt=args.prompt,
            output_dir=args.o,
            image_pattern=args.image_pattern,
            checkpoint=args.checkpoint,
            device=args.device,
            threshold=args.threshold,
            resize_width=args.resize_width,
            force=args.force,
        )
    except ImportError:
        print(INSTALL_HINT, file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0 if result["failed"] == 0 else 1


def _cmd_plot(args: argparse.Namespace) -> int:
    """Handler for ``plot``."""
    from contour_reg.viz.plots import plot_pareto, plot_sweep_heatmap

    results_dir = Path(args.results)
    out_dir = Path(args.o)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "sweep_summary.json"
    with open(summary_path) as fh:
        summary = json.load(fh)

    plot_sweep_heatmap(summary, out_dir / "sweep_heatmap.png")
    plot_pareto(summary, out_dir / "pareto.png")

    return 0


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def _add_dof_sobel_margin(p: argparse.ArgumentParser) -> None:
    """Options shared by ``sweep`` and ``pipeline``."""
    p.add_argument("--dof", type=int, choices=[6, 7], default=6,
                   help="6 = rigid; 7 = similarity (rigid + uniform scale) (default: 6)")
    p.add_argument("--sobel", type=int, default=3,
                   help="Sobel kernel size of the Canny detector (default: 3)")
    p.add_argument("--margin", type=int, default=20,
                   help="Half-width in px of the search band around the mask boundary (default: 20)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contour-reg",
        description="Contour-based 2D-3D registration toolkit.",
    )
    sub = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")

    # ------------------------------------------------------------------ extract-contours
    p_ec = sub.add_parser("extract-contours", help="Extract contours from masks and save as .npy")
    p_ec.add_argument("--masks", required=True, metavar="DIR", help="Directory of mask images")
    p_ec.add_argument("--images", default=None, metavar="DIR", help="Directory of source images (for Canny mode)")
    p_ec.add_argument("--mode", choices=["canny", "mask-boundary"], default="canny",
                      help="Contour extraction mode (default: canny)")
    p_ec.add_argument("--canny-low", type=int, default=50, dest="canny_low", help="Canny lower threshold (default: 50)")
    p_ec.add_argument("--canny-high", type=int, default=150, dest="canny_high", help="Canny upper threshold (default: 150)")
    p_ec.add_argument("--margin", type=int, default=20, help="Margin size in pixels (default: 20)")
    p_ec.add_argument("--sobel", type=int, default=3, help="Sobel kernel size (default: 3)")
    p_ec.add_argument("--component-mode", choices=["hybrid", "center_point", "smallest_area"],
                      default="hybrid", dest="component_mode", help="Connected component filter mode (default: hybrid)")
    p_ec.add_argument("--include-inner-contours", action="store_true", dest="include_inner_contours",
                      help="Include inner (hole) contours")
    p_ec.add_argument("--no-inner", action="store_false", dest="include_inner_contours",
                      help="Exclude inner contours (default)")
    p_ec.set_defaults(include_inner_contours=False)
    p_ec.add_argument("-o", required=True, metavar="DIR", help="Output directory for .npy files")

    # ------------------------------------------------------------------ register
    p_reg = sub.add_parser("register", help="Run contour-based ICP registration")
    p_reg.add_argument("--poses", required=True, metavar="PATH", help="Camera poses file")
    p_reg.add_argument("--format", default="auto", choices=["auto", "meshroom", "colmap", "idr"],
                       help="Camera format (default: auto)")
    p_reg.add_argument("--mesh", required=True, metavar="FILE", help="Mesh file (.ply)")
    p_reg.add_argument("--contours", required=True, metavar="DIR", help="Directory of contour .npy files")
    p_reg.add_argument("--init-pose", nargs=6, type=float, default=None, metavar="FLOAT",
                       dest="init_pose", help="Initial pose as 6 floats (rx ry rz tx ty tz)")
    p_reg.add_argument("--init-transform", default=None, metavar="FILE", dest="init_transform",
                       help="Initial GT->camera alignment as a 4x4 similarity in a .npy file "
                            "(v -> s*R@v + t, e.g. a MeshLab mlp_transform.npy). Decomposed into "
                            "an initial pose AND scale; the scale seeds the ICP (which --init-pose "
                            "cannot set). Mutually exclusive with --init-pose.")
    p_reg.add_argument("--dof", type=int, choices=[6, 7], default=6,
                       help="Degrees of freedom: 6 (rigid) or 7 (rigid+scale) (default: 6)")
    p_reg.add_argument("--max-iterations", type=int, default=50, dest="max_iterations",
                       help="Maximum ICP iterations (default: 50)")
    p_reg.add_argument("--error-percentile", type=float, default=75.0, dest="error_percentile",
                       help="Error threshold percentile for trimmed ICP; 100 keeps all points (default: 75)")
    p_reg.add_argument("--max-correspondence-distance", type=float, default=0.0,
                       dest="max_correspondence_distance",
                       help="Max correspondence distance in px, 0=disabled (default: 0)")
    p_reg.add_argument("--min-canny-points", type=int, default=0, dest="min_canny_points",
                       help="Minimum contour points per view (default: 0)")
    p_reg.add_argument("--loss", default="soft_l1",
                       choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
                       help="Least-squares robust loss (default: soft_l1)")
    p_reg.add_argument("--f-scale", type=float, default=6.0, dest="f_scale",
                       help="Robust-loss transition scale in px. scipy default 1.0 down-weights "
                            "all >1px residuals as outliers; ~6 keeps typical inliers in the "
                            "quadratic regime while still suppressing gross outliers (default: 6.0)")
    p_reg.add_argument("--include-inner-contours", action="store_true", dest="include_inner_contours",
                       help="Include inner contours in ICP")
    p_reg.add_argument("-o", required=True, metavar="FILE", help="Output result JSON file")

    # ------------------------------------------------------------------ sweep
    p_sw = sub.add_parser("sweep", help="Canny parameter sweep")
    p_sw.add_argument("--poses", required=True, metavar="PATH", help="Camera poses file")
    p_sw.add_argument("--format", default="auto", choices=["auto", "meshroom", "colmap", "idr"],
                      help="Camera format (default: auto)")
    p_sw.add_argument("--mesh", required=True, metavar="FILE", help="Mesh file (.ply)")
    p_sw.add_argument("--masks", required=True, metavar="DIR", help="Directory of mask images")
    p_sw.add_argument("--images", default=None, metavar="DIR", help="Directory of source images")
    p_sw.add_argument("--grid", default="lo=5,10,20,50 hi=20,30,40,60,80,120",
                      help="Grid string, e.g. \"lo=5,10,20 hi=20,30,40,60\" (default: full grid)")
    p_sw.add_argument("--error-percentile", type=float, default=75.0, dest="error_percentile",
                      help="ICP error percentile; 100 keeps all points (default: 75)")
    plot_group_sw = p_sw.add_mutually_exclusive_group()
    plot_group_sw.add_argument("--plot", action="store_true", default=True, dest="plot",
                               help="Write diagnostic plots (default: ON)")
    plot_group_sw.add_argument("--no-plot", action="store_false", dest="plot",
                               help="Skip plots")
    _add_dof_sobel_margin(p_sw)
    p_sw.add_argument("-o", required=True, metavar="DIR", help="Output directory")

    # ------------------------------------------------------------------ refine
    p_ref = sub.add_parser("refine", help="Coarse-then-fine refinement from a sweep result")
    p_ref.add_argument("--poses", required=True, metavar="PATH", help="Camera poses file")
    p_ref.add_argument("--format", default="auto", choices=["auto", "meshroom", "colmap", "idr"],
                       help="Camera format (default: auto)")
    p_ref.add_argument("--mesh", required=True, metavar="FILE", help="Mesh file (.ply)")
    p_ref.add_argument("--contours", required=True, metavar="DIR", help="Directory of contour .npy files")
    p_ref.add_argument("--sweep-dir", required=True, metavar="DIR", dest="sweep_dir",
                       help="Sweep output directory containing sweep_summary.json")
    p_ref.add_argument("--keep-pct", type=float, default=90.0, dest="keep_pct",
                       help="Percentile of correspondences to keep in fine phase (default: 90)")
    p_ref.add_argument("--error-percentile", type=float, default=75.0, dest="error_percentile",
                       help="Error percentile for coarse ICP phase (default: 75)")
    p_ref.add_argument("--sweep-config", default=None, metavar="SLUG", dest="sweep_config",
                       help="Start from this sweep config (e.g. the best of eval-canny) "
                            "instead of the lowest-error one")
    p_ref.add_argument("--dof", type=int, choices=[6, 7], default=6,
                       help="6 = rigid; 7 = similarity, starting from the sweep scale (default: 6)")
    p_ref.add_argument("-o", required=True, metavar="FILE", help="Output result JSON file")

    # ------------------------------------------------------------------ eval-canny
    p_evc = sub.add_parser("eval-canny", help="Rank swept Canny configs by contour F-beta")
    p_evc.add_argument("--poses", required=True, metavar="PATH", help="Camera poses file")
    p_evc.add_argument("--format", default="auto", choices=["auto", "meshroom", "colmap", "idr"],
                       help="Camera format (default: auto)")
    p_evc.add_argument("--mesh", required=True, metavar="FILE", help="Mesh file (.ply)")
    p_evc.add_argument("--masks", required=True, metavar="DIR", help="Directory of mask images")
    p_evc.add_argument("--images", default=None, metavar="DIR", help="Directory of source images")
    p_evc.add_argument("--sweep-dir", required=True, metavar="DIR", dest="sweep_dir",
                       help="Sweep output directory containing sweep_summary.json")
    p_evc.add_argument("--epsilons", nargs="+", type=float, default=[1.0, 2.0, 3.0, 5.0],
                       help="Pixel tolerances to evaluate (default: 1 2 3 5)")
    p_evc.add_argument("--select-epsilon", type=float, default=2.0, dest="select_epsilon",
                       help="Epsilon at which to select the best config (default: 2)")
    p_evc.add_argument("--beta", type=float, default=0.5,
                       help="F-beta beta (default: 0.5, precision-weighted; F1 is always reported)")
    p_evc.add_argument("--trim-percentile", type=float, default=75.0, dest="trim_percentile",
                       help="Coarse trim percentile used to form C_trim (default: 75)")
    p_evc.add_argument("-o", required=True, metavar="FILE", help="Output eval JSON file")

    # ------------------------------------------------------------------ eval-loo
    p_loo = sub.add_parser("eval-loo", help="Leave-one-view-out registration evaluation")
    p_loo.add_argument("--poses", required=True, metavar="PATH", help="Camera poses file")
    p_loo.add_argument("--format", default="auto", choices=["auto", "meshroom", "colmap", "idr"],
                       help="Camera format (default: auto)")
    p_loo.add_argument("--mesh", required=True, metavar="FILE", help="Mesh file (.ply)")
    p_loo.add_argument("--contours", required=True, metavar="DIR", help="Directory of contour .npy files")
    p_loo.add_argument("--init-pose", nargs=6, type=float, default=None, metavar="FLOAT",
                       dest="init_pose", help="Initial pose as 6 floats (rx ry rz tx ty tz)")
    p_loo.add_argument("--min-canny-points", type=int, default=0, dest="min_canny_points",
                       help="Min contour points per training view (default: 0)")
    p_loo.add_argument("--max-correspondence-distance", type=float, default=0.0,
                       dest="max_correspondence_distance",
                       help="Max correspondence distance in px (default: 0)")
    p_loo.add_argument("--optimize-scale", action="store_true", dest="optimize_scale",
                       help="Also optimize a uniform scale during LOO registration")
    p_loo.add_argument("--initial-scale", type=float, default=1.0, dest="initial_scale",
                       help="Initial scale when --optimize-scale is set (default: 1.0)")
    plot_group_loo = p_loo.add_mutually_exclusive_group()
    plot_group_loo.add_argument("--plot", action="store_true", default=False, dest="plot",
                                help="Write plots (default: OFF)")
    plot_group_loo.add_argument("--no-plot", action="store_false", dest="plot",
                                help="Skip plots (default)")
    p_loo.add_argument("-o", required=True, metavar="FILE", help="Output LOO JSON file")

    # ------------------------------------------------------------------ pipeline
    p_pip = sub.add_parser(
        "pipeline",
        help="Full end-to-end contour registration pipeline "
             "(masks are an INPUT; never runs segmentation; use "
             "'contour-reg segment' separately to generate masks)",
    )
    p_pip.add_argument("--poses", required=True, metavar="PATH", help="Camera poses file")
    p_pip.add_argument("--format", default="auto", choices=["auto", "meshroom", "colmap", "idr"],
                       help="Camera format (default: auto)")
    p_pip.add_argument("--mesh", required=True, metavar="FILE", help="Mesh file (.ply)")
    p_pip.add_argument("--masks", required=True, metavar="DIR", help="Directory of mask images")
    p_pip.add_argument("--images", default=None, metavar="DIR", help="Directory of source images")
    plot_group_pip = p_pip.add_mutually_exclusive_group()
    plot_group_pip.add_argument("--plot", action="store_true", default=True, dest="plot",
                                help="Write diagnostic plots (default: ON)")
    plot_group_pip.add_argument("--no-plot", action="store_false", dest="plot",
                                help="Skip plots")
    _add_dof_sobel_margin(p_pip)
    p_pip.add_argument("--select-by", choices=["fbeta", "error"], default="fbeta", dest="select_by",
                       help="Sweep config to refine: highest contour F0.5 at 2 px (default, as "
                            "eval-canny) or lowest reprojection error")
    p_pip.add_argument("-o", required=True, metavar="DIR", help="Output directory")

    # ------------------------------------------------------------------ plot
    p_plt = sub.add_parser("plot", help="Write sweep heatmap and pareto PNGs from a results dir")
    p_plt.add_argument("--results", required=True, metavar="DIR",
                       help="Results directory containing sweep_summary.json")
    p_plt.add_argument("-o", required=True, metavar="DIR", help="Output directory for PNGs")

    # ------------------------------------------------------------------ segment
    p_seg = sub.add_parser(
        "segment",
        help="Generate binary masks from images with a text prompt (SAM 3). "
             "Requires `pip install -e '.[segmentation]'` and gated HF weights "
             "(facebook/sam3).",
    )
    p_seg.add_argument("--images", required=True, metavar="DIR",
                       help="Directory of input images")
    p_seg.add_argument("--prompt", required=True, metavar="STR",
                       help="Text concept to segment, e.g. \"object\" or \"lego brick\"")
    p_seg.add_argument("--output", required=True, metavar="DIR", dest="o",
                       help="Output directory for binary masks (0/255 PNG)")
    p_seg.add_argument("--image-pattern", default="*.png", dest="image_pattern",
                       help="Glob pattern for input images (default: \"*.png\")")
    p_seg.add_argument("--checkpoint", default="facebook/sam3",
                       help="HF repo id or local path (default: facebook/sam3)")
    p_seg.add_argument("--device", choices=["cuda", "cpu"], default=None,
                       help="Compute device (default: auto-detect)")
    p_seg.add_argument("--threshold", type=float, default=0.5,
                       help="Instance detection score threshold (default: 0.5)")
    p_seg.add_argument("--resize-width", type=int, default=None, metavar="N",
                       dest="resize_width",
                       help="Downscale images to this width before inference")
    p_seg.add_argument("--force", action="store_true",
                       help="Overwrite existing masks")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_HANDLERS = {
    "extract-contours": _cmd_extract_contours,
    "register": _cmd_register,
    "sweep": _cmd_sweep,
    "refine": _cmd_refine,
    "eval-canny": _cmd_eval_canny,
    "eval-loo": _cmd_eval_loo,
    "pipeline": _cmd_pipeline,
    "plot": _cmd_plot,
    "segment": _cmd_segment,
}


def main(argv=None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).

    Returns:
        Exit code: 0 on success, non-zero on error.
    """
    # Configure root logger so progress messages from the library reach the
    # terminal.  ``logging.basicConfig`` is a no-op when handlers are already
    # configured (e.g. during pytest), so this is safe in all contexts.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        return 2

    from contour_reg.io.matching import MatchingError

    handler = _HANDLERS[args.subcommand]
    try:
        return handler(args)
    except MatchingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
