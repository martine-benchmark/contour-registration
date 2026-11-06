"""LOO integration test on real surgery_scissors fixture (Task 8.3).

Exercises include_inner_contours=True, min_canny_points=100, and
max_correspondence_distance=100.0 end-to-end on real data.
"""
from pathlib import Path

import cv2
import numpy as np

from contour_reg.config import ICPConfig
from contour_reg.contours.canny import mask_boundary_contour
from contour_reg.io.mesh import load_mesh
from contour_reg.io.poses import load_poses
from contour_reg.pipeline.loo import run_loo

DATA_SS = Path(__file__).parent / "data" / "surgery_scissors"


def test_loo_surgery_scissors_inner_contours():
    """LOO on surgery_scissors with inner contours, min_canny_points, max_corr_dist."""
    cameras = load_poses(DATA_SS / "colmap")
    mesh = load_mesh(DATA_SS / "mesh.ply")

    # Build per-view target contours with inner contours enabled
    mask_paths = sorted((DATA_SS / "masks").iterdir())
    targets = [
        mask_boundary_contour(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), include_inner=True)
        for p in mask_paths
    ]

    out = run_loo(
        mesh,
        cameras,
        targets,
        np.zeros(6),
        ICPConfig(max_iterations=6, include_inner_contours=True),
        min_canny_points=100,
        max_correspondence_distance=100.0,
    )

    # Structure checks
    assert len(out["per_view"]) == len(cameras), (
        f"Expected {len(cameras)} per-view entries, got {len(out['per_view'])}"
    )
    assert "median" in out, "Output dict missing 'median'"
    assert "iqr" in out, "Output dict missing 'iqr'"

    # View indices must match positions 0..N-1
    assert [e["view"] for e in out["per_view"]] == list(range(len(cameras)))

    # Median must be finite (all 8 views have >100 contour points, so none are skipped)
    assert np.isfinite(out["median"]), f"LOO median is non-finite: {out['median']}"
