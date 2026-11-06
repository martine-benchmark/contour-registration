"""ICP integration test on real lego fixture (Task 8.3).

Registers from a small perturbation and asserts reprojection error improves.
"""
from pathlib import Path

import cv2
import numpy as np

from contour_reg.contours.canny import mask_boundary_contour
from contour_reg.io.mesh import load_mesh
from contour_reg.io.poses import load_poses
from contour_reg.metrics import reprojection_error
from contour_reg.registration.icp import contour_based_icp

DATA = Path(__file__).parent / "data" / "lego"


def test_icp_improves_from_small_perturbation():
    """Register lego from a perturbed init; reprojection error must decrease."""
    cameras = load_poses(DATA / "colmap")
    mesh = load_mesh(DATA / "mesh.ply")

    # Build per-view target contours from sorted mask files
    mask_paths = sorted((DATA / "masks").iterdir())
    targets = [
        mask_boundary_contour(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        for p in mask_paths
    ]

    # Step 1: get a baseline converged pose starting from identity
    pose_baseline, _, _ = contour_based_icp(
        mesh,
        np.zeros(6),
        cameras,
        targets,
        max_iterations=10,
        verbose=False,
    )

    # Step 2: perturb the baseline pose by 5 % of the mesh x-extent in tx
    perturb_amount = 0.05 * mesh.bounding_box.extents[0]
    perturbed_init = pose_baseline.copy()
    perturbed_init[3] += perturb_amount

    reproj_before = reprojection_error(perturbed_init, mesh, cameras, targets)["mean"]

    # Step 3: register from the perturbed init
    pose_registered, _, _ = contour_based_icp(
        mesh,
        perturbed_init,
        cameras,
        targets,
        max_iterations=10,
        verbose=False,
    )

    reproj_after = reprojection_error(pose_registered, mesh, cameras, targets)["mean"]

    # The registered pose must yield lower reprojection error than the perturbed init
    assert np.isfinite(reproj_before), "reprojection error before registration is non-finite"
    assert np.isfinite(reproj_after), "reprojection error after registration is non-finite"
    assert reproj_after < reproj_before, (
        f"ICP did not improve reprojection: before={reproj_before:.3f} px, "
        f"after={reproj_after:.3f} px"
    )
