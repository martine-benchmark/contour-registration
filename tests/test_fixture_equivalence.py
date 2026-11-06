"""Cross-format projection equivalence test.

Verifies that the same 3D points project identically across all three
camera-pose formats (Meshroom, COLMAP, IDR). This guards against regressions
in the loaders, particularly sign/convention errors that would pass other tests.
"""
from pathlib import Path

import numpy as np
import pytest
import trimesh

from contour_reg.io.poses import load_poses

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("obj", ["lego", "surgery_scissors", "spine"])
def test_three_formats_project_equally(obj):
    """Load poses from 3 formats and verify projections match."""
    d = DATA / obj

    # Load poses from all 3 formats
    cams_m = load_poses(d / "meshroom" / "cameras.json")
    cams_c = load_poses(d / "colmap")
    cams_i = load_poses(d / "idr" / "cameras.npz")

    # Verify all formats loaded the same number of cameras
    assert len(cams_m) == len(cams_c) == len(cams_i), (
        f"{obj}: camera count mismatch: "
        f"meshroom={len(cams_m)}, colmap={len(cams_c)}, idr={len(cams_i)}"
    )

    # Load mesh and sample vertices as test points
    mesh = trimesh.load(d / "mesh.ply", force="mesh")
    rng = np.random.default_rng(0)
    P = mesh.vertices[rng.choice(len(mesh.vertices), size=30, replace=False)]

    # For each camera, verify projections match across all formats
    deviations = []
    for cam_idx, (cm, cc, ci) in enumerate(zip(cams_m, cams_c, cams_i)):
        pm, pc, pi = cm.project_points(P), cc.project_points(P), ci.project_points(P)

        # Compute max deviations for diagnostic output
        dev_mc = np.max(np.abs(pm - pc))
        dev_mi = np.max(np.abs(pm - pi))
        deviations.append((dev_mc, dev_mi))

        # Assertions with diagnostic messages
        assert np.allclose(pm, pc, atol=1.0), (
            f"{obj} camera {cam_idx}: meshroom vs colmap deviation {dev_mc:.4f} px"
        )
        assert np.allclose(pm, pi, atol=1.0), (
            f"{obj} camera {cam_idx}: meshroom vs idr deviation {dev_mi:.4f} px"
        )

    # Log max observed deviations
    if deviations:
        max_dev_mc = max(d[0] for d in deviations)
        max_dev_mi = max(d[1] for d in deviations)
        print(
            f"\n{obj}: max deviation meshroom vs colmap={max_dev_mc:.6f} px, "
            f"meshroom vs idr={max_dev_mi:.6f} px"
        )
