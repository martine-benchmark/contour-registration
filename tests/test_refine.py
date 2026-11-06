import numpy as np
import trimesh

from contour_reg.io.camera import Camera
from contour_reg.pipeline.refine import run_refine
from contour_reg.registration.correspondences import extract_silhouette_edges_fast


def test_refine_runs_and_returns_history():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    cam = Camera(position=np.array([0,0,-3.0]), rotation_world2cam=np.eye(3),
                 focal_length=800.0, principal_point=np.array([320.0,240.0]),
                 width=640, height=480)
    tgt = extract_silhouette_edges_fast(mesh, cam)[0]
    pose, hist, scale = run_refine(mesh, [cam], [tgt], np.array([0,0,0,0.03,0,0]),
                            keep_pct=90.0, max_iterations=10)
    assert len(hist) >= 1 and hist[-1] <= hist[0] + 1e-6
    assert scale == 1.0  # rigid by default
