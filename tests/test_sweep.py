import numpy as np
import trimesh

from contour_reg.config import ICPConfig, SweepGrid
from contour_reg.io.camera import Camera
from contour_reg.pipeline.sweep import run_sweep


def test_sweep_writes_summary(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    cam = Camera(position=np.array([0, 0, -3.0]), rotation_world2cam=np.eye(3),
                 focal_length=800.0, principal_point=np.array([320.0, 240.0]),
                 width=640, height=480)
    # 1 view, a precomputed contour as "mask" passthrough is out of scope here:
    # use a coarse grid with 2 valid configs
    grid = SweepGrid(lows=[5], highs=[20, 30])
    # masks/images None -> sweep falls back to mask-boundary using provided contour arrays
    from contour_reg.registration.correspondences import extract_silhouette_edges_fast
    contour = extract_silhouette_edges_fast(mesh, cam)[0]
    res = run_sweep(mesh, [cam], masks=[contour], images=None, grid=grid,
                    icp_config=ICPConfig(max_iterations=5),
                    init_pose=np.zeros(6), out_dir=tmp_path, n_workers=1)
    assert (tmp_path / "sweep_summary.json").exists()
    assert len(res) == 2
