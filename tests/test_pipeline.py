"""End-to-end pipeline tests: synthetic (Task 5.5) + real fixture (Task 8.3)."""
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from contour_reg.config import ICPConfig, SweepGrid
from contour_reg.io.camera import Camera
from contour_reg.io.converters import write_colmap
from contour_reg.pipeline.run import run_pipeline

DATA = Path(__file__).parent / "data" / "lego"


# ---------------------------------------------------------------------------
# Helper: render a dilated silhouette mask for a camera
# ---------------------------------------------------------------------------

def _render_mask(mesh: trimesh.Trimesh, cam: Camera) -> np.ndarray:
    """Paint silhouette edges onto a blank image and dilate to form a binary mask."""
    from contour_reg.registration.correspondences import extract_silhouette_edges_fast

    pts, _ = extract_silhouette_edges_fast(mesh, cam)
    img = np.zeros((cam.height, cam.width), np.uint8)
    if len(pts) > 0:
        pts_i = np.round(pts).astype(np.int32)
        pts_i[:, 0] = np.clip(pts_i[:, 0], 0, cam.width - 1)
        pts_i[:, 1] = np.clip(pts_i[:, 1], 0, cam.height - 1)
        img[pts_i[:, 1], pts_i[:, 0]] = 255
    return cv2.dilate(img, np.ones((5, 5), np.uint8))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_pipeline_synthetic(tmp_path):
    """Full synthetic end-to-end: load → sweep → refine → result.json."""
    # ------------------------------------------------------------------
    # Build scene: icosphere + 3 cameras around the Y axis
    # ------------------------------------------------------------------
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    (tmp_path / "mesh.ply").write_bytes(mesh.export(file_type="ply"))

    cams = []
    for ang in (0.0, 1.2, 2.4):
        R = trimesh.transformations.rotation_matrix(ang, [0, 1, 0])[:3, :3]
        cams.append(
            Camera(
                position=R @ np.array([0, 0, -3.0]),
                rotation_world2cam=R.T,
                focal_length=400.0,
                principal_point=np.array([128.0, 128.0]),
                width=256,
                height=256,
            )
        )

    # ------------------------------------------------------------------
    # Write COLMAP poses
    # ------------------------------------------------------------------
    posedir = tmp_path / "poses"
    posedir.mkdir()
    write_colmap(cams, posedir, image_names=[f"{i}.png" for i in range(len(cams))])

    # ------------------------------------------------------------------
    # Render and save masks
    # ------------------------------------------------------------------
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    for i, cam in enumerate(cams):
        mask = _render_mask(mesh, cam)
        cv2.imwrite(str(masks_dir / f"{i}.png"), mask)

    # ------------------------------------------------------------------
    # Run pipeline (tiny grid + few iterations for speed)
    # ------------------------------------------------------------------
    out = tmp_path / "out"
    res = run_pipeline(
        posedir,
        tmp_path / "mesh.ply",
        masks_dir,
        out,
        grid=SweepGrid(lows=[5], highs=[20, 30]),
        icp_config=ICPConfig(max_iterations=5),
        plot=True,
    )

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    # result.json must exist and be valid JSON
    result_json = out / "result.json"
    assert result_json.exists(), "result.json was not written"

    with open(result_json) as fh:
        on_disk = json.load(fh)

    # returned dict and on-disk dict must agree on the pose
    assert "final_pose" in res, "run_pipeline did not return 'final_pose'"
    assert len(res["final_pose"]) == 6, "final_pose must have 6 elements"
    assert np.all(np.isfinite(res["final_pose"])), "final_pose contains non-finite values"

    assert on_disk["final_pose"] == res["final_pose"]
    assert "best_config" in res
    assert "error_history" in res
    assert "result_path" in res

    # sweep sub-directory must contain the summary
    assert (out / "sweep" / "sweep_summary.json").exists()

    # plots directory must exist and contain at least one rendered plot
    # (a silent viz failure swallowed by the plot guard would fail here)
    assert (out / "plots").is_dir()
    assert (out / "plots" / "sweep_heatmap.png").exists()

    # result_path must be JSON-serialisable (a str, not a Path)
    assert isinstance(res["result_path"], str)
    json.dumps(res)


# ---------------------------------------------------------------------------
# Real-fixture test (Task 8.3)
# ---------------------------------------------------------------------------

def test_pipeline_lego_fixture(tmp_path):
    """Full pipeline on the committed lego fixture (mask-boundary mode)."""
    res = run_pipeline(
        poses_path=DATA / "colmap",
        mesh_path=DATA / "mesh.ply",
        masks_dir=DATA / "masks",
        out_dir=tmp_path,
        grid=SweepGrid(lows=[5], highs=[20, 30]),
        icp_config=ICPConfig(max_iterations=8),
        plot=True,
    )

    # result.json must exist on disk
    assert (tmp_path / "result.json").exists(), "result.json was not written"

    # Returned pose must be length-6 and all finite
    assert "final_pose" in res
    assert len(res["final_pose"]) == 6
    assert np.all(np.isfinite(res["final_pose"])), "final_pose contains non-finite values"

    # Sweep heatmap must be written
    assert (tmp_path / "plots" / "sweep_heatmap.png").exists(), "sweep_heatmap.png missing"
