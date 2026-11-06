import numpy as np
import trimesh

from contour_reg.config import ICPConfig
from contour_reg.io.camera import Camera
from contour_reg.pipeline.loo import run_loo
from contour_reg.registration.correspondences import extract_silhouette_edges_fast


def test_loo_reports_per_view():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    cams, tgts = [], []
    for ang in (0.0, 1.2, 2.4):
        R = trimesh.transformations.rotation_matrix(ang,[0,1,0])[:3,:3]
        c = Camera(position=R@np.array([0,0,-3.0]), rotation_world2cam=R.T,
                   focal_length=800.0, principal_point=np.array([320.,240.]),
                   width=640, height=480)
        cams.append(c)
        tgts.append(extract_silhouette_edges_fast(mesh, c)[0])
    out = run_loo(mesh, cams, tgts, np.zeros(6), ICPConfig(max_iterations=5))
    assert len(out['per_view']) == 3 and 'median' in out
    assert 'iqr' in out
    assert [e['view'] for e in out['per_view']] == [0, 1, 2]
    assert all(np.isfinite(e['held_out_error_px']) for e in out['per_view'])


def test_loo_scale_and_thresholds():
    """LOO with optimize_scale reports scale-aware held-out threshold stats."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    cams, tgts = [], []
    for ang in (0.0, 1.2, 2.4):
        R = trimesh.transformations.rotation_matrix(ang, [0, 1, 0])[:3, :3]
        c = Camera(position=R @ np.array([0, 0, -3.0]), rotation_world2cam=R.T,
                   focal_length=800.0, principal_point=np.array([320., 240.]),
                   width=640, height=480)
        cams.append(c)
        tgts.append(extract_silhouette_edges_fast(mesh, c)[0])
    out = run_loo(mesh, cams, tgts, np.zeros(6), ICPConfig(max_iterations=5),
                  optimize_scale=True, initial_scale=1.0)
    for e in out['per_view']:
        assert {'pct_lt_1px', 'pct_lt_2px', 'pct_lt_5px', 'median_px'} <= set(e)
        assert 0.0 <= e['pct_lt_5px'] <= 100.0
        assert e['pct_lt_5px'] >= e['pct_lt_1px']  # monotone in threshold
