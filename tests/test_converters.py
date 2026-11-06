"""Round-trip projection-equivalence tests for COLMAP and IDR writers."""
import numpy as np

from contour_reg.io.camera import Camera
from contour_reg.io.converters import write_colmap, write_idr
from contour_reg.io.loaders.colmap import load_colmap
from contour_reg.io.loaders.idr import load_idr


def _cam():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    return Camera(position=np.array([0.1, 0.2, -3.0]), rotation_world2cam=R,
                  focal_length=1000.0, principal_point=np.array([320.0, 240.0]),
                  width=640, height=480)


def _proj(cam):
    pts = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0, 0.05, 0]])
    return cam.project_points(pts)


def test_colmap_roundtrip_projection(tmp_path):
    c0 = _cam()
    write_colmap([c0], tmp_path)
    c1 = load_colmap(tmp_path)[0]
    assert np.allclose(_proj(c0), _proj(c1), atol=1e-4)


def test_idr_roundtrip_projection(tmp_path):
    c0 = _cam()
    write_idr([c0], tmp_path / "cameras.npz")
    c1 = load_idr(tmp_path / "cameras.npz")[0]
    assert np.allclose(_proj(c0), _proj(c1), atol=1e-3)
