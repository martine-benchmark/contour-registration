import numpy as np

from contour_reg.io.loaders.colmap import load_colmap, qvec2rotmat


def _write_model(d):
    (d / "cameras.txt").write_text(
        "# id model w h params\n1 PINHOLE 640 480 1000 1000 320 240\n")
    (d / "images.txt").write_text(
        "# id qw qx qy qz tx ty tz cam name\n"
        "1 1 0 0 0 0 0 5 1 view0.png\n\n")
    return d


def test_qvec_identity():
    assert np.allclose(qvec2rotmat([1, 0, 0, 0]), np.eye(3))


def test_colmap_loads_one_camera(tmp_path):
    cams = load_colmap(_write_model(tmp_path))
    assert len(cams) == 1
    assert cams[0].width == 640 and np.isclose(cams[0].focal_length, 1000.0)
    # identity rotation, tvec [0,0,5] -> position -R^T t = [0,0,-5]
    assert np.allclose(cams[0].position, [0, 0, -5])
