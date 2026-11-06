import numpy as np

from contour_reg.io.loaders.idr import load_idr


def test_decompose_roundtrip(tmp_path):
    K = np.array([[1000,0,320],[0,1000,240],[0,0,1]], float)
    R = np.eye(3)
    t = np.array([0,0,5.0])
    P = K @ np.hstack([R, t[:,None]])
    np.savez(tmp_path/"cameras.npz", world_mat_0=P, scale_mat_0=np.eye(4))
    cams = load_idr(tmp_path/"cameras.npz")
    assert len(cams) == 1
    assert np.isclose(cams[0].focal_length, 1000.0, atol=1e-3)
    assert np.allclose(cams[0].position, [0,0,-5], atol=1e-3)  # C = -R^T t
