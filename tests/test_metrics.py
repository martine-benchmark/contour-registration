import numpy as np

from contour_reg.metrics import add_metric, rotation_error_deg, translation_error


def test_add_zero_for_identical_pose():
    verts = np.random.default_rng(0).random((50,3))
    pose = np.array([0.1,0.2,0.3, 1.0,2.0,3.0])
    assert add_metric(pose, pose, verts) < 1e-9

def test_translation_error_value():
    a = np.array([0,0,0, 0,0,0.0])
    b = np.array([0,0,0, 0,0,2.0])
    assert np.isclose(translation_error(a,b), 2.0)

def test_rotation_error_90deg():
    a = np.zeros(6)
    b = np.array([0,0,np.pi/2, 0,0,0.0])
    assert np.isclose(rotation_error_deg(a,b), 90.0, atol=1e-6)
