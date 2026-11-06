import numpy as np

from contour_reg.io.camera import Camera


def test_project_center_point_lands_at_principal_point():
    # Camera at origin looking down +z, point on optical axis projects to (cx, cy)
    cam = Camera(position=np.zeros(3), rotation_world2cam=np.eye(3),
                 focal_length=1000.0, principal_point=np.array([320.0, 240.0]),
                 width=640, height=480)
    p = cam.project_points(np.array([[0.0, 0.0, 5.0]]))
    assert np.allclose(p[0], [320.0, 240.0], atol=1e-6)

def test_project_offaxis_uses_focal():
    cam = Camera(position=np.zeros(3), rotation_world2cam=np.eye(3),
                 focal_length=1000.0, principal_point=np.array([0.0, 0.0]),
                 width=640, height=480)
    p = cam.project_points(np.array([[1.0, 0.0, 10.0]]))
    assert np.allclose(p[0], [100.0, 0.0], atol=1e-6)  # f * x/z = 1000 * 0.1
