import json

import numpy as np

from contour_reg.io.loaders.meshroom import load_meshroom


def _write_sfm(tmp_path):
    data = {
        "intrinsics": [{"width": 640, "height": 480, "focalLength": 30.0,
                        "sensorWidth": 36.0, "principalPoint": [0.0, 0.0],
                        "distortionParams": [], "distortionType": "none"}],
        "poses": [{"poseId": "0", "pose": {"transform": {
            "rotation": ["1","0","0","0","1","0","0","0","1"],
            "center": ["0","0","-5"]}}}],
        "views": [{"viewId": "0", "poseId": "0", "path": "0.png"}],
    }
    p = tmp_path / "sfm.json"
    p.write_text(json.dumps(data))
    return p

def test_meshroom_focal_px_and_count(tmp_path):
    cams = load_meshroom(_write_sfm(tmp_path))
    assert len(cams) == 1
    assert np.isclose(cams[0].focal_length, 30.0 * 640 / 36.0)

def test_meshroom_world_flip_applied(tmp_path):
    cams = load_meshroom(_write_sfm(tmp_path), world_flip=True)
    # center [0,0,-5] flipped by diag(1,-1,-1) -> [0,0,5]
    assert np.allclose(cams[0].position, [0.0, 0.0, 5.0])

def test_meshroom_world_flip_rotation_and_disable(tmp_path):
    # 90° rotation about Z stored row-major as cam2world in the JSON
    angle = np.pi / 2
    R_cam2world = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle),  np.cos(angle), 0.0],
        [0.0,            0.0,           1.0],
    ])
    rotation_flat = [str(v) for v in R_cam2world.flatten()]
    center = [0.0, 0.0, -5.0]
    data = {
        "intrinsics": [{"width": 640, "height": 480, "focalLength": 30.0,
                        "sensorWidth": 36.0, "principalPoint": [0.0, 0.0],
                        "distortionParams": [], "distortionType": "none"}],
        "poses": [{"poseId": "0", "pose": {"transform": {
            "rotation": rotation_flat,
            "center": [str(v) for v in center]}}}],
        "views": [{"viewId": "0", "poseId": "0", "path": "0.png"}],
    }
    p = tmp_path / "sfm.json"
    p.write_text(json.dumps(data))

    world_correction = np.diag([1.0, -1.0, -1.0])

    # world_flip=True (default): R_w2c = R_c2w.T @ diag(1,-1,-1),
    # position = diag(1,-1,-1) @ center => [0, 0, 5]
    cams = load_meshroom(p, world_flip=True)
    expected_rot = R_cam2world.T @ world_correction
    expected_pos = world_correction @ np.array(center)
    assert np.allclose(cams[0].rotation_world2cam, expected_rot)
    assert np.allclose(cams[0].position, expected_pos)

    # world_flip=False: R_w2c = R_c2w.T, position unchanged = [0, 0, -5]
    cams = load_meshroom(p, world_flip=False)
    assert np.allclose(cams[0].rotation_world2cam, R_cam2world.T)
    assert np.allclose(cams[0].position, center)


def test_meshroom_multi_intrinsic(tmp_path):
    data = {
        "intrinsics": [
            {"intrinsicId": "100", "width": 640, "height": 480, "focalLength": 30.0,
             "sensorWidth": 36.0, "principalPoint": [0.0, 0.0],
             "distortionParams": [], "distortionType": "none"},
            {"intrinsicId": "200", "width": 800, "height": 600, "focalLength": 50.0,
             "sensorWidth": 36.0, "principalPoint": [0.0, 0.0],
             "distortionParams": [], "distortionType": "none"},
        ],
        "poses": [{"poseId": "0", "pose": {"transform": {
            "rotation": ["1","0","0","0","1","0","0","0","1"],
            "center": ["0","0","-5"]}}}],
        # view references the SECOND intrinsic (id "200")
        "views": [{"viewId": "0", "poseId": "0", "intrinsicId": "200", "path": "0.png"}],
    }
    p = tmp_path / "sfm.json"
    p.write_text(json.dumps(data))
    cams = load_meshroom(p)
    assert len(cams) == 1
    # focal must come from the SECOND intrinsic: 50 * 800 / 36
    assert np.isclose(cams[0].focal_length, 50.0 * 800 / 36.0)
    assert cams[0].width == 800
    assert cams[0].height == 600
