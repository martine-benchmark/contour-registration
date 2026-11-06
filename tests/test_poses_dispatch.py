import json

import pytest

from contour_reg.io.poses import load_poses


def test_auto_detects_meshroom_json(tmp_path):
    p = tmp_path/"sfm.json"
    p.write_text(json.dumps({"intrinsics":[{"width":64,"height":48,"focalLength":30,
        "sensorWidth":36,"principalPoint":[0,0]}],
        "poses":[{"poseId":"0","pose":{"transform":{"rotation":["1","0","0","0","1","0","0","0","1"],"center":["0","0","0"]}}}],
        "views":[{"viewId":"0","poseId":"0","path":"0.png"}]}))
    assert len(load_poses(p)) == 1

def test_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError):
        load_poses(tmp_path/"nope.xyz")
