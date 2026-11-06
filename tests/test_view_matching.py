"""Masks, images and contours are matched to the views by file name (stem), not by sorted order."""
import logging
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from contour_reg.cli import main
from contour_reg.io.matching import MatchingError, match_to_cameras

DATA = Path(__file__).parent / "data"


def _cams(*names):
    return [SimpleNamespace(name=n) for n in names]


# ---------------------------------------------------------------------------
# match_to_cameras
# ---------------------------------------------------------------------------

def test_files_follow_the_camera_order_not_the_sorted_order():
    paths = [Path("m/a.png"), Path("m/b.png"), Path("m/c.png")]
    assert match_to_cameras(paths, _cams("b", "c", "a")) == [Path("m/b.png"), Path("m/c.png"), Path("m/a.png")]


def test_match_ignores_extensions():
    assert match_to_cameras([Path("m/IMG_1.png")], _cams("IMG_1")) == [Path("m/IMG_1.png")]


def test_missing_file_is_an_error():
    with pytest.raises(MatchingError, match="No mask file for 1 of 2 view"):
        match_to_cameras([Path("m/a.png")], _cams("a", "b"), "mask")


def test_two_files_for_one_view_is_ambiguous():
    with pytest.raises(MatchingError, match="[Aa]mbiguous"):
        match_to_cameras([Path("m/a.jpg"), Path("m/a.png")], _cams("a"))


def test_two_views_with_the_same_name_is_ambiguous():
    with pytest.raises(MatchingError, match="[Aa]mbiguous"):
        match_to_cameras([Path("m/a.png")], _cams("a", "a"))


def test_no_file_matches_any_view_names_the_rule():
    with pytest.raises(MatchingError, match="matched to the views by file name"):
        match_to_cameras([Path("m/0.png"), Path("m/1.png")], _cams("IMG_7", "IMG_8"))


def test_extra_files_are_ignored_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        out = match_to_cameras([Path("m/a.png"), Path("m/z.png")], _cams("a"))
    assert out == [Path("m/a.png")]
    assert "z.png" in caplog.text


def test_unnamed_cameras_use_the_sorted_order():
    paths = [Path("m/0.png"), Path("m/1.png")]
    assert match_to_cameras(paths, [object(), object()]) == paths
    with pytest.raises(MatchingError, match="2 camera"):
        match_to_cameras(paths[:1], [object(), object()])


# ---------------------------------------------------------------------------
# Loaders expose the view name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt, sub", [("meshroom", "meshroom/cameras.json"), ("colmap", "colmap")])
def test_loaders_set_view_names(fmt, sub):
    from contour_reg.io.poses import load_poses

    cams = load_poses(DATA / "lego" / sub, fmt=fmt)
    assert sorted(c.name for c in cams) == [str(i) for i in range(8)]


def test_meshroom_names_follow_view_ids(tmp_path):
    """viewId order (the loader's camera order) differs from the file-name order."""
    import json

    from contour_reg.io.poses import load_poses

    data = json.loads((DATA / "lego" / "meshroom" / "cameras.json").read_text())
    view_ids = [int(v["viewId"]) for v in data["views"]]
    names = [Path(v["path"]).stem for v in data["views"]]
    expected = [n for _, n in sorted(zip(view_ids, names))]
    assert [c.name for c in load_poses(DATA / "lego" / "meshroom" / "cameras.json")] == expected


def test_meshroom_windows_paths(tmp_path):
    import json

    from contour_reg.io.poses import load_poses

    data = json.loads((DATA / "lego" / "meshroom" / "cameras.json").read_text())
    for v in data["views"]:
        v["path"] = "C:\\capture\\" + Path(v["path"]).stem + ".JPG"
    p = tmp_path / "sfm.json"
    p.write_text(json.dumps(data))
    assert sorted(c.name for c in load_poses(p)) == [str(i) for i in range(8)]


# ---------------------------------------------------------------------------
# Commands use the name matching
# ---------------------------------------------------------------------------

def _write_masks(d, names):
    d.mkdir()
    for k, n in enumerate(names):
        m = np.zeros((8, 8), np.uint8)
        m[0, 0] = 10 * (k + 1)  # identifies the file
        cv2.imwrite(str(d / f"{n}.png"), m)


def test_pipeline_matches_masks_by_view_name(tmp_path, monkeypatch):
    from contour_reg.pipeline import run

    _write_masks(tmp_path / "masks", ["a", "b"])  # sorted: a, b
    captured = {}

    def stop(**kw):
        captured.update(kw)
        raise RuntimeError("stop after the sweep inputs")

    monkeypatch.setattr(run, "load_poses", lambda *a, **k: _cams("b", "a"))
    monkeypatch.setattr(run, "load_mesh", lambda *a, **k: SimpleNamespace(vertices=np.zeros((1, 3))))
    monkeypatch.setattr(run, "run_sweep", stop)
    with pytest.raises(RuntimeError, match="stop"):
        run.run_pipeline(tmp_path / "p", tmp_path / "m.ply", tmp_path / "masks", tmp_path / "out", plot=False)
    assert [int(m[0, 0]) for m in captured["masks"]] == [20, 10]  # b first, then a


def test_cli_sweep_matches_masks_by_view_name(tmp_path, monkeypatch):
    _write_masks(tmp_path / "masks", ["a", "b"])
    captured = {}
    monkeypatch.setattr("contour_reg.io.poses.load_poses", lambda *a, **k: _cams("b", "a"))
    monkeypatch.setattr("contour_reg.io.mesh.load_mesh", lambda *a, **k: object())
    monkeypatch.setattr("contour_reg.pipeline.sweep.run_sweep", lambda **kw: captured.update(kw) or {})
    assert main(["sweep", "--poses", "p", "--mesh", "m.ply", "--masks", str(tmp_path / "masks"),
                 "-o", str(tmp_path / "o"), "--no-plot"]) == 0
    assert [int(m[0, 0]) for m in captured["masks"]] == [20, 10]


def test_cli_register_matches_contours_by_view_name(tmp_path, monkeypatch):
    cdir = tmp_path / "contours"
    cdir.mkdir()
    np.save(cdir / "a.npy", np.full((5, 2), 1.0))
    np.save(cdir / "b.npy", np.full((7, 2), 2.0))
    captured = {}

    def fake_icp(mesh, init_pose, cameras, target_contours, **kwargs):
        captured["lens"] = [len(c) for c in target_contours]
        return np.zeros(6), [1.0], 1.0

    monkeypatch.setattr("contour_reg.io.poses.load_poses", lambda *a, **k: _cams("b", "a"))
    monkeypatch.setattr("contour_reg.io.mesh.load_mesh", lambda *a, **k: object())
    monkeypatch.setattr("contour_reg.registration.icp.contour_based_icp", fake_icp)
    assert main(["register", "--poses", "p", "--mesh", "m.ply", "--contours", str(cdir),
                 "-o", str(tmp_path / "r.json")]) == 0
    assert captured["lens"] == [7, 5]


def test_cli_missing_mask_exits_1_with_message(tmp_path, monkeypatch, capsys):
    _write_masks(tmp_path / "masks", ["a"])
    monkeypatch.setattr("contour_reg.pipeline.run.load_poses", lambda *a, **k: _cams("a", "b"))
    monkeypatch.setattr("contour_reg.pipeline.run.load_mesh",
                        lambda *a, **k: SimpleNamespace(vertices=np.zeros((1, 3))))
    rc = main(["pipeline", "--poses", "p", "--mesh", "m.ply", "--masks", str(tmp_path / "masks"),
               "-o", str(tmp_path / "o")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "No mask file for 1 of 2 view" in err and "b" in err


def test_cli_extract_contours_matches_images_to_masks_by_name(tmp_path, capsys):
    _write_masks(tmp_path / "masks", ["a", "b"])
    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a.png"), np.zeros((8, 8), np.uint8))
    rc = main(["extract-contours", "--masks", str(tmp_path / "masks"), "--images", str(tmp_path / "images"),
               "-o", str(tmp_path / "c")])
    assert rc == 1
    assert "No image file for 1 of 2" in capsys.readouterr().err
