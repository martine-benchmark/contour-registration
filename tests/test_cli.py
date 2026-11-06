"""Tests for contour_reg.cli (Task 7.1)."""
import numpy as np

import contour_reg.cli as cli
from contour_reg.cli import _parse_grid, main


def test_help_lists_subcommands(capsys):
    try:
        main(["--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    for sub in ["extract-contours", "register", "sweep", "refine", "eval-canny",
                "eval-loo", "pipeline", "plot"]:
        assert sub in out, f"Subcommand '{sub}' not found in --help output"


def test_eval_canny_parses_and_registered():
    """`eval-canny` parses its args and has a registered handler."""
    args = cli._build_parser().parse_args([
        "eval-canny", "--poses", "p.json", "--mesh", "m.ply", "--masks", "md",
        "--sweep-dir", "sd", "--select-epsilon", "2", "--beta", "0.5", "-o", "out.json",
    ])
    assert args.subcommand == "eval-canny"
    assert args.select_epsilon == 2.0 and args.beta == 0.5
    assert cli._HANDLERS["eval-canny"] is cli._cmd_eval_canny


def test_segment_subcommand_in_help(capsys):
    """`contour-reg --help` must list the `segment` subcommand."""
    try:
        main(["--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "segment" in out, "Subcommand 'segment' not found in --help output"


def test_segment_missing_deps_message(capsys, monkeypatch):
    """When the optional segmentation deps are missing, `segment` returns
    non-zero and prints the install hint, without loading any model."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("contour_reg.segmentation.sam3_segmenter"):
            raise ImportError("No module named 'transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    rc = main(["segment", "--images", "x", "--prompt", "y", "--output", "z"])

    assert rc != 0
    err = capsys.readouterr().err
    assert "pip install -e '.[segmentation]'" in err
    assert "segmentation dependencies" in err


def test_register_requires_args(capsys):
    rc = 0
    try:
        rc = main(["register"])
    except SystemExit as e:
        rc = e.code
    assert rc != 0  # missing required args


def test_parse_grid():
    grid = _parse_grid("lo=5,10 hi=20,30")
    assert grid.lows == [5, 10]
    assert grid.highs == [20, 30]


def test_register_wires_dof_inner_and_view_filter(tmp_path, monkeypatch):
    """register --dof 7 → optimize_scale=True; inner-contours + min-canny filter honored."""
    captured = {}

    def fake_icp(mesh, init_pose, cameras, target_contours, **kwargs):
        captured["n_views"] = len(cameras)
        captured["n_contours"] = len(target_contours)
        captured.update(kwargs)
        return np.zeros(6), [1.0, 0.5], 1.0

    # Three views; one contour too small to survive min-canny filtering.
    contours = [np.zeros((50, 2)), np.zeros((2, 2)), np.zeros((40, 2))]

    monkeypatch.setattr("contour_reg.io.poses.load_poses", lambda *a, **k: ["c0", "c1", "c2"])
    monkeypatch.setattr("contour_reg.io.mesh.load_mesh", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_load_contour_dir", lambda d, cams=None: contours)
    monkeypatch.setattr("contour_reg.registration.icp.contour_based_icp", fake_icp)

    out = tmp_path / "result.json"
    rc = main([
        "register",
        "--poses", "poses.json",
        "--mesh", "m.ply",
        "--contours", str(tmp_path),
        "--dof", "7",
        "--min-canny-points", "10",
        "--include-inner-contours",
        "-o", str(out),
    ])

    assert rc == 0
    assert captured["optimize_scale"] is True
    assert captured["include_inner_contours"] is True
    assert captured["n_views"] == 2  # one view dropped by min-canny filter
    assert captured["n_contours"] == 2
    assert out.exists()


def test_extract_contours_wires_component_mode(tmp_path, monkeypatch):
    """extract-contours --component-mode center_point → filter receives mode='center_point'."""
    import cv2

    captured = {}

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    cv2.imwrite(str(masks_dir / "0.png"), mask)

    def fake_filter(m, mode="hybrid"):
        captured["mode"] = mode
        return m

    monkeypatch.setattr("contour_reg.contours.canny.filter_connected_components", fake_filter)

    out_dir = tmp_path / "out"
    rc = main([
        "extract-contours",
        "--masks", str(masks_dir),
        "--mode", "mask-boundary",
        "--component-mode", "center_point",
        "-o", str(out_dir),
    ])

    assert rc == 0
    assert captured["mode"] == "center_point"
    assert (out_dir / "0.npy").exists()
