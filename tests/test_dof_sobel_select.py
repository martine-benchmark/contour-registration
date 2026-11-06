"""--dof / --sobel / --margin in sweep, refine and pipeline; pipeline selection by contour F-beta."""
import json

import numpy as np
import pytest

from contour_reg.cli import main
from contour_reg.config import ICPConfig, SweepGrid
from contour_reg.contour_metrics import view_counts

SLUG_A = "cl5_ch20_m20_sk3"   # lowest reprojection error, poor contour overlap
SLUG_B = "cl5_ch30_m20_sk3"   # higher reprojection error, best contour overlap
POSE_A = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
POSE_B = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0]


def _cfg(high):
    return {"canny_low": 5, "canny_high": high, "margin_size": 20, "sobel_kernel": 3,
            "include_inner_contours": False}


def _summary():
    return {
        "best": SLUG_A,
        "configs": {
            SLUG_A: {"pose": POSE_A, "scale": 1.0, "error": 1.0, "config": _cfg(20)},
            SLUG_B: {"pose": POSE_B, "scale": 1.3, "error": 2.0, "config": _cfg(30)},
        },
    }


# ---------------------------------------------------------------------------
# Library level
# ---------------------------------------------------------------------------

def test_grid_carries_sobel_and_margin():
    cfgs = list(SweepGrid(lows=[5], highs=[20], sobel_kernel=15, margin_size=10).configs())
    assert len(cfgs) == 1
    assert cfgs[0].sobel_kernel == 15 and cfgs[0].margin_size == 10
    assert cfgs[0].slug == "cl5_ch20_m10_sk15"


def test_sweep_dof7_optimizes_scale(tmp_path, monkeypatch):
    from contour_reg.pipeline import sweep

    captured = {}

    def fake_icp(mesh, init_pose, cameras, target_contours, **kwargs):
        captured.update(kwargs)
        return np.zeros(6), [1.0], 1.05

    monkeypatch.setattr(sweep, "contour_based_icp", fake_icp)
    monkeypatch.setattr(sweep, "reprojection_error", lambda *a, **k: {"mean": 1.0})
    res = sweep.run_sweep(None, [object()], masks=[np.zeros((10, 2))], images=None,
                          grid=SweepGrid(lows=[5], highs=[20]), icp_config=ICPConfig(dof=7),
                          init_pose=np.zeros(6), out_dir=tmp_path)
    assert captured["optimize_scale"] is True
    assert res[SLUG_A]["scale"] == pytest.approx(1.05)


def test_refine_dof7_carries_scale_between_phases(monkeypatch):
    from contour_reg.pipeline import refine

    calls = []

    def fake_icp(mesh, init_pose, cameras, target_contours, **kwargs):
        calls.append(kwargs)
        return np.asarray(init_pose) + 1, [2.0], kwargs["initial_scale"] * 1.1

    monkeypatch.setattr(refine, "contour_based_icp", fake_icp)
    pose, hist, scale = refine.run_refine(None, [], [], np.zeros(6), optimize_scale=True, initial_scale=2.0)
    assert [c["optimize_scale"] for c in calls] == [True, True]
    assert calls[0]["initial_scale"] == pytest.approx(2.0)
    assert calls[1]["initial_scale"] == pytest.approx(2.2)
    assert scale == pytest.approx(2.42)
    assert hist == [2.0, 2.0]


# ---------------------------------------------------------------------------
# run_pipeline selection (sweep, contours and ICP stubbed out)
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_pipeline(tmp_path, monkeypatch):
    from contour_reg.pipeline import run

    refine_calls = []
    far = {"value": False}  # when True, no config's contour reaches the silhouette

    def fake_sweep(**kwargs):
        out = kwargs["out_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "sweep_summary.json").write_text(json.dumps(_summary()))
        return _summary()["configs"]

    def fake_view_counts(mesh, cameras, contours, pose, scale, trim_percentile=75.0, **kw):
        sil = np.stack([np.arange(50.0), np.zeros(50)], axis=1)
        if far["value"]:
            contour = sil + 10.0
        elif pose[5] == 2.0:
            contour = sil                       # config B: contour on the silhouette
        else:
            contour = np.concatenate([sil[:10], sil[:40] + 10.0])  # config A: mostly off
        return [view_counts(sil, contour, contour) for _ in cameras]

    def fake_refine(mesh, cameras, target_contours, init_pose, **kwargs):
        refine_calls.append({"init_pose": list(init_pose), **kwargs})
        return np.asarray(init_pose, float), [1.0], kwargs.get("initial_scale", 1.0)

    monkeypatch.setattr(run, "load_poses", lambda *a, **k: [object(), object()])
    monkeypatch.setattr(run, "load_mesh", lambda *a, **k: type("M", (), {"vertices": np.zeros((1, 3))})())
    monkeypatch.setattr(run, "_list_images", lambda d: ["0.png", "1.png"])
    monkeypatch.setattr(run, "_load_grayscale", lambda p: np.zeros((8, 8), np.uint8))
    monkeypatch.setattr(run, "run_sweep", fake_sweep)
    monkeypatch.setattr(run, "build_target_contours", lambda cfg, masks, images: [np.zeros((3, 2))] * 2)
    monkeypatch.setattr(run, "config_view_counts", fake_view_counts)
    monkeypatch.setattr(run, "run_refine", fake_refine)

    def call(**kwargs):
        return run.run_pipeline(tmp_path / "poses", tmp_path / "m.ply", tmp_path / "masks", tmp_path / "out",
                                plot=False, **kwargs)

    return call, refine_calls, far, tmp_path / "out"


def test_pipeline_selects_by_fbeta_by_default(stub_pipeline):
    call, refine_calls, _, out = stub_pipeline
    res = call()
    assert res["best_config"] == SLUG_B
    assert refine_calls[0]["init_pose"] == POSE_B
    on_disk = json.loads((out / "result.json").read_text())
    assert on_disk["best_config"] == SLUG_B
    assert on_disk["select_by"] == "fbeta"
    assert on_disk["selection_score"] > 0.9
    assert (out / "sweep" / "eval_canny.json").exists()


def test_pipeline_select_by_error_keeps_lowest_reprojection(stub_pipeline):
    call, refine_calls, _, _ = stub_pipeline
    res = call(select_by="error")
    assert res["best_config"] == SLUG_A
    assert refine_calls[0]["init_pose"] == POSE_A


def test_pipeline_fails_when_no_config_reaches_silhouette(stub_pipeline):
    from contour_reg.pipeline.run import SelectionError

    call, refine_calls, far, _ = stub_pipeline
    far["value"] = True
    with pytest.raises(SelectionError, match="within 2 px"):
        call()
    assert refine_calls == []


def test_pipeline_dof7_refines_from_sweep_scale(stub_pipeline):
    call, refine_calls, _, out = stub_pipeline
    res = call(icp_config=ICPConfig(dof=7))
    assert refine_calls[0]["optimize_scale"] is True
    assert refine_calls[0]["initial_scale"] == pytest.approx(1.3)
    assert res["scale"] == pytest.approx(1.3)
    on_disk = json.loads((out / "result.json").read_text())
    assert on_disk["dof"] == 7 and on_disk["scale"] == pytest.approx(1.3)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def test_cli_pipeline_forwards_dof_sobel_margin_select(monkeypatch):
    captured = {}
    monkeypatch.setattr("contour_reg.pipeline.run.run_pipeline", lambda **kw: captured.update(kw) or {})
    rc = main(["pipeline", "--poses", "p", "--mesh", "m.ply", "--masks", "d", "-o", "o",
               "--dof", "7", "--sobel", "15", "--margin", "10", "--select-by", "error"])
    assert rc == 0
    assert captured["icp_config"].dof == 7
    assert captured["grid"].sobel_kernel == 15 and captured["grid"].margin_size == 10
    assert captured["select_by"] == "error"


def test_cli_pipeline_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr("contour_reg.pipeline.run.run_pipeline", lambda **kw: captured.update(kw) or {})
    assert main(["pipeline", "--poses", "p", "--mesh", "m.ply", "--masks", "d", "-o", "o"]) == 0
    assert captured["icp_config"].dof == 6
    assert captured["grid"].sobel_kernel == 3 and captured["grid"].margin_size == 20
    assert captured["select_by"] == "fbeta"


def test_cli_pipeline_selection_error_exits_1(monkeypatch, capsys):
    from contour_reg.pipeline.run import SelectionError

    def boom(**kw):
        raise SelectionError("No sweep config reaches the silhouette within 2 px")

    monkeypatch.setattr("contour_reg.pipeline.run.run_pipeline", boom)
    rc = main(["pipeline", "--poses", "p", "--mesh", "m.ply", "--masks", "d", "-o", "o"])
    assert rc == 1
    assert "within 2 px" in capsys.readouterr().err


def test_cli_sweep_forwards_dof_sobel_margin(tmp_path, monkeypatch):
    import cv2

    masks = tmp_path / "masks"
    masks.mkdir()
    cv2.imwrite(str(masks / "0.png"), np.zeros((8, 8), np.uint8))
    captured = {}
    monkeypatch.setattr("contour_reg.io.poses.load_poses", lambda *a, **k: [object()])
    monkeypatch.setattr("contour_reg.io.mesh.load_mesh", lambda *a, **k: object())
    monkeypatch.setattr("contour_reg.pipeline.sweep.run_sweep", lambda **kw: captured.update(kw) or {})
    rc = main(["sweep", "--poses", "p", "--mesh", "m.ply", "--masks", str(masks), "-o", str(tmp_path / "o"),
               "--no-plot", "--dof", "7", "--sobel", "15", "--margin", "10"])
    assert rc == 0
    assert captured["icp_config"].dof == 7
    assert captured["grid"].sobel_kernel == 15 and captured["grid"].margin_size == 10


@pytest.fixture
def refine_setup(tmp_path, monkeypatch):
    import contour_reg.cli as cli

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(_summary()))
    captured = {}

    def fake_refine(mesh, cameras, target_contours, init_pose, **kwargs):
        captured.update({"init_pose": list(init_pose), **kwargs})
        return np.asarray(init_pose, float), [1.0], kwargs.get("initial_scale", 1.0)

    monkeypatch.setattr("contour_reg.io.poses.load_poses", lambda *a, **k: [object()])
    monkeypatch.setattr("contour_reg.io.mesh.load_mesh", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_load_contour_dir", lambda d, cams=None: [np.zeros((3, 2))])
    monkeypatch.setattr("contour_reg.pipeline.refine.run_refine", fake_refine)
    base = ["refine", "--poses", "p", "--mesh", "m.ply", "--contours", str(tmp_path),
            "--sweep-dir", str(sweep_dir), "-o", str(tmp_path / "r.json")]
    return base, captured, tmp_path / "r.json"


def test_cli_refine_sweep_config_and_dof7(refine_setup):
    base, captured, out = refine_setup
    assert main(base + ["--sweep-config", SLUG_B, "--dof", "7"]) == 0
    assert captured["init_pose"] == POSE_B
    assert captured["optimize_scale"] is True and captured["initial_scale"] == pytest.approx(1.3)
    res = json.loads(out.read_text())
    assert res["scale"] == pytest.approx(1.3) and res["best_sweep_config"] == SLUG_B


def test_cli_refine_defaults_to_sweep_best(refine_setup):
    base, captured, _ = refine_setup
    assert main(base) == 0
    assert captured["init_pose"] == POSE_A
    assert captured["optimize_scale"] is False


def test_cli_refine_rigid_rejects_scaled_sweep_pose(refine_setup, capsys):
    base, _, _ = refine_setup
    assert main(base + ["--sweep-config", SLUG_B]) == 2
    assert "--dof 7" in capsys.readouterr().err
