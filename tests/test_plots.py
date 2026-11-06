# tests/test_plots.py
from contour_reg.viz.plots import plot_convergence, plot_loo, plot_pareto, plot_sweep_heatmap

# Tiny synthetic summary in the canonical sweep_summary.json shape:
# { "best": "<slug>", "configs": { "<slug>": {"error": <float>, "pose": [...], "config": {...}}, ... } }
_SUMMARY = {
    "best": "cl5_ch20_m10_sk3",
    "configs": {
        "cl5_ch20_m10_sk3": {
            "error": 0.05,
            "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "config": {
                "canny_low": 5, "canny_high": 20,
                "margin_size": 10, "sobel_kernel": 3,
                "include_inner_contours": False,
            },
        },
        "cl5_ch30_m10_sk3": {
            "error": 0.12,
            "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "config": {
                "canny_low": 5, "canny_high": 30,
                "margin_size": 10, "sobel_kernel": 3,
                "include_inner_contours": False,
            },
        },
        "cl10_ch20_m10_sk3": {
            "error": 0.08,
            "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "config": {
                "canny_low": 10, "canny_high": 20,
                "margin_size": 10, "sobel_kernel": 3,
                "include_inner_contours": False,
            },
        },
    },
}


def test_convergence_writes_png(tmp_path):
    p = plot_convergence([5.0, 3.0, 2.0, 1.5], tmp_path / "conv.png")
    assert p.exists() and p.suffix == ".png"


def test_sweep_heatmap_writes_png(tmp_path):
    p = plot_sweep_heatmap(_SUMMARY, tmp_path / "heatmap.png")
    assert p.exists() and p.suffix == ".png"


def test_pareto_writes_png(tmp_path):
    p = plot_pareto(_SUMMARY, tmp_path / "pareto.png")
    assert p.exists() and p.suffix == ".png"


def test_loo_writes_png(tmp_path):
    loo_result = {
        "per_view": [
            {"view": 0, "held_out_error_px": 1.2},
            {"view": 1, "held_out_error_px": float("inf")},
            {"view": 2, "held_out_error_px": 0.8},
        ],
        "median": 1.0,
        "iqr": 0.4,
    }
    p = plot_loo(loo_result, tmp_path / "loo.png")
    assert p.exists() and p.suffix == ".png"
