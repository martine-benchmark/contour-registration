"""Matplotlib plotting utilities for the contour-registration toolkit.

All functions use the Agg (non-interactive) backend so they work on headless
servers without a display.  Each function:
  - Creates a figure
  - Renders the requested plot
  - Saves a PNG to ``out_png`` (parent directory is created if needed)
  - Closes the figure (no resource leak)
  - Returns ``Path(out_png)``
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(error_history: list[float], out_png, title: str | None = None) -> Path:
    """Line plot of ICP error (y) vs iteration index (x).

    Parameters
    ----------
    error_history:
        Sequence of per-iteration error values.
    out_png:
        Destination path for the PNG (str or Path).
    title:
        Optional plot title; defaults to ``"ICP convergence"``.

    Returns
    -------
    Path
        Resolved path of the written PNG.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(error_history)), error_history, marker="o", linewidth=1.5)
    ax.set_title(title or "ICP convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def plot_sweep_heatmap(summary: dict, out_png) -> Path:
    """2-D heatmap of registration error over (canny_low rows × canny_high cols).

    Consumes ``summary["configs"]`` for per-config data and ``summary["best"]``
    to annotate the best configuration in the title.

    Parameters
    ----------
    summary:
        Dict in the canonical sweep_summary.json shape::

            {
                "best": "<slug>",
                "configs": {
                    "<slug>": {
                        "error": <float>,
                        "pose": [...],
                        "config": {
                            "canny_low": int, "canny_high": int,
                            "margin_size": int, "sobel_kernel": int,
                            "include_inner_contours": bool
                        }
                    },
                    ...
                }
            }

    out_png:
        Destination path for the PNG.

    Returns
    -------
    Path
        Resolved path of the written PNG.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    configs = summary["configs"]
    best_slug = summary["best"]

    # Collect unique low/high values (sorted)
    lo_vals = sorted({v["config"]["canny_low"] for v in configs.values()})
    hi_vals = sorted({v["config"]["canny_high"] for v in configs.values()})

    nlow = len(lo_vals)
    nhigh = len(hi_vals)
    grid = np.full((nlow, nhigh), np.nan)

    lo_idx = {v: i for i, v in enumerate(lo_vals)}
    hi_idx = {v: j for j, v in enumerate(hi_vals)}

    for slug, data in configs.items():
        lo = data["config"]["canny_low"]
        hi = data["config"]["canny_high"]
        grid[lo_idx[lo], hi_idx[hi]] = data["error"]

    fig, ax = plt.subplots(figsize=(max(6, nhigh * 1.0), max(5, nlow * 0.8)))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis_r",
                   interpolation="nearest")
    ax.set_xticks(range(nhigh))
    ax.set_xticklabels(hi_vals)
    ax.set_yticks(range(nlow))
    ax.set_yticklabels(lo_vals)
    ax.set_xlabel("Canny high threshold")
    ax.set_ylabel("Canny low threshold")
    ax.set_title(f"Sweep heatmap, best: {best_slug}")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Error")

    # Annotate non-NaN cells with their error value. Pick each label's color by
    # the cell's actual rendered luminance so it stays legible regardless of the
    # colormap: dark text on light cells, white text on dark cells.
    cmap = im.get_cmap()
    norm = im.norm
    for i in range(nlow):
        for j in range(nhigh):
            val = grid[i, j]
            if not np.isnan(val):
                r, g, b = cmap(norm(val))[:3]
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def plot_loo(loo_result: dict, out_png) -> Path:
    """Bar chart of leave-one-view-out held-out error per view.

    Parameters
    ----------
    loo_result:
        Dict in the shape returned by
        :func:`contour_reg.pipeline.loo.run_loo`, i.e. with a ``"per_view"``
        list of ``{"view": int, "held_out_error_px": float}`` entries.
        Non-finite errors (``inf``) are plotted as zero-height bars and
        annotated as ``"inf"``.
    out_png:
        Destination path for the PNG (str or Path).

    Returns
    -------
    Path
        Resolved path of the written PNG.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    per_view = loo_result.get("per_view", [])
    views = [entry["view"] for entry in per_view]
    raw_errors = [entry["held_out_error_px"] for entry in per_view]
    heights = [e if np.isfinite(e) else 0.0 for e in raw_errors]

    fig, ax = plt.subplots(figsize=(max(6, len(views) * 0.4), 4))
    bars = ax.bar(views, heights, color="steelblue")

    # Annotate non-finite (inf) bars.
    for bar, raw in zip(bars, raw_errors):
        if not np.isfinite(raw):
            ax.text(bar.get_x() + bar.get_width() / 2, 0, "inf",
                    ha="center", va="bottom", fontsize=7, color="red")

    ax.set_title("Leave-one-view-out error")
    ax.set_xlabel("View index")
    ax.set_ylabel("Held-out error (px)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def plot_pareto(summary: dict, out_png) -> Path:
    """Scatter plot of error vs canny_high for every config; best config highlighted.

    Consumes ``summary["configs"]`` and ``summary["best"]`` in the canonical
    sweep_summary.json shape.

    Parameters
    ----------
    summary:
        Dict in the canonical sweep_summary.json shape (see :func:`plot_sweep_heatmap`).
    out_png:
        Destination path for the PNG.

    Returns
    -------
    Path
        Resolved path of the written PNG.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    configs = summary["configs"]
    best_slug = summary["best"]

    slugs = list(configs.keys())
    hi_vals = [configs[s]["config"]["canny_high"] for s in slugs]
    errors = [configs[s]["error"] for s in slugs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(hi_vals, errors, s=50, alpha=0.7, label="configs")

    # Highlight the best config
    if best_slug in configs:
        bhi = configs[best_slug]["config"]["canny_high"]
        berr = configs[best_slug]["error"]
        ax.scatter([bhi], [berr], s=120, color="red", zorder=5, label=f"best: {best_slug}")
        ax.annotate(best_slug, (bhi, berr), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color="red")

    ax.set_xlabel("canny_high")
    ax.set_ylabel("Error")
    ax.set_title("Configs: error vs canny_high")
    ax.legend(fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png
