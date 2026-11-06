"""Generate docs/assets/registration.png: target contour, initial pose and registered pose on one view.

The ground-truth mesh of a Martine object is moved away from its registered pose by a fixed rotation
and translation, then registered back with the coarse-to-fine contour ICP of `contour-reg refine`.
Panel titles give the mean distance (image px) from the silhouette to the silhouette at the published
registration. Needs the Martine dataset (published layout: <root>/<object>/mvs/d8/{images,masks,sfm.json}
and <root>/<object>/gt/gt.ply); not run by the tests.

    uv run --with fast-simplification --with openexr python docs/assets/make_figure.py \\
        --root /path/to/martine --every 4

Tone mapping (common to the Martine repositories): the 99th percentile of the object pixels maps to 1,
then clipping and sRGB encoding. Colours and line styles follow the contour-registration banner.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from contour_reg.config import CannyConfig  # noqa: E402
from contour_reg.io.mesh import decimate_mesh, load_mesh  # noqa: E402
from contour_reg.io.poses import load_poses  # noqa: E402
from contour_reg.pipeline._contours import build_target_contours  # noqa: E402
from contour_reg.pipeline.refine import run_refine  # noqa: E402
from contour_reg.registration.transforms import pose_to_matrix  # noqa: E402

BACKGROUND = "#111111"
TARGET = "#56B4E9"  # Canny target contour (Okabe-Ito sky blue), drawn with alpha 0.8
START = "#c2577f"  # silhouette at the initial pose, dashed 7 px on / 5 px off
FINAL = "#f8a700"  # silhouette after contour ICP
TITLE, LEGEND = "#e6edf3", "#9198a1"
PANEL_W = 600  # px per panel; the height follows the object's aspect ratio
SUPER = 4  # drawing resolution factor
PAD = 0.08  # crop padding around the object, as a fraction of its size


def read_exr(path: Path) -> np.ndarray:
    """Linear BGR float32 image; OpenCV when built with OpenEXR, else the `openexr` package."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img[..., :3].astype(np.float32)
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        ch = f.channels()
        rgb = ch["RGB"].pixels if "RGB" in ch else ch["RGBA"].pixels[..., :3] if "RGBA" in ch else \
            np.stack([ch[c].pixels for c in "RGB"], axis=-1)
    return np.ascontiguousarray(rgb[..., ::-1], dtype=np.float32)


def tone_map(hdr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linear RGB -> 8-bit sRGB, the 99th percentile of the object pixels mapped to 1."""
    scale = np.percentile(hdr[mask].max(axis=1), 99) if mask.any() else hdr.max()
    x = np.clip(hdr / max(scale, 1e-12), 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (srgb * 255 + 0.5).astype(np.uint8)


def crop(mask: np.ndarray) -> tuple:
    """Panel size (w, h) at SUPER resolution and the 2x3 affine from image pixels to it (bbox + PAD)."""
    ys, xs = np.nonzero(mask)
    bw, bh = (np.ptp(xs) + 1) * (1 + 2 * PAD), (np.ptp(ys) + 1) * (1 + 2 * PAD)
    w = PANEL_W * SUPER
    h = int(round(w * bh / bw))
    k = w / bw
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    return (w, h), np.array([[k, 0, (w - 1) / 2 - k * cx], [0, k, (h - 1) / 2 - k * cy]])


def silhouette_contour(mesh, pose: np.ndarray, camera, affine: np.ndarray, size: tuple) -> np.ndarray:
    """Outer contour (N, 2) of the projected mesh at *pose*, in panel pixels (sub-pixel rasterization)."""
    T = pose_to_matrix(pose)
    uv = camera.project_points(mesh.vertices @ T[:3, :3].T + T[:3, 3])
    uv = uv @ affine[:, :2].T + affine[:, 2]
    shift = 4
    tris = np.round(uv[mesh.faces] * (1 << shift)).astype(np.int32)
    sil = np.zeros(size[::-1], np.uint8)
    cv2.fillPoly(sil, list(tris), 255, lineType=cv2.LINE_8, shift=shift)  # union of the triangles
    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=len)[:, 0, :]


def mean_distance(contour: np.ndarray, reference: np.ndarray, size: tuple, scale: float) -> float:
    """Mean distance (image px) from *contour* points to the *reference* contour, both in panel pixels."""
    ref = np.full(size[::-1], 255, np.uint8)
    cv2.polylines(ref, [reference.reshape(-1, 1, 2)], True, 0, 1)
    dist = cv2.distanceTransform(ref, cv2.DIST_L2, 5)
    x = contour[:, 0].clip(0, size[0] - 1)
    y = contour[:, 1].clip(0, size[1] - 1)
    return float(dist[y, x].mean() / scale)


def hex_bgr(h: str) -> tuple:
    return tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))


def dashed(img: np.ndarray, contour: np.ndarray, color: tuple, width: int, on: float, off: float) -> None:
    """Closed dashed polyline along *contour* (pixel steps); dash lengths in pixels."""
    pts = np.vstack([contour, contour[:1]]).astype(np.float64)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    drawn = np.mod(s, on + off) < on
    for i in np.flatnonzero(drawn[:-1]):
        p, q = np.round(pts[i]).astype(int), np.round(pts[i + 1]).astype(int)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, width, cv2.LINE_AA)


def panel(photo, mask, affine, size, target, contour=None, style=None) -> np.ndarray:
    """Photo cut out by its mask on the background, target contour, optional silhouette."""
    img = cv2.warpAffine(photo, affine, size, flags=cv2.INTER_CUBIC).astype(np.float32)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.8)
    alpha = cv2.warpAffine(alpha, affine, size, flags=cv2.INTER_LINEAR)[..., None]
    img = (img * alpha + np.array(hex_bgr(BACKGROUND), np.float32) * (1 - alpha)).astype(np.uint8)
    dots = img.copy()
    pts = target @ affine[:, :2].T + affine[:, 2]
    for x, y in np.round(pts * 16).astype(np.int32):
        cv2.circle(dots, (int(x), int(y)), int(0.8 * SUPER * 16), hex_bgr(TARGET), -1, cv2.LINE_AA, shift=4)
    img = cv2.addWeighted(img, 0.2, dots, 0.8, 0)
    if style == "start":
        dashed(img, contour, hex_bgr(START), int(round(2.0 * SUPER)), 7 * SUPER, 5 * SUPER)
    elif style == "final":
        cv2.polylines(img, [contour.reshape(-1, 1, 2)], True, hex_bgr(FINAL), int(round(2.4 * SUPER)), cv2.LINE_AA)
    w, h = size
    small = cv2.resize(img, (w // SUPER, h // SUPER), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="Martine dataset root (published layout)")
    ap.add_argument("--object", default="lego")
    ap.add_argument("--view", default="000", help="View shown in the figure")
    ap.add_argument("--every", type=int, default=1, help="ICP on one view out of N (the shown view is kept)")
    ap.add_argument("--canny", type=int, nargs=2, default=(20, 60), metavar=("LOW", "HIGH"))
    ap.add_argument("--margin", type=int, default=10,
                    help="Canny band half-width (px); the published images are black from 16 px outside the mask")
    ap.add_argument("--rotation-deg", type=float, default=6.0, help="Start: rotation about the mesh centroid")
    ap.add_argument("--translation", type=float, default=0.03, help="Start: translation, fraction of bbox diagonal")
    ap.add_argument("-o", "--output", type=Path, default=Path(__file__).with_name("registration.png"))
    args = ap.parse_args()

    d8 = args.root / args.object / "mvs" / "d8"
    cameras = [c for i, c in enumerate(load_poses(d8 / "sfm.json", fmt="meshroom"))
               if i % args.every == 0 or c.name == args.view]
    photos, masks = [], []
    for cam in cameras:
        hdr = read_exr(d8 / "images" / f"{cam.name}.exr")
        masks.append(cv2.imread(str(d8 / "masks" / f"{cam.name}.png"), cv2.IMREAD_GRAYSCALE))
        photos.append(tone_map(hdr, masks[-1] > 0))
    print(f"{len(cameras)} views loaded", flush=True)
    cfg = CannyConfig(canny_low=args.canny[0], canny_high=args.canny[1], margin_size=args.margin)
    targets = build_target_contours(cfg, masks, [cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) for p in photos])

    mesh = decimate_mesh(load_mesh(args.root / args.object / "gt" / "gt.ply"), 30000)
    print(f"mesh: {len(mesh.vertices)} vertices", flush=True)

    # Start pose: the registered mesh rotated about its centroid and translated (fixed directions).
    c = mesh.vertices.mean(axis=0)
    axis = np.array([1.0, 0.6, -0.4]) / np.linalg.norm([1.0, 0.6, -0.4])
    R = Rotation.from_rotvec(np.deg2rad(args.rotation_deg) * axis).as_matrix()
    direction = np.array([0.7, -0.5, 0.5]) / np.linalg.norm([0.7, -0.5, 0.5])
    shift = args.translation * float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))  # pose units (mm)
    t = c - R @ c + shift * direction
    start = np.concatenate([Rotation.from_matrix(R).as_euler("xyz"), t])

    final, history, _ = run_refine(mesh, cameras, targets, start)
    print(f"mean contour error: {history[0]:.2f} -> {history[-1]:.2f} px", flush=True)

    # Shown view; distances are measured to the silhouette at the published (reference) registration.
    k = next(i for i, cam in enumerate(cameras) if cam.name == args.view)
    size, affine = crop(masks[k] > 0)
    sil = {name: silhouette_contour(mesh, pose, cameras[k], affine, size)
           for name, pose in [("ref", np.zeros(6)), ("start", start), ("final", final)]}
    d_start = mean_distance(sil["start"], sil["ref"], size, affine[0, 0])
    d_final = mean_distance(sil["final"], sil["ref"], size, affine[0, 0])
    m = masks[k] > 0
    panels = [
        ("Target contour", panel(photos[k], m, affine, size, targets[k])),
        (f"Start: {args.rotation_deg:g}° and {shift:.0f} mm off ({d_start:.1f} px)",
         panel(photos[k], m, affine, size, targets[k], sil["start"], "start")),
        (f"After ICP ({d_final:.1f} px)", panel(photos[k], m, affine, size, targets[k], sil["final"], "final")),
    ]

    # Layout in pixels at 200 dpi: 1800 px wide, three panels, title band above, legend band below.
    dpi, width, gap, title_h, legend_h = 200, 1800, 12, 46, 52
    pw = (width - 4 * gap) / 3
    ph = pw * size[1] / size[0]
    height = title_h + ph + legend_h
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=BACKGROUND)
    for i, (title, img) in enumerate(panels):
        ax = fig.add_axes(((gap + i * (pw + gap)) / width, legend_h / height, pw / width, ph / height))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, color=TITLE, fontsize=10, pad=6)
    handles = [
        Line2D([], [], color=TARGET, alpha=0.8, marker="o", linestyle="", markersize=4,
               label="Target contour (Canny edges near the mask boundary)"),
        Line2D([], [], color=START, linewidth=1.6, linestyle=(0, (3.5, 2.5)), label="initial pose"),
        Line2D([], [], color=FINAL, linewidth=2, label="after contour ICP"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=LEGEND,
               bbox_to_anchor=(0.5, 0.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=BACKGROUND)
    meta = {"object": args.object, "view": args.view, "canny": list(args.canny), "margin": args.margin,
            "views": len(cameras), "start_pose": start.tolist(), "final_pose": final.tolist(),
            "mean_contour_error_px": [history[0], history[-1]], "distance_to_reference_px": [d_start, d_final]}
    print(json.dumps(meta))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
