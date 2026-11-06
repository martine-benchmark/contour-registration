"""Build the README banner of contour-registration: six objects with their initial and registered mesh contours.

Usage:
    python make_banner.py --data <martine root> --contours <dir> --out assets/banner.png

<martine root>/<object>/mvs/d8/{images/000.exr, masks/000.png} give the photo and its mask; <dir>/<object>/ holds
start_contour.npy and final_contour.npy, the outer silhouette contour (N, 2) in d8 pixels of the mesh projected
with the perturbed initial pose and with the pose after contour ICP. Tone mapping: exposure so that the 99th
percentile inside the mask is 1, clip, sRGB curve. The banner is a 1600x400 RGBA PNG: rounded #111111 card with a
#30363d border (readable on GitHub light and dark themes) and the Martine palette gradient rule, shared by the
banners of the three Martine repositories. Requires numpy, opencv-python (with OpenEXR) and Pillow.
"""

import argparse
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

W, H, RADIUS = 1600, 400, 24
CARD, BORDER, MUTED = "#111111", "#30363d", "#9198a1"
PALETTE = ["#003d5c", "#5749a5", "#b35280", "#f64a4b", "#f8a700"]
START, FINAL = "#c2577f", "#f8a700"
OBJECTS = ["rock", "glass", "knife", "lego", "plant", "whisk"]
FONTS = ["DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
SS = 3  # supersampling for the vector strokes


def rgb(h, a=255):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (a,)


def font(size, path=None):
    for f in ([path] if path else []) + FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def photo(root, obj):
    im = cv2.imread(os.path.join(root, obj, "mvs/d8/images/000.exr"), cv2.IMREAD_UNCHANGED)[..., ::-1]
    m = cv2.imread(os.path.join(root, obj, "mvs/d8/masks/000.png"), cv2.IMREAD_GRAYSCALE)
    x = np.clip(im / np.percentile(im[m > 127], 99), 0, 1)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return (x * 255 + 0.5).astype(np.uint8), m


def card():
    base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, H - 1], RADIUS, fill=255)
    base.paste(Image.new("RGBA", (W, H), rgb(CARD)), (0, 0), mask)
    return base, mask


def finish(base, mask, rule_h=6):
    xs = np.linspace(0, len(PALETTE) - 1, W)
    cols = np.array([rgb(c)[:3] for c in PALETTE], float)
    i = np.clip(xs.astype(int), 0, len(PALETTE) - 2)
    t = (xs - i)[:, None]
    row = cols[i] * (1 - t) + cols[i + 1] * t
    rule = np.dstack([np.repeat(row[None], rule_h, 0), np.full((rule_h, W), 255)]).astype(np.uint8)
    base.alpha_composite(Image.fromarray(rule, "RGBA"), (0, H - rule_h))
    ImageDraw.Draw(base).rounded_rectangle([0, 0, W - 1, H - 1], RADIUS, outline=rgb(BORDER), width=2)
    base.putalpha(Image.fromarray(np.minimum(np.array(base.getchannel("A")), np.array(mask))))
    return base


def polyline(draw, pts, color, width, dash=None):
    pts = [tuple(p) for p in pts] + [tuple(pts[0])]
    if dash is None:
        draw.line(pts, fill=color, width=width, joint="curve")
        return
    on, off = dash
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(np.array(pts), axis=0).T))]
    for s0 in np.arange(0, seg[-1], on + off):
        sel = (seg >= s0) & (seg <= s0 + on)
        if sel.sum() >= 2:
            draw.line([pts[k] for k in np.nonzero(sel)[0]], fill=color, width=width, joint="curve")


def object_layer(rgb_img, m, start, final, cell_w, cell_h):
    """Cut-out photo and both contours, scaled to fit the cell; returns an RGBA image."""
    allpts = np.vstack([start, final, np.argwhere(m > 127)[:, ::-1]])
    x0, y0 = allpts.min(0) - 4
    x1, y1 = allpts.max(0) + 4
    s = min(cell_w / (x1 - x0), cell_h / (y1 - y0))
    size = (round((x1 - x0) * s), round((y1 - y0) * s))
    M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    px = cv2.warpAffine(rgb_img, M, size, flags=cv2.INTER_AREA)
    a = cv2.GaussianBlur(cv2.warpAffine(m, M, size, flags=cv2.INTER_AREA), (0, 0), 0.8)
    layer = Image.fromarray(np.dstack([px, a]), "RGBA")
    big = Image.new("RGBA", (size[0] * SS, size[1] * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    tr = lambda c: (np.asarray(c, float) - [x0, y0]) * s * SS  # noqa: E731
    polyline(d, tr(start), rgb(START), 2 * SS, dash=(7 * SS, 5 * SS))
    polyline(d, tr(final), rgb(FINAL), round(2.4 * SS))
    layer.alpha_composite(big.resize(size, Image.LANCZOS))
    return layer


def legend(base, x1, y, f):
    d = ImageDraw.Draw(base)
    items = [("initial pose", START, True), ("after contour ICP", FINAL, False)]
    x = x1
    for label, col, dashed in reversed(items):
        x -= d.textlength(label, font=f)
        d.text((x, y), label, fill=rgb(MUTED), font=f, anchor="lm")
        x -= 10 + 34
        if dashed:
            for k in range(0, 34, 12):
                d.line([(x + k, y), (x + min(k + 7, 34), y)], fill=rgb(col), width=3)
        else:
            d.line([(x, y), (x + 34, y)], fill=rgb(col), width=3)
        x -= 30


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="Martine dataset root")
    ap.add_argument("--contours", required=True, help="directory with <object>/{start,final}_contour.npy")
    ap.add_argument("--out", default="banner.png")
    ap.add_argument("--font", default=None, help="TrueType font (default: DejaVu Sans)")
    ap.add_argument("--objects", nargs="+", default=OBJECTS, help="objects, left to right")
    a = ap.parse_args()

    base, mask = card()
    pad, gap, top, cell_h = 34, 14, 24, 300
    cw = (W - 2 * pad - gap * (len(a.objects) - 1)) / len(a.objects)
    for i, obj in enumerate(a.objects):
        img, m = photo(a.data, obj)
        start = np.load(os.path.join(a.contours, obj, "start_contour.npy")).reshape(-1, 2)
        final = np.load(os.path.join(a.contours, obj, "final_contour.npy")).reshape(-1, 2)
        lay = object_layer(img, m, start, final, cw, cell_h)
        x = pad + i * (cw + gap)
        base.alpha_composite(lay, (round(x + (cw - lay.width) / 2), round(top + (cell_h - lay.height) / 2)))
    legend(base, W - pad, H - 46, font(19, a.font))
    out = finish(base, mask)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.save(a.out, optimize=True)
    print(f"wrote {a.out} ({W}x{H})")


if __name__ == "__main__":
    main()
