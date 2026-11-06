# Command-line reference

Every `contour-reg` command with its main options and defaults. `contour-reg <command> --help` gives
the full list. The method behind each step is explained in [method.md](method.md).

Options shared by the commands that read a capture:

| Option | Default | Meaning |
| :-- | :-- | :-- |
| `--poses PATH` | required | `sfm.json` (Meshroom), COLMAP model directory, or IDR `cameras.npz` |
| `--format {auto,meshroom,colmap,idr}` | `auto` | Force the pose format (auto: `.json` → Meshroom, `.npz` → IDR, directory → COLMAP) |
| `--mesh FILE` | required | Mesh to register (`.ply`, `.obj`) |
| `--masks DIR` | required | One binary mask per view, matched to the views by file name (see [below](#matching-masks-and-images-to-views)) |
| `--images DIR` | none | One image per view, matched the same way; contours are then Canny edges instead of mask boundaries |

### Matching masks and images to views

A mask (or image) belongs to the view whose image has the same file name, extension aside: view
`IMG_0001.JPG` ↔ mask `IMG_0001.png` (Meshroom: the view `path`; COLMAP: the image name). A view
without a mask, or two masks for one view, is an error; masks of views absent from the poses are
ignored with a warning. IDR poses have no image names: files are then taken in sorted file-name order
(zero-pad numbers) and must match the number of cameras.

Meshroom poses include the format's world flip `diag(1, -1, -1)`, which the loader applies (see
[output.md](output.md#camera-conventions)).

## `pipeline`: one command from masks to a registered pose

```bash
contour-reg pipeline --poses sfm.json --mesh mesh.ply --masks masks/ --images images/ -o out/
```

Runs the Canny sweep over the default grid, selects a configuration, refines it and writes
`out/result.json` (see [output.md](output.md)).

| Option | Default | Meaning |
| :-- | :-- | :-- |
| `--dof {6,7}` | 6 | 6: rigid; 7: similarity (rigid + uniform scale), e.g. metric mesh vs up-to-scale SfM |
| `--sobel K` | 3 | Sobel kernel size of the Canny detector (odd, up to 31 in OpenCV) |
| `--margin PX` | 20 | Half-width of the band around the mask boundary in which Canny edges are kept |
| `--select-by {fbeta,error}` | `fbeta` | Configuration to refine: highest contour F0.5 at 2 px, or lowest reprojection error |
| `--plot` / `--no-plot` | plots on | Write `plots/{sweep_heatmap,convergence,pareto}.png` |

Grid: Canny low ∈ {5, 10, 20, 50} × high ∈ {20, 30, 40, 60, 80, 120} with low < high, i.e. 20
configurations. Without `--images`, every configuration gives the same mask-boundary contour, so the
sweep only repeats the same registration. With `--select-by fbeta` the pipeline stops with an error
when no configuration has a contour point within 2 px of the projected silhouette: check the initial
pose of the mesh or the contours.

## `sweep`: one registration per Canny configuration

```bash
contour-reg sweep --poses sfm.json --mesh mesh.ply --masks masks/ --images images/ \
    --grid "lo=5,10,20,50 hi=20,30,40,60,80,120" -o sweep/
```

Writes `sweep/<slug>.json` per configuration and `sweep/sweep_summary.json` (sorted by reprojection
error). A slug reads `cl<low>_ch<high>_m<margin>_sk<sobel>`. Options: `--grid`, `--dof`, `--sobel`,
`--margin`, `--error-percentile` (75), `--plot`/`--no-plot`.

## `eval-canny`: rank the swept configurations by contour F-beta

```bash
contour-reg eval-canny --poses sfm.json --mesh mesh.ply --masks masks/ --images images/ \
    --sweep-dir sweep/ -o eval_canny.json
```

Options: `--epsilons` (1 2 3 5 px), `--select-epsilon` (2), `--beta` (0.5), `--trim-percentile` (75).
The best configuration is printed and stored under `best`; F1 is reported alongside.

## `refine`: coarse-to-fine ICP from a sweep result

```bash
contour-reg extract-contours --masks masks/ --images images/ --canny-low 10 --canny-high 60 \
    --sobel 15 -o contours/
contour-reg refine --poses sfm.json --mesh mesh.ply --contours contours/ --sweep-dir sweep/ \
    --sweep-config cl10_ch60_m20_sk15 --dof 7 -o refined.json
```

Starts from the pose (and scale) of `--sweep-config`, by default the lowest-error configuration of
`sweep_summary.json`. The contours must be those of that configuration. Options: `--dof` (6; a
scaled sweep pose requires 7), `--error-percentile` (75, coarse phase), `--keep-pct` (90, fine phase).

## `extract-contours`: per-view contours as `.npy`

| Option | Default | Meaning |
| :-- | :-- | :-- |
| `--mode {canny,mask-boundary}` | `canny` | Canny needs `--images`; otherwise the mask boundary is used |
| `--canny-low`, `--canny-high` | 50, 150 | Canny hysteresis thresholds |
| `--sobel`, `--margin` | 3, 20 | As for `pipeline` |
| `--component-mode {hybrid,center_point,smallest_area}` | `hybrid` | Which connected component of the mask to keep |
| `--include-inner-contours` | off | Also keep hole boundaries |

## `register`: one ICP run on pre-extracted contours

Options: `--init-pose rx ry rz tx ty tz` or `--init-transform T.npy` (4×4 similarity), `--dof` (6),
`--max-iterations` (50), `--error-percentile` (75), `--max-correspondence-distance` (0 = off, px),
`--min-canny-points` (0: drop views with fewer contour points), `--loss` (`soft_l1`), `--f-scale` (6 px),
`--include-inner-contours`.

## `eval-loo`: leave-one-view-out error

```bash
contour-reg eval-loo --poses sfm.json --mesh mesh.ply --contours contours/ --init-pose 0 0 0 0 0 0 -o loo.json
```

Registers on all views but one and measures the contour error on the held-out view, for every view.
Options: `--optimize-scale` (with `--initial-scale`), `--max-correspondence-distance`,
`--min-canny-points`, `--plot` (off by default).

## `plot` and `segment`

`contour-reg plot --results sweep/ -o plots/` re-draws the sweep heatmap and Pareto plot.
`contour-reg segment` produces masks from a text prompt with SAM 3: see [segmentation.md](segmentation.md).

## Troubleshooting

| Symptom | Likely cause |
| :-- | :-- |
| `No mask file for N of M view(s): …` | Masks must be named after the view images (`IMG_0001.JPG` ↔ `IMG_0001.png`) |
| `Found N mask file(s) but M camera(s)` | IDR poses (no image names): exactly one mask per camera, in sorted order |
| `No sweep config reaches the silhouette within 2 px` | Mesh too far from its true pose, mesh in another frame than the poses, or wrong or empty masks |
| Mesh projects upside down or mirrored | The mesh is not in the frame of the poses (for Meshroom, the flipped frame of the format; see [output.md](output.md)) |
