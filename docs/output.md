# Outputs and pose convention

What `pipeline`, `sweep`, `refine` and `register` write, how to read the pose, and how to apply it to
the mesh.

## `pipeline` output directory

```
out/
├── result.json              selected configuration, final pose and scale
├── sweep/
│   ├── <slug>.json          one registration per Canny configuration
│   ├── sweep_summary.json   all configurations, sorted by reprojection error
│   └── eval_canny.json      contour F-beta ranking (with --select-by fbeta)
└── plots/                   sweep_heatmap.png, convergence.png, pareto.png
```

`result.json`:

| Key | Meaning |
| :-- | :-- |
| `best_config` | Slug of the refined configuration, `cl<low>_ch<high>_m<margin>_sk<sobel>` |
| `final_pose` | `[rx, ry, rz, tx, ty, tz]` (see below) |
| `scale` | Uniform scale `s` (1.0 with `--dof 6`) |
| `dof` | 6 or 7 |
| `select_by`, `selection_score` | Selection criterion and its value (F0.5 at 2 px; `null` with `error`) |
| `error_history` | Mean contour error (px) along the coarse then fine ICP runs |

`refine` and `register` write the same pose under `pose`, with `scale`, `dof` and `error_history`.

## Example: the `lego` test fixture

The [Try it](../README.md#try-it) command registers `tests/data/lego` (8 low-resolution
views, masks only) and writes this `out/lego/result.json`. The figures below are those of this
fixture, not of the full-resolution dataset.

```json
{
  "best_config": "cl10_ch120_m20_sk3",
  "final_pose": [-0.00058, -0.00293, 0.00091, 0.08991, -0.09933, -0.03491],
  "scale": 1.0,
  "dof": 6,
  "select_by": "fbeta",
  "selection_score": 0.9994,
  "error_history": [1.515, "…", 1.112]
}
```

- `final_pose` is the transform to apply to the mesh vertices (see [Pose convention](#pose-convention)).
- `error_history`: the mean contour error went from 1.52 to 1.11 px.
- `best_config`: without images, all 20 Canny configurations share the mask-boundary contour, so
  `best_config` is just the first of equal scores. Add `--images DIR` to use Canny edges of the
  images instead of the mask boundaries, which is more accurate when the masks are coarse.

## Pose convention

The transform maps mesh vertices into the world frame of the cameras:

```
v' = s · R · v + t,    R = Rotation.from_euler("xyz", [rx, ry, rz])   (SciPy, radians)
```

`t` is in the units of the camera poses. Rotation is about the mesh origin, not its centroid.

```python
import json
import trimesh
from contour_reg.io.mesh import apply_transform_to_mesh_with_scale

res = json.load(open("out/result.json"))
mesh = trimesh.load("mesh.ply")
pose, scale = res["final_pose"], res["scale"]
aligned = apply_transform_to_mesh_with_scale(mesh, pose[:3], pose[3:], scale)
aligned.export("mesh_registered.ply")
```

As a 4×4 matrix: `contour_reg.registration.transforms.pose_to_matrix(pose)` gives `[R | t]`; multiply
its rotation block by `s` for the similarity.

## Camera conventions

Cameras follow the OpenCV convention (x right, y down, z forward); the pixel `(0, 0)` is the centre of
the top-left pixel. The world flip `diag(1, -1, -1)` is part of the Meshroom/AliceVision `sfm.json`
format: the loader always applies it to the rotations and camera centres, so the mesh is expected in
the flipped world frame (`X_cam = Rᵀ(diag(1, -1, -1)·X − C)` with `R`, `C` read from the file).
COLMAP and IDR poses are used as they are.
