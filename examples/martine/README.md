# Example: a Martine object

[`register.sh`](register.sh) runs `contour-reg pipeline` with the settings used for the
[Martine benchmark](https://martine-benchmark.github.io/):

| Option | Value |
| :-- | :-- |
| `--dof 7` | similarity transform: the ground-truth scans are metric, the SfM poses are up to scale |
| `--images` | Canny contours of the images, kept in a band around the masks |
| `--margin 20` | 20 px band |
| `--sobel 15` | Sobel kernel 15 |
| (default grid) | 20 Canny configurations: low ∈ {5, 10, 20, 50}, high ∈ {20, 30, 40, 60, 80, 120}, low < high |
| `--select-by fbeta` | configuration with the highest contour F0.5 at 2 px ([docs/method.md](../../docs/method.md)) |

```bash
examples/martine/register.sh path/to/object/sfm.json path/to/object/gt.ply \
    path/to/object/masks path/to/object/images out/object
```

The data layout and camera conventions of Martine are documented in
[martine-dataset](https://github.com/martine-benchmark/martine-dataset).
