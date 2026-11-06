# Masks with SAM 3 (`contour-reg segment`)

Optional upstream step that produces the per-view masks the registration consumes, from a text prompt,
with SAM 3 through Hugging Face `transformers`. `pipeline` never runs it.

## Install and weights

```bash
uv sync --extra segmentation      # transformers>=5.2, torch, torchvision, huggingface_hub, pillow
```

The weights `facebook/sam3` are gated: request access at <https://huggingface.co/facebook/sam3>, then
authenticate with `hf auth login` or `export HF_TOKEN=hf_...`. Nothing is downloaded until the model is
loaded. Without the extra, `contour-reg segment` exits with status 1 and prints the install command;
the other commands do not need it.

## Usage

```bash
uv run contour-reg segment --images images/ --prompt "object" --output masks/
```

| Option | Default | Meaning |
| :-- | :-- | :-- |
| `--images DIR`, `--output DIR` | required | Input images; masks are written as `<stem>.png` (0/255) |
| `--prompt TEXT` | required | One text concept, e.g. `"lego brick"` |
| `--image-pattern GLOB` | `*.png` | Input files |
| `--checkpoint ID` | `facebook/sam3` | Hugging Face repo id or local directory |
| `--device {cuda,cpu}` | auto | CUDA if available |
| `--threshold` | 0.5 | Instance score threshold; all detected instances are merged into one mask |
| `--resize-width N` | off | Downscale to N px wide before inference (masks are written at that size) |
| `--force` | off | Overwrite existing masks (otherwise they are skipped) |

A GPU is strongly recommended. Weights are bfloat16; on CUDA, inference runs under
`torch.autocast(dtype=torch.bfloat16)` automatically. Masks keep the image stems, so they match
the views by name (see [cli.md](cli.md#matching-masks-and-images-to-views)).

## Python API

```python
from contour_reg.segmentation.sam3_segmenter import load_sam3, segment_directory, segment_image

model, processor = load_sam3("facebook/sam3", device="cuda")
mask = segment_image("images/0.png", "object", model, processor, "cuda", threshold=0.5)  # (H, W) uint8
counts = segment_directory("images/", "object", "masks/", resize_width=2048)  # {"processed", "skipped", "failed"}
```

## On a SLURM cluster

[`examples/slurm/segment.sbatch`](../examples/slurm/segment.sbatch) runs `segment` on one GPU; set the
account and partition in its header, then:

```bash
IMAGES=images/ PROMPT="object" OUTPUT=masks/ HF_TOKEN=hf_... sbatch examples/slurm/segment.sbatch
```

Extra options go through `EXTRA="--resize-width 2048 --force"`.
