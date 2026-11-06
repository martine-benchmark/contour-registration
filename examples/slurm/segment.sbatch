#!/usr/bin/env bash
# segment.sbatch: GPU job for SAM 3 text-promptable segmentation
#
# MUST run on a GPU compute node, NOT the login node.
#
# Required env vars:
#   IMAGES:   input image directory
#   PROMPT:   text concept (e.g. "object" or "lego brick")
#   OUTPUT:   output mask directory
#
# Optional env vars:
#   HF_TOKEN:      HuggingFace token for gated facebook/sam3 weights
#   EXTRA:         any additional CLI flags (e.g. "--resize-width 2048 --force")
#   CHECKPOINT:    HF repo id or local path (default: facebook/sam3)
#   IMAGE_PATTERN: glob pattern for images (default: *.png)
#
# Example:
#   IMAGES=/data/renders/lego \
#   PROMPT="lego brick" \
#   OUTPUT=/data/masks/lego \
#   sbatch examples/slurm/segment.sbatch
#
# With HF_TOKEN:
#   HF_TOKEN=hf_... \
#   IMAGES=/data/renders/object \
#   PROMPT="object" \
#   OUTPUT=/data/masks/object \
#   EXTRA="--resize-width 2048 --threshold 0.3" \
#   sbatch examples/slurm/segment.sbatch

#SBATCH --job-name=cr_segment
#SBATCH --account=<your_account>      # adapt to your cluster
#SBATCH --partition=<gpu_partition>
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# ── Validate required env vars ───────────────────────────────────────────
: "${IMAGES:?IMAGES env var must be set to the input image directory}"
: "${PROMPT:?PROMPT env var must be set to the text concept to segment}"
: "${OUTPUT:?OUTPUT env var must be set to the output mask directory}"

CHECKPOINT="${CHECKPOINT:-facebook/sam3}"
IMAGE_PATTERN="${IMAGE_PATTERN:-*.png}"
EXTRA="${EXTRA:-}"

echo "[cr_segment] Job ${SLURM_JOB_ID} starting on $(hostname)"
echo "[cr_segment]   IMAGES  = ${IMAGES}"
echo "[cr_segment]   PROMPT  = ${PROMPT}"
echo "[cr_segment]   OUTPUT  = ${OUTPUT}"
echo "[cr_segment]   CHECKPOINT = ${CHECKPOINT}"

# ── Python environment: run from the repository root after
#    `uv sync --extra segmentation` (or activate your own environment and drop `uv run`).

# ── HuggingFace authentication ───────────────────────────────────────────
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[cr_segment] HF_TOKEN is set, using it for gated model access."
    export HF_TOKEN
fi

# ── Create output / log directories ─────────────────────────────────────
mkdir -p "${OUTPUT}"
mkdir -p "${SLURM_SUBMIT_DIR}/slurm_logs"

# ── Run segmentation (first-class CLI subcommand) ────────────────────────
uv run --extra segmentation contour-reg segment \
    --images "${IMAGES}" \
    --prompt "${PROMPT}" \
    --output "${OUTPUT}" \
    --image-pattern "${IMAGE_PATTERN}" \
    --checkpoint "${CHECKPOINT}" \
    --device cuda \
    ${EXTRA}

echo "[cr_segment] Done."
