#!/usr/bin/env bash
# Register one Martine object with the settings of the Martine benchmark (paper, appendix):
#   7-DoF similarity (metric ground-truth scan vs up-to-scale SfM poses), Canny contours from
#   images within a 20 px band around the masks, Sobel kernel 15, the default 20-configuration
#   threshold grid, configuration selected by contour F0.5 at 2 px (the pipeline default).
#
# usage: examples/martine/register.sh SFM_JSON MESH MASKS_DIR IMAGES_DIR OUT_DIR
#   SFM_JSON    Meshroom sfm.json of the object
#   MESH        ground-truth mesh, roughly aligned with the cameras
#   MASKS_DIR   one mask per view, named after the view images of SFM_JSON (extension aside)
#   IMAGES_DIR  one image per view, named the same way
set -euo pipefail
[ $# -eq 5 ] || { sed -n "2,13p" "$0"; exit 2; }
SFM=$1; MESH=$2; MASKS=$3; IMAGES=$4; OUT=$5

uv run contour-reg pipeline \
    --poses "$SFM" --format meshroom --mesh "$MESH" \
    --masks "$MASKS" --images "$IMAGES" \
    --dof 7 --sobel 15 --margin 20 --select-by fbeta \
    -o "$OUT"

