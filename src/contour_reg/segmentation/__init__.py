"""Text-promptable SAM 3 segmentation (optional add-on).

Requires the ``[segmentation]`` extra:  ``uv sync --extra segmentation``
(provides transformers, torch, torchvision, pillow).

The heavy module ``sam3_segmenter`` needs torch/transformers for inference,
so it is imported lazily by the ``contour-reg segment`` CLI subcommand, never
at ``contour_reg.cli`` module top-level.
"""

INSTALL_HINT = (
    "The 'segment' command needs the optional segmentation dependencies "
    "(torch, torchvision, transformers). Install them with:  "
    "uv sync --extra segmentation   (or: pip install -e '.[segmentation]')"
)
