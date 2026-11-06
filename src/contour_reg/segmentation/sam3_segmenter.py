"""SAM 3 segmenter: text-promptable mask generation via HuggingFace transformers.

Public API
----------
load_sam3(checkpoint, device)        -> (model, processor)
segment_image(image, prompt, ...)    -> np.ndarray (HxW, uint8, 0/255)
segment_directory(image_dir, ...)    -> {"processed", "skipped", "failed"}

Internal helper (also importable for testing)
---------------------------------------------
_union_masks(masks, H, W)            -> np.ndarray (HxW, uint8, 0/255)

Notes
-----
- SAM 3 weights are GATED on HuggingFace (https://huggingface.co/facebook/sam3).
  Request access, then authenticate with `hf auth login` or HF_TOKEN env var.
- Model weights are bfloat16; inference on CUDA must use torch.autocast +
  torch.inference_mode (handled internally; callers need not worry about this).
- Importing this module does NOT trigger a weight download.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from contour_reg.segmentation import INSTALL_HINT

try:  # torch is only needed for live SAM 3 inference, not for mask post-processing
    import torch
except ImportError:  # pragma: no cover - exercised in torch-free environments
    torch = None

logger = logging.getLogger(__name__)

def _require_segmentation_deps(need_torch: bool = True) -> None:
    """Raise :class:`ImportError` with an install hint if transformers (or torch) is missing."""
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    if need_torch and torch is None:
        raise ImportError(INSTALL_HINT)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ImageLike = Union[str, Path, "Image.Image", np.ndarray]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _bfloat16_ctx(device: str):
    """Context manager: bfloat16 autocast on CUDA, no-grad on CPU.

    Falls back to a plain context when torch is unavailable; the caller then
    supplies its own model, and there is no autograd state to guard.
    """
    if torch is None:
        yield
        return
    if device.startswith("cuda"):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        with torch.inference_mode():
            yield


def _to_pil(image: ImageLike) -> Tuple["Image.Image", int, int]:
    """Convert path / PIL / ndarray to a PIL RGB image and return (img, H, W)."""
    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            # Float array → scale to uint8
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)
        # ndarray may be BGR (from OpenCV): assume RGB per docstring (HxWx3 uint8/float)
        pil = Image.fromarray(image, mode="RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    w, h = pil.size
    return pil, h, w


def _mask_to_bool_ndarray(m) -> np.ndarray:
    """Convert a single mask (torch tensor or array-like) to a 2-D bool ndarray.

    Handles CUDA tensors and non-numpy-friendly dtypes (e.g. bfloat16) by
    moving to CPU and casting to bool before the numpy conversion.  Numpy
    arrays and other array-likes are passed straight to ``np.asarray``.
    A leading singleton dimension (1, H, W) is squeezed to (H, W).
    """
    if torch is not None and isinstance(m, torch.Tensor):
        # .detach().cpu() is required for CUDA tensors; .to(torch.bool) avoids
        # the "can't convert bfloat16 to numpy" path for half-precision masks.
        arr = m.detach().to(torch.bool).cpu().numpy()
    else:
        arr = np.asarray(m, dtype=bool)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)  # (1, H, W) -> (H, W)
    return arr


def _union_masks(masks: Sequence, H: int, W: int) -> np.ndarray:
    """Union a sequence of boolean (or 0/1) instance masks into a single HxW uint8 mask.

    Each mask may be a numpy array OR a torch tensor (CPU or CUDA); tensors are
    moved to CPU and cast to bool before conversion.  Each mask must be 2-D
    (or (1, H, W)) and broadcastable to (H, W).
    Returns an HxW uint8 array with values 0 or 255.
    If *masks* is empty, returns an all-zero array of shape (H, W).
    """
    result = np.zeros((H, W), dtype=np.uint8)
    for m in masks:
        arr = _mask_to_bool_ndarray(m)
        result[arr] = 255
    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_sam3(
    checkpoint: str = "facebook/sam3",
    device: Optional[str] = None,
):
    """Load SAM 3 model and processor from HuggingFace or a local path.

    Parameters
    ----------
    checkpoint:
        HuggingFace repo id (e.g., "facebook/sam3") or an absolute local
        directory that contains the model files.  The HuggingFace Hub cache
        and HF_TOKEN / `hf auth login` credentials are honoured automatically.
    device:
        "cuda", "cpu", or None (auto-detect: CUDA if available, else CPU).

    Returns
    -------
    (model, processor), both ready for inference (.eval() already called).

    Raises
    ------
    RuntimeError
        If the checkpoint is gated (HTTP 401/403) and the user has not
        authenticated. The error message includes the URL and remediation steps.
    ImportError
        If the optional ``[segmentation]`` dependencies are not installed.
    """
    _require_segmentation_deps(need_torch=device is None)  # torch only for device auto-detection
    from transformers import Sam3Model, Sam3Processor  # lazy import

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading SAM 3 processor from %r …", checkpoint)
    try:
        processor = Sam3Processor.from_pretrained(checkpoint)
    except Exception as exc:
        _reraise_if_gated(exc, checkpoint)
        raise

    logger.info("Loading SAM 3 model from %r onto %s …", checkpoint, device)
    try:
        model = Sam3Model.from_pretrained(checkpoint)
    except Exception as exc:
        _reraise_if_gated(exc, checkpoint)
        raise

    model.to(device).eval()
    logger.info("SAM 3 ready on %s.", device)
    return model, processor


def _reraise_if_gated(exc: Exception, checkpoint: str) -> None:
    """Re-raise a gated / auth error with a user-friendly message."""
    msg = str(exc).lower()
    if any(kw in msg for kw in ("401", "403", "gated", "access", "unauthorized", "forbidden")):
        raise RuntimeError(
            f"Cannot load SAM 3 weights from '{checkpoint}': access denied.\n"
            "The SAM 3 weights are GATED on HuggingFace. To gain access:\n"
            "  1. Request access at https://huggingface.co/facebook/sam3\n"
            "  2. Authenticate: run `hf auth login` or set HF_TOKEN=<your_token>\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Single-image segmentation
# ---------------------------------------------------------------------------

def segment_image(
    image: ImageLike,
    prompt: str,
    model,
    processor,
    device: str,
    *,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> np.ndarray:
    """Segment *image* using a text *prompt* and return a binary mask.

    Parameters
    ----------
    image:
        A file path (str or Path), a PIL.Image, or an HxWx3 uint8/float ndarray.
    prompt:
        A single text concept (e.g., "lego brick").
    model, processor:
        Returned by :func:`load_sam3`.
    device:
        "cuda" or "cpu".
    threshold:
        Score threshold for instance detection (passed to post_process).
    mask_threshold:
        Pixel-level binarisation threshold (passed to post_process).

    Returns
    -------
    np.ndarray
        HxW uint8 array with values 0 or 255. All-zero if nothing is detected.
    """
    pil_img, H, W = _to_pil(image)

    inputs = processor(images=pil_img, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with _bfloat16_ctx(device):
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=[(H, W)],
    )
    item = results[0]
    masks_tensor = item.get("masks", [])

    if len(masks_tensor) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # _union_masks moves any CUDA / bfloat16 tensors to CPU bool before numpy
    # conversion, so masks_tensor can be passed through directly.
    return _union_masks(masks_tensor, H, W)


# ---------------------------------------------------------------------------
# Directory-level segmentation
# ---------------------------------------------------------------------------

def segment_directory(
    image_dir: Union[str, Path],
    prompt: str,
    output_dir: Union[str, Path],
    *,
    image_pattern: str = "*.png",
    checkpoint: str = "facebook/sam3",
    device: Optional[str] = None,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    resize_width: Optional[int] = None,
    force: bool = False,
) -> Dict[str, int]:
    """Run SAM 3 segmentation on all images in *image_dir*.

    Parameters
    ----------
    image_dir:
        Directory containing input images.
    prompt:
        Text concept for SAM 3 (e.g., "object").
    output_dir:
        Masks are written as ``{stem}.png`` (0/255 grayscale).
    image_pattern:
        Glob pattern relative to *image_dir* (default: "*.png").
    checkpoint:
        HuggingFace repo id or local path (forwarded to :func:`load_sam3`).
    device:
        "cuda", "cpu", or None (auto-detect).
    threshold:
        Instance detection score threshold.
    mask_threshold:
        Pixel binarisation threshold.
    resize_width:
        If set, downscale each image to this width before inference (keeps
        aspect ratio).  Masks are written at the downscaled size.
    force:
        If False (default), skip images whose output mask already exists.

    Returns
    -------
    dict with keys "processed", "skipped", "failed" (counts).

    Raises
    ------
    ImportError
        If the optional ``[segmentation]`` dependencies are not installed.
    """
    _require_segmentation_deps()
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = load_sam3(checkpoint=checkpoint, device=device)

    image_files = sorted(image_dir.glob(image_pattern))
    if not image_files:
        logger.warning("No images matching %r found in %s", image_pattern, image_dir)

    processed = skipped = failed = 0

    for img_path in image_files:
        out_path = output_dir / f"{img_path.stem}.png"
        if out_path.exists() and not force:
            logger.info("Skipping %s (already exists)", img_path.name)
            skipped += 1
            continue

        logger.info("Segmenting %s …", img_path.name)
        try:
            pil_img = Image.open(img_path).convert("RGB")
            if resize_width is not None:
                orig_w, orig_h = pil_img.size
                new_h = int(resize_width * orig_h / orig_w)
                pil_img = pil_img.resize((resize_width, new_h), Image.LANCZOS)

            mask = segment_image(
                pil_img, prompt, model, processor, device,
                threshold=threshold,
                mask_threshold=mask_threshold,
            )
            cv2.imwrite(str(out_path), mask)
            processed += 1
        except Exception as exc:
            logger.error("Failed to segment %s: %s", img_path.name, exc)
            failed += 1

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    logger.info(
        "Done. Processed: %d, skipped: %d, failed: %d",
        processed, skipped, failed,
    )
    return {"processed": processed, "skipped": skipped, "failed": failed}
