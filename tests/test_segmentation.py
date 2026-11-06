"""Model-free tests for the SAM 3 segmentation module.

These tests do NOT require the facebook/sam3 weights, network access,
or a GPU.  They exercise only the pure-Python logic by:
  - Testing the _union_masks helper directly.
  - Monkeypatching transformers / model calls in segment_image.
  - Verifying CLI --help output without loading any model.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# _union_masks: pure function, no model needed
# ---------------------------------------------------------------------------

class TestUnionMasks:
    def test_empty_returns_zero_array(self):
        from contour_reg.segmentation.sam3_segmenter import _union_masks
        result = _union_masks([], 10, 12)
        assert result.shape == (10, 12)
        assert result.dtype == np.uint8
        assert result.sum() == 0

    def test_single_mask_sets_255(self):
        from contour_reg.segmentation.sam3_segmenter import _union_masks
        mask = np.zeros((4, 6), dtype=bool)
        mask[1:3, 2:5] = True
        result = _union_masks([mask], 4, 6)
        assert result.dtype == np.uint8
        assert result[1, 2] == 255
        assert result[0, 0] == 0

    def test_two_masks_union(self):
        from contour_reg.segmentation.sam3_segmenter import _union_masks
        H, W = 8, 8
        m1 = np.zeros((H, W), dtype=bool)
        m2 = np.zeros((H, W), dtype=bool)
        m1[0:4, 0:4] = True   # top-left quadrant
        m2[4:8, 4:8] = True   # bottom-right quadrant
        result = _union_masks([m1, m2], H, W)
        # Top-left and bottom-right should be 255
        assert result[0, 0] == 255
        assert result[7, 7] == 255
        # Top-right should be 0
        assert result[0, 7] == 0
        # Bottom-left should be 0
        assert result[7, 0] == 0

    def test_overlapping_masks_yield_255(self):
        from contour_reg.segmentation.sam3_segmenter import _union_masks
        m1 = np.ones((5, 5), dtype=bool)
        m2 = np.zeros((5, 5), dtype=bool)
        m2[2, 2] = True
        result = _union_masks([m1, m2], 5, 5)
        assert (result == 255).all()

    def test_values_are_0_or_255_only(self):
        """uint8 output contains only 0 and 255, nothing in between."""
        from contour_reg.segmentation.sam3_segmenter import _union_masks
        masks = [np.eye(6, dtype=bool)]
        result = _union_masks(masks, 6, 6)
        unique_vals = set(result.flat)
        assert unique_vals <= {0, 255}

    def test_union_accepts_torch_tensors(self):
        """Regression: masks may arrive as torch tensors (CUDA in production).

        Passing torch tensors must exercise the .detach().cpu().numpy() path and
        produce the correct 0/255 numpy mask.  CPU tensors suffice to catch a
        missing CPU/numpy conversion (np.asarray(tensor) on a CUDA tensor raises).
        """
        torch = pytest.importorskip("torch")
        from contour_reg.segmentation.sam3_segmenter import _union_masks

        H, W = 8, 8
        t1 = torch.zeros((H, W), dtype=torch.bool)
        t2 = torch.zeros((H, W), dtype=torch.bool)
        t1[0:4, 0:4] = True   # top-left quadrant
        t2[4:8, 4:8] = True   # bottom-right quadrant

        result = _union_masks([t1, t2], H, W)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert set(result.flat) <= {0, 255}
        assert result[0, 0] == 255
        assert result[7, 7] == 255
        assert result[0, 7] == 0
        assert result[7, 0] == 0

    def test_union_squeezes_leading_singleton_tensor(self):
        """A (1, H, W) torch mask is squeezed to (H, W) before union."""
        torch = pytest.importorskip("torch")
        from contour_reg.segmentation.sam3_segmenter import _union_masks

        H, W = 5, 5
        t = torch.zeros((1, H, W), dtype=torch.bool)
        t[0, 2, 2] = True
        result = _union_masks([t], H, W)
        assert result.shape == (H, W)
        assert result[2, 2] == 255
        assert result[0, 0] == 0


# ---------------------------------------------------------------------------
# segment_image: monkeypatched processor, no weights loaded
# ---------------------------------------------------------------------------

class TestSegmentImage:
    """segment_image must union instance masks returned by the processor."""

    def _make_fake_processor_and_outputs(self, H: int, W: int, instance_masks: List[np.ndarray]):
        """Build a fake processor whose post_process returns known masks."""
        outputs_stub = object()  # sentinel, content irrelevant

        def fake_post_process(outputs, threshold, mask_threshold, target_sizes):
            # Convert bool arrays to torch-like objects (just np arrays work too)
            return [{"masks": instance_masks, "boxes": [], "scores": []}]

        fake_processor = SimpleNamespace(
            __call__=MagicMock(return_value={"pixel_values": MagicMock()}),
            post_process_instance_segmentation=fake_post_process,
        )
        fake_processor.__call__.return_value = {"pixel_values": MagicMock(to=MagicMock())}

        return fake_processor, outputs_stub

    def _make_callable_processor(self, instance_masks):
        """Return a MagicMock processor that is callable and has post_process."""
        inputs_dict = {"pixel_values": MagicMock(**{"to.return_value": MagicMock()})}
        proc = MagicMock()
        proc.return_value = inputs_dict  # processor(images=...) -> inputs_dict

        def fake_post_process(outputs, threshold, mask_threshold, target_sizes):
            return [{"masks": instance_masks, "boxes": [], "scores": []}]

        proc.post_process_instance_segmentation = fake_post_process
        return proc

    def test_two_instance_masks_are_unioned(self, tmp_path):
        """segment_image correctly unions 2 instance masks via _union_masks."""
        from PIL import Image as PILImage

        from contour_reg.segmentation.sam3_segmenter import segment_image

        H, W = 16, 20
        m1 = np.zeros((H, W), dtype=bool)
        m1[0:8, 0:10] = True  # top-left half
        m2 = np.zeros((H, W), dtype=bool)
        m2[8:16, 10:20] = True  # bottom-right half

        fake_processor = self._make_callable_processor([m1, m2])
        fake_model = MagicMock(return_value=SimpleNamespace())
        pil_img = PILImage.new("RGB", (W, H), color=(128, 128, 128))

        result = segment_image(
            pil_img,
            "object",
            fake_model,
            fake_processor,
            "cpu",
            threshold=0.5,
            mask_threshold=0.5,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (H, W)
        assert result.dtype == np.uint8
        # Top-left area should be 255 (from m1)
        assert result[0, 0] == 255
        # Bottom-right area should be 255 (from m2)
        assert result[15, 19] == 255
        # Top-right area should be 0 (neither mask covers it)
        assert result[0, 19] == 0

    def test_no_detections_returns_zero_mask(self, tmp_path):
        """When no instance masks are found, a zero mask is returned."""
        from PIL import Image as PILImage

        from contour_reg.segmentation.sam3_segmenter import segment_image

        H, W = 10, 10
        fake_processor = self._make_callable_processor([])  # empty masks
        fake_model = MagicMock(return_value=SimpleNamespace())
        pil_img = PILImage.new("RGB", (W, H))

        result = segment_image(pil_img, "object", fake_model, fake_processor, "cpu")

        assert result.shape == (H, W)
        assert result.sum() == 0


# ---------------------------------------------------------------------------
# 'contour-reg segment --help': no model, no network
# ---------------------------------------------------------------------------

class TestSegmentSubcommandHelp:
    def test_segment_help_lists_required_flags(self, capsys):
        from contour_reg.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["segment", "--help"])
        assert exc_info.value.code == 0
        output = capsys.readouterr().out
        for flag in ["--images", "--prompt", "--output", "--checkpoint"]:
            assert flag in output, f"'{flag}' not found in segment --help output"

    def test_segment_help_lists_optional_flags(self, capsys):
        from contour_reg.cli import main
        with pytest.raises(SystemExit):
            main(["segment", "--help"])
        out = capsys.readouterr().out
        for flag in ["--resize-width", "--force", "--threshold", "--device"]:
            assert flag in out, f"'{flag}' not found in segment --help output"


# ---------------------------------------------------------------------------
# load_sam3 gated-error handling: monkeypatched from_pretrained
# ---------------------------------------------------------------------------

class TestLoadSam3GatedError:
    def test_gated_401_raises_runtime_error_with_url(self):
        """A 401-style error from from_pretrained should be re-raised as RuntimeError."""
        from contour_reg.segmentation import sam3_segmenter

        gated_exc = Exception("401 Client Error: Unauthorized for gated repo")

        with patch.dict(
            "sys.modules",
            {"transformers": MagicMock(
                Sam3Model=MagicMock(
                    from_pretrained=MagicMock(side_effect=gated_exc)
                ),
                Sam3Processor=MagicMock(
                    from_pretrained=MagicMock(side_effect=gated_exc)
                ),
            )},
        ):
            with pytest.raises(RuntimeError) as exc_info:
                sam3_segmenter.load_sam3("facebook/sam3", device="cpu")

        msg = str(exc_info.value)
        assert "request access" in msg.lower() or "huggingface.co/facebook/sam3" in msg

    def test_gated_403_raises_runtime_error(self):
        """A 403 forbidden error is also handled."""
        from contour_reg.segmentation import sam3_segmenter

        gated_exc = Exception("403 Forbidden: access restricted to gated model")

        with patch.dict(
            "sys.modules",
            {"transformers": MagicMock(
                Sam3Model=MagicMock(
                    from_pretrained=MagicMock(side_effect=gated_exc)
                ),
                Sam3Processor=MagicMock(
                    from_pretrained=MagicMock(side_effect=gated_exc)
                ),
            )},
        ):
            with pytest.raises(RuntimeError) as exc_info:
                sam3_segmenter.load_sam3("facebook/sam3", device="cpu")

        assert "https://huggingface.co/facebook/sam3" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 'contour-reg segment' without the [segmentation] extra
# ---------------------------------------------------------------------------

def _hide_segmentation_deps(monkeypatch):
    """Simulate an install without torch/transformers."""
    from contour_reg.segmentation import sam3_segmenter

    monkeypatch.setattr(sam3_segmenter, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)  # makes 'import transformers' raise ImportError


class TestSegmentWithoutExtra:
    def test_segment_directory_raises_import_error(self, tmp_path, monkeypatch):
        from contour_reg.segmentation.sam3_segmenter import segment_directory

        _hide_segmentation_deps(monkeypatch)
        with pytest.raises(ImportError, match="segmentation"):
            segment_directory(tmp_path, "object", tmp_path / "out")

    def test_load_sam3_raises_import_error_even_with_device(self, monkeypatch):
        from contour_reg.segmentation.sam3_segmenter import load_sam3

        _hide_segmentation_deps(monkeypatch)
        with pytest.raises(ImportError, match="segmentation"):
            load_sam3(device="cpu")

    def test_cli_exits_1_with_install_hint(self, tmp_path, monkeypatch, capsys):
        from contour_reg.cli import main

        _hide_segmentation_deps(monkeypatch)
        code = main(["segment", "--images", str(tmp_path), "--prompt", "object",
                     "--output", str(tmp_path / "out")])
        assert code == 1
        err = capsys.readouterr().err
        assert "uv sync --extra segmentation" in err
        assert "Traceback" not in err
