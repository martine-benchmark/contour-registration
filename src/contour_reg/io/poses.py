"""Unified camera-pose loader.

Public entry point:  ``load_poses(path, fmt="auto", **kw) -> List[Camera]``

Format detection (``fmt="auto"``):
- ``.json``  → Meshroom/AliceVision SfM (passes ``**kw``, e.g. ``world_flip``)
- ``.npz``   → IDR (kwargs ignored)
- directory containing ``cameras.txt`` or ``cameras.bin`` → COLMAP (kwargs ignored)
- anything else → :class:`ValueError`

Explicit ``fmt`` values: ``"meshroom"``, ``"colmap"``, ``"idr"``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

from contour_reg.io.camera import Camera
from contour_reg.io.loaders.colmap import load_colmap
from contour_reg.io.loaders.idr import load_idr
from contour_reg.io.loaders.meshroom import load_meshroom

_SUPPORTED = ("meshroom", "colmap", "idr")


def _detect_fmt(path: Path) -> str:
    """Infer format from *path*; raise ValueError when undetermined."""
    if path.suffix.lower() == ".json":
        return "meshroom"
    if path.suffix.lower() == ".npz":
        return "idr"
    if path.is_dir() and (
        (path / "cameras.txt").exists() or (path / "cameras.bin").exists()
    ):
        return "colmap"
    raise ValueError(
        f"Cannot determine camera-pose format for {path!r}. "
        f"Supported: JSON (Meshroom), NPZ (IDR), directory with cameras.txt/.bin (COLMAP). "
        f"Pass fmt= explicitly if the path is unconventional."
    )


def load_poses(path: Union[str, Path], fmt: str = "auto", **kw) -> List[Camera]:
    """Load camera poses from *path* in the requested format.

    Args:
        path: Path to the SfM file or directory. Accepts ``str`` or
            :class:`pathlib.Path`.
        fmt: One of ``"auto"``, ``"meshroom"``, ``"colmap"``, ``"idr"``.
            ``"auto"`` (default) detects the format from the path.
        **kw: Extra keyword arguments forwarded to the underlying loader.
            Only Meshroom accepts kwargs (e.g. ``world_flip=False``).
            Passing kwargs for COLMAP or IDR raises :class:`TypeError`.

    Returns:
        List of :class:`~contour_reg.io.camera.Camera` objects.

    Raises:
        ValueError: When ``fmt="auto"`` and the format cannot be determined,
            or when an unrecognised explicit ``fmt`` is supplied.
        TypeError: When kwargs are supplied for a non-Meshroom format.
    """
    path = Path(path)

    if fmt == "auto":
        fmt = _detect_fmt(path)
    elif fmt not in _SUPPORTED:
        raise ValueError(
            f"Unknown fmt={fmt!r}. Supported values: {_SUPPORTED}."
        )

    if fmt != "meshroom" and kw:
        raise TypeError(
            f"Unexpected keyword arguments for fmt={fmt!r}: {list(kw)}"
        )

    if fmt == "meshroom":
        return load_meshroom(path, **kw)
    if fmt == "colmap":
        return load_colmap(path)
    if fmt == "idr":
        return load_idr(path)
    raise AssertionError(f"Unreachable: {fmt!r}")
