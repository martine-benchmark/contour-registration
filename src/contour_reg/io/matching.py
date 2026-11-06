"""Match per-view files (masks, images, contours) to the cameras by file name.

Each loader that knows the image of a view sets ``camera.name`` to the stem of
that image file (Meshroom: the view ``path``; COLMAP: the image name). A file
``<stem>.<any extension>`` then belongs to the camera named ``<stem>``:
``IMG_0001.JPG`` (view) ↔ ``IMG_0001.png`` (mask). The result follows the camera
order; ambiguous or missing matches raise :class:`MatchingError`.

Cameras without names (IDR ``cameras.npz`` has no image names) are matched to
the files in sorted file-name order, which then must have the same count.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_SHOW = 5  # names listed in error messages


class MatchingError(ValueError):
    """Per-view files cannot be matched unambiguously to the cameras."""


def _preview(items: Sequence[str]) -> str:
    items = list(items)
    more = f", … (+{len(items) - _SHOW})" if len(items) > _SHOW else ""
    return ", ".join(items[:_SHOW]) + more


def view_name(path: str) -> str:
    """Stem of an image path, for POSIX or Windows separators."""
    return Path(path.replace("\\", "/")).stem


def match_by_name(paths: Sequence[Path], names: Sequence[Optional[str]], what: str = "mask",
                  owner: str = "view") -> List[Path]:
    """Return one path per entry of *names*, matched by file stem.

    Args:
        paths: Candidate files (any order).
        names: Expected stem per view, in the order of the result; all ``None``
            means the views have no names and *paths* (sorted) are used in order.
        what: Kind of file, for messages (``"mask"``, ``"image"``, ``"contour"``).
        owner: What the names belong to, for messages (``"view"``, ``"mask"``).

    Raises:
        MatchingError: Count mismatch (unnamed views), duplicate names, several
            files for one view, or views without a file.
    """
    paths = sorted(Path(p) for p in paths)
    if all(n is None for n in names):
        if len(paths) != len(names):
            raise MatchingError(
                f"Found {len(paths)} {what} file(s) but {len(names)} camera(s), which carry no image "
                f"names: provide exactly one {what} per camera (matched in sorted file-name order)."
            )
        logger.info("Cameras carry no image names: %s files matched in sorted file-name order.", what)
        return paths
    if any(n is None for n in names):
        raise MatchingError(f"Some {owner}s have no name: cannot match {what} files by name.")

    dup_names = sorted(n for n, k in Counter(names).items() if k > 1)
    if dup_names:
        raise MatchingError(f"Ambiguous: several {owner}s share the name(s) {_preview(dup_names)}.")

    by_stem: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        by_stem[p.stem].append(p)
    wanted = set(names)
    dup_files = sorted(s for s in wanted if len(by_stem.get(s, ())) > 1)
    if dup_files:
        examples = _preview([p.name for s in dup_files for p in by_stem[s]])
        raise MatchingError(f"Ambiguous: several {what} files for one {owner} ({examples}).")

    missing = [n for n in names if n not in by_stem]
    if missing:
        rule = (f"{what.capitalize()} files are matched to the {owner}s by file name: {owner} image "
                f"'{names[0]}.<ext>' ↔ {what} '{names[0]}.<ext>'.")
        if len(missing) == len(names):
            raise MatchingError(
                f"No {what} file matches any of the {len(names)} {owner}(s) ({_preview(names)}); "
                f"found {_preview([p.name for p in paths]) or 'none'}. {rule}"
            )
        raise MatchingError(
            f"No {what} file for {len(missing)} of {len(names)} {owner}(s): {_preview(missing)}. {rule}"
        )

    extra = sorted(p.name for s, ps in by_stem.items() if s not in wanted for p in ps)
    if extra:
        logger.warning("Ignoring %d %s file(s) that match no %s: %s", len(extra), what, owner, _preview(extra))
    return [by_stem[n][0] for n in names]


def match_to_cameras(paths: Sequence[Path], cameras: Sequence, what: str = "mask") -> List[Path]:
    """:func:`match_by_name` with the names of *cameras* (``camera.name``, ``None`` if unknown)."""
    return match_by_name(paths, [getattr(c, "name", None) for c in cameras], what)
