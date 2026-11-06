"""Mesh IO utilities: load, transform, and decimate trimesh meshes."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh from *path*, forcing a single Trimesh."""
    return trimesh.load(str(path), force='mesh')


def apply_transform_to_mesh(
    mesh: trimesh.Trimesh,
    euler_xyz: np.ndarray,
    translation: np.ndarray,
) -> trimesh.Trimesh:
    """Apply rotation (euler_xyz) then translation to *mesh*, returning a new Trimesh.

    Ported verbatim from ``utils.py::apply_transform_to_mesh``.
    The input mesh is never mutated.
    """
    mesh_transformed = mesh.copy()
    rot_matrix = Rotation.from_euler('xyz', euler_xyz).as_matrix()
    mesh_transformed.vertices = (rot_matrix @ mesh_transformed.vertices.T).T
    mesh_transformed.vertices += translation
    return mesh_transformed


def apply_transform_to_mesh_with_scale(
    mesh: trimesh.Trimesh,
    euler_xyz: np.ndarray,
    translation: np.ndarray,
    scale: float,
) -> trimesh.Trimesh:
    """Apply uniform scale, rotation (euler_xyz), then translation to *mesh*.

    The transform is ``T(v) = scale * R @ v + t``.
    Ported verbatim from ``utils.py::apply_transform_to_mesh_with_scale``.
    The input mesh is never mutated.
    """
    mesh_transformed = mesh.copy()
    rot_matrix = Rotation.from_euler('xyz', euler_xyz).as_matrix()
    mesh_transformed.vertices = scale * (rot_matrix @ mesh_transformed.vertices.T).T
    mesh_transformed.vertices += translation
    return mesh_transformed


def decimate_mesh(mesh: trimesh.Trimesh, target_vertices: int) -> trimesh.Trimesh:
    """Decimate *mesh* to approximately *target_vertices* vertices.

    Uses trimesh quadric decimation when available.  Converts the vertex target
    to a topology-aware proportional face count (matching ``decimate_mesh_for_icp``).
    Falls back gracefully if the decimation API is unavailable or raises, returning
    the original mesh with a logged warning.

    The result always has ``<= len(mesh.vertices)`` vertices.
    """
    current_vertices = max(1, len(mesh.vertices))
    if current_vertices <= target_vertices:
        return mesh

    # Topology-aware proportional face target (matches decimate_mesh_for_icp)
    current_faces = len(mesh.faces)
    target_faces = max(100, int(current_faces * (target_vertices / current_vertices)))

    # Try trimesh quadric decimation: API varies across versions
    try:
        try:
            decimated = mesh.simplify_quadric_decimation(face_count=target_faces)
        except TypeError:
            # Older trimesh uses a positional argument
            decimated = mesh.simplify_quadric_decimation(target_faces)

        if decimated is not None and len(decimated.vertices) > 0:
            return decimated
        logger.warning("decimate_mesh: trimesh returned empty mesh; using original")
        return mesh
    except Exception as exc:
        logger.warning("decimate_mesh: trimesh decimation failed (%s); returning original mesh", exc)
        return mesh
