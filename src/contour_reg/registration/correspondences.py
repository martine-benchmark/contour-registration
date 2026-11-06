"""Silhouette extraction and contour-ICP correspondence machinery.

Ported verbatim from ``utils_optimized.py`` (lines 33-576).
Import rewiring: Camera → contour_reg.io.camera; apply_transform_to_mesh →
contour_reg.io.mesh.  No imports from utils / utils_optimized / pyalicevisionlib.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
import trimesh
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

from contour_reg.io.camera import Camera

# =============================================================================
# Global cache for mesh structures (avoids recomputation)
# =============================================================================

class MeshStructureCache:
    """Cache for mesh structures that are expensive to compute.

    Avoids recomputing edge_to_faces, face_adjacency, etc. at every iteration.
    """

    def __init__(self):
        self._cache = {}

    def _mesh_key(self, mesh: trimesh.Trimesh) -> str:
        """Build a unique key from the mesh geometry (not the pose)."""
        return f"{len(mesh.vertices)}_{len(mesh.faces)}_{id(mesh.faces.data)}"

    def get_edge_to_faces(self, mesh: trimesh.Trimesh) -> Dict:
        """Return the edge -> faces mapping, from the cache or computed."""
        key = self._mesh_key(mesh) + "_edge_to_faces"

        if key not in self._cache:
            edge_to_faces = {}
            faces = mesh.faces
            for face_idx in range(len(faces)):
                face = faces[face_idx]
                for i in range(3):
                    v0, v1 = int(face[i]), int(face[(i + 1) % 3])
                    edge_key = (min(v0, v1), max(v0, v1))
                    if edge_key not in edge_to_faces:
                        edge_to_faces[edge_key] = []
                    edge_to_faces[edge_key].append(face_idx)
            self._cache[key] = edge_to_faces

        return self._cache[key]

    def get_adjacency_set(self, mesh: trimesh.Trimesh) -> set:
        """Return the set of edges shared between two faces."""
        key = self._mesh_key(mesh) + "_adjacency_set"

        if key not in self._cache:
            face_adjacency_edges = mesh.face_adjacency_edges
            adjacency_set = set(map(tuple, np.sort(face_adjacency_edges, axis=1)))
            self._cache[key] = adjacency_set

        return self._cache[key]

    def get_border_edges(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Return the border edges (not shared)."""
        key = self._mesh_key(mesh) + "_border_edges"

        if key not in self._cache:
            adjacency_set = self.get_adjacency_set(mesh)
            all_edges = mesh.edges_unique
            all_edges_sorted = np.sort(all_edges, axis=1)
            border_mask = np.array(
                [tuple(e) not in adjacency_set for e in all_edges_sorted]
            )
            self._cache[key] = all_edges[border_mask] if np.any(border_mask) else np.array([])

        return self._cache[key]

    def clear(self):
        """Clear the cache."""
        self._cache.clear()


# Global instance of the structure cache
_mesh_cache = MeshStructureCache()


def clear_mesh_cache() -> None:
    """Clear the mesh structure cache."""
    _mesh_cache.clear()


# =============================================================================
# Helpers
# =============================================================================

def subsample_indices(indices: np.ndarray, max_points: int = 2000) -> np.ndarray:
    """Subsample an index array to limit the number of points."""
    if len(indices) <= max_points:
        return indices
    step = len(indices) // max_points
    return indices[::step]


# =============================================================================
# Vectorized mask rendering
# =============================================================================

def render_mesh_mask_fast(mesh: trimesh.Trimesh, camera: Camera) -> np.ndarray:
    """Render an object mask by rasterization.

    Ported from ``utils_optimized.py:render_mesh_mask_fast`` (line 237).
    """
    width, height = camera.width, camera.height

    # Project all mesh vertices (once)
    vertices_3d = mesh.vertices
    projected_2d = camera.project_points(vertices_3d)

    # Back-face culling
    face_centers = mesh.triangles_center
    view_vectors = camera.position - face_centers
    view_vectors_normalized = view_vectors / (
        np.linalg.norm(view_vectors, axis=1, keepdims=True) + 1e-8
    )
    dot_products = np.sum(mesh.face_normals * view_vectors_normalized, axis=1)
    front_facing_mask = dot_products > 0

    visible_faces = mesh.faces[front_facing_mask]

    if len(visible_faces) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    # Get the projected triangles of the visible faces
    triangles_2d = projected_2d[visible_faces]  # (N, 3, 2)

    # Filter out triangles entirely outside the image
    min_coords = triangles_2d.min(axis=1)  # (N, 2)
    max_coords = triangles_2d.max(axis=1)  # (N, 2)

    in_image = (
        (max_coords[:, 0] >= 0) & (min_coords[:, 0] < width) &
        (max_coords[:, 1] >= 0) & (min_coords[:, 1] < height)
    )

    valid_triangles = triangles_2d[in_image].astype(np.int32)

    if len(valid_triangles) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    # Create the mask: a single fillPoly call with all triangles
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, valid_triangles, 255)

    return mask


# =============================================================================
# Optimized silhouette extraction
# =============================================================================

def filter_occluding_contours_fast(
    mesh: trimesh.Trimesh,
    vertices_indices: np.ndarray,
    camera: Camera,
    distance_threshold: float = 1.5,
    include_inner_contours: bool = False,
) -> np.ndarray:
    """Filter silhouette vertices to keep only those on the visible contour.

    Ported from ``utils_optimized.py:filter_occluding_contours_fast`` (line 398).
    ``include_inner_contours`` controls ``cv2.RETR_TREE`` vs ``cv2.RETR_EXTERNAL``
    (source line 417).
    """
    if len(vertices_indices) == 0:
        return vertices_indices

    mask = render_mesh_mask_fast(mesh, camera)

    # Extract the mask contour
    retrieval_mode = cv2.RETR_TREE if include_inner_contours else cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours(mask, retrieval_mode, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.array([], dtype=vertices_indices.dtype)

    # Contour image
    contour_img = np.zeros_like(mask)
    cv2.drawContours(contour_img, contours, -1, 255, thickness=1)

    # Distance transform
    dist_transform = cv2.distanceTransform(255 - contour_img, cv2.DIST_L2, 5)

    # Project the vertices
    vertices = mesh.vertices[vertices_indices]
    projected_2d = camera.project_points(vertices)

    # Vectorized filtering
    xi = np.round(projected_2d[:, 0]).astype(np.int32)
    yi = np.round(projected_2d[:, 1]).astype(np.int32)

    # Validity mask (inside the image)
    in_bounds = (
        (xi >= 0) & (xi < camera.width) &
        (yi >= 0) & (yi < camera.height)
    )

    valid_mask = np.zeros(len(vertices_indices), dtype=bool)
    valid_mask[in_bounds] = dist_transform[yi[in_bounds], xi[in_bounds]] <= distance_threshold

    return vertices_indices[valid_mask]


def extract_silhouette_edges_fast(
    mesh: trimesh.Trimesh,
    camera: Camera,
    filter_occluding: bool = True,
    include_inner_contours: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the silhouette vertices.

    Ported from ``utils_optimized.py:extract_silhouette_edges_fast`` (line 299).

    Args:
        mesh: Trimesh mesh.
        camera: Camera object.
        filter_occluding: If True, filter out self-occluding contours.
        include_inner_contours: If True, include inner contours (holes).

    Returns:
        (points_2d, indices_3d): Projected 2D points and 3D vertex indices.
    """
    # Compute the face normals if needed
    if mesh.face_normals is None or len(mesh.face_normals) == 0:
        mesh.fix_normals()

    # Face centers
    face_centers = mesh.triangles_center

    # Vectors from the centers to the camera (vectorized)
    view_vectors = camera.position - face_centers
    norms = np.linalg.norm(view_vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    view_vectors_normalized = view_vectors / norms

    # Vectorized dot product
    dot_products = np.einsum('ij,ij->i', mesh.face_normals, view_vectors_normalized)
    front_facing = dot_products > 0

    # Find the silhouette edges
    face_adjacency = mesh.face_adjacency
    face_adjacency_edges = mesh.face_adjacency_edges

    silhouette_edges_list = []

    # 1. Classic silhouette edges (XOR of the visibility)
    if len(face_adjacency) > 0:
        face1_visible = front_facing[face_adjacency[:, 0]]
        face2_visible = front_facing[face_adjacency[:, 1]]
        silhouette_mask = face1_visible != face2_visible
        if np.any(silhouette_mask):
            silhouette_edges_list.append(face_adjacency_edges[silhouette_mask])

    # 2. Border edges, using the global cache
    if len(face_adjacency_edges) > 0:
        border_edges = _mesh_cache.get_border_edges(mesh)

        if len(border_edges) > 0:
            edge_to_faces = _mesh_cache.get_edge_to_faces(mesh)

            valid_border = []
            for edge in border_edges:
                edge_key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
                face_indices = edge_to_faces.get(edge_key, [])
                if any(front_facing[f] for f in face_indices):
                    valid_border.append(edge)

            if valid_border:
                silhouette_edges_list.append(np.array(valid_border))

    # Combine
    if not silhouette_edges_list:
        return np.array([]), np.array([])

    silhouette_edges = np.vstack(silhouette_edges_list)
    silhouette_vertex_indices = np.unique(silhouette_edges.flatten())

    # Filter self-occluding contours
    if filter_occluding and len(silhouette_vertex_indices) > 0:
        silhouette_vertex_indices = filter_occluding_contours_fast(
            mesh, silhouette_vertex_indices, camera,
            include_inner_contours=include_inner_contours,
        )

    if len(silhouette_vertex_indices) == 0:
        return np.array([]), np.array([])

    silhouette_vertices = mesh.vertices[silhouette_vertex_indices]
    projected_2d = camera.project_points(silhouette_vertices)

    # Filter out of image
    valid_mask = (
        (projected_2d[:, 0] >= 0) & (projected_2d[:, 0] < camera.width) &
        (projected_2d[:, 1] >= 0) & (projected_2d[:, 1] < camera.height)
    )

    return projected_2d[valid_mask], silhouette_vertex_indices[valid_mask]


# =============================================================================
# Parallel processing of views
# =============================================================================

def process_single_view(
    cam_idx: int,
    mesh_vertices: np.ndarray,
    camera: Camera,
    target_contour: np.ndarray,
    indices_3d: np.ndarray,
) -> Tuple[int, float, np.ndarray]:
    """Process a single view for the error computation.

    Ported from ``utils_optimized.py:process_single_view`` (line 451).

    Returns:
        (cam_idx, mean_error, distances)
    """
    if len(indices_3d) == 0 or len(target_contour) == 0:
        return cam_idx, 0.0, np.array([])

    vertices_3d = mesh_vertices[indices_3d]
    projected_2d = camera.project_points(vertices_3d)

    # Distance from each contour point to its nearest projected mesh-silhouette point.
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    nn.fit(projected_2d)
    distances, _ = nn.kneighbors(target_contour)

    return cam_idx, np.mean(distances), distances.flatten()


def compute_correspondences_parallel(
    mesh: trimesh.Trimesh,
    cameras: List[Camera],
    target_contours: List[np.ndarray],
    n_jobs: int = -1,
    include_inner_contours: bool = False,
) -> Dict:
    """Compute the correspondences for all views in parallel.

    Ported from ``utils_optimized.py:compute_correspondences_parallel`` (line 476).

    Args:
        mesh: Current mesh (already transformed).
        cameras: List of cameras.
        target_contours: List of target contours.
        n_jobs: Number of parallel jobs (-1 = all cores).
        include_inner_contours: If True, include inner contours (holes).

    Returns:
        Dict[camera_idx] -> {'indices_3d': array, 'points_2d': array}
    """

    def process_view(cam_idx: int):
        camera = cameras[cam_idx]
        target_contour = target_contours[cam_idx]

        projected_2d, silhouette_indices = extract_silhouette_edges_fast(
            mesh, camera, include_inner_contours=include_inner_contours
        )

        if len(projected_2d) == 0 or len(target_contour) == 0:
            return None

        return cam_idx, {
            'indices_3d': silhouette_indices,
            'points_2d': target_contour,
        }

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_view)(i) for i in range(len(cameras))
    )

    correspondences: Dict = {}
    for result in results:
        if result is not None:
            cam_idx, corr = result
            correspondences[cam_idx] = corr

    return correspondences


def compute_error_parallel(
    mesh_vertices: np.ndarray,
    cameras: List[Camera],
    fixed_correspondences: Dict,
    n_jobs: int = -1,
    return_distances: bool = False,
) -> Tuple:
    """Compute the mean error over all views in parallel.

    Ported from ``utils_optimized.py:compute_error_parallel`` (line 526).

    Returns:
        (mean_error, per_view_errors)                             if return_distances=False
        (mean_error, per_view_errors, per_view_distances)         if return_distances=True
    """
    view_data = []
    for cam_idx in fixed_correspondences:
        corr = fixed_correspondences[cam_idx]
        view_data.append((
            cam_idx,
            mesh_vertices,
            cameras[cam_idx],
            corr['points_2d'],
            corr['indices_3d'],
        ))

    if not view_data:
        if return_distances:
            return 1e6, {}, {}
        return 1e6, {}

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_single_view)(*data) for data in view_data
    )

    per_view_errors: Dict = {}
    per_view_distances: Dict = {}
    total_error = 0.0
    count = 0

    for cam_idx, error, distances in results:
        if distances.size > 0:
            per_view_errors[cam_idx] = error
            per_view_distances[cam_idx] = distances
            total_error += error
            count += 1

    mean_error = total_error / count if count > 0 else 1e6
    if return_distances:
        return mean_error, per_view_errors, per_view_distances
    return mean_error, per_view_errors
