import numpy as np
import trimesh

from contour_reg.io.masks import load_contour_points
from contour_reg.io.mesh import apply_transform_to_mesh, decimate_mesh


def test_translation_moves_centroid():
    m = trimesh.creation.box(extents=[1,1,1])
    out = apply_transform_to_mesh(m, np.zeros(3), np.array([1.0,2.0,3.0]))
    assert np.allclose(out.vertices.mean(axis=0), [1,2,3], atol=1e-6)
    assert np.allclose(m.vertices.mean(axis=0), [0, 0, 0], atol=1e-6), "input mesh must not be mutated"

def test_decimate_reduces_vertices():
    m = trimesh.creation.icosphere(subdivisions=4)  # ~2562 verts
    out = decimate_mesh(m, 200)
    assert len(out.vertices) <= len(m.vertices)

def test_load_contour_from_npy(tmp_path):
    pts = np.array([[1.0,2.0],[3.0,4.0]])
    np.save(tmp_path/"c.npy", pts)
    assert np.allclose(load_contour_points(tmp_path/"c.npy"), pts)
