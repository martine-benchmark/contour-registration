import numpy as np
import trimesh

from contour_reg.io.camera import Camera
from contour_reg.registration.correspondences import extract_silhouette_edges_fast
from contour_reg.registration.icp import contour_based_icp

# Final translation must land within this distance of GT (= zero). Observed
# converged norm is ~3e-11; 1e-3 (1 mm) recovers >98% of the 50 mm offset while
# staying non-flaky across BLAS/platform variation.
CONVERGED_T = 1e-3


def _cams():
    cams = []
    for ang in (0.0, 1.2, 2.4):
        R = trimesh.transformations.rotation_matrix(ang, [0,1,0])[:3,:3]
        pos = R @ np.array([0,0,-3.0])
        cams.append(Camera(position=pos, rotation_world2cam=R.T,
            focal_length=800.0, principal_point=np.array([320.0,240.0]),
            width=640, height=480))
    return cams


def _targets(mesh, cams):
    return [extract_silhouette_edges_fast(mesh, c)[0] for c in cams]


def test_icp_recovers_small_translation():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    cams = _cams()
    # target = silhouettes at GT (identity) pose
    targets = _targets(mesh, cams)
    init = np.array([0,0,0, 0.05,0.0,0.0])  # 5cm off
    pose, hist, _ = contour_based_icp(mesh, init, cams, targets,
                                      max_iterations=30, verbose=False)
    final_t = np.linalg.norm(pose[3:6])
    assert final_t < np.linalg.norm(init[3:6])           # moved toward GT
    assert final_t < CONVERGED_T                          # actually converged to GT
    assert hist[-1] <= hist[0]


def test_icp_with_max_correspondence_distance():
    # Exercises the Phase-1 per-view distance-filter branch (icp.py).
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    cams = _cams()
    targets = _targets(mesh, cams)
    init = np.array([0,0,0, 0.05,0.0,0.0])  # 5cm off
    pose, hist, _ = contour_based_icp(mesh, init, cams, targets,
                                      max_correspondence_distance=5.0,
                                      max_iterations=10, verbose=False)
    assert np.isfinite(hist[-1])                          # finite final error
    assert np.linalg.norm(pose[3:6]) < np.linalg.norm(init[3:6])  # moved toward GT


def test_icp_with_keep_pct():
    # Exercises the epsilon-cutoff + top keep_pct% branch (icp.py Phase-1 filter).
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    cams = _cams()
    targets = _targets(mesh, cams)
    init = np.array([0,0,0, 0.05,0.0,0.0])  # 5cm off
    pose, hist, _ = contour_based_icp(mesh, init, cams, targets,
                                      max_correspondence_distance=5.0, keep_pct=80.0,
                                      max_iterations=10, verbose=False)
    assert np.isfinite(hist[-1])                          # finite final error
    assert np.linalg.norm(pose[3:6]) < np.linalg.norm(init[3:6])  # moved toward GT


def test_icp_optimize_scale_returns_scale():
    # Exercises the optimize_scale branch (residual_function_with_scale path).
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    cams = _cams()
    targets = _targets(mesh, cams)
    init = np.array([0,0,0, 0.02,0.0,0.0])  # small perturbation
    pose, hist, final_scale = contour_based_icp(mesh, init, cams, targets,
                                                optimize_scale=True, initial_scale=1.0,
                                                max_iterations=10, verbose=False)
    assert isinstance(final_scale, float)                 # sane scalar (np.float64 is a float)
    assert np.isfinite(final_scale) and final_scale > 0   # positive, finite
    assert np.isfinite(hist[-1])


def test_default_trim_percentile_is_75():
    import inspect

    from contour_reg.config import ICPConfig
    assert inspect.signature(contour_based_icp).parameters["error_threshold_percentile"].default == 75.0
    assert ICPConfig().error_percentile == 75.0


def test_trimmed_phase_invariant_to_error_offset(monkeypatch):
    # The trimmed error lives on its own scale (another correspondence set): adding a
    # constant to it must not change the returned pose.
    import contour_reg.registration.icp as icp_mod
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    cams = _cams()
    rng = np.random.default_rng(0)
    targets = []
    for S in _targets(mesh, cams):
        c = S.mean(0)
        clutter = c + (S[rng.choice(len(S), len(S) // 3)] - c) * rng.uniform(0.2, 0.7, (len(S) // 3, 1))
        targets.append(np.vstack([S, clutter]))
    init = np.array([0, 0, 0, 0.03, 0.0, 0.0])

    def run():
        return contour_based_icp(mesh, init, cams, targets, error_threshold_percentile=75.0,
                                 optimize_scale=True, initial_scale=1.0,
                                 max_iterations=15, verbose=False)

    ref_pose, _, ref_scale = run()
    trimmed_ids = set()
    refine, error = icp_mod._refine_correspondences, icp_mod.compute_error_parallel

    def tagged_refine(*a, **k):
        out = refine(*a, **k)
        trimmed_ids.add(id(out))
        return out

    def offset_error(verts, cams_, corr, *a, **k):
        res = error(verts, cams_, corr, *a, **k)
        return (res[0] + 100.0,) + tuple(res[1:]) if id(corr) in trimmed_ids else res

    monkeypatch.setattr(icp_mod, "_refine_correspondences", tagged_refine)
    monkeypatch.setattr(icp_mod, "compute_error_parallel", offset_error)
    pose, _, scale = run()
    np.testing.assert_allclose(pose, ref_pose, atol=1e-9)
    assert np.isclose(scale, ref_scale)


def test_icp_rejects_camera_contour_count_mismatch():
    import pytest
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    cams = _cams()
    targets = _targets(mesh, cams)
    with pytest.raises(ValueError, match="same length"):
        contour_based_icp(mesh, np.zeros(6), cams[:2], targets, max_iterations=1, verbose=False)


def test_cutoff_step_accepted_only_if_better_on_same_points(monkeypatch):
    # Clutter beyond the cutoff inflates the error over ALL points; a step that
    # worsens the fit on the kept points must still be rejected.
    import contour_reg.registration.icp as icp_mod
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    cams = _cams()
    targets = [np.vstack([S, S + 80.0]) for S in _targets(mesh, cams)]   # clutter 80+ px away
    init = np.zeros(6)
    real_ls = icp_mod.least_squares

    def worse_step(fun, x0, *a, **k):
        r = real_ls(fun, x0, *a, **{**k, "max_nfev": 1})
        r.x = np.asarray(x0, float) + np.array([0, 0, 0, 0.02, 0, 0])   # 2 cm off: worse
        return r

    monkeypatch.setattr(icp_mod, "least_squares", worse_step)
    pose, _, _ = contour_based_icp(mesh, init, cams, targets, error_threshold_percentile=100.0,
                                   max_correspondence_distance=5.0, max_iterations=5, verbose=False)
    np.testing.assert_allclose(pose, init, atol=1e-12)
