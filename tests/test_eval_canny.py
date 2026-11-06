"""Tests for Canny-config evaluation and ranking (silhouette injected)."""
import numpy as np

from contour_reg.contour_metrics import view_counts
from contour_reg.pipeline.eval_canny import (
    c_trim,
    config_view_counts,
    eval_configs,
    trim_contour,
)


def test_c_trim_absolute_cutoff():
    silhouette = np.array([[0.0, 0.0], [10.0, 0.0]])
    contour = np.array([[0.0, 1.0], [10.0, 3.0], [50.0, 50.0]])  # dists 1, 3, big
    kept = c_trim(contour, silhouette, epsilon=2.0)
    assert len(kept) == 1 and np.allclose(kept[0], [0.0, 1.0])
    assert len(c_trim(contour, np.empty((0, 2)), 2.0)) == 0  # empty silhouette


def test_trim_contour_keeps_closest_fraction():
    silhouette = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    # 4 on-silhouette points (dist 0) + 4 far ones
    contour = np.vstack([silhouette, silhouette + [0.0, 50.0]])
    kept = trim_contour(contour, silhouette, percentile=50.0)
    assert len(kept) == 4
    assert np.allclose(kept, silhouette)


def test_trim_contour_percentile_100_keeps_all():
    contour = np.array([[0.0, 0.0], [100.0, 100.0]])
    sil = np.array([[0.0, 0.0]])
    assert len(trim_contour(contour, sil, percentile=100.0)) == 2


def test_config_view_counts_with_injected_silhouette():
    # A real tiny mesh (apply_transform needs one); the silhouette_fn is mocked.
    import trimesh
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    sil = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    def fake_sil(_m, _cam):
        return sil, np.arange(len(sil))

    cams = [object(), object()]  # unused by fake_sil
    contours = [sil.copy(), np.vstack([sil, [[80.0, 80.0]]])]
    views = config_view_counts(
        mesh, cams, contours, pose=np.zeros(6), scale=1.0,
        trim_percentile=75.0, silhouette_fn=fake_sil,
    )
    assert len(views) == 2
    # view 0: contour == silhouette -> perfect recall/precision
    assert views[0].n_S == 3 and views[0].n_C == 3


def test_eval_configs_selects_best_fbeta():
    good = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    clean = [view_counts(good, good, good)]
    noisy = [view_counts(
        good,
        np.vstack([good, [[70.0, 70.0], [80.0, 80.0]]]),
        np.vstack([good, [[70.0, 70.0]]]),
    )]
    out = eval_configs(
        {"clean": clean, "noisy": noisy},
        epsilons=(1.0, 2.0), select_epsilon=2.0,
    )
    assert out["best"] == "clean"
    assert "eps_2" in out["configs"]["clean"]
    assert out["scores"]["clean"] > out["scores"]["noisy"]


def test_eval_configs_empty():
    out = eval_configs({}, epsilons=(2.0,), select_epsilon=2.0)
    assert out["best"] is None


def test_eval_configs_inserts_missing_select_epsilon():
    """select_epsilon absent from epsilons must not KeyError; it is evaluated."""
    good = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out = eval_configs(
        {"a": [view_counts(good, good, good)]},
        epsilons=(1.0, 3.0, 5.0), select_epsilon=2.0,
    )
    assert out["best"] == "a"
    assert "eps_2" in out["configs"]["a"]        # inserted
    assert out["scores"]["a"] == out["configs"]["a"]["eps_2"]["F_beta"]


def test_config_view_counts_length_mismatch_raises():
    import trimesh
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    with np.testing.assert_raises(ValueError):
        config_view_counts(mesh, [object()], [np.zeros((1, 2)), np.zeros((1, 2))],
                           pose=np.zeros(6), scale=1.0,
                           silhouette_fn=lambda m, c: (np.zeros((1, 2)), np.array([0])))


def test_eval_configs_reports_f1_ranking():
    good = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out = eval_configs({"a": [view_counts(good, good, good)]}, epsilons=(2.0,), select_epsilon=2.0)
    assert out["beta"] == 0.5 and out["best_f1"] == "a" and "a" in out["scores_f1"]
