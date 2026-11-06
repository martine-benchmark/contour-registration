"""Tests for contour-vs-silhouette overlap metrics (Canny-config selection)."""
import numpy as np
import pytest

from contour_reg.contour_metrics import (
    aggregate,
    f_beta,
    nn_distances,
    select_best,
    view_counts,
)


def test_nn_distances_basic():
    q = np.array([[0.0, 0.0], [10.0, 0.0]])
    r = np.array([[0.0, 3.0], [10.0, 4.0]])
    d = nn_distances(q, r)
    assert np.allclose(d, [3.0, 4.0])


def test_nn_distances_empty_reference_is_inf():
    d = nn_distances(np.array([[0.0, 0.0]]), np.empty((0, 2)))
    assert d.shape == (1,) and np.isinf(d[0])
    assert nn_distances(np.empty((0, 2)), np.array([[0.0, 0.0]])).shape == (0,)


def test_f_beta_extremes_and_precision_weighting():
    assert f_beta(1.0, 1.0, 0.5) == pytest.approx(1.0)
    assert f_beta(1.0, 0.0, 0.5) == 0.0
    assert f_beta(0.0, 0.0, 0.5) == 0.0
    # beta<1 favours precision: high-precision/low-recall beats the mirror case.
    assert f_beta(0.9, 0.3, 0.5) > f_beta(0.3, 0.9, 0.5)


def test_f_beta_matches_reference_formula():
    p, r, b = 0.8, 0.6, 0.5
    expected = (1 + b * b) * p * r / (b * b * p + r)
    assert f_beta(p, r, b) == pytest.approx(expected)


def test_perfect_overlap_gives_unit_metrics():
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    vc = view_counts(silhouette=pts, canny=pts, canny_trim=pts)
    agg = aggregate([vc], epsilon=0.5)
    assert agg["T_R"] == pytest.approx(1.0)
    assert agg["P_C"] == pytest.approx(1.0)
    assert agg["P_trim"] == pytest.approx(1.0)
    assert agg["F_beta"] == pytest.approx(1.0)


def test_trimming_improves_precision():
    silhouette = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    # raw Canny = the 3 on-silhouette points + 2 far outliers
    canny = np.vstack([silhouette, [[100.0, 100.0], [200.0, 200.0]]])
    trimmed = silhouette  # outliers removed by trimming
    agg = aggregate([view_counts(silhouette, canny, trimmed)], epsilon=1.0)
    assert agg["P_C"] == pytest.approx(3 / 5)   # 3 of 5 raw points near S
    assert agg["P_trim"] == pytest.approx(1.0)  # all trimmed points near S
    assert agg["T_R"] == pytest.approx(1.0)     # every S point covered


def test_micro_average_over_views():
    a = view_counts(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]),
                    np.array([[0.0, 0.0]]))
    b = view_counts(np.array([[0.0, 0.0], [50.0, 50.0]]),
                    np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]))
    # total S = 3, covered = 2 -> T_R = 2/3 (counts summed before ratio)
    assert aggregate([a, b], epsilon=1.0)["T_R"] == pytest.approx(2 / 3)


def test_select_best_picks_highest_fbeta():
    good = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    clean = view_counts(good, good, good)
    noisy = view_counts(
        good,
        np.vstack([good, [[99.0, 99.0], [88.0, 88.0]]]),
        np.vstack([good, [[99.0, 99.0]]]),
    )
    best, scores = select_best({"clean": [clean], "noisy": [noisy]}, epsilon=1.0)
    assert best == "clean"
    assert scores["clean"] > scores["noisy"]


def test_select_best_empty():
    assert select_best({}, epsilon=2.0) == ("", {})


def test_residual_is_per_contour_point_and_directional():
    """Pin the residual direction: one residual per CONTOUR point (data->model)."""
    from contour_reg.registration.residuals import _view_residuals
    silhouette = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])  # 3 pts
    contour = np.array([[0.0, 1.0], [10.0, 1.0]])                  # 2 pts
    r = _view_residuals(projected_2d=silhouette, contour_2d=contour)
    assert len(r) == len(contour)          # contour -> nearest silhouette
    # Swapping roles changes the length -> the residual is genuinely directional.
    r_swapped = _view_residuals(projected_2d=contour, contour_2d=silhouette)
    assert len(r_swapped) == len(silhouette) and len(r_swapped) != len(r)


def test_default_selection_is_f05_and_f1_is_reported():
    """Default beta = 0.5 (precision-weighted); aggregate also reports F1."""
    assert f_beta(0.8, 0.4) == pytest.approx(1.25 * 0.8 * 0.4 / (0.25 * 0.8 + 0.4))
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
    agg = aggregate([view_counts(pts[:2], pts, pts)], epsilon=0.5)
    assert agg["F1"] == pytest.approx(f_beta(agg["P_trim"], agg["T_R"], 1.0))
    assert agg["F_beta"] == pytest.approx(f_beta(agg["P_trim"], agg["T_R"], 0.5))
