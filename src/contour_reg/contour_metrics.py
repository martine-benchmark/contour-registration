"""Contour-vs-silhouette overlap metrics for Canny-config selection.

These metrics compare, for a registered pose, three 2D point sets at a pixel
tolerance ``epsilon``:

* ``S``     : the projected mesh silhouette,
* ``C``     : the raw Canny contour,
* ``C_trim``: the Canny contour after registration trimming.

They are *overlap* metrics (symmetric in spirit; they do not depend on the ICP
residual direction), used only to rank Canny configurations against one another:

    T_R    = |{s in S : d(s, C) <= eps}| / |S|            silhouette coverage (recall)
    P_C    = |{c in C : d(c, S) <= eps}| / |C|            raw-Canny precision
    P_trim = |{c in C_trim : d(c, S) <= eps}| / |C_trim|  trimmed precision
    F_beta = (1 + b^2) * P_trim * T_R / (b^2 * P_trim + T_R)   (default b = 0.5)
    F1     = F_beta with b = 1 (reported alongside)

Counts are summed over views before ratioing (micro-average). ``select_best``
picks the configuration with the highest ``F_beta`` at a chosen epsilon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def nn_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distance from each point in *query* to its nearest point in *reference*.

    An empty *reference* yields ``+inf`` for every query point (nothing matches);
    an empty *query* yields an empty array.
    """
    query = np.asarray(query, dtype=float).reshape(-1, 2)
    reference = np.asarray(reference, dtype=float).reshape(-1, 2)
    if len(query) == 0:
        return np.empty(0, dtype=float)
    if len(reference) == 0:
        return np.full(len(query), np.inf, dtype=float)
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(reference)
    distances, _ = nn.kneighbors(query)
    return distances.ravel()


@dataclass(frozen=True)
class ViewCounts:
    """Per-view nearest-neighbour distances, thresholded lazily per epsilon.

    Storing the distances (not per-epsilon counts) lets a whole epsilon sweep be
    evaluated from a single nearest-neighbour pass per view.
    """

    d_S_to_C: np.ndarray      # each silhouette point -> nearest Canny point
    d_C_to_S: np.ndarray      # each Canny point -> nearest silhouette point
    d_Ctrim_to_S: np.ndarray  # each trimmed Canny point -> nearest silhouette point

    @property
    def n_S(self) -> int:
        return len(self.d_S_to_C)

    @property
    def n_C(self) -> int:
        return len(self.d_C_to_S)

    @property
    def n_C_trim(self) -> int:
        return len(self.d_Ctrim_to_S)


def view_counts(
    silhouette: np.ndarray, canny: np.ndarray, canny_trim: np.ndarray
) -> ViewCounts:
    """Compute the three nearest-neighbour distance arrays for one view."""
    return ViewCounts(
        d_S_to_C=nn_distances(silhouette, canny),
        d_C_to_S=nn_distances(canny, silhouette),
        d_Ctrim_to_S=nn_distances(canny_trim, silhouette),
    )


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """F-beta score; ``beta < 1`` weights *precision* over *recall*."""
    denom = beta * beta * precision + recall
    if denom <= 0:
        return 0.0
    return (1.0 + beta * beta) * precision * recall / denom


def aggregate(
    views: Sequence[ViewCounts], epsilon: float, beta: float = 0.5
) -> Dict[str, float]:
    """Micro-averaged contour metrics at *epsilon* over a set of views."""
    s_f = sum(int(np.count_nonzero(v.d_S_to_C <= epsilon)) for v in views)
    c_f = sum(int(np.count_nonzero(v.d_C_to_S <= epsilon)) for v in views)
    ct_f = sum(int(np.count_nonzero(v.d_Ctrim_to_S <= epsilon)) for v in views)
    n_s = sum(v.n_S for v in views)
    n_c = sum(v.n_C for v in views)
    n_ct = sum(v.n_C_trim for v in views)

    t_r = s_f / n_s if n_s else 0.0
    p_c = c_f / n_c if n_c else 0.0
    p_trim = ct_f / n_ct if n_ct else 0.0
    return {
        "epsilon": float(epsilon),
        "beta": float(beta),
        "T_R": t_r,
        "P_C": p_c,
        "P_trim": p_trim,
        "T_FP_res": 1.0 - p_trim,
        "F_beta": f_beta(p_trim, t_r, beta),
        "F1": f_beta(p_trim, t_r, 1.0),
        "n_S": n_s,
        "n_C": n_c,
        "n_C_trim": n_ct,
    }


def best_by_score(scores: Mapping[str, float]) -> str:
    """Deterministic argmax over a ``{name: score}`` map (ties break on name).

    Returns ``""`` for an empty mapping. Shared by :func:`select_best` and
    :func:`contour_reg.pipeline.eval_canny.eval_configs` so their ranking cannot
    drift apart.
    """
    if not scores:
        return ""
    return max(sorted(scores), key=scores.__getitem__)


def select_best(
    per_config: Mapping[str, Sequence[ViewCounts]],
    epsilon: float = 2.0,
    beta: float = 0.5,
) -> Tuple[str, Dict[str, float]]:
    """Rank configs by ``F_beta`` at *epsilon*.

    Returns ``(best_config, {config: F_beta})``. Ties break on the config name
    for determinism; an empty mapping returns ``("", {})``.
    """
    scores = {
        name: aggregate(views, epsilon, beta)["F_beta"]
        for name, views in per_config.items()
    }
    return best_by_score(scores), scores
