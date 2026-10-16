"""Learned baselines on the same clusters: gradient-boosted re-scoring and OBBStacking."""
import math

import numpy as np

from .clusters import CLASS_COL

OBBS_MISSING = math.log(1e-3 / (1 - 1e-3))  # logit of the export confidence floor


def gbdt_model(seed=0):
    """Histogram GBDT on the cluster descriptor with the class as a categorical feature."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=50, l2_regularization=1.0,
                                          categorical_features=[CLASS_COL], random_state=seed)


def gbdt_scores(train, test):
    """Fit on training-split clusters (target IoU >= 0.5) and score the test clusters."""
    model = gbdt_model().fit(train["X"], train["y"] >= 0.5)
    return model.predict_proba(test["X"])[:, 1]


def _view_logits(z, n_views):
    """[n, V]: logit of each view's most confident member in the cluster (missing -> floor)."""
    n = len(z["y"])
    cid = np.repeat(np.arange(n), z["mcount"])
    best = np.zeros((n, n_views), np.float32)
    np.maximum.at(best, (cid, z["mview"]), z["mconf"])
    c = np.clip(best, 1e-6, 1 - 1e-6)
    return np.where(best > 0, np.log(c / (1 - c)), OBBS_MISSING), cid


def _poly_to_xywht(p, asp):
    """[m, 8] normalised polygons -> centre, long side, short side, angle of the long side in [0, pi)."""
    q = p.reshape(-1, 4, 2).astype(np.float64).copy()
    q[..., 1] *= asp[:, None]
    e1, e2 = q[:, 1] - q[:, 0], q[:, 3] - q[:, 0]
    l1, l2 = np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1)
    long_e = np.where((l1 >= l2)[:, None], e1, e2)
    th = np.mod(np.arctan2(long_e[:, 1], long_e[:, 0]), np.pi)
    return q.mean(1), np.maximum(l1, l2), np.minimum(l1, l2), th


def obbstacking(train, test, n_views):
    """OBBStacking (Lin et al., arXiv:2209.13369) with the views as member models.

    A logistic regression over the confidence logit of each view's most confident box in the
    cluster (the logit of 0.001 when the view has none), fitted on training-split clusters,
    gives the score. Boxes are fused as in the original: centre, long and short side averaged with the
    calibrated member scores s*_p = sigmoid(z_p w_view + b), the angle averaged relative to the
    orientation of the best member. Returns (scores, fused polygons)."""
    from sklearn.linear_model import LogisticRegression
    Ztr, _ = _view_logits(train, n_views)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(Ztr, train["y"] >= 0.5)
    w, b = lr.coef_[0], lr.intercept_[0]
    Zte, cid = _view_logits(test, n_views)
    score = 1 / (1 + np.exp(-(Zte @ w + b)))
    c = np.clip(test["mconf"], 1e-6, 1 - 1e-6)
    s = 1 / (1 + np.exp(-(np.log(c / (1 - c)) * w[test["mview"]] + b)))
    ctr, L_, S_, th = _poly_to_xywht(test["mpoly"], test["aspect"][cid])
    n = len(test["y"])
    ws = np.bincount(cid, s, n)
    order = np.lexsort((-s, cid))
    first = order[np.r_[0, np.flatnonzero(np.diff(cid[order])) + 1]]
    th_mj = np.zeros(n)
    th_mj[cid[first]] = th[first]
    r = th - th_mj[cid]
    r = np.where(r > np.pi / 2, r - np.pi, np.where(r < -np.pi / 2, r + np.pi, r))
    f = lambda x: np.bincount(cid, s * x, n) / ws  # noqa: E731
    cx, cy, Lf, Sf, tf = f(ctr[:, 0]), f(ctr[:, 1]), f(L_), f(S_), f(r) + th_mj
    u = np.stack([np.cos(tf), np.sin(tf)], 1) * (Lf / 2)[:, None]
    v = np.stack([-np.sin(tf), np.cos(tf)], 1) * (Sf / 2)[:, None]
    m = np.stack([cx, cy], 1)
    pts = np.stack([m + u + v, m - u + v, m - u - v, m + u - v], 1)
    pts[..., 1] /= test["aspect"][:, None]
    return score, pts.reshape(n, 8).astype(np.float32)
