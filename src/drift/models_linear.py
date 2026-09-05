"""Track B: shrinkage LDA / LR with class-balanced self-training and kNN propagation."""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def make_lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def make_lr(C: float = 1.0):
    return LogisticRegression(C=C, max_iter=5000)


def cbst(make, Xs, ys, Xt, rounds=(0.3, 0.5, 0.7), margin_min: float = 0.0):
    """Class-balanced self-training.

    Each round adds the top `frac` most confident target rows *per predicted class*
    with their hard pseudo-labels and refits.  Returns (P0 base probs, P adapted
    probs, classes).
    """
    clf = make().fit(Xs, ys)
    P0 = clf.predict_proba(Xt)
    P = P0
    classes = clf.classes_
    for frac in rounds:
        pred = classes[P.argmax(1)]
        srt = np.sort(P, axis=1)
        conf = srt[:, -1]
        margin = srt[:, -1] - srt[:, -2]
        sel = np.zeros(len(Xt), bool)
        for c in classes:
            idx = np.where((pred == c) & (margin >= margin_min))[0]
            if len(idx) == 0:
                continue
            k = max(1, int(frac * len(idx)))
            sel[idx[np.argsort(-conf[idx])[:k]]] = True
        clf = make().fit(np.vstack([Xs, Xt[sel]]), np.concatenate([ys, pred[sel]]))
        P = clf.predict_proba(Xt)
    return P0, P, classes


def knn_propagate(P: np.ndarray, Xt: np.ndarray, k: int = 10, alpha: float = 0.8,
                  iters: int = 20, tau: float = 0.05) -> np.ndarray:
    """Label propagation over the target feature kNN graph (cosine). Features only."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Xt)
    dist, ind = nn.kneighbors(Xt)
    n = len(Xt)
    W = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k)
    W[rows, ind[:, 1:].ravel()] = np.exp(-dist[:, 1:].ravel() / tau)
    W = (W + W.T) / 2
    d = W.sum(1) + 1e-9
    S = W / np.sqrt(d[:, None] * d[None, :])
    F = P.copy()
    for _ in range(iters):
        F = alpha * S @ F + (1 - alpha) * P
    return F / F.sum(1, keepdims=True)
