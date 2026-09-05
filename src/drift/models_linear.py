"""Track B: shrinkage LDA / LR with class-balanced self-training (optionally under a known class
marginal), transductive LDA class-mean re-estimation, and kNN propagation."""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from drift.assign import sinkhorn_marginal


def make_lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def make_lr(C: float = 1.0):
    return LogisticRegression(C=C, max_iter=5000)


def cbst(make, Xs, ys, Xt, rounds=(0.3, 0.5, 0.7), margin_min: float = 0.0, marginal=None, tau: float = 1.0):
    """Class-balanced self-training.

    Each round adds the top `frac` most confident target rows *per predicted class* with hard
    pseudo-labels and refits.  With `marginal` (expected class counts of the target, e.g. uniform for
    the real test) the pseudo-labels and the confidence ranking come from the Sinkhorn-balanced
    posterior instead of the raw one, so the selection cannot amplify the classifier's own class-mass
    leakage (raw LDA predicts 650 Acetaldehyde on the test, plain cbst pumps it down to 441).
    Returns (P0 base probs, P adapted probs, classes).
    """
    clf = make().fit(Xs, ys)
    P0 = clf.predict_proba(Xt)
    P = P0
    classes = clf.classes_
    for frac in rounds:
        R = sinkhorn_marginal(P, marginal, tau=tau) if marginal is not None else P
        pred = classes[R.argmax(1)]
        srt = np.sort(R, axis=1)
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


def lda_em(Xs, ys, Xt, prior=None, iters: int = 10, trust: float = 1.0, shrinkage="auto"):
    """Transductive LDA: re-estimate the class means on the unlabeled target by EM with the mixing
    weights FIXED at `prior` (uniform by default, the organisers' 600-per-class statement) and the
    tied within-class covariance taken from the source.  Drift moves whole class clusters; with the
    weights fixed no component can swallow another, and each mean may move at most `trust`
    Mahalanobis units per iteration.  Returns (P0 source-model probs, P adapted probs, classes)."""
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=shrinkage).fit(Xs, ys)
    classes = lda.classes_
    C = len(classes)
    Zs, Zt = lda.transform(Xs), lda.transform(Xt)
    mu = np.stack([Zs[ys == c].mean(0) for c in classes])
    R = np.vstack([Zs[ys == c] - mu[i] for i, c in enumerate(classes)])
    Sw = R.T @ R / len(R) + 1e-6 * np.eye(Zs.shape[1])
    Si = np.linalg.inv(Sw)
    pi = np.full(C, 1.0 / C) if prior is None else np.asarray(prior, float) / np.sum(prior)

    def posterior(m):
        d = Zt[:, None, :] - m[None, :, :]
        ll = -0.5 * np.einsum("ncd,de,nce->nc", d, Si, d) + np.log(pi)[None, :]
        ll -= ll.max(1, keepdims=True)
        p = np.exp(ll)
        return p / p.sum(1, keepdims=True)

    P0 = posterior(mu)
    # initialise with the global shift of the target cloud (weighted by the fixed prior)
    m = mu + (Zt.mean(0) - (pi[:, None] * mu).sum(0))[None, :]
    for _ in range(iters):
        P = posterior(m)
        new = (P.T @ Zt) / np.maximum(P.sum(0)[:, None], 1e-9)
        step = new - m
        dist = np.sqrt(np.einsum("cd,de,ce->c", step, Si, step))
        scale = np.minimum(1.0, trust / np.maximum(dist, 1e-12))
        m = m + step * scale[:, None]
    return P0, posterior(m), classes


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
