"""Balanced assignment of rows to classes under a KNOWN class marginal.

The organisers state that the hidden test has exactly 600 rows per class; this module turns that
statement into an optimal-transport assignment on the model's log-probabilities.  On leave-one-batch-out
folds the same functions are used with the held-out batch's TRUE class counts as the marginal: the labels
are used only to set the column masses (evaluation design), never to move an individual row.
Only features/probabilities enter; nothing here looks at row order or measurement ids.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment


def _column_mass(N: int, C: int, marginal) -> np.ndarray:
    if marginal is None:
        return np.full(C, N / C)
    m = np.asarray(marginal, float)
    if m.shape != (C,) or m.sum() <= 0:
        raise ValueError("marginal must be a positive vector with one entry per class")
    return m * N / m.sum()


def sinkhorn_marginal(P: np.ndarray, marginal=None, tau: float = 1.0, n_iter: int = 5000,
                      tol: float = 1e-3) -> np.ndarray:
    """Entropic optimal-transport plan between rows (mass 1 each) and classes (mass `marginal`,
    default N/C each) with cost -log P / tau.  Returns Q with rows summing to 1 and columns to the
    marginal (to relative tolerance `tol`).  tau -> 0 approaches the hard assignment, tau = 1 is the
    posterior with per-class log-bias corrections (Saerens-style prior shift, but with the marginal
    enforced exactly instead of estimated)."""
    P = np.asarray(P, float)
    N, C = P.shape
    col = _column_mass(N, C, marginal)
    logK = np.log(np.clip(P, 1e-12, 1.0)) / tau
    logK = logK - logK.max(axis=1, keepdims=True)
    K = np.exp(logK)
    u = np.ones(N)
    v = np.ones(C)
    dev = np.inf
    for it in range(n_iter):
        u = 1.0 / np.maximum(K @ v, 1e-300)
        v = col / np.maximum(K.T @ u, 1e-300)
        if it % 10 == 9:
            Q = u[:, None] * K * v[None, :]
            Q = Q / Q.sum(axis=1, keepdims=True)
            dev = np.abs(Q.sum(axis=0) - col).max() / col.max()
            if dev < tol:
                break
    Q = u[:, None] * K * v[None, :]
    Q = Q / Q.sum(axis=1, keepdims=True)
    dev = np.abs(Q.sum(axis=0) - col).max() / col.max()
    if dev >= tol:
        warnings.warn(f"sinkhorn_marginal: column deviation {dev:.3%} after {n_iter} iterations", stacklevel=2)
    return Q


def hungarian_assign(P: np.ndarray, marginal=None) -> np.ndarray:
    """Exact minimum-cost assignment (cost -log P) with integer class quotas. Returns 0-based labels."""
    P = np.asarray(P, float)
    N, C = P.shape
    col = _column_mass(N, C, marginal)
    counts = np.floor(col).astype(int)
    rem = N - counts.sum()
    if rem > 0:                                    # distribute rounding remainder to the largest fractions
        order = np.argsort(-(col - counts))
        counts[order[:rem]] += 1
    cols = np.repeat(np.arange(C), counts)
    cost = -np.log(np.clip(P, 1e-12, 1.0))[:, cols]
    r, c = linear_sum_assignment(cost)
    out = np.empty(N, int)
    out[r] = cols[c]
    return out


def balanced_predict(P: np.ndarray, marginal=None, method: str = "sinkhorn", tau: float = 1.0) -> np.ndarray:
    """0-based labels after enforcing the marginal ('sinkhorn' argmax of the plan, or 'hungarian')."""
    if method == "none":
        return np.asarray(P).argmax(1)
    if method == "hungarian":
        return hungarian_assign(P, marginal)
    if method == "sinkhorn":
        return sinkhorn_marginal(P, marginal, tau=tau).argmax(1)
    raise ValueError(method)


def flow_table(pred_before: np.ndarray, pred_after: np.ndarray, n_classes: int = 6) -> np.ndarray:
    """6x6 counts of rows moved from class i (rows) to class j (cols); diagonal = unchanged."""
    M = np.zeros((n_classes, n_classes), int)
    np.add.at(M, (np.asarray(pred_before, int), np.asarray(pred_after, int)), 1)
    return M


def hist_l1(pred: np.ndarray, n_classes: int = 6, marginal=None) -> float:
    """L1 distance between the predicted class histogram and the expected one, as a fraction of N."""
    pred = np.asarray(pred, int)
    h = np.bincount(pred, minlength=n_classes).astype(float)
    col = _column_mass(len(pred), n_classes, marginal)
    return float(np.abs(h - col).sum() / len(pred))
