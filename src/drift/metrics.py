"""Metrics and label-free sanity diagnostics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

CLASSES = np.arange(1, 7)
NAMES = {1: "EtOH", 2: "Ethyl", 3: "NH3", 4: "Acetal", 5: "Aceton", 6: "Tolu"}


def macro_f1(y_true, y_pred, labels=None) -> float:
    """Macro-F1 over the classes present in y_true (proxy batches may lack a gas)."""
    if labels is None:
        labels = np.unique(np.asarray(y_true))
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))


def per_class_f1(y_true, y_pred) -> np.ndarray:
    return f1_score(y_true, y_pred, average=None, labels=CLASSES, zero_division=0)


def pred_hist(pred) -> np.ndarray:
    return np.bincount(np.asarray(pred, int), minlength=7)[1:]


def agreement(p1, p2) -> float:
    return float(np.mean(np.asarray(p1) == np.asarray(p2)))


def em_prior(P: np.ndarray, prior_src: np.ndarray, iters: int = 100):
    """Saerens et al. 2002 EM re-estimation of the target prior.

    Returns (adjusted posteriors, estimated prior).
    """
    pt = prior_src.copy()
    W = P
    for _ in range(iters):
        W = P * (pt / prior_src)
        W = W / W.sum(axis=1, keepdims=True)
        pt = W.mean(axis=0)
    return W, pt


def reweight_prior(P: np.ndarray, prior_src: np.ndarray, prior_tgt: np.ndarray) -> np.ndarray:
    W = P * (prior_tgt / prior_src)
    return W / W.sum(axis=1, keepdims=True)


def sanity_check(pred, n_expected: int, lo: int = 520, hi: int = 680) -> list:
    """Return a list of problems (empty = OK)."""
    problems = []
    pred = np.asarray(pred)
    if len(pred) != n_expected:
        problems.append(f"row count {len(pred)} != {n_expected}")
    h = pred_hist(pred)
    for c, cnt in zip(CLASSES, h):
        if cnt < lo or cnt > hi:
            problems.append(f"class {c} ({NAMES[c]}) predicted {cnt} times, outside [{lo},{hi}]")
    if set(np.unique(pred).tolist()) - set(CLASSES.tolist()):
        problems.append("labels outside 1..6")
    return problems


# ---- label-free validators for the unlabeled target (Musgrave et al. 2022: BNM / IM / ClassAMI rank well)
def bnm(P: np.ndarray) -> float:
    """Batch nuclear-norm of the prediction matrix, normalised by sqrt(N); higher = more confident AND diverse."""
    return float(np.linalg.norm(P, "nuc") / np.sqrt(len(P)))


def info_max(P: np.ndarray) -> float:
    """H(mean prediction) - mean H(prediction); higher = confident per row, balanced overall."""
    eps = 1e-9
    h_mean = -(P.mean(0) * np.log(P.mean(0) + eps)).sum()
    mean_h = -(P * np.log(P + eps)).sum(1).mean()
    return float(h_mean - mean_h)


def class_ami(P: np.ndarray, F: np.ndarray, k: int = 6, seed: int = 0) -> float:
    """Adjusted mutual information between argmax predictions and a k-means clustering of the target features."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score
    lab = KMeans(k, n_init=4, random_state=seed).fit_predict(F)
    return float(adjusted_mutual_info_score(P.argmax(1), lab))
