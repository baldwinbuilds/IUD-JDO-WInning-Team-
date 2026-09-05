"""Self-training on the unlabeled target: class-balanced selection PER TARGET DOMAIN with soft (KD)
targets; optional Sinkhorn equipartition applied only to the real test domain (batch id 9), whose
class prior is stated by the organisers to be uniform."""
from __future__ import annotations

import numpy as np

from drift.nn.train import RunConfig, train_one

TEST_DOMAIN = 9   # 0-based batch id of the real test batch (batch 10)


def soften(P: np.ndarray, T: float) -> np.ndarray:
    q = np.power(np.clip(P, 1e-9, 1.0), 1.0 / T)
    return q / q.sum(1, keepdims=True)


def sinkhorn_balanced(P: np.ndarray, tau: float = 0.1, n_iter: int = 100) -> np.ndarray:
    """Equipartition assignment: rows sum to 1, columns each get N/C mass (uniform target prior)."""
    N, C = P.shape
    Q = np.exp(np.log(np.clip(P, 1e-9, 1.0)) / tau)
    for _ in range(n_iter):
        Q = Q / Q.sum(0, keepdims=True) * (N / C)
        Q = Q / Q.sum(1, keepdims=True)
    return Q


def select_pseudo(P: np.ndarray, frac: float, margin_min: float, conf_src: np.ndarray | None = None):
    """Top `frac` rows per predicted class of P (margin filtered on P). Ranking/weight confidence =
    conf_src[i, argmax P[i]] when given (model probability of the assigned class; needed when P is a
    near-one-hot Sinkhorn assignment), else max P. Returns local idx, conf."""
    pred = P.argmax(1)
    srt = np.sort(P, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    conf = srt[:, -1] if conf_src is None else conf_src[np.arange(len(P)), pred]
    sel = []
    for c in range(P.shape[1]):
        idx = np.where((pred == c) & (margin >= margin_min))[0]
        if len(idx) == 0:
            continue
        k = max(1, int(frac * len(idx)))
        sel.append(idx[np.argsort(-conf[idx])[:k]])
    idx = np.concatenate(sel) if sel else np.array([], int)
    return idx, conf[idx]


def self_train(cfg: RunConfig, Xs, ys, bs, Xt, bt, base_probs: np.ndarray, n_domains: int = 10,
               device: str = "cpu", y_eval=None, n_eval=None, verbose: bool = False, log=print) -> dict:
    """Run cfg.selftrain_rounds rounds of retraining with pseudo-labelled target rows (per domain)."""
    bt = np.asarray(bt)
    P = np.asarray(base_probs, np.float64)
    out = {"rounds": []}
    for r, frac in enumerate(cfg.selftrain_rounds):
        T = cfg.pseudo_T if r == 0 else 1.0    # later teachers already reproduce softened targets: no compounding
        idx_all, Q_all, W_all = [], [], []
        for d in np.unique(bt):
            m = np.where(bt == d)[0]
            Pd = P[m]
            use_sk = cfg.sinkhorn and d == TEST_DOMAIN
            src = sinkhorn_balanced(Pd) if use_sk else Pd
            loc, conf = select_pseudo(src, frac, cfg.pseudo_margin, conf_src=Pd if use_sk else None)
            idx_all.append(m[loc])
            Q_all.append(soften(src[loc], T))       # target = the (balanced) posterior the row was selected under
            W_all.append(cfg.pseudo_weight * conf)  # weight = model's own probability of that class
        idx = np.concatenate(idx_all)
        Q = np.vstack(Q_all)
        W = np.concatenate(W_all)
        res = train_one(cfg, Xs, ys, bs, Xt, bt, n_domains, device, pseudo=(idx, Q, W),
                        y_eval=y_eval, n_eval=n_eval, verbose=verbose)
        P = np.asarray(res["probs"], np.float64)
        hist = np.bincount(P[bt == TEST_DOMAIN].argmax(1), minlength=6).tolist() if (bt == TEST_DOMAIN).any() \
            else np.bincount(P.argmax(1), minlength=6).tolist()
        out["rounds"].append({"frac": frac, "n_pseudo": int(len(idx)), "test_hist": hist,
                              "history": res["history"], "seconds": res["seconds"]})
        if log:
            log(f"    self-train round {r+1}: frac={frac} pseudo={len(idx)} test_hist={hist}")
    out["probs"] = P.astype(np.float32)
    return out
