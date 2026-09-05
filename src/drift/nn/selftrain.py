"""Self-training on the unlabeled target: class-balanced selection PER TARGET DOMAIN with soft (KD)
targets.  With cfg.sinkhorn the pseudo-labels of a domain whose class marginal is known come from the
Sinkhorn-balanced posterior: the real test domain (batch id 9; organisers state 600 rows per class ->
uniform) and, in leave-one-batch-out runs, the held-out batch with its true class counts (labels used
only to set the six column masses, so the OOF score exercises the same code path as the test)."""
from __future__ import annotations

import numpy as np

from drift.assign import sinkhorn_marginal
from drift.nn.train import RunConfig, train_one

TEST_DOMAIN = 9   # 0-based batch id of the real test batch (batch 10)


def soften(P: np.ndarray, T: float) -> np.ndarray:
    q = np.power(np.clip(P, 1e-9, 1.0), 1.0 / T)
    return q / q.sum(1, keepdims=True)


def sinkhorn_balanced(P: np.ndarray, tau: float = 0.1, marginal=None) -> np.ndarray:
    """Balanced assignment posterior (rows sum to 1, columns to the marginal; uniform by default)."""
    return sinkhorn_marginal(P, marginal, tau=tau)


def select_pseudo(P: np.ndarray, frac: float, margin_min: float, conf_src: np.ndarray | None = None,
                  tiebreak: np.ndarray | None = None):
    """Top `frac` rows per predicted class of P (margin filtered on P). Ranking/weight confidence =
    conf_src[i, argmax P[i]] when given (model probability of the assigned class), else max P; ties are
    broken by `tiebreak[i, argmax P[i]]` (deterministic, never by row position). Returns local idx, conf."""
    pred = P.argmax(1)
    srt = np.sort(P, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    rows = np.arange(len(P))
    conf = srt[:, -1] if conf_src is None else conf_src[rows, pred]
    tb = np.zeros(len(P)) if tiebreak is None else tiebreak[rows, pred]
    sel = []
    for c in range(P.shape[1]):
        idx = np.where((pred == c) & (margin >= margin_min))[0]
        if len(idx) == 0:
            continue
        k = max(1, int(frac * len(idx)))
        order = np.lexsort((-tb[idx], -conf[idx]))      # primary: conf desc; secondary: tiebreak desc
        sel.append(idx[order[:k]])
    idx = np.concatenate(sel) if sel else np.array([], int)
    return idx, conf[idx]


def self_train(cfg: RunConfig, Xs, ys, bs, Xt, bt, base_probs: np.ndarray, n_domains: int = 10,
               device: str = "cpu", y_eval=None, n_eval=None, verbose: bool = False, log=print,
               heldout_marginal=None) -> dict:
    """Run cfg.selftrain_rounds rounds of retraining with pseudo-labelled target rows (per domain).

    heldout_marginal: class counts (6,) of the held-out domain = bt[0] (LOBO runs), used as that domain's
    Sinkhorn marginal when cfg.sinkhorn; the test domain always uses the uniform marginal.
    cfg.sinkhorn_rank: 'model' ranks/weights reassigned rows by the model's own probability of the
    assigned class (rows the prior wants to move rank last); 'q' ranks/weights by the balanced posterior.
    """
    bt = np.asarray(bt)
    P = np.asarray(base_probs, np.float64)
    out = {"rounds": [], "alt": None}
    held_dom = int(bt[0]) if (n_eval and heldout_marginal is not None) else None
    prev_pred = P.argmax(1)
    for r, frac in enumerate(cfg.selftrain_rounds):
        T = cfg.pseudo_T if r == 0 else 1.0    # later teachers already reproduce softened targets: no compounding
        idx_all, Q_all, W_all = [], [], []
        n_moved = 0
        for d in np.unique(bt):
            m = np.where(bt == d)[0]
            Pd = P[m]
            marg = None
            use_sk = False
            if cfg.sinkhorn and d == TEST_DOMAIN:
                use_sk, marg = True, None
            elif cfg.sinkhorn and held_dom is not None and d == held_dom:
                use_sk, marg = True, np.asarray(heldout_marginal, float)
            src = sinkhorn_balanced(Pd, cfg.sinkhorn_tau, marg) if use_sk else Pd
            conf_src = Pd if (use_sk and cfg.sinkhorn_rank == "model") else None
            loc, conf = select_pseudo(src, frac, cfg.pseudo_margin, conf_src=conf_src, tiebreak=Pd)
            n_moved += int((src.argmax(1) != Pd.argmax(1)).sum())
            idx_all.append(m[loc])
            Q_all.append(soften(src[loc], T))       # target = the (balanced) posterior the row was selected under
            W_all.append(cfg.pseudo_weight * conf)  # weight = confidence under the ranking posterior
        idx = np.concatenate(idx_all)
        Q = np.vstack(Q_all)
        W = np.concatenate(W_all)
        res = train_one(cfg, Xs, ys, bs, Xt, bt, n_domains, device, pseudo=(idx, Q, W),
                        y_eval=y_eval, n_eval=n_eval, verbose=verbose)
        use_swa = cfg.adabn and cfg.selftrain_output == "swa_raw"
        P_new = np.asarray(res["probs_swa_raw"] if use_swa else res["probs"], np.float64)
        test_m = bt == TEST_DOMAIN
        hist = np.bincount(P_new[test_m].argmax(1), minlength=6).tolist() if test_m.any() \
            else np.bincount(P_new.argmax(1), minlength=6).tolist()
        agree = float(np.mean(P_new.argmax(1) == prev_pred))
        agree_test = float(np.mean(P_new[test_m].argmax(1) == prev_pred[test_m])) if test_m.any() else agree
        info = {"frac": frac, "n_pseudo": int(len(idx)), "n_reassigned": n_moved, "test_hist": hist,
                "agree_prev": agree, "agree_prev_test": agree_test, "history": res["history"], "seconds": res["seconds"]}
        if log:
            log(f"    self-train round {r+1}: frac={frac} pseudo={len(idx)} reassigned={n_moved} "
                f"test_hist={hist} agree_prev_test={agree_test:.3f}")
        if cfg.selftrain_guard and agree_test < cfg.selftrain_guard:
            info["rejected"] = True
            out["rounds"].append(info)
            if log:
                log(f"    self-train round {r+1} REJECTED (agreement {agree_test:.3f} < guard {cfg.selftrain_guard}); keeping previous round")
            break
        out["rounds"].append(info)
        P, prev_pred = P_new, P_new.argmax(1)
        out["alt"] = {"last": res["probs_last"], "last_adabn": res.get("probs_last_adabn")}
        if cfg.adabn:                                    # the other read-out of the same network
            out["alt"]["adabn" if use_swa else "swa_raw"] = res["probs"] if use_swa else res["probs_swa_raw"]
    out["probs"] = P.astype(np.float32)
    return out
