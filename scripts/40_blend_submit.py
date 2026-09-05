"""Blend Track B and NN models on leave-one-batch-out OOF predictions (batches 6-9), apply the balanced
assignment implied by the organisers' 600-per-class statement, run label-free gates, write the submission.

Pipeline
  1. members: trackB:<variant> (runs/trackB/oof_lobo{k}.npz, test_probs.npz) and nn:<variant>
     (runs/<nn_tag>/<variant>_fold{k}_seed*.npz; OOF seed-averaged, test averaged over all folds & seeds
     with fold-0 models - trained on all nine batches - weighted --fold0-weight);
  2. per-member temperature calibration (fitted on the OOF of the OTHER folds when scoring a fold, on all
     folds for the test) so that over-confident LDA posteriors do not dominate soft NN posteriors;
  3. weighted blend in log space -> softmax;
  4. decision rule = balanced assignment (Sinkhorn, tau) with the fold's true class counts as marginal on
     OOF folds (labels used only for the six column masses) and the uniform marginal on the test;
  5. hill-climb the member weights on the weighted post-assignment macro-F1 (batches 6/7/8/9, w 1/2/1/2),
     fall back to the best single member if the blend loses > 0.5 pt on any fold;
  6. label-free gates on the test: pre-assignment histogram, fraction of rows moved, flow matrix,
     agreement with every member; refuse to write when they look like a broken member set (--force).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift.assign import balanced_predict, flow_table, hist_l1  # noqa: E402
from drift.data import N_TEST, REPORTS, RUNS, SUBS, load_test, write_submission  # noqa: E402
from drift.metrics import CLASSES, macro_f1, per_class_f1, pred_hist, sanity_check  # noqa: E402
from drift.splits import PROXY_WEIGHTS, weighted_score  # noqa: E402

FOLDS = (6, 7, 8, 9)
EPS = 1e-9


def load_models(nn_tag: str, trackb_variants, nn_variants, fold0_weight: float):
    """Return dict name -> {'oof': {k: probs}, 'test': probs}, y dict k -> labels (1..6)."""
    models, y = {}, {}
    tb = RUNS / "trackB"
    if tb.exists() and trackb_variants:
        tp = np.load(tb / "test_probs.npz", allow_pickle=True)
        for v in trackb_variants:
            if v not in tp:
                print(f"  (trackB:{v} not in test_probs.npz - skipped)")
                continue
            m = {"oof": {}, "test": tp[v]}
            for k in FOLDS:
                z = np.load(tb / f"oof_lobo{k}.npz", allow_pickle=True)
                m["oof"][k] = z[v]
                y[k] = z["y"].astype(int)
            models[f"trackB:{v}"] = m
    nd = RUNS / nn_tag
    if nd.exists():
        oof = defaultdict(lambda: defaultdict(list))
        test, tw = defaultdict(list), defaultdict(list)
        for f in sorted(nd.glob("*_fold*_seed*.npz")):
            z = np.load(f, allow_pickle=True)
            v, k = str(z["variant"]), int(z["fold"])
            if nn_variants and v not in nn_variants:
                continue
            if k in FOLDS:
                oof[v][k].append(z["oof"])
                y[k] = z["y_oof"].astype(int)
            test[v].append(z["test"])
            tw[v].append(fold0_weight if k == 0 else 1.0)
        for v in oof:
            if all(k in oof[v] for k in FOLDS):
                w = np.asarray(tw[v]) / np.sum(tw[v])
                models[f"nn:{v}"] = {"oof": {k: np.mean(oof[v][k], axis=0) for k in FOLDS},
                                     "test": np.tensordot(w, np.stack(test[v]), axes=1),
                                     "n_test_models": len(test[v]),
                                     "n_seeds": {k: len(oof[v][k]) for k in FOLDS}}
            else:
                print(f"  (skipping nn:{v}: folds present {sorted(oof[v])}, need {FOLDS})")
    return models, y


def fit_temperature(P: np.ndarray, y0: np.ndarray) -> float:
    """Temperature T minimising the NLL of softmax(log P / T); T > 1 softens over-confident members."""
    L = np.log(np.clip(P, EPS, 1.0))

    def nll(logT):
        Z = L / np.exp(logT)
        Z = Z - Z.max(1, keepdims=True)
        lse = np.log(np.exp(Z).sum(1))
        return float(np.mean(lse - Z[np.arange(len(y0)), y0]))
    r = minimize_scalar(nll, bounds=(-2.0, 3.5), method="bounded")
    return float(np.exp(r.x))


def calibrate(models, y, enabled: bool):
    """temps[m][k] = T fitted on the other folds (honest for scoring fold k); temps[m]['test'] on all."""
    temps = {}
    for m, d in models.items():
        temps[m] = {}
        for k in FOLDS:
            if not enabled:
                temps[m][k] = 1.0
                continue
            P = np.vstack([d["oof"][j] for j in FOLDS if j != k])
            yy = np.concatenate([y[j] - 1 for j in FOLDS if j != k])
            temps[m][k] = fit_temperature(P, yy)
        temps[m]["test"] = fit_temperature(np.vstack([d["oof"][j] for j in FOLDS]),
                                           np.concatenate([y[j] - 1 for j in FOLDS])) if enabled else 1.0
    return temps


def blend(models, weights: dict, key, temps) -> np.ndarray:
    L = 0.0
    for m, w in weights.items():
        P = models[m]["oof"][key] if key != "test" else models[m]["test"]
        L = L + w * np.log(np.clip(P, EPS, 1.0)) / temps[m][key]
    L = L - L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)


def score(models, y, weights: dict, temps, assign: str, tau: float):
    per, per_pre = {}, {}
    for k in FOLDS:
        P = blend(models, weights, k, temps)
        marg = np.bincount(y[k], minlength=7)[1:]
        per_pre[k] = macro_f1(y[k], P.argmax(1) + 1)
        per[k] = macro_f1(y[k], balanced_predict(P, marg, assign, tau) + 1)
    return weighted_score(per), per, per_pre


def hill_climb(models, y, temps, assign, tau, n_iter: int = 40):
    names = list(models)
    singles = {m: score(models, y, {m: 1.0}, temps, assign, tau) for m in names}
    best = max(singles, key=lambda m: singles[m][0])
    counts = defaultdict(int)
    counts[best] += 1
    cur = singles[best][0]
    for _ in range(n_iter):
        cand = {}
        for m in names:
            c = defaultdict(int, counts)
            c[m] += 1
            tot = sum(c.values())
            cand[m] = score(models, y, {mm: cc / tot for mm, cc in c.items()}, temps, assign, tau)[0]
        m = max(cand, key=cand.get)
        if cand[m] < cur + 1e-5:
            break
        counts[m] += 1
        cur = cand[m]
    tot = sum(counts.values())
    return {m: c / tot for m, c in counts.items()}, singles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-tag", default="nn")
    ap.add_argument("--trackb", nargs="*", default=["ens_cbstb", "em_ensb", "lda_em", "ens_cbst"])
    ap.add_argument("--nn", nargs="*", default=None, help="restrict NN variants (default: all with folds 6,7,8,9)")
    ap.add_argument("--assign", choices=["sinkhorn", "hungarian", "none"], default="sinkhorn")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--no-calib", action="store_true", help="skip per-member temperature calibration")
    ap.add_argument("--fold0-weight", type=float, default=2.0)
    ap.add_argument("--equal", action="store_true", help="equal weights over all members instead of hill-climb")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--name", default="final")
    args = ap.parse_args()
    models, y = load_models(args.nn_tag, args.trackb, args.nn, args.fold0_weight)
    if not models:
        sys.exit("no models found")
    print("members:", ", ".join(f"{m}" + (f"[{d['n_test_models']} test nets]" if "n_test_models" in d else "") for m, d in models.items()))
    temps = calibrate(models, y, not args.no_calib)
    print("temperatures (fitted on all folds):", {m: round(t["test"], 2) for m, t in temps.items()})
    weights, singles = hill_climb(models, y, temps, args.assign, args.tau)
    print(f"\nsingle members: weighted post-assignment macro-F1 over folds {FOLDS} (weights {PROXY_WEIGHTS}); per fold post / pre")
    for m, (s, per, pre) in sorted(singles.items(), key=lambda t: -t[1][0]):
        print(f"  {m:20s} {s:.4f}  " + " ".join(f"b{k} {per[k]:.3f}/{pre[k]:.3f}" for k in FOLDS))
    if args.equal:
        weights = {m: 1 / len(models) for m in models}
    hc_w, hc_per, hc_pre = score(models, y, weights, temps, args.assign, args.tau)
    mean_w = {m: 1 / len(models) for m in models}
    mean_s, mean_per, _ = score(models, y, mean_w, temps, args.assign, args.tau)
    print(f"\nblend {json.dumps({m: round(w, 3) for m, w in weights.items()})}: {hc_w:.4f} per fold "
          + " ".join(f"b{k} {hc_per[k]:.3f}/{hc_pre[k]:.3f}" for k in FOLDS))
    print(f"equal-weight mean of all: {mean_s:.4f} per fold " + " ".join(f"b{k} {mean_per[k]:.3f}" for k in FOLDS))
    best_single = max(singles, key=lambda m: singles[m][0])
    single_per = singles[best_single][1]
    if any(hc_per[k] < single_per[k] - 0.005 for k in FOLDS):
        print("blend fails the robustness rule (loses > 0.5 pt to the best single on a fold) -> best single", best_single)
        weights = {best_single: 1.0}
        hc_w, hc_per, hc_pre = score(models, y, weights, temps, args.assign, args.tau)
    for k in (9, 7):
        P = blend(models, weights, k, temps)
        marg = np.bincount(y[k], minlength=7)[1:]
        print(f"  per-class F1 b{k} post-assignment: {np.round(per_class_f1(y[k], balanced_predict(P, marg, args.assign, args.tau) + 1), 3).tolist()}")

    # ---- test
    P = blend(models, weights, "test", temps)
    pred_pre = P.argmax(1)
    pred0 = balanced_predict(P, None, args.assign, args.tau)
    pred = CLASSES[pred0]
    moved = float(np.mean(pred0 != pred_pre))
    hist_pre = pred_hist(pred_pre + 1).tolist()
    print(f"\nTEST pre-assignment histogram: {hist_pre} (hist_l1 {hist_l1(pred_pre):.3f}) | mean max-prob {P.max(1).mean():.3f}")
    print(f"TEST post-assignment histogram: {pred_hist(pred).tolist()} | moved {moved:.3f}")
    print("flow (rows: pre class 1..6, cols: assigned):\n", flow_table(pred_pre, pred0))
    agree = {}
    for m in models:
        Pm = models[m]["test"]
        agree[m] = {"argmax": float(np.mean(Pm.argmax(1) == pred0)),
                    "post": float(np.mean(balanced_predict(Pm, None, args.assign, args.tau) == pred0))}
        print(f"  agreement with {m:20s}: argmax {agree[m]['argmax']:.3f} | post-assignment {agree[m]['post']:.3f}")
    conf_moved = P.max(1)[pred0 != pred_pre]
    if len(conf_moved):
        print("  max-prob quantiles of moved rows (10/50/90 %):", np.round(np.quantile(conf_moved, [0.1, 0.5, 0.9]), 3).tolist())
    problems = sanity_check(pred, N_TEST)
    if moved > 0.25:
        problems.append(f"assignment moved {moved:.1%} of rows (> 25 %): member set looks broken")
    if min(hist_pre) < 350 or max(hist_pre) > 900:
        problems.append(f"pre-assignment histogram {hist_pre} outside [350, 900]")
    if agree[best_single]["post"] < 0.75:
        problems.append(f"agreement with best single member {best_single} is {agree[best_single]['post']:.3f} < 0.75")
    for p in problems:
        print("GATE:", p)
    existing = sorted(SUBS.glob(f"sub_{args.name}_v*.csv"))
    n = len(existing) + 1
    path = SUBS / f"sub_{args.name}_v{n}.csv"
    report = {"weights": weights, "temperatures": {m: t["test"] for m, t in temps.items()}, "assign": args.assign,
              "tau": args.tau, "score_post": hc_w, "per_fold_post": hc_per, "per_fold_pre": hc_pre,
              "singles": {m: s[0] for m, s in singles.items()}, "test_hist_pre": hist_pre,
              "test_hist_post": pred_hist(pred).tolist(), "moved": moved, "agreement": agree,
              "flow": flow_table(pred_pre, pred0).tolist(), "problems": problems}
    REPORTS.mkdir(exist_ok=True)
    json.dump(report, open(REPORTS / f"blend_{args.name}_v{n}.json", "w"), indent=1, default=float)
    if problems and not args.force:
        print("NOT written (gates failed); use --force to write anyway. Report saved.")
        return
    test = load_test()
    write_submission(test["measurement_id"], pred, path)
    np.save(RUNS / f"blend_{args.name}_v{n}_test_probs.npy", P)
    print("wrote", path)


if __name__ == "__main__":
    main()
