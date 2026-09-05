"""Blend Track B and NN models on leave-one-batch-out OOF predictions (batches 6-8),
optionally tune per-class logit biases for macro-F1, sanity-check, and write the submission.

Models available for blending:
  trackB:<variant>   from runs/trackB/oof_lobo{k}.npz and runs/trackB/test_probs.npz
  nn:<variant>       from runs/<nn_tag>/<variant>_fold{k}_seed*.npz (seed-averaged OOF; test probs averaged over all folds & seeds)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift.data import N_TEST, REPORTS, RUNS, SUBS, load_test, write_submission  # noqa: E402
from drift.metrics import CLASSES, em_prior, macro_f1, per_class_f1, pred_hist, reweight_prior, sanity_check  # noqa: E402
from drift.splits import PROXY_WEIGHTS, weighted_score  # noqa: E402

FOLDS = (6, 7, 8)


def load_models(nn_tag: str, trackb_variants, nn_variants):
    """Return dict name -> {'oof': {k: probs}, 'test': probs}, and y dict k -> labels."""
    models, y = {}, {}
    tb = RUNS / "trackB"
    if tb.exists():
        tp = np.load(tb / "test_probs.npz", allow_pickle=True)
        for v in trackb_variants:
            m = {"oof": {}, "test": tp[v]}
            for k in FOLDS:
                z = np.load(tb / f"oof_lobo{k}.npz", allow_pickle=True)
                m["oof"][k] = z[v]
                y[k] = z["y"]
            models[f"trackB:{v}"] = m
    nd = RUNS / nn_tag
    if nd.exists():
        oof = defaultdict(lambda: defaultdict(list))
        test = defaultdict(list)
        for f in sorted(nd.glob("*_fold*_seed*.npz")):
            z = np.load(f, allow_pickle=True)
            v, k = str(z["variant"]), int(z["fold"])
            if nn_variants and v not in nn_variants:
                continue
            if k in FOLDS:
                oof[v][k].append(z["oof"])
                y[k] = z["y_oof"].astype(int)
            test[v].append(z["test"])
        for v in oof:
            if all(k in oof[v] for k in FOLDS):
                models[f"nn:{v}"] = {"oof": {k: np.mean(oof[v][k], axis=0) for k in FOLDS},
                                     "test": np.mean(test[v], axis=0), "n_test_models": len(test[v])}
            else:
                print(f"  (skipping nn:{v}: folds present {sorted(oof[v])}, need {FOLDS})")
    return models, y


def score(models, y, weights: dict, bias=None):
    per = {}
    for k in FOLDS:
        P = sum(w * models[m]["oof"][k] for m, w in weights.items())
        if bias is not None:
            P = np.exp(np.log(P + 1e-9) + bias)
        per[k] = macro_f1(y[k], P.argmax(1) + 1)
    return weighted_score(per), per


def hill_climb(models, y, n_iter: int = 30):
    names = list(models)
    singles = {m: score(models, y, {m: 1.0})[0] for m in names}
    best = max(singles, key=singles.get)
    counts = defaultdict(int)
    counts[best] += 1
    cur, _ = score(models, y, {best: 1.0})
    for _ in range(n_iter):
        cand = {}
        for m in names:
            c = defaultdict(int, counts)
            c[m] += 1
            tot = sum(c.values())
            cand[m] = score(models, y, {mm: cc / tot for mm, cc in c.items()})[0]
        m = max(cand, key=cand.get)
        if cand[m] < cur - 1e-9:
            break
        counts[m] += 1
        cur = cand[m]
    tot = sum(counts.values())
    return {m: c / tot for m, c in counts.items()}, singles


def tune_bias(models, y, weights, steps=np.arange(-1.0, 1.01, 0.1), rounds: int = 2):
    """Coordinate descent on per-class log-prob biases; accepted only if no proxy batch gets worse."""
    bias = np.zeros(6)
    base_w, base_per = score(models, y, weights)
    for _ in range(rounds):
        for c in range(6):
            best_s, best_v = None, bias[c]
            for s in steps:
                b = bias.copy()
                b[c] = s
                w, per = score(models, y, weights, b)
                if all(per[k] >= base_per[k] - 1e-9 for k in FOLDS) and (best_s is None or w > best_s):
                    best_s, best_v = w, s
            bias[c] = best_v
    tuned_w, tuned_per = score(models, y, weights, bias)
    if tuned_w <= base_w + 1e-4:
        return np.zeros(6), base_w, base_per
    return bias, tuned_w, tuned_per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-tag", default="nn")
    ap.add_argument("--trackb", nargs="*", default=["ens_cbst", "lda_cbst"])
    ap.add_argument("--nn", nargs="*", default=None, help="restrict NN variants (default: all with folds 6,7,8)")
    ap.add_argument("--bias", action="store_true", help="tune per-class log biases on proxies (off by default: proxies are near-saturated and it over-fits)")
    ap.add_argument("--prior", choices=["none", "uniform"], default="none")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--name", default="final")
    args = ap.parse_args()
    models, y = load_models(args.nn_tag, args.trackb, args.nn)
    if not models:
        sys.exit("no models found")
    print("models:", ", ".join(models))
    weights, singles = hill_climb(models, y)
    print("\nsingle-model weighted LOBO(6,7,8) macro-F1:")
    for m, s in sorted(singles.items(), key=lambda t: -t[1]):
        print(f"  {m:24s} {s:.4f}  per-batch {[round(score(models, y, {m: 1.0})[1][k], 4) for k in FOLDS]}")
    hc_w, hc_per = score(models, y, weights)
    mean_w = {m: 1 / len(models) for m in models}
    mean_s, mean_per = score(models, y, mean_w)
    print(f"\nhill-climb blend {json.dumps({m: round(w, 3) for m, w in weights.items()})}: {hc_w:.4f} per-batch {[round(hc_per[k], 4) for k in FOLDS]}")
    print(f"simple mean of all: {mean_s:.4f} per-batch {[round(mean_per[k], 4) for k in FOLDS]}")
    best_single = max(singles, key=singles.get)
    # robustness rule: blend must not lose > 0.5pt to the best single on any proxy batch
    single_per = score(models, y, {best_single: 1.0})[1]
    if any(hc_per[k] < single_per[k] - 0.005 for k in FOLDS):
        print("hill-climb blend fails the robustness rule -> falling back to best single", best_single)
        weights = {best_single: 1.0}
    bias = np.zeros(6)
    if args.bias:
        bias, bw, bper = tune_bias(models, y, weights)
        print(f"per-class log-bias {np.round(bias, 2).tolist()} -> {bw:.4f} per-batch {[round(bper[k], 4) for k in FOLDS]}")
    # ---- test
    P = sum(w * models[m]["test"] for m, w in weights.items())
    P = np.exp(np.log(P + 1e-9) + bias)
    P = P / P.sum(1, keepdims=True)
    prior_src = None
    if args.prior == "uniform":
        # source prior of the blended models is ~uniform already (class-balanced training) for NN; use the
        # empirical EM prior as the reference instead to avoid double counting
        _, pt = em_prior(P, np.full(6, 1 / 6))
        print("EM prior estimate on test:", np.round(pt, 3).tolist())
        P = reweight_prior(P, pt, np.full(6, 1 / 6))
    pred = CLASSES[P.argmax(1)]
    _, pt = em_prior(P, np.full(6, 1 / 6))
    print("\nTEST histogram:", pred_hist(pred).tolist(), "| EM prior:", np.round(pt, 3).tolist(),
          "| mean max-prob:", round(float(P.max(1).mean()), 3))
    for m in models:
        print(f"  agreement with {m:24s}: {np.mean(CLASSES[models[m]['test'].argmax(1)] == pred):.3f}")
    problems = sanity_check(pred, N_TEST)
    for p in problems:
        print("SANITY:", p)
    existing = sorted(SUBS.glob(f"sub_{args.name}_v*.csv"))
    n = len(existing) + 1
    path = SUBS / f"sub_{args.name}_v{n}.csv"
    report = {"weights": weights, "bias": bias.tolist(), "score_weighted_678": hc_w, "per_batch": hc_per,
              "singles": singles, "test_hist": pred_hist(pred).tolist(), "problems": problems, "prior": args.prior}
    REPORTS.mkdir(exist_ok=True)
    json.dump(report, open(REPORTS / f"blend_{args.name}_v{n}.json", "w"), indent=1, default=float)
    if problems and not args.force:
        print("NOT written (sanity failed); use --force to write anyway. Report saved.")
        return
    test = load_test()
    write_submission(test["measurement_id"], pred, path)
    np.save(RUNS / f"blend_{args.name}_v{n}_test_probs.npy", P)
    print("wrote", path)


if __name__ == "__main__":
    main()
