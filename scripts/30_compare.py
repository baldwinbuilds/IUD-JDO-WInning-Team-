"""Build the LOBO comparison matrix (variants x held-out batches) from runs/<tag>/*.npz,
plus test-side diagnostics (predicted histograms, agreement between variants)."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift.data import REPORTS, RUNS  # noqa: E402
from drift.metrics import agreement, macro_f1, per_class_f1, pred_hist  # noqa: E402
from drift.splits import weighted_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="nn")
    ap.add_argument("--extra", nargs="*", default=[], help="other run dirs to include, e.g. trackB")
    args = ap.parse_args()
    files = sorted((RUNS / args.tag).glob("*_fold*_seed*.npz"))
    if not files:
        sys.exit(f"no runs in {RUNS / args.tag}")
    oof = defaultdict(lambda: defaultdict(list))     # variant -> fold -> list of prob arrays (seeds)
    y_of = {}
    test_probs = defaultdict(list)                     # variant -> list of test prob arrays
    for f in files:
        z = np.load(f, allow_pickle=True)
        v, k = str(z["variant"]), int(z["fold"])
        if k > 0:
            oof[v][k].append(z["oof"])
            y_of[k] = z["y_oof"]
        test_probs[v].append(z["test"])
    rows = []
    for v in sorted(oof):
        row = {"variant": v}
        for k in sorted(oof[v]):
            P = np.mean(oof[v][k], axis=0)
            row[f"b{k}"] = macro_f1(y_of[k], P.argmax(1) + 1)
            row[f"n_seeds_b{k}"] = len(oof[v][k])
        row["weighted_678"] = weighted_score({k: row[f"b{k}"] for k in (6, 7, 8) if f"b{k}" in row})
        row["b7_only"] = row.get("b7", float("nan"))   # batch 7 is large and class-balanced: fairest proxy for AdaBN/DA
        rows.append(row)
    table = pd.DataFrame(rows).set_index("variant").sort_values("weighted_678", ascending=False)
    pd.set_option("display.width", 200)
    print("\nheld-out batch class shares (min..max over 6 classes) - small/skewed batches under-rate AdaBN/DANN:")
    for k in sorted(y_of):
        sh = np.bincount(y_of[k].astype(int), minlength=7)[1:] / len(y_of[k])
        print(f"  b{k}: n={len(y_of[k])} shares {np.round(sh, 2).tolist()}")
    print("\n=== LOBO macro-F1 (seed-averaged probabilities) ===")
    print(table[[c for c in table.columns if not c.startswith("n_seeds")]].round(4).to_string())
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / f"{args.tag}_compare.csv")
    # per-class F1 on batches 6-8 for the top variants
    print("\n=== per-class F1 on held-out batches 6/7/8 ===")
    for v in table.index[:8]:
        for k in (6, 7, 8):
            if k in oof[v]:
                P = np.mean(oof[v][k], axis=0)
                print(f"  {v:5s} b{k}: {np.round(per_class_f1(y_of[k], P.argmax(1)+1), 3).tolist()}")
    # test-side diagnostics
    print("\n=== test predicted histograms (all folds & seeds averaged; organisers: 600 per class) ===")
    preds = {}
    for v in sorted(test_probs):
        P = np.mean(test_probs[v], axis=0)
        preds[v] = P.argmax(1) + 1
        print(f"  {v:5s} n_models={len(test_probs[v]):3d} hist={pred_hist(preds[v]).tolist()} mean_maxp={P.max(1).mean():.3f}")
    for extra in args.extra:
        z = np.load(RUNS / extra / "test_probs.npz", allow_pickle=True)
        for key in ("ens_cbst", "lda_cbst", "ens_cbst_prop"):
            if key in z:
                preds[f"{extra}:{key}"] = z["classes"][z[key].argmax(1)]
                print(f"  {extra}:{key:14s} hist={pred_hist(preds[f'{extra}:{key}']).tolist()}")
    names = list(preds)
    if len(names) > 1:
        print("\n=== test agreement between variants ===")
        M = pd.DataFrame([[agreement(preds[a], preds[b]) for b in names] for a in names], index=names, columns=names)
        print(M.round(3).to_string())
        M.to_csv(REPORTS / f"{args.tag}_test_agreement.csv")


if __name__ == "__main__":
    main()
