"""Build the LOBO comparison matrix (variants x held-out batches) from runs/<tag>/*.npz, plus
test-side diagnostics (predicted histograms, label-free validators, agreement between variants).

Every held-out batch is scored twice: by argmax and after the balanced optimal-transport assignment
with the batch's true class counts as marginal (`_ot`; the analogue of the organisers' 600-per-class
statement, which is what the submission uses).  The decision score is the weighted `_ot` score over
batches 6/7/8/9 (weights 1/2/1/2); batch 9 is the only proxy with drift comparable to the test."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift.assign import balanced_predict, hist_l1  # noqa: E402
from drift.data import REPORTS, RUNS  # noqa: E402
from drift.metrics import agreement, macro_f1, per_class_f1, pred_hist  # noqa: E402
from drift.splits import PROXY_WEIGHTS, weighted_score  # noqa: E402

FOLDS = (6, 7, 8, 9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="nn")
    ap.add_argument("--extra", nargs="*", default=[], help="other run dirs to include, e.g. trackB")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--no-alts", dest="alts", action="store_false", help="hide the <variant>@<alt> read-out rows")
    args = ap.parse_args()
    files = sorted((RUNS / args.tag).glob("*_fold*_seed*.npz"))
    if not files:
        sys.exit(f"no runs in {RUNS / args.tag}")
    oof = defaultdict(lambda: defaultdict(list))     # variant -> fold -> list of prob arrays (seeds)
    y_of = {}
    test_probs = defaultdict(list)                     # variant -> list of test prob arrays
    validators = defaultdict(list)
    for f in files:
        z = np.load(f, allow_pickle=True)
        v, k = str(z["variant"]), int(z["fold"])
        if k > 0:
            oof[v][k].append(z["oof"])
            y_of[k] = z["y_oof"].astype(int)
        test_probs[v].append(z["test"])
        if args.alts:                                  # other read-outs of the same network (e.g. @swa_raw / @adabn)
            for key in z.files:
                if key.startswith("alt_") and key.endswith("_oof") and not key.endswith("last_oof") and not key.endswith("last_adabn_oof"):
                    name = key[len("alt_"):-len("_oof")]
                    if np.array_equal(z["oof"], z[key]) and np.array_equal(z["test"], z[f"alt_{name}_test"]):
                        continue
                    va = f"{v}@{name}"
                    if k > 0:
                        oof[va][k].append(z[key])
                    test_probs[va].append(z[f"alt_{name}_test"])
        if "validators" in z:
            validators[v].append(json.loads(str(z["validators"])))
    rows = []
    for v in sorted(oof):
        row = {"variant": v}
        for k in sorted(oof[v]):
            P = np.mean(oof[v][k], axis=0)
            marg = np.bincount(y_of[k], minlength=7)[1:]
            row[f"b{k}"] = macro_f1(y_of[k], P.argmax(1) + 1)
            row[f"b{k}_ot"] = macro_f1(y_of[k], balanced_predict(P, marg, "sinkhorn", args.tau) + 1)
            row[f"n_seeds_b{k}"] = len(oof[v][k])
        row["score_ot"] = weighted_score({k: row[f"b{k}_ot"] for k in FOLDS if f"b{k}_ot" in row})
        row["score_argmax"] = weighted_score({k: row[f"b{k}"] for k in FOLDS if f"b{k}" in row})
        rows.append(row)
    table = pd.DataFrame(rows).set_index("variant")
    table = table.sort_values(["score_ot", "b9_ot" if "b9_ot" in table else "b7_ot"], ascending=False, na_position="last")
    pd.set_option("display.width", 250)
    print("\nheld-out batch class shares (min..max over 6 classes) - small/skewed batches under-rate AdaBN/DANN:")
    for k in sorted(y_of):
        sh = np.bincount(y_of[k], minlength=7)[1:] / len(y_of[k])
        print(f"  b{k}: n={len(y_of[k])} shares {np.round(sh, 2).tolist()}")
    print(f"\n=== LOBO macro-F1 (seed-averaged probabilities); _ot = after balanced assignment with the true marginal; "
          f"score weights {PROXY_WEIGHTS} ===")
    cols = [c for c in table.columns if not c.startswith("n_seeds")]
    print(table[cols].round(4).to_string())
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / f"{args.tag}_compare.csv")
    print("\n=== per-class F1 (argmax / _ot) on held-out batches 9 and 7 for the top variants ===")
    for v in table.index[:8]:
        for k in (9, 7):
            if k in oof[v]:
                P = np.mean(oof[v][k], axis=0)
                marg = np.bincount(y_of[k], minlength=7)[1:]
                a = np.round(per_class_f1(y_of[k], P.argmax(1) + 1), 3).tolist()
                b = np.round(per_class_f1(y_of[k], balanced_predict(P, marg, "sinkhorn", args.tau) + 1), 3).tolist()
                print(f"  {v:5s} b{k}: {a} / {b}")
    print("\n=== test predicted histograms (all folds & seeds averaged; organisers: 600 per class) ===")
    preds = {}
    for v in sorted(test_probs):
        P = np.mean(test_probs[v], axis=0)
        preds[v] = P.argmax(1) + 1
        val = ""
        if validators[v]:
            m = {k: np.mean([d[k] for d in validators[v]]) for k in validators[v][0]}
            val = " | BNM %.3f IM %.3f AMI %.3f" % (m["bnm"], m["im"], m["ami"])
        print(f"  {v:5s} n_models={len(test_probs[v]):3d} hist={pred_hist(preds[v]).tolist()} hist_l1={hist_l1(P.argmax(1)):.3f}"
              f" mean_maxp={P.max(1).mean():.3f}{val}")
    for extra in args.extra:
        z = np.load(RUNS / extra / "test_probs.npz", allow_pickle=True)
        for key in ("ens_cbst", "ens_cbstb", "lda_em", "em_ensb"):
            if key in z:
                preds[f"{extra}:{key}"] = z["classes"][z[key].argmax(1)]
                print(f"  {extra}:{key:10s} hist={pred_hist(preds[f'{extra}:{key}']).tolist()} hist_l1={hist_l1(z[key].argmax(1)):.3f}")
    names = list(preds)
    if len(names) > 1:
        print("\n=== test agreement between variants (argmax) ===")
        M = pd.DataFrame([[agreement(preds[a], preds[b]) for b in names] for a in names], index=names, columns=names)
        print(M.round(3).to_string())
        M.to_csv(REPORTS / f"{args.tag}_test_agreement.csv")


if __name__ == "__main__":
    main()
