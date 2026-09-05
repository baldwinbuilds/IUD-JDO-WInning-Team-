"""Track B end-to-end: pattern+state features -> shrinkage LDA + LR -> class-balanced self-training
(plain, and under the known class marginal) + transductive LDA-EM -> submission.

Evaluates on leave-one-batch-out (all 9 batches) and forward-chaining (7,8,9), saves OOF
probabilities for blending, then fits on all labelled data and predicts batch 10.

Marginals: the organisers state 600 rows per class in the test, so the test-side self-training /
EM use a uniform marginal.  On held-out batches the batch's true class counts play the same role
(labels used only to set the six column masses - evaluation design, never per-row information); pass
--no-proxy-marginal to disable.  Every variant is scored both by argmax and after the balanced
optimal-transport assignment (`+ot`), which is what the final submission uses.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift.assign import balanced_predict, flow_table, hist_l1  # noqa: E402
from drift.data import (N_TEST, RUNS, SUBS, REPORTS, class_prior, load_test, load_train,  # noqa: E402
                        write_submission)
from drift.features import DEFAULT_LINEAR, FeatureBuilder  # noqa: E402
from drift.metrics import NAMES, macro_f1, per_class_f1, pred_hist, sanity_check  # noqa: E402
from drift.models_linear import cbst, knn_propagate, lda_em, make_lda, make_lr  # noqa: E402
from drift.splits import forward_splits, lobo_splits, weighted_score  # noqa: E402

VARIANTS = ["lda", "lr", "lda_cbst", "lr_cbst", "ens_cbst", "ens_cbst_prop",
            "lda_cbstb", "lr_cbstb", "ens_cbstb", "lda_em", "em_ensb"]
CANDIDATES = ["lda_cbst", "ens_cbst", "ens_cbstb", "lda_em", "em_ensb"]   # eligible for the submission


def run_split(fb_blocks, df_s, df_t, marginal, tau: float):
    fb = FeatureBuilder(fb_blocks).fit(df_s)
    Xs, Xt = fb.transform(df_s), fb.transform(df_t)
    sc = StandardScaler().fit(Xs)
    Xs, Xt = sc.transform(Xs), sc.transform(Xt)
    ys = df_s["gas_class"].to_numpy(int)
    P0l, Pl, classes = cbst(make_lda, Xs, ys, Xt)
    P0r, Pr, _ = cbst(make_lr, Xs, ys, Xt)
    _, Plb, _ = cbst(make_lda, Xs, ys, Xt, marginal=marginal, tau=tau)
    _, Prb, _ = cbst(make_lr, Xs, ys, Xt, marginal=marginal, tau=tau)
    _, Pem, _ = lda_em(Xs, ys, Xt, prior=marginal)
    Pe = (Pl + Pr) / 2
    Peb = (Plb + Prb) / 2
    probs = {"lda": P0l, "lr": P0r, "lda_cbst": Pl, "lr_cbst": Pr, "ens_cbst": Pe,
             "ens_cbst_prop": knn_propagate(Pe, Xt), "lda_cbstb": Plb, "lr_cbstb": Prb, "ens_cbstb": Peb,
             "lda_em": Pem, "em_ensb": (Pem + Peb) / 2}
    return probs, classes, class_prior(ys)


def evaluate(probs, classes, yt, marginal, tau: float):
    res = {}
    for k, P in probs.items():
        res[k] = macro_f1(yt, classes[P.argmax(1)])
        res[k + "+ot"] = macro_f1(yt, classes[balanced_predict(P, marginal, "sinkhorn", tau)])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", default=list(DEFAULT_LINEAR))
    ap.add_argument("--tag", default="trackB")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="Sinkhorn temperature for pseudo-labels and the final assignment")
    ap.add_argument("--no-proxy-marginal", action="store_true",
                    help="do not use the held-out batch's class counts as its marginal (uniform instead)")
    ap.add_argument("--force", action="store_true", help="write the argmax submission even if sanity checks fail")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    train, test = load_train(), load_test()
    batch = train["batch"].to_numpy(int)
    y = train["gas_class"].to_numpy(int)
    out_dir = RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    uniform = np.ones(6)

    rows = []
    if not args.skip_eval:
        splits = [(f"lobo{k}", k, tr, te) for k, tr, te in lobo_splits(batch)]
        splits += [(f"fwd{k}", k, tr, te) for k, tr, te in forward_splits(batch)]
        for name, k, trm, tem in splits:
            yt = y[tem]
            marg = uniform if args.no_proxy_marginal else np.bincount(yt, minlength=7)[1:].astype(float)
            probs, classes, prior_src = run_split(args.blocks, train[trm], train[tem], marg, args.tau)
            res = evaluate(probs, classes, yt, marg, args.tau)
            res.update(split=name, batch=k, n=int(tem.sum()))
            rows.append(res)
            pcf = per_class_f1(yt, classes[probs["ens_cbstb"].argmax(1)])
            line = f"{name:7s} n={tem.sum():4d} "
            line += " ".join(f"{v}={res[v]:.3f}/{res[v + '+ot']:.3f}" for v in VARIANTS)
            line += " | per-class F1(ens_cbstb) " + " ".join(f"{NAMES[c]}={f:.2f}" for c, f in zip(range(1, 7), pcf))
            line += f"  t={time.time() - t0:.0f}s"
            print(line, flush=True)
            np.savez_compressed(out_dir / f"oof_{name}.npz", classes=classes, y=yt, idx=np.where(tem)[0],
                                prior_src=prior_src, marginal=marg, **probs)
        table = pd.DataFrame(rows).set_index("split")
        table.to_csv(REPORTS / f"{args.tag}_lobo.csv")
        lobo = table[table.index.str.startswith("lobo")].set_index("batch")
        print("\nweighted LOBO score (6:1, 7:2, 8:1, 9:2)  argmax / +ot   [b9 argmax/+ot]:")
        for v in VARIANTS:
            print(f"  {v:14s} {weighted_score(lobo[v].to_dict()):.4f} / {weighted_score(lobo[v + '+ot'].to_dict()):.4f}"
                  f"   [{lobo.loc[9, v]:.4f} / {lobo.loc[9, v + '+ot']:.4f}]")

    # ---- final fit on all labelled batches -> batch 10 (uniform marginal: 600 per class)
    table_path = REPORTS / f"{args.tag}_lobo.csv"
    best_variant = "ens_cbstb"
    if table_path.exists():
        lobo = pd.read_csv(table_path).query("split.str.startswith('lobo')", engine="python").set_index("batch")
        scores = {v: weighted_score(lobo[v + "+ot"].to_dict()) for v in CANDIDATES}
        best_variant = max(scores, key=scores.get)
        print(f"\nselected variant by weighted LOBO (+ot): {best_variant} "
              + json.dumps({k: round(s, 4) for k, s in scores.items()}))
    probs, classes, prior_src = run_split(args.blocks, train, test, uniform, args.tau)
    np.savez_compressed(out_dir / "test_probs.npz", classes=classes, prior_src=prior_src, **probs)
    Pb = probs[best_variant]
    pred_raw = classes[Pb.argmax(1)]
    pred_ot = classes[balanced_predict(Pb, uniform, "sinkhorn", args.tau)]
    print("\nTEST predicted histograms (argmax):")
    for v, P in probs.items():
        print(f"   {v:14s}", pred_hist(classes[P.argmax(1)]).tolist(), f" hist_l1={hist_l1(P.argmax(1)):.3f}")
    print(f"\n{best_variant}: argmax hist {pred_hist(pred_raw).tolist()} -> +ot hist {pred_hist(pred_ot).tolist()}"
          f" | moved {np.mean(pred_raw != pred_ot):.3f} | mean max-prob {Pb.max(1).mean():.3f}")
    print("flow (rows: argmax class 1..6, cols: assigned class):\n", flow_table(pred_raw - 1, pred_ot - 1))
    problems = sanity_check(pred_raw, N_TEST)
    for p in problems:
        print("SANITY(argmax):", p)
    path = SUBS / f"sub_{args.tag}_v1.csv"
    if problems and not args.force:
        print("argmax submission NOT written (sanity failed). Re-run with --force to write anyway.")
    else:
        write_submission(test["measurement_id"], pred_raw, path)
        print("wrote", path)
    path_ot = SUBS / f"sub_{args.tag}_ot_v1.csv"
    write_submission(test["measurement_id"], pred_ot, path_ot)
    print("wrote", path_ot)
    json.dump({"blocks": args.blocks, "tau": args.tau, "best_variant": best_variant,
               "n_features": int(FeatureBuilder(args.blocks).fit(train).transform(train).shape[1])},
              open(out_dir / "config.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
