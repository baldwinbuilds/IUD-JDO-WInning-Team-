"""Track B end-to-end: pattern features -> shrinkage LDA + LR -> class-balanced
self-training -> kNN propagation -> (prior reweighting) -> submission.

Evaluates on leave-one-batch-out (all 9 batches) and forward-chaining (7,8,9),
saves OOF probabilities for blending, then fits on all labelled data and
predicts batch 10.
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
from drift.data import (N_TEST, RUNS, SUBS, REPORTS, class_prior, load_test, load_train,  # noqa: E402
                        write_submission)
from drift.features import DEFAULT_LINEAR, FeatureBuilder  # noqa: E402
from drift.metrics import (NAMES, agreement, em_prior, macro_f1, per_class_f1, pred_hist,  # noqa: E402
                           reweight_prior, sanity_check)
from drift.models_linear import cbst, knn_propagate, make_lda, make_lr  # noqa: E402
from drift.splits import forward_splits, lobo_splits, weighted_score  # noqa: E402

VARIANTS = ["lda", "lda_cbst", "lr", "lr_cbst", "ens_cbst", "ens_cbst_prop"]


def run_split(fb_blocks, df_s, df_t):
    fb = FeatureBuilder(fb_blocks).fit(df_s)
    Xs, Xt = fb.transform(df_s), fb.transform(df_t)
    sc = StandardScaler().fit(Xs)
    Xs, Xt = sc.transform(Xs), sc.transform(Xt)
    ys = df_s["gas_class"].to_numpy(int)
    P0l, Pl, classes = cbst(make_lda, Xs, ys, Xt)
    P0r, Pr, _ = cbst(make_lr, Xs, ys, Xt)
    Pe = (Pl + Pr) / 2
    Pp = knn_propagate(Pe, Xt)
    probs = {"lda": P0l, "lda_cbst": Pl, "lr": P0r, "lr_cbst": Pr, "ens_cbst": Pe, "ens_cbst_prop": Pp}
    return probs, classes, class_prior(ys)


def evaluate(probs, classes, prior_src, yt):
    res = {}
    for k, P in probs.items():
        res[k] = macro_f1(yt, classes[P.argmax(1)])
    Pp = probs["ens_cbst_prop"]
    prior_t = class_prior(yt)
    res["prop+oracle_prior"] = macro_f1(yt, classes[reweight_prior(Pp, prior_src, prior_t).argmax(1)])
    W, _ = em_prior(Pp, prior_src)
    res["prop+em_prior"] = macro_f1(yt, classes[W.argmax(1)])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", default=list(DEFAULT_LINEAR))
    ap.add_argument("--tag", default="trackB")
    ap.add_argument("--force", action="store_true", help="write submission even if sanity checks fail")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    train, test = load_train(), load_test()
    batch = train["batch"].to_numpy(int)
    y = train["gas_class"].to_numpy(int)
    out_dir = RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    rows = []
    if not args.skip_eval:
        splits = [(f"lobo{k}", k, tr, te) for k, tr, te in lobo_splits(batch)]
        splits += [(f"fwd{k}", k, tr, te) for k, tr, te in forward_splits(batch)]
        for name, k, trm, tem in splits:
            probs, classes, prior_src = run_split(args.blocks, train[trm], train[tem])
            res = evaluate(probs, classes, prior_src, y[tem])
            res.update(split=name, batch=k, n=int(tem.sum()))
            rows.append(res)
            pcf = per_class_f1(y[tem], classes[probs["ens_cbst_prop"].argmax(1)])
            print(f"{name:7s} n={tem.sum():4d} " + " ".join(f"{v}={res[v]:.3f}" for v in VARIANTS + ["prop+oracle_prior", "prop+em_prior"])
                  + " | per-class F1(prop) " + " ".join(f"{NAMES[c]}={f:.2f}" for c, f in zip(range(1, 7), pcf))
                  + f"  t={time.time()-t0:.0f}s", flush=True)
            np.savez_compressed(out_dir / f"oof_{name}.npz", classes=classes, y=y[tem], idx=np.where(tem)[0],
                                prior_src=prior_src, **probs)
        table = pd.DataFrame(rows).set_index("split")
        table.to_csv(REPORTS / f"{args.tag}_lobo.csv")
        lobo = table[table.index.str.startswith("lobo")].set_index("batch")
        print("\nweighted LOBO score (6:1, 7:2, 8:2):")
        for v in VARIANTS + ["prop+oracle_prior", "prop+em_prior"]:
            print(f"  {v:20s} {weighted_score(lobo[v].to_dict()):.4f}")

    # ---- final fit on all labelled batches -> batch 10
    table_path = REPORTS / f"{args.tag}_lobo.csv"
    best_variant, use_prior = "ens_cbst", False
    if table_path.exists():
        lobo = pd.read_csv(table_path).query("split.str.startswith('lobo')", engine="python").set_index("batch")
        scores = {v: weighted_score(lobo[v].to_dict()) for v in ["lda_cbst", "ens_cbst", "ens_cbst_prop"]}
        best_variant = max(scores, key=scores.get)
        use_prior = weighted_score(lobo["prop+oracle_prior"].to_dict()) > weighted_score(lobo["ens_cbst_prop"].to_dict()) + 0.002
        print(f"\nselected variant by weighted LOBO: {best_variant} {scores} | prior reweighting helps on proxies: {use_prior}")
    probs, classes, prior_src = run_split(args.blocks, train, test)
    np.savez_compressed(out_dir / "test_probs.npz", classes=classes, prior_src=prior_src, **probs)
    Pb = probs[best_variant]
    uniform = np.full(6, 1 / 6)
    P_final = reweight_prior(Pb, prior_src, uniform) if use_prior else Pb
    pred_raw = classes[Pb.argmax(1)]
    pred = classes[P_final.argmax(1)]
    _, pt = em_prior(Pb, prior_src)
    print("\nTEST predicted histogram (argmax):        ", pred_hist(pred_raw).tolist())
    print("TEST predicted histogram (final):         ", pred_hist(pred).tolist())
    print("TEST EM-estimated prior:                  ", np.round(pt, 3).tolist())
    print("mean max-prob:", round(float(Pb.max(1).mean()), 3))
    for v, P in probs.items():
        print(f"   histogram {v:14s}", pred_hist(classes[P.argmax(1)]).tolist())
    problems = sanity_check(pred, N_TEST)
    for p in problems:
        print("SANITY:", p)
    path = SUBS / f"sub_{args.tag}_v1.csv"
    if problems and not args.force:
        print("NOT written (sanity failed). Re-run with --force to write anyway.")
    else:
        write_submission(test["measurement_id"], pred, path)
        print("wrote", path)
    json.dump({"blocks": args.blocks, "n_features": int(FeatureBuilder(args.blocks).fit(train).transform(train).shape[1])},
              open(out_dir / "config.json", "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
