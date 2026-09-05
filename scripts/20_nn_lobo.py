"""Track A: train NN variants in leave-one-batch-out fashion.

For each (variant, held-out batch k, seed): source = labelled batches != k; unlabeled
target = held-out batch k  +  the real test batch (both get domain ids, no gas labels).
Saves runs/<tag>/<variant>_fold<k>_seed<s>.npz with OOF probs on batch k and probs on the test.
With --folds 0 the source is all 9 batches and the target is the test only (final fit).

Example (GPU box):
  python scripts/20_nn_lobo.py --variants a0 a1 a2 --folds 1 2 3 4 5 6 7 8 9 --seeds 0 1 2 3 4 --device cuda
Smoke test (CPU):
  python scripts/20_nn_lobo.py --variants a0 a1 --folds 8 --seeds 0 --epochs 3 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift.data import RUNS, load_test, load_train  # noqa: E402
from drift.features import DEFAULT_NN  # noqa: E402
from drift.metrics import bnm, class_ami, info_max, macro_f1, per_class_f1, pred_hist  # noqa: E402
from drift.nn.data import build_arrays  # noqa: E402
from drift.nn.selftrain import self_train  # noqa: E402
from drift.nn.train import RunConfig, train_one  # noqa: E402

VARIANTS = {
    # reference / ablations
    "a0": dict(arch="mlp", da="none"),
    "a0nb": dict(arch="mlp", da="none", balanced=False),
    "a1": dict(arch="mlp", da="dann", lam_max=0.3),
    "a1l": dict(arch="mlp", da="dann", lam_max=0.1),
    "a2": dict(arch="mlp", da="cdan", lam_max=0.3),
    "a4": dict(arch="mlp", da="coral", coral_w=1.0),
    # AdaBN family (test-domain BatchNorm statistics) - proxy-validated: fold8 .93 -> .956
    "a3": dict(arch="mlp", da="none", adabn=True),
    "a3d": dict(arch="mlp", da="dann", lam_max=0.3, adabn=True),
    "a3dl": dict(arch="mlp", da="dann", lam_max=0.1, adabn=True),
    "a2b": dict(arch="mlp", da="cdan", lam_max=0.3, adabn=True),
    "a2bl": dict(arch="mlp", da="cdan", lam_max=0.1, adabn=True),
    "a4b": dict(arch="mlp", da="coral", coral_w=1.0, adabn=True),
    "a5b": dict(arch="cnn", da="none", adabn=True, hidden=(256,)),
    "a6": dict(arch="tabm", da="none", hidden=(512, 512, 512)),      # TabM has no BatchNorm: AdaBN impossible
    "a7b": dict(arch="mlp", da="none", adabn=True, selftrain_rounds=(0.5,)),
    "a7b3": dict(arch="mlp", da="none", adabn=True, selftrain_rounds=(0.3, 0.5, 0.7)),
    "a9b": dict(arch="mlp", da="none", adabn=True, selftrain_rounds=(0.3, 0.5, 0.7), sinkhorn=True),
    # Sinkhorn pseudo-labels ranked/weighted by the balanced posterior (the prior actually acts), softer tau
    "a9q": dict(arch="mlp", da="none", adabn=True, selftrain_rounds=(0.3, 0.5, 0.7), sinkhorn=True, sinkhorn_rank="q"),
    "a9s": dict(arch="mlp", da="none", adabn=True, selftrain_rounds=(0.5, 0.7, 0.9), sinkhorn=True, sinkhorn_rank="q", sinkhorn_tau=0.4),
    "a9t": dict(arch="mlp", da="none", adabn=True, selftrain_rounds=(0.5, 0.7, 0.9), sinkhorn=True, sinkhorn_rank="q", sinkhorn_tau=1.0),
    "a3w": dict(arch="mlp", da="none", adabn=True, hidden=(1024, 1024, 1024), dropout=0.25),
    "a3nb": dict(arch="mlp", da="none", adabn=True, balanced=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["a0", "a1"])
    ap.add_argument("--folds", nargs="+", type=int, default=[6, 7, 8, 9])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--blocks", nargs="+", default=list(DEFAULT_NN))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="nn")
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads (GPU runs need few; 0 = torch default)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    out_dir = RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    train, test = load_train(), load_test()
    batch = train["batch"].to_numpy(int)
    log_path = out_dir / "log.txt"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    for k in args.folds:
        if k == 0:
            src_df, held_df = train, None
        else:
            src_df, held_df = train[batch != k], train[batch == k]
        tgt_df = test if held_df is None else pd.concat([held_df, test], ignore_index=True)
        n_eval = 0 if held_df is None else len(held_df)
        y_eval = None if held_df is None else held_df["gas_class"].to_numpy(int)
        for seed in args.seeds:
            arr = build_arrays(src_df, tgt_df, blocks=args.blocks, seed=seed)
            for v in args.variants:
                path = out_dir / f"{v}_fold{k}_seed{seed}.npz"
                if path.exists() and not args.overwrite:
                    log(f"skip {path.name} (exists)")
                    continue
                cfg = RunConfig(variant=v, seed=seed, blocks=tuple(args.blocks), **VARIANTS[v])
                if cfg.arch == "cnn":
                    names = arr["names"]
                    assert names[0].startswith("slog_s0_") and names[128].startswith("pat_s0_"),                         "cnn expects blocks to start with slog then pattern (16x8 each)"
                if args.epochs:
                    cfg.epochs = args.epochs
                t0 = time.time()
                log(f"== {v} fold{k} seed{seed} device={args.device} n_src={len(src_df)} n_tgt={len(tgt_df)} "
                    f"in_dim={arr['Xs'].shape[1]} epochs={cfg.epochs}")
                res = train_one(cfg, arr["Xs"], arr["ys"], arr["bs"], arr["Xt"], arr["bt"], n_domains=10,
                                device=args.device, y_eval=y_eval, n_eval=n_eval or None, verbose=args.verbose)
                probs = res["probs"]
                extra = {}
                if cfg.selftrain_rounds:
                    held_marg = np.bincount(y_eval - 1, minlength=6) if y_eval is not None else None
                    st = self_train(cfg, arr["Xs"], arr["ys"], arr["bs"], arr["Xt"], arr["bt"], probs,
                                    n_domains=10, device=args.device, y_eval=y_eval, n_eval=n_eval or None,
                                    verbose=args.verbose, log=log, heldout_marginal=held_marg)
                    extra["probs_base"] = probs
                    probs = st["probs"]
                    extra["selftrain"] = json.dumps(st["rounds"])
                oof = probs[:n_eval] if n_eval else np.zeros((0, 6))
                tst = probs[n_eval:]
                val = {"bnm": bnm(tst), "im": info_max(tst), "ami": class_ami(tst, arr["Xt"][n_eval:], seed=seed)}
                extra["validators"] = json.dumps(val)
                msg = (f"   done in {time.time()-t0:.0f}s | test hist={pred_hist(tst.argmax(1)+1).tolist()}"
                       f" | test validators BNM={val['bnm']:.3f} IM={val['im']:.3f} ClassAMI={val['ami']:.3f}")
                if n_eval:
                    f1 = macro_f1(y_eval, oof.argmax(1) + 1)
                    pcf = per_class_f1(y_eval, oof.argmax(1) + 1)
                    msg += f" | heldout batch{k} macroF1={f1:.4f} per-class={np.round(pcf, 3).tolist()}"
                    alts = {"last": res["probs_last"], "swa_raw": res.get("probs_swa_raw"), "last_adabn": res.get("probs_last_adabn")}
                    msg += " | alt heldout F1: " + " ".join(f"{a}={macro_f1(y_eval, P[:n_eval].argmax(1)+1):.4f}" for a, P in alts.items() if P is not None)
                for a in ("probs_last", "probs_swa_raw", "probs_last_adabn"):
                    if res.get(a) is not None:
                        extra[a.replace("probs_", "alt_")] = res[a]
                log(msg)
                np.savez_compressed(path, oof=oof, test=tst, y_oof=(y_eval if y_eval is not None else np.zeros(0)),
                                    fold=k, seed=seed, variant=v, config=json.dumps(cfg.to_dict()),
                                    history=json.dumps(res["history"]), seconds=res["seconds"], **extra)


if __name__ == "__main__":
    main()
