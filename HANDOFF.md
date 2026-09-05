# HANDOFF — Inter-uni Datathon Stream 3 (gas-sensor drift, batch 10)

Updated 2026-09-05 ~21:00 local on the GPU desktop (RTX 3060 Ti). Everything below was verified in-session
unless marked otherwise. Read this file, then `README_GPU.md`.

## 0. Deadline, deliverable, rules

- **Deadline ≈ 2026-09-07 01:00 local.** Keep a 4–6 h buffer.
- Deliverable: one CSV exactly like `data/sample_submission.csv` — header `measurement_id,gas_class`, 3,600 rows,
  same order as the test file, integers 1–6. `drift.data.write_submission` enforces the id order.
- Metric: **macro-F1** over 6 classes (1 Ethanol, 2 Ethylene, 3 Ammonia, 4 Acetaldehyde, 5 Acetone, 6 Toluene).
- Organisers state the hidden test has **600 rows per class**. We use this as a *known class marginal* (see §2).
- **Rules (hard):** no identifying the source dataset, no external copies, no recovering hidden labels.
  Never use `measurement_id` / row position for prediction. Unlabeled test *features* may be used for domain
  adaptation. Do not name the source dataset anywhere. Leak audit before submitting:
  `grep -rn "measurement_id\|idnum\|sort_values" src scripts` — only reading/writing CSVs may match.
- `data/test.csv` is the real 3,600-row file (copied from `data from kaggle/`); the old 2,288-row placeholder is gone.

## 1. What exists (all committed)

```
data/                train.csv (10,310 rows, batches 1–9), test.csv (3,600 rows, batch 10), sample_submission.csv
src/drift/
  assign.py          sinkhorn_marginal / hungarian_assign / balanced_predict: optimal-transport assignment of rows to
                     classes under a KNOWN marginal (600/class on the test; the held-out batch's true counts on LOBO
                     folds — labels used only as the six column masses); flow_table, hist_l1
  features.py        FeatureBuilder(blocks): slog, pattern, logscale, logconc, state, shape, typemed
  splits.py          lobo/forward splits; PROXY_WEIGHTS {6:1, 7:2, 8:1, 9:2}; strict weighted_score (NaN if a fold is missing)
  metrics.py         macro_f1, per_class_f1, pred_hist, sanity_check, label-free validators bnm / info_max / class_ami
  models_linear.py   LDA/LR; cbst(marginal=...) = class-balanced self-training whose pseudo-labels come from the
                     Sinkhorn-balanced posterior; lda_em = transductive class-mean EM with a fixed prior
  nn/nets.py         MLP / TabM / 1-D CNN encoders, DANN/CDAN heads, adabn() (raises if no BatchNorm)
  nn/train.py        train_one(): class-balanced index batching, label smoothing, AdamW+OneCycle, SWA (hand-written
                     running mean: torch 2.11 broke AveragedModel(use_buffers)), DA losses, per-domain AdaBN
  nn/selftrain.py    self_train(): per-domain pseudo-labels; with cfg.sinkhorn the test domain uses the uniform
                     marginal and the held-out batch its true counts; rank/weight by the balanced posterior
                     (sinkhorn_rank='q'); rounds ≥1 read out RUNNING BN statistics (selftrain_output='swa_raw')
scripts/
  10_lda_track.py    Track B → runs/trackB/, reports/trackB_lobo.csv, submissions/sub_trackB_v1.csv (argmax) and
                     sub_trackB_ot_v1.csv (balanced assignment). ~140 s CPU.
  20_nn_lobo.py      NN variants × folds × seeds → runs/<tag>/<variant>_fold<k>_seed<s>.npz (+ alt_* read-outs, log.txt)
  30_compare.py      LOBO matrix with `_ot` columns (post-assignment), @alt rows, test histograms/validators/agreement
  40_blend_submit.py temperature-calibrated log-space blend, hill-climb on post-assignment score over folds 6–9 with a
                     label-free test-histogram penalty, final balanced assignment, gates → submissions/sub_<name>_vN.csv
runs/ (git-ignored)  trackB/, trackB_noconc/, nn/ (Stage 1+2 npz files), smoke*/
```

## 2. What we learned today (the important part)

1. **Batch 9 is the only proxy that behaves like the test.** Batches 6/7/8 saturate (.96–.99 for anything decent);
   batch 9 (470 rows, fairly balanced, most recent) is where drift bites (raw LDA .73, LR .88). Batch 10 is 2–3×
   further from the training manifold than batch 9 (agent-measured nearest-neighbour distances). Model selection and
   blending now use folds 6/7/8/9 with weights 1/2/1/2 and are scored **after** the balanced assignment.
2. **The 600-per-class statement is the biggest lever.** With the true marginal, the optimal-transport assignment
   lifts Track B on batch 9 from .909 to .973 while leaving batch 7 at .998. On the test, the OT step moves ~6–11 % of
   rows, mostly Ethanol→Acetaldehyde. Soft prior *reweighting* (tried earlier) hurt; the hard marginal does not.
3. **Plain class-balanced self-training amplified the test skew** (raw LDA predicts 650 Acetaldehyde on the test,
   plain cbst 441). Selecting pseudo-labels under the Sinkhorn-balanced posterior fixes it: Track B `ens_cbstb`
   gets batch 9 .968 (argmax) / .980 (+ot) and an unforced test histogram of 621/562/650/646/538/583
   (was 928/519/568/413/567/605).
4. **AdaBN and self-training must not be stacked at read-out.** AdaBN is essential for the raw MLP (batch 9 .78 → .97),
   but once target rows are trained in (pseudo-labels), re-estimating BN on the target double-counts the shift:
   fold 9 .85 (AdaBN) vs .99 (running statistics) for the same network. Self-trained variants now read out running
   statistics; the AdaBN read-out is saved as `alt_adabn_*`.
5. **Which NN variant.** Seed 0, primary read-out, batch 9 / batch 7 / unforced test histogram:
   a9s (AdaBN teacher, 3 Sinkhorn rounds, τ=0.4, rank by balanced posterior) .989 / .999 / 657-535-667-605-540-596;
   a9t (τ=1) .989 / .999 / 855 Ethanol; a9b (rank by model prob, τ=0.1) .989 / .997 / 779 Ethanol;
   a7b (1 round, no Sinkhorn) .989 / .972 / 822 Ethanol; a3 (AdaBN only) .974 / .975 / 747 Ethanol;
   a5b (CNN) .957 / .966. Only a9s balances the test by itself; the blend's histogram penalty handles the others.
   All adversarial variants (DANN/CDAN/CORAL) lost on every proxy — dropped. TabM has no BatchNorm (AdaBN impossible).
6. **The two families disagree on ~23 % of test rows** (NN vs Track B; 91–95 % within the NN family) although both
   score ≥ .98 on every proxy: the largest blocks are Acetaldehyde↔Toluene, Acetaldehyde↔Ammonia and
   Ethanol↔Acetaldehyde at 10–250 ppm; at ≥ 400 ppm both predict only Ammonia/Acetone (8 % disagreement). Dropping
   the concentration feature from Track B (`runs/trackB_noconc`) changes nothing on the proxies and only slightly
   raises agreement with the NN (.753 → .766) — concentration kept.
7. Refuted (measured by the design agents on Track B artefacts): recency weighting (forward-9 much worse with recent
   batches only), kNN / RBF-SVM / boosting members (.57 / .75 on batch 9, non-diverse), pattern-only features,
   a linear Ethanol-vs-Acetaldehyde expert.

## 3. How to produce the submission

```bash
# Track B (CPU, ~140 s)                       -> runs/trackB, sub_trackB_v1.csv, sub_trackB_ot_v1.csv
.venv/Scripts/python.exe scripts/10_lda_track.py --force
# NN (GPU): promoted variants, all folds, 3 seeds (fold 0 = fit on all 9 batches, test-only target)
.venv/Scripts/python.exe scripts/20_nn_lobo.py --tag nn --device cuda --seeds 0 1 2 --folds 9 7 6 8 0 --variants a3 a9s a9t a7b a9b a5b
.venv/Scripts/python.exe scripts/30_compare.py --tag nn --extra trackB
# Blend (members: trackB ens_cbstb/em_ensb/lda_em + every NN variant with folds 6-9; add --alts for @adabn read-outs)
.venv/Scripts/python.exe scripts/40_blend_submit.py --name final            # --nn a9s a3 ... to restrict
```
`40_blend_submit.py` prints per-member post/pre-assignment scores per fold, the temperatures, the hill-climb
objective (proxy score − 0.05 × test hist_l1), the blend's pre/post test histograms, the flow matrix, the moved
fraction and the agreement with every member; it refuses to write if > 25 % of rows move, the pre-assignment
histogram is outside [350, 900] for a class, or agreement with the best single member is < .75 (`--force`).
Fallbacks, in order: `submissions/sub_trackB_ot_v1.csv` (Track B ens_cbstb + assignment), `sub_trackB_v1.csv`.

## 4. Decision rule for the final CSV

Prefer the blend whose (a) post-assignment weighted score over folds 6–9 is within noise of the best,
(b) unforced test histogram is closest to 600/class (hist_l1), (c) OT step moves the fewest rows, and (d) agrees
most with the other family. Report all four for every candidate in the write-up.

## 5. Pitfalls verified the hard way

- torch 2.11: `AveragedModel(use_buffers=True)` crashes on BatchNorm's integer buffer (fixed with `_swa_update`).
- The machine had < 1 GB free RAM and ~7 GB free disk; agent fan-outs running CPU experiments slowed GPU runs 5×.
- `weighted_score` returns NaN when a fold is missing — a partially run variant must not sort to the top.
- Sinkhorn at τ=0.1 is one-hot in float64: ranking ties, margin filter inert — use τ ≥ 0.4 with rank='q'.
- The label-free validators BNM/IM prefer sharper models, not better ones (a7b scored lower on them than a3).

## 6. Test-set structure (EDA only — never use for prediction)
Rows are in acquisition order; concentrations form repeated programmes. This is why row order would leak labels and
why the rules forbid using it. Keep this out of the model and the write-up.
