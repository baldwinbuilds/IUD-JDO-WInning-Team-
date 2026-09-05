# Running the pipeline on the GPU machine

Self-contained folder (data in `data/`; the venv interpreter is `.venv/Scripts/python.exe`, torch 2.11 cu128).
See `HANDOFF.md` for the state of the project and the reasoning behind the protocol.

## 1. Environment check

```bash
.venv/Scripts/python.exe scripts/00_env_check.py          # must print torch ... and cuda True
```

## 2. Track B (linear, CPU, ~140 s)

```bash
.venv/Scripts/python.exe scripts/10_lda_track.py --force
```
Writes `runs/trackB/{oof_lobo1..9,oof_fwd7..9,test_probs}.npz`, `reports/trackB_lobo.csv`,
`submissions/sub_trackB_v1.csv` (argmax of the selected variant) and `submissions/sub_trackB_ot_v1.csv`
(after the balanced 600-per-class assignment — the fallback submission).
Variants: `lda, lr, lda_cbst, lr_cbst, ens_cbst, ens_cbst_prop` (plain) and `lda_cbstb, lr_cbstb, ens_cbstb`
(self-training under the known marginal), `lda_em` (class-mean EM with fixed prior), `em_ensb`.
Every variant is scored by argmax and `+ot` (after assignment with the held-out batch's true counts as marginal).

## 3. NN track (GPU)

```bash
# variant comparison / full matrix: folds 9 and 7 are the proxies that matter, 6 and 8 are guards, 0 = fit on all 9 batches
.venv/Scripts/python.exe scripts/20_nn_lobo.py --tag nn --device cuda --seeds 0 1 2 --folds 9 7 6 8 0 \
    --variants a3 a9s a9t a7b a9b a5b
.venv/Scripts/python.exe scripts/30_compare.py --tag nn --extra trackB
```
Variant meanings (`scripts/20_nn_lobo.py::VARIANTS`): a0 plain MLP; a3 +per-domain AdaBN; a5b 1-D CNN+AdaBN;
a7b AdaBN + 1 self-training round; a9b/a9s/a9t AdaBN teacher + 3 Sinkhorn self-training rounds (a9b: rank by
model probability, τ=0.1; a9s: rank by balanced posterior, τ=0.4, fractions .5/.7/.9; a9t: same with τ=1);
a6 TabM (no BatchNorm, no AdaBN); a1l/a2/a4/a2b/a2bl/a3dl/a4b adversarial or CORAL alignment (all lost — do not rerun).
Self-trained variants read out running BN statistics; the AdaBN read-out is stored as `alt_adabn_*` and appears
as `<variant>@adabn` in `30_compare.py` (`--no-alts` hides it) and in the blend with `--alts`.
Each run appends to `runs/nn/log.txt` (held-out F1, alternative read-outs, per-round reassignments/agreement,
test histogram, validators). Existing npz files are skipped (`--overwrite` to redo). `--threads 2` is the default
so the GPU job does not hog the CPU.

## 4. Blend and submit

```bash
.venv/Scripts/python.exe scripts/40_blend_submit.py --name final
#   --nn a9s a3          restrict NN members     --alts    also offer @adabn read-outs
#   --no-nn              Track B only            --assign hungarian|none   --tau 1.0   --hist-penalty 0.05
#   --test-models lobo|fold0   which nets' test predictions are averaged     --force   ignore the gates
```
Prints per-member post/pre-assignment macro-F1 per fold (6/7/8/9), fitted temperatures, the hill-climb result,
pre/post test histograms, the flow matrix, moved fraction and member agreement; writes
`submissions/sub_<name>_vN.csv`, `reports/blend_<name>_vN.json`, `runs/blend_<name>_vN_test_probs.npy`
(never overwrites an existing version).

Rules reminder: nothing here uses `measurement_id` order or test labels; test *features* are used only as
unlabeled adaptation data, and the 600-per-class statement only as a class marginal.
