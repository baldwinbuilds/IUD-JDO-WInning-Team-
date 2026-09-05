# Running the NN track on the GPU machine

The whole folder is self-contained (data in `data/`). Copy it (OneDrive sync or zip) and run from the folder root.

## 1. Environment (once)

```bash
conda env create -f environment.yml          # or: conda create -n drift python=3.11 pandas numpy scipy scikit-learn
conda activate drift
pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick the CUDA build for your driver
python scripts/00_env_check.py                # must print cuda: True
```

## 2. Stage 1 — compare variants (held-out batches 6, 7, 8; 1 seed)  ~1 GPU-hour

```bash
python scripts/20_nn_lobo.py --tag nn --device cuda --seeds 0 --folds 7 6 8 \
  --variants a3 a2b a2bl a3dl a4b a5b a6b a7b a9b a3w a0 a1l a2 a4
python scripts/30_compare.py --tag nn --extra trackB
```
Read `reports/nn_compare.csv`. Two scores: `weighted_678` and `b7_only`. **Batch 7 is the fairest proxy** (3,613 rows, all six classes reasonably balanced); batches 6 and 8 are class-skewed (batch 8: 294 rows, 49 % Acetone) and systematically *under-rate* AdaBN/DANN variants, whose favourable case is exactly the real test (3,600 rows, 600 per class). Track B reference (same protocol): lobo6 .988, lobo7 .996, lobo8 .962 (weighted .981).
Also read the per-run `test validators` in `runs/nn/log.txt` (BNM, IM, ClassAMI — higher is better, label-free) and the test histogram (organisers: 600 per class; flag <520 or >680).
Promote a variant if it is competitive with the best on b7, not catastrophic on b6/b8, and has a plausible test histogram / good validators.

## 3. Stage 2 — the 9 leave-one-batch-out networks for the promoted variants (× seeds)  ~1 GPU-hour per variant per seed-triple

```bash
python scripts/20_nn_lobo.py --tag nn --device cuda --seeds 0 1 2 --folds 1 2 3 4 5 6 7 8 9 0 \
  --variants <promoted, e.g. a3d a7s a9>
python scripts/30_compare.py --tag nn --extra trackB
```
`--folds 0` = fit on all 9 batches with the test as the only unlabeled target (extra ensemble member).
Every run writes `runs/nn/<variant>_fold<k>_seed<s>.npz` (OOF probs on batch k + test probs) and appends to `runs/nn/log.txt`; re-running skips existing files (`--overwrite` to redo).

## 4. Optional knobs

* `--epochs N` (default 150), `--blocks ...` to ablate feature blocks (default: slog pattern logscale logconc state shape).
* Add variants in `scripts/20_nn_lobo.py::VARIANTS` (arch mlp|tabm|cnn, da none|dann|cdan|coral, lam_max, adabn, selftrain_rounds, sinkhorn).

## 5. Bring results back

Copy `runs/nn/` (and `reports/`) back to the main machine; `scripts/40_blend_submit.py` blends the promoted models with Track B on the batch 6–8 OOF predictions and writes the submission.

Rules reminder: nothing in this pipeline uses `measurement_id` order or test labels; test *features* are used only as unlabeled domain-adaptation data.
