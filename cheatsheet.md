# Cheatsheet — gas-sensor drift classification (batches 1–9 → batch 10)

A guide for redoing the pipeline by hand **using only the competition files** (`train.csv`, `test.csv`,
`sample_submission.csv`, the overview text). Every claim about the data below is derived from those files with the code
shown; every code block was run end-to-end on this machine (`.venv/Scripts/python.exe`, numpy/pandas/scikit-learn/scipy;
torch only for §10) and reproduces the numbers quoted. "Batch 9 .968 → .980" always means macro-F1 on held-out batch 9,
before → after the balanced assignment.

Provenance note: an earlier version of this file described the eight descriptors with physical names taken from prior
knowledge of similar sensor datasets. That is outside information and has been removed. The descriptors are now
called d0–d7 and characterised only by what the competition data shows (§1b). No external data or labels were ever used.

---

## 0. The whole recipe in ten lines

1. Understand the 128 features as 16 sensors × 8 descriptors (stated in the data dictionary; the layout is confirmed from the data, §1b); batches are time; batch 10 is far drifted.
2. Clean nothing; check everything (3,600 test rows, id order, no NaN/inf, class counts per batch).
3. Validate with leave-one-batch-out; **batch 9 is the only proxy that behaves like the test**; batch 7 is the big sanity fold.
4. Build scale-free features: sensor pattern normalised per descriptor, log magnitude, log concentration, plus a per-sensor ratio of the two large descriptors.
5. Baselines: shrinkage LDA and logistic regression on standardised features (batch 9 ≈ .73 / .88).
6. Turn "600 rows per class" into a **Sinkhorn balanced assignment** of test rows to classes (batch 9 .909 → .973 for the ensemble).
7. Self-train on the unlabeled target, but pick pseudo-labels from the **balanced** posterior (batch 9 .968 / .980; test histogram balanced).
8. Optional NN: MLP + BatchNorm + class-balanced batches + label smoothing; AdaBN once, then Sinkhorn self-training read out with running statistics (batch 9 .99).
9. Blend in calibrated log space, choose weights on post-assignment scores, apply the assignment at τ = 0.5.
10. Check: 3,600 rows, id order, histogram near 600/class, < 10 % rows moved by the assignment, leak audit. Keep fallbacks.

---

## 1. Understand the data before touching a model

**Given by the competition.** `measurement_id`, `batch` (1–9 train, 10 test, chronological), `concentration`,
`feat_1 … feat_128` = "eight response descriptors for each of 16 chemical sensors", `gas_class` 1–6
(1 Ethanol, 2 Ethylene, 3 Ammonia, 4 Acetaldehyde, 5 Acetone, 6 Toluene). Metric macro-F1. The hidden test has
**600 rows per class** (stated). Rules: do not identify the source dataset, no external data, do not recover labels.

**Per-batch class counts** (rows = batch; class 1..6):

```
b1  90/ 98/ 83/ 30/ 70/ 74      b6 514/574/110/ 29/606/467
b2 164/334/100/109/532/  5      b7 649/662/360/744/630/568   <- big, balanced: sanity fold
b3 365/490/216/240/275/  0      b8  30/ 30/ 40/ 33/143/ 18   <- tiny, 49 % Acetone: guard only
b4  64/ 43/ 12/ 30/ 12/  0      b9  61/ 55/100/ 75/ 78/101   <- most recent, balanced: THE drift proxy
b5  28/ 40/ 20/ 46/ 63/  0      test 600 x 6 (stated)
```

**Facts that shaped every decision (all measured on the competition files)**

- Batches 6/7/8 are easy for anything decent (.96–.99); batch 9 is not (raw LDA .73). Batch 10 is 2–3× further from
  the training cloud than batch 9 (nearest-neighbour distance in pattern space), so expect real scores well below proxies.
- Concentration is class-specific in train (`train.groupby("gas_class").concentration.agg(["min","median","max"])`:
  Toluene ≤ 230, Ammonia up to 1000, Acetone ≤ 500). The test has 800 rows at 400–1000 and 140 at 1–2.5. Concentration
  is a legitimate feature, but it extrapolates on batch 10.
- The 600-per-class statement is a *known class marginal* — the single most valuable piece of information.
- Test rows are in acquisition order and the concentration sequence repeats in programmes. **Never** use row
  position / `measurement_id` for prediction (rules) — only for reading and writing the CSV.

### 1b. Working out the descriptor structure from the data alone

The data dictionary says 16 sensors × 8 descriptors but not the order or the meaning. Three cheap checks settle what
you need:

```python
X = train[FEAT].to_numpy(float)
pos = (X > 0).mean(0)                                   # fraction of positive values per column
print(np.round(pos[:16], 2))                             # [1 1 1 1 1 0 0 0  1 1 1 1 1 0 0 0]
print(np.allclose(pos.reshape(16, 8), pos.reshape(16, 8)[0], atol=0.02))   # True -> period-8 layout
cube = X.reshape(len(X), 16, 8)                         # (rows, sensor, descriptor d0..d7)
for d in range(8):                                      # sign, size, and co-movement with d0
    print(d, (cube[:, :, d] > 0).mean().round(3), np.median(np.abs(cube[:, :, d])).round(2),
          np.mean([np.corrcoef(np.log1p(np.abs(cube[:, s, d])), np.log1p(np.abs(cube[:, s, 0])))[0, 1] for s in range(16)]).round(2))
```

Result (train.csv):

| position | sign | median size | corr with log\|d0\| | what the data says |
|---|---|---|---|---|
| d0 | + (99.8 %) | 13 000 | 1.00 | the large "response magnitude" of the sensor |
| d1 | + (≥ 0.993, median 3.7) | 3.7 | .75 | a normalised magnitude, never below ~1 |
| d2, d3, d4 | + | 3.8, 7.2, 10.4 | .88, .78, .71 | a positive family, ordered \|d2\| < \|d3\| < \|d4\| in 99.9 % of rows |
| d5, d6, d7 | − | 2.5, 3.9, 9.2 | .91, .87, .63 | a negative family, ordered \|d5\| < \|d6\| < \|d7\| in 99.8 % of rows |

So each sensor block is: one big magnitude, one normalised magnitude, two ordered triplets of smaller descriptors that
all scale with the magnitude. That is all the feature engineering needs: (a) divide by a scale to get a fingerprint,
(b) keep the scale separately, (c) ratios inside a sensor block are scale-free shape descriptors. The 8-per-sensor
layout also tells you that "sensor-major 16×8" reshape above is correct (a 1-D CNN over sensors uses it).

Do **not** try to name the descriptors physically; nothing in the pipeline needs it.

---

## 2. Cleaning (there is almost nothing to clean)

- No missing values, no duplicates, no inf in either file. Do **not** drop "outliers": the high-magnitude rows are real.
- Check the test file has exactly **3,600** rows (`G_B10_0001 … G_B10_3600`) and the same ids as `sample_submission.csv`,
  in the same order. (A partial 2,288-row copy existed for a while — that cost time.)
- Force numeric dtypes, keep `batch` as an id, never as a model feature.
- The only "imputation": masked cells of the derived state feature (§4) are filled with the **training** median.

---

## 3. Validation protocol (the part people get wrong)

- **Leave-one-batch-out (LOBO)**: train on the other 8 batches, predict the held-out batch. Never random K-fold, never
  early-stop on the held-out batch (it makes LOBO scores dishonest).
- Decision score = weighted macro-F1 over held-out batches **6/7/8/9 with weights 1/2/1/2**, computed **after** the
  balanced assignment (§7) using that batch's true class counts as the marginal. The labels are used only to set six
  column masses — the exact analogue of the organisers' "600 per class".
- Batch 9 decides; batch 7 must not drop; 6 and 8 are no-catastrophe guards (both are class-skewed and *under-rate*
  any method that uses target statistics — AdaBN, Sinkhorn, EM — because the real test is balanced and they are not).
- Forward-chaining (train < k, test k) for k = 7, 8, 9 is a useful second view for anything time-related.
- Two label-free validators on the real test: the unforced predicted histogram (distance to 600/class) and the fraction
  of rows the assignment has to move. Both dropped from .19 / 11 % to .08 / 5 % during the work.

---

## 4. Features — verified code

```python
import numpy as np, pandas as pd
FEAT = [f"feat_{i}" for i in range(1, 129)]

def features(df, state_fill=None):
    cube = df[FEAT].to_numpy(float).reshape(len(df), 16, 8)     # (rows, sensors, descriptors d0..d7)
    d0, d1 = cube[:, :, 0], cube[:, :, 1]
    l1 = np.abs(cube).sum(axis=1, keepdims=True) + 1e-6          # per-descriptor L1 over the 16 sensors
    pattern = (cube / l1).reshape(len(df), -1)                   # 128 scale-free "fingerprint" values
    logscale = np.log1p(np.abs(d0)).mean(axis=1, keepdims=True)  # overall response magnitude
    logconc = np.log(df["concentration"].to_numpy(float))[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        S = d0 / (d1 - 1.0)                                      # per-sensor ratio of the two large descriptors
        logS = np.log(np.where((d0 > 0) & (d1 > 1.02) & (S > 0), S, np.nan))   # guard: d1 can sit at ~1
    if state_fill is None:
        state_fill = np.nanmedian(logS, axis=0)                  # learned on TRAIN only
    logS = np.where(np.isfinite(logS), logS, state_fill)
    X = np.hstack([pattern, logscale, logconc, logS, logS - logS.mean(axis=1, keepdims=True)])
    return X, state_fill
```

| block | dims | idea (derived from §1b) |
|---|---|---|
| `pattern` | 128 | which sensors respond relative to the others, per descriptor — a fingerprint independent of overall scale |
| `logscale` | 1 | how strongly the array responded overall |
| `logconc` | 1 | concentration is informative in-distribution; risky at 400–1000 on the test |
| `logS`, sensor-centred `logS` | 32 | `d0/(d1−1)` per sensor: its batch-level median drifts with time (sensor-0 medians 9.8 → 6.9 over batches 1–9), so it carries "sensor state"; kept because the ablation below says so, not because of any physical story |
| (NN only) signed-log of the raw 128, and "shape" = log(\|d2..d7\| / \|d0\|) plus the pairwise ratio of the negative family to the positive family | 128 + 96 + 48 | scale-free within-sensor shape; they hurt the *linear* model (.981 → .968) but help the NN |

**Ablation, honest version** (linear recipe of §8 with self-training under the marginal; batch 9 / batch 7, argmax →
after assignment):

| feature set | batch 9 | batch 7 |
|---|---|---|
| full (pattern + logscale + logconc + state) | .968 → **.980** | .998 → .999 |
| without the state block | .976 → .972 | .996 → .997 |
| without log concentration | .968 → .980 | .997 → .999 |
| raw signed-log instead of pattern | .975 → .978 | .994 → .993 |
| pattern only | .971 → .969 | .995 → .999 |

Read: once the marginal-aware adaptation is in place the feature choice moves batch 9 by ≈ 1 point. Any of these is a
fine starting point; the full set is best by a small margin. Scaling: `StandardScaler` on training rows for linear
models; for the NN a `QuantileTransformer(output_distribution="normal")` fitted on **train + test pooled** (legitimate:
unlabeled test features only) removes scale drift.

---

## 5. Linear baselines

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
lda = lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")   # shrinkage matters (162 dims, 10k rows)
lr  = lambda: LogisticRegression(C=1.0, max_iter=5000)
```

Expected LOBO macro-F1 (argmax): batch 7 ≈ .99 for both; batch 9 LDA .73, LR .88. LDA is the drift-fragile one; LR
the robust one. LDA is **over-confident under drift** (median max-prob .998 on the test while wrong on ~25 % of rows) —
never use its probabilities as confidence; temperature scaling gives T ≈ 1.2–3 for LDA-based members vs ≈ 0.3–0.6 for
label-smoothed NNs.

---

## 6. Recognise the failure mode before fixing it

On the test the raw models predicted 928 Ethanol / 413 Acetaldehyde (should be 600/600). Diagnostics that showed it:

- Predicted histogram vs 600/class (`np.bincount(pred)`), and an EM estimate of the class prior — both said
  "Ethanol ≫ Acetaldehyde".
- The runner-up class of the Ethanol-predicted rows was Acetaldehyde for 702 of 928 rows.
- Plain class-balanced self-training made it *worse* (raw LDA 650 Acetaldehyde → 441): selecting the top-k *per
  predicted class* freezes whatever skew the model already has.
- The wrong rows were **not** low-confidence (LDA over-confidence), so thresholding on confidence cannot find them.

---

## 7. The 600-per-class prior → balanced assignment (verified code)

```python
def sinkhorn(P, marginal=None, tau=1.0, n_iter=2000, tol=1e-3):
    """Entropic optimal transport: rows (mass 1) -> classes (mass `marginal`, default N/C), cost -log P / tau.
    Returns Q with rows summing to 1 and columns to the marginal.  argmax(Q) is the balanced prediction."""
    P = np.asarray(P, float); N, C = P.shape
    col = np.full(C, N / C) if marginal is None else np.asarray(marginal, float) * N / np.sum(marginal)
    K = np.exp((np.log(np.clip(P, 1e-12, 1)) / tau) - (np.log(np.clip(P, 1e-12, 1)) / tau).max(1, keepdims=True))
    u, v = np.ones(N), np.ones(C)
    for _ in range(n_iter):
        u = 1 / np.maximum(K @ v, 1e-300); v = col / np.maximum(K.T @ u, 1e-300)
        Q = u[:, None] * K * v[None, :]; Q /= Q.sum(1, keepdims=True)
        if np.abs(Q.sum(0) - col).max() / col.max() < tol: break
    return Q
```

How to use it:

- **Final decision** on the test: `pred = classes[sinkhorn(P_test, None, tau).argmax(1)]`. τ = 1 corrects the class
  masses softly (per-class log-bias correction); τ = 0.5 enforces the marginal more tightly and scored best on the
  proxies for the final blend; the exact Hungarian version (`scipy.optimize.linear_sum_assignment` on −log P with
  each class column repeated 600×) hits 600/600/… exactly but scored slightly lower (b9 .955 vs .973 for τ = 1).
- **Evaluation** on a held-out batch: pass `marginal = np.bincount(y_held)[1:]` (its true counts).
- **Effect**: ensemble batch 9 .909 → .973, batch 7 .996 → .998, batch 6 .988 → .991, batch 8 (294 rows) .962 → .945
  (noise). Soft prior *reweighting* (multiplying by prior ratios) hurt on batch 6 (.988 → .954) — do not use it.
- Watch the flow matrix (which class → which) and the max-prob of moved rows: healthy moves are Ethanol→Acetaldehyde,
  Ammonia↔Ethylene, Acetaldehyde→Toluene/Ammonia at low-confidence rows (max-prob ≈ .5–.7 after calibration).

---

## 8. Self-training under the marginal (verified code — this is the core of the linear track)

```python
def self_train(make, Xs, ys, Xt, marginal, rounds=(0.3, 0.5, 0.7)):
    clf = make().fit(Xs, ys); P = clf.predict_proba(Xt); classes = clf.classes_
    for frac in rounds:
        Q = sinkhorn(P, marginal)                                # balanced posterior (test: uniform marginal)
        pred, conf = classes[Q.argmax(1)], Q.max(1)              # labels AND ranking from the balanced posterior
        keep = np.zeros(len(Xt), bool)
        for c in classes:
            idx = np.where(pred == c)[0]
            if len(idx): keep[idx[np.argsort(-conf[idx])[:max(1, int(frac * len(idx)))]]] = True
        clf = make().fit(np.vstack([Xs, Xt[keep]]), np.concatenate([ys, pred[keep]]))
        P = clf.predict_proba(Xt)
    return P, classes

def run(df_s, df_t, marginal):
    Xs, fill = features(df_s); Xt, _ = features(df_t, fill)
    sc = StandardScaler().fit(Xs); Xs, Xt = sc.transform(Xs), sc.transform(Xt)
    ys = df_s.gas_class.to_numpy()
    Pl, classes = self_train(lda, Xs, ys, Xt, marginal)
    Pr, _ = self_train(lr, Xs, ys, Xt, marginal)
    return (Pl + Pr) / 2, classes

# validation
for k in (9, 7):
    s, t = train[train.batch != k], train[train.batch == k]
    marg = np.bincount(t.gas_class, minlength=7)[1:]
    P, classes = run(s, t, marg)
    print(k, f1_score(t.gas_class, classes[P.argmax(1)], average="macro"),
             f1_score(t.gas_class, classes[sinkhorn(P, marg).argmax(1)], average="macro"))
# final
P, classes = run(train, test, np.ones(6))
pred = classes[sinkhorn(P, None, tau=1.0).argmax(1)]
```

Verified output: batch 9 **.968 / .980**, batch 7 **.998 / .999**; test histogram 621/562/650/646/538/583 before,
552/626/641/626/540/615 after the assignment. This alone is a strong submission (`sub_trackB_ot_v1.csv`).

Ranking by the **balanced** posterior is essential: ranking by the model's own probability of the re-assigned class
puts exactly the rows the prior wants to move at the bottom of the list, and the prior never acts.

Optional extra member, same idea in closed form: **LDA-EM** — project with LDA (`solver="eigen"`), keep the tied
within-class covariance from the source, re-estimate the six class means on the unlabeled target by EM with the
mixing weights *fixed* at the marginal (uniform on the test), clip each mean's move to ≤ 1 Mahalanobis unit per
iteration, 10 iterations. Batch 9 .96; unforced test histogram 714/553/548/658/556/571.

---

## 9. What the two families disagree about (know this before blending)

The linear ensemble and the neural nets agree on only ~77 % of test rows even though both are ≥ .98 on every proxy.
Disagreement lives at concentrations 10–250 (25–28 % of rows there; 8 % at ≥ 400) in the pairs Acetaldehyde↔Toluene,
Acetaldehyde↔Ammonia, Ethanol↔Acetaldehyde. No proxy can arbitrate it; the blend and the label-free checks are the hedge.

---

## 10. Neural-net track — a simpler verified version

Design (each item was needed): MLP 3×512 with **BatchNorm**, dropout .15, AdamW + one-cycle LR (2e-3), **class-balanced
sampling** (uniform training prior), **label smoothing .1**, 150 epochs (60 is enough to see the effects),
**AdaBN** = re-estimate BatchNorm statistics on each target domain separately (one exact pass over that domain's rows),
then **Sinkhorn self-training** rounds (fractions .5/.7/.9, τ = 0.4) whose pseudo-labels come from the balanced posterior
per domain — and whose output is read with the **running** BatchNorm statistics, *not* AdaBN.

```python
import copy, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k = 9; src, held = train[train.batch != k], train[train.batch == k]
target = pd.concat([held, test], ignore_index=True)             # unlabeled target = held-out batch + real test
Xs, fill = features(src); Xt, _ = features(target, fill)
qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0).fit(np.vstack([Xs, Xt]))
Xs, Xt = qt.transform(Xs).astype(np.float32), qt.transform(Xt).astype(np.float32)
ys = src.gas_class.to_numpy() - 1
dom = np.r_[np.zeros(len(held), int), np.ones(len(test), int)]  # target domain ids: 0 = held-out batch, 1 = test

def make_mlp(d, h=512, p=0.15):
    return nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p),
                         nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p),
                         nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 6)).to(dev)

def train_mlp(X, Q, W, epochs=60, bs=256, seed=0):
    """X features, Q soft targets (n,6), W row weights. Class-balanced sampling by argmax(Q)."""
    torch.manual_seed(seed)
    X, Q, W = (torch.as_tensor(a, dtype=torch.float32, device=dev) for a in (X, Q, W))
    counts = np.bincount(Q.argmax(1).cpu().numpy(), minlength=6).astype(float)
    sample_w = torch.as_tensor(1.0 / counts[Q.argmax(1).cpu().numpy()])
    model = make_mlp(X.shape[1]); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = len(X) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        idx = torch.multinomial(sample_w, steps * bs, replacement=True, generator=g).view(steps, bs).to(dev)
        for b in idx:
            logp = F.log_softmax(model(X[b]), 1)
            q = Q[b]; hard = (q.max(1, keepdim=True).values > 0.999).float()
            q = q * (1 - 0.1 * hard) + (0.1 / 6) * hard                 # label smoothing on hard rows only
            loss = -((q * logp).sum(1) * W[b]).sum() / W[b].sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
    return model

@torch.no_grad()
def predict(model, X):
    model.eval(); return F.softmax(model(torch.as_tensor(X, device=dev)), 1).cpu().numpy()

@torch.no_grad()
def adabn_predict(model, X, dom):
    """Re-estimate BatchNorm statistics on EACH target domain separately, then predict that domain."""
    out = np.zeros((len(X), 6), np.float32)
    for d in np.unique(dom):
        m = dom == d; net = copy.deepcopy(model)
        for mod in net.modules():
            if isinstance(mod, nn.BatchNorm1d): mod.reset_running_stats(); mod.momentum = None
        net.train()
        for mod in net.modules():
            if isinstance(mod, nn.Dropout): mod.eval()
        net(torch.as_tensor(X[m], device=dev))                         # one pass = exact domain statistics
        out[m] = predict(net, X[m])
    return out

# round 0: source only; the AdaBN read-out is the teacher
Q0 = np.eye(6)[ys]; W0 = np.ones(len(ys))
model = train_mlp(Xs, Q0, W0)
P = adabn_predict(model, Xt, dom)

# self-training rounds: pseudo-labels from the Sinkhorn-balanced posterior per domain; read out running statistics
y_held = held.gas_class.to_numpy(); marg_held = np.bincount(y_held - 1, minlength=6)   # evaluation design only
for r, frac in enumerate((0.5, 0.7, 0.9)):
    idx, Q, W = [], [], []
    for d, marg in ((0, marg_held), (1, None)):                      # test domain: uniform marginal (600 per class)
        m = np.where(dom == d)[0]; Qd = sinkhorn(P[m], marg, tau=0.4)
        pred, conf = Qd.argmax(1), Qd.max(1)
        for c in range(6):
            cand = np.where(pred == c)[0]
            keep = cand[np.argsort(-conf[cand])[:max(1, int(frac * len(cand)))]]
            idx.append(m[keep]); Q.append(Qd[keep] ** (0.5 if r == 0 else 1.0)); W.append(0.5 * conf[keep])
    idx, Q, W = np.concatenate(idx), np.vstack(Q), np.concatenate(W); Q /= Q.sum(1, keepdims=True)
    model = train_mlp(np.vstack([Xs, Xt[idx]]), np.vstack([Q0, Q]), np.concatenate([W0, W]), seed=r + 1)
    P = predict(model, Xt)                                            # running statistics, NOT AdaBN, after self-training
```

Verified with 60 epochs on held-out batch 9: raw .80 → AdaBN .905 → self-training rounds .974 → .989 → **.993**
(the AdaBN read-out of the same self-trained nets would be .92–.94). With 150 epochs and the full feature set the
repo's variant `a9s` reaches .990 on batch 9 and .999 on batch 7, and its unforced test histogram is 657/535/667/605/540/596.

Why the read-out rule: pseudo-labelled target rows are trained in mini-batches whose BatchNorm statistics are
dominated by source rows; re-estimating BN on the target alone at inference shifts them a second time. AdaBN is
right *before* the target is trained in, wrong *after*.

Practical: one 150-epoch run ≈ 20–40 s on an RTX 3060 Ti (self-training ≈ 4× that); use 3 seeds; average test
probabilities over all fold models (models trained on all 9 batches count double). Keep `torch.set_num_threads(2)`
so a GPU job does not pin the CPU. Variants that lost on every proxy and are not worth re-running: DANN/CDAN/CORAL
domain-adversarial losses, a TabM ensemble (no BatchNorm → no AdaBN), wider MLPs. A 1-D CNN over the 16 sensors
(channels = the 8 descriptors) is a reasonable diversity member (batch 9 .96).

---

## 11. Blending

1. Members: linear `ens_cbstb` (+ `lda_em`), NN self-trained variants, maybe the CNN. Exclude the plain (unbalanced)
   self-training members — their test histogram is known to be skewed.
2. **Temperature-calibrate** each member on out-of-fold predictions of the *other* folds (fit T minimising NLL of
   softmax(log P / T)); LDA members get T ≈ 1.2–3, NN members ≈ 0.3–0.6.
3. Blend in log space: `L = Σ_m w_m · log P_m / T_m`, softmax.
4. Weights: hill-climb on the weighted post-assignment score over folds 6–9, started from both the best single member
   and equal weights, with a penalty of 0.05 × L1 distance of the blend's **unforced test histogram** to 600/class
   (label-free; the proxies are saturated and cannot see the Ethanol/Acetaldehyde leakage, the histogram can).
   Fall back to the best single member if the blend loses > 0.5 pt on any fold.
5. Final assignment: `sinkhorn(P_test, None, tau=0.5).argmax(1)`.
6. The greedy hill-climb chases proxy noise: seven configurations scored .988–.991 while their unforced histograms
   ranged from 615 to 815 Ethanol. Judge candidates by score *and* histogram distance *and* moved fraction *and*
   cross-family agreement, not by score alone.

Final blend chosen: NN a9s .29, CNN .29, NN a9b .14, NN a7b .14, linear ens_cbstb .14; score .9906
(batch 9 .998, batch 7 .999); unforced histogram 667/533/679/599/536/586 → 602/572/605/603/613/605 after the
assignment (4.7 % of rows moved).

---

## 12. Checks before uploading

```python
sub = pd.read_csv("submission.csv"); sample = pd.read_csv("data/sample_submission.csv")
assert list(sub.columns) == ["measurement_id", "gas_class"] and len(sub) == 3600
assert (sub.measurement_id.values == sample.measurement_id.values).all()
assert sub.gas_class.between(1, 6).all() and sub.gas_class.dtype.kind == "i"
print(np.bincount(sub.gas_class, minlength=7)[1:])          # every class 520-680 after the assignment
```

- Leak audit: `grep -rn "measurement_id\|sort_values\|iloc" src scripts` — only CSV read/write may appear.
- Unforced histogram (before the assignment) inside roughly 350–900 per class; the assignment should move < 10–15 % of
  rows; agreement with the best single member ≥ .75; moved rows should have low max-prob (≈ .5–.7).
- Keep every candidate CSV; keep the linear-track-plus-assignment file as the fallback.

---

## 13. Expected numbers (sanity table)

| stage | held-out batch 9 | held-out batch 7 | unforced test histogram |
|---|---|---|---|
| raw LDA / LR (argmax) | .73 / .88 | .99 / .97 | 829/513/556/650/504/548 (LDA) |
| LDA+LR average, plain self-training | .909 | .996 | 928/519/568/413/567/605 |
| + balanced assignment at the end | .973 | .998 | (forced) |
| self-training under the marginal (`ens_cbstb`) | .968 → .980 | .998 → .999 | 621/562/650/646/538/583 |
| MLP + AdaBN (1 seed, 150 ep) | .974 | .975 | 747/548/618/463/617/607 |
| MLP + AdaBN + 3 Sinkhorn rounds, running-stats read-out (`a9s`) | .990 | .999 | 657/535/667/605/540/596 |
| final calibrated blend + assignment τ=0.5 | .998 | .999 | 667/533/679/599/536/586 → 602/572/605/603/613/605 |

Real batch-10 macro-F1 is unknown until the leaderboard; expect it well below the proxies (the families still
disagree on ~22 % of rows).

---

## 14. What did not work (do not spend time here)

- Prior *reweighting* (multiplying probabilities by target/source prior ratios): batch 6 .988 → .954.
- Recency weighting / training only on recent batches: forward-9 LR .88 → .60–.72. Old batches supply drift diversity.
- kNN in pattern space (.57 on batch 9), RBF-SVM (.75, 96 % agreement with LDA), boosting (trees cannot extrapolate).
- kNN label propagation: hurt batch 8; shape features in the *linear* model: .981 → .968.
- DANN / CDAN / CORAL alignment: lost on batch 7 and 9; marginal alignment on class-skewed proxies is confounded.
- Per-class bias tuning on saturated proxies: over-fits. A linear Ethanol-vs-Acetaldehyde expert: no split on the test.
- Sinkhorn at τ = 0.1: the posterior is exactly one-hot in float64 → ranking ties, margin filter inert. Use τ ≥ 0.4.
- Label-free validators BNM / InfoMax: they reward sharp models, not correct ones (the best self-trained nets scored lower).

---

## 15. Decision rule for the final CSV

Among candidates whose post-assignment weighted score is within ~0.2 pt of the best: prefer the one with the smallest
unforced-histogram distance to 600/class, then the fewest rows moved by the assignment, then the highest agreement with
the *other* model family. Report all four numbers for every candidate. Upload the fallback (linear track + assignment)
as a second submission if the platform allows it — it is the cheapest way to learn which family is right on the
disputed rows.

---

## 16. Glossary

- **LOBO** leave-one-batch-out validation. **Proxy** a held-out training batch standing in for the test batch.
- **Marginal** the vector of class counts (600 × 6 on the test). **Sinkhorn** iterative scaling that finds the
  entropic optimal-transport plan between rows and classes under a given marginal.
- **AdaBN** re-estimating BatchNorm mean/variance on the target data at inference. **Running statistics** the
  BatchNorm buffers accumulated during training.
- **Self-training / pseudo-labels** retraining with confident target rows labelled by the model (here: by the balanced posterior).
- **Temperature T** divides log-probabilities before softmax; T > 1 softens, T < 1 sharpens.
- **hist_l1** Σ\|count_c − 600\| / 3600 of the unforced test predictions — the label-free skew measure.
