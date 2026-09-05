"""
Leave-one-batch-out (LOBO) training and evaluation for the drift-robust MLP,
plus a linear baseline for comparison, plus final ensembled test predictions.

Why leave-one-batch-out instead of a random/stratified split: the actual
test set is an entire unseen batch (10), which never appears during
training. A random split would let the model see every batch's
distribution during training and overstate how well it generalizes to a
genuinely new batch -- LOBO is the closest simulation of the real
competition condition available from the labelled data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from features import FEATURE_COLS, build_model_inputs
from model import DriftRobustMLP

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4


def to_tensors(df):
    x = torch.tensor(df[FEATURE_COLS].to_numpy(), dtype=torch.float32)
    log_conc = torch.tensor(df["log_concentration"].to_numpy(), dtype=torch.float32)
    return x, log_conc


def class_weights_from(y, n_classes=6):
    # Inverse-frequency weighting so the loss doesn't just optimize for the
    # majority classes -- macro-F1 scores every class equally regardless of
    # how often it appears in training.
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.clip(counts, 1, None))
    return torch.tensor(weights, dtype=torch.float32)


def train_one_fold(train_df, val_df):
    """Train a DriftRobustMLP on train_df, early-stopping on val_df's macro-F1.
    Returns (best_state_dict, best_val_f1)."""
    x_train, conc_train = to_tensors(train_df)
    y_train = torch.tensor(train_df["gas_class"].to_numpy() - 1, dtype=torch.long)  # 0-indexed
    x_val, conc_val = to_tensors(val_df)
    y_val_np = val_df["gas_class"].to_numpy() - 1

    x_train, conc_train, y_train = x_train.to(DEVICE), conc_train.to(DEVICE), y_train.to(DEVICE)
    x_val, conc_val = x_val.to(DEVICE), conc_val.to(DEVICE)

    model = DriftRobustMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from(y_train.cpu().numpy()).to(DEVICE))

    n = x_train.shape[0]
    best_f1, best_state, epochs_without_improvement = -1.0, None, 0

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            if len(idx) <= 1:  # BatchNorm1d needs >1 sample per batch in train mode
                continue
            optimizer.zero_grad()
            logits = model(x_train[idx], conc_train[idx])
            loss = criterion(logits, y_train[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(x_val, conc_val).argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(y_val_np, val_preds, average="macro")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_preds = val_preds
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    return best_state, best_f1, best_preds, y_val_np


def baseline_f1(train_df, val_df):
    """Multinomial logistic regression on the same aligned features, as a
    sanity-check floor: if the MLP can't beat this, the added complexity
    isn't earning its keep."""
    x_train = train_df[FEATURE_COLS + ["log_concentration"]].to_numpy()
    y_train = train_df["gas_class"].to_numpy()
    x_val = val_df[FEATURE_COLS + ["log_concentration"]].to_numpy()
    y_val = val_df["gas_class"].to_numpy()

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_train, y_train)
    preds = clf.predict(x_val)
    return f1_score(y_val, preds, average="macro")


def run_lobo_cv(train_raw):
    fold_states = {}
    mlp_scores, baseline_scores = {}, {}
    all_val_preds, all_val_true = [], []

    for held_out_batch in sorted(train_raw["batch"].unique()):
        fold_train_raw = train_raw[train_raw["batch"] != held_out_batch]
        fold_val_raw = train_raw[train_raw["batch"] == held_out_batch]

        # Alignment stats are computed per-batch inside build_model_inputs,
        # so calling it separately on train/val here means the held-out
        # batch is scaled using ONLY its own (unlabeled-equivalent) feature
        # distribution -- no leakage of its labels or of other batches' stats.
        fold_train = build_model_inputs(fold_train_raw)
        fold_val = build_model_inputs(fold_val_raw)

        state, mlp_f1, val_preds, val_true = train_one_fold(fold_train, fold_val)
        base_f1 = baseline_f1(fold_train, fold_val)

        fold_states[held_out_batch] = state
        mlp_scores[held_out_batch] = mlp_f1
        baseline_scores[held_out_batch] = base_f1
        all_val_preds.append(val_preds)
        all_val_true.append(val_true)
        print(f"held-out batch {held_out_batch}: MLP macro-F1={mlp_f1:.3f}  |  LogisticRegression macro-F1={base_f1:.3f}")

    print(f"\nmean MLP macro-F1:  {np.mean(list(mlp_scores.values())):.3f}")
    print(f"mean baseline macro-F1: {np.mean(list(baseline_scores.values())):.3f}")

    # Pooled per-class breakdown across all LOBO folds (every training row
    # was in exactly one fold's held-out set, so this covers all of train_clean.csv).
    pooled_true = np.concatenate(all_val_true)
    pooled_preds = np.concatenate(all_val_preds)
    print("\nper-class report, pooled across all LOBO held-out folds:")
    print(classification_report(pooled_true, pooled_preds, target_names=[f"class_{i+1}" for i in range(6)], digits=3))

    print("confusion matrix (rows=true, cols=predicted, 1-indexed classes):")
    cm = confusion_matrix(pooled_true, pooled_preds)
    print(pd.DataFrame(cm, index=[f"true_{i+1}" for i in range(6)], columns=[f"pred_{i+1}" for i in range(6)]))

    return fold_states


def predict_test(fold_states, train_raw, test_raw):
    """Ensemble the 9 LOBO fold models (average softmax probabilities) and
    predict on test.csv, which is aligned using its own batch-10 stats."""
    test_aligned = build_model_inputs(test_raw)
    x_test, conc_test = to_tensors(test_aligned)
    x_test, conc_test = x_test.to(DEVICE), conc_test.to(DEVICE)

    probs_sum = torch.zeros(len(test_raw), 6, device=DEVICE)
    model = DriftRobustMLP().to(DEVICE)
    with torch.no_grad():
        for state in fold_states.values():
            model.load_state_dict(state)
            model.eval()
            logits = model(x_test, conc_test)
            probs_sum += torch.softmax(logits, dim=1)

    preds = probs_sum.argmax(dim=1).cpu().numpy() + 1  # back to 1-indexed gas_class
    return preds


def main():
    train_raw = pd.read_csv("train_clean.csv")
    test_raw = pd.read_csv("test_clean.csv")

    print(f"device: {DEVICE}\n")
    fold_states = run_lobo_cv(train_raw)

    preds = predict_test(fold_states, train_raw, test_raw)
    submission = pd.DataFrame({"measurement_id": test_raw["measurement_id"], "gas_class": preds})
    submission.to_csv("submission.csv", index=False)
    print(f"\nwrote submission.csv ({len(submission)} rows)")
    print(submission["gas_class"].value_counts().sort_index())


if __name__ == "__main__":
    main()
