"""Batch-aware validation splits. Never random K-fold."""
from __future__ import annotations

import numpy as np

PROXY_WEIGHTS = {6: 1.0, 7: 2.0, 8: 2.0}   # decision score; batch 9 is reported only


def lobo_splits(batch: np.ndarray, batches=None):
    """Leave-one-batch-out: yield (k, train_mask, test_mask)."""
    ks = sorted(np.unique(batch)) if batches is None else list(batches)
    for k in ks:
        yield int(k), batch != k, batch == k


def forward_splits(batch: np.ndarray, ks=(7, 8, 9)):
    """Train on all batches before k, test on k."""
    for k in ks:
        yield int(k), batch < k, batch == k


def weighted_score(scores: dict, weights: dict = PROXY_WEIGHTS) -> float:
    num = sum(weights[k] * scores[k] for k in weights if k in scores)
    den = sum(weights[k] for k in weights if k in scores)
    return num / den if den else float("nan")
