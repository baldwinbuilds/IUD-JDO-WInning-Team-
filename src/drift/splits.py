"""Batch-aware validation splits. Never random K-fold."""
from __future__ import annotations

import numpy as np

# Decision score. Batch 9 is the most drifted / most recent labelled batch and the only proxy with
# headroom (linear models .73-.91 there vs .96-.99 elsewhere); batch 7 is large and class-balanced.
# Batches 6 and 8 are class-skewed (b8: 294 rows, 49 % Acetone) and act mainly as no-catastrophe guards.
PROXY_WEIGHTS = {6: 1.0, 7: 2.0, 8: 1.0, 9: 2.0}
LEGACY_PROXY_WEIGHTS = {6: 1.0, 7: 2.0, 8: 2.0}


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
    """Weighted mean over the proxy batches; NaN if any weighted batch is missing (partial runs must
    not sort above complete ones)."""
    if any(k not in scores for k in weights):
        return float("nan")
    num = sum(w * scores[k] for k, w in weights.items())
    return num / sum(weights.values())
