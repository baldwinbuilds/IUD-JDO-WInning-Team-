"""
Per-batch feature alignment for the Gas Sensor Array Drift datathon.

EDA (see conversation / plan doc) showed batches differ systematically in
their feature distributions -- not just at the outlier tails, batch 4's
*typical* rows already run "hot" relative to other batches. A model trained
on raw features risks learning batch-specific quirks instead of the
underlying gas-class signal, which is exactly what falls apart on the
hidden batch-10 test set.

The fix used here: standardize each batch against its OWN distribution
before doing anything else. This is legitimate for test.csv too, since it
only uses test's unlabeled feature values (never gas_class), so it doesn't
leak any label information -- it's the same information a deployed model
would have (a new batch's raw sensor readings) plus the (physically
reasonable) assumption you can compute summary statistics over it.

Median/IQR (not mean/std) on purpose: the ~1.8% of rows flagged as
statistical outliers during cleaning are real high-concentration sensor
responses, not data errors (see cleaner.py's report_outliers). Mean/std
would let those rows drag a batch's scaling around; median/IQR is robust
to them while still correcting the batch-level shift.
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [f"feat_{i}" for i in range(1, 129)]


def per_batch_robust_align(df: pd.DataFrame, feature_cols=FEATURE_COLS, batch_col="batch") -> pd.DataFrame:
    """Return a copy of df with feature_cols replaced by (x - median) / IQR,
    computed independently within each batch group."""
    df = df.copy()

    def align_group(g):
        median = g[feature_cols].median()
        q1 = g[feature_cols].quantile(0.25)
        q3 = g[feature_cols].quantile(0.75)
        iqr = (q3 - q1).replace(0, 1.0)  # guard against a zero-variance feature within a batch
        g[feature_cols] = (g[feature_cols] - median) / iqr
        return g

    return df.groupby(batch_col, group_keys=False, observed=True).apply(align_group)


def add_log_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """concentration spans 1-1000 and is heavily right-skewed; log1p compresses
    that range so the model doesn't need to learn a huge linear scale."""
    df = df.copy()
    df["log_concentration"] = np.log1p(df["concentration"])
    return df


def build_model_inputs(df: pd.DataFrame, feature_cols=FEATURE_COLS, batch_col="batch") -> pd.DataFrame:
    """Full alignment pipeline: per-batch robust scaling of features + log concentration.

    Deliberately does NOT include batch_col in the output -- the model must
    not be able to key off batch identity as a feature, since the real test
    batch (10) is never seen during training and any reliance on it as a
    learned category cannot generalize.
    """
    df = per_batch_robust_align(df, feature_cols, batch_col)
    df = add_log_concentration(df)
    return df
