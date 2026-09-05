"""Data loading. The test file is read in its given order only to write the submission."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
SUBS = ROOT / "submissions"
REPORTS = ROOT / "reports"
N_TEST = 3600
CLASSES = np.arange(1, 7)


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA / "train.csv")


def load_test() -> pd.DataFrame:
    df = pd.read_csv(DATA / "test.csv")
    if len(df) != N_TEST:
        import warnings
        warnings.warn(f"data/test.csv has {len(df)} rows, expected {N_TEST}: this is NOT the full competition test file",
                      stacklevel=2)
    return df


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(DATA / "sample_submission.csv")


def class_prior(y: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(y, int), minlength=7)[1:] / len(y)


def write_submission(ids, pred, path: Path) -> None:
    sub = pd.DataFrame({"measurement_id": ids, "gas_class": np.asarray(pred, int)})
    sample = load_sample_submission()
    if list(sub.measurement_id) != list(sample.measurement_id):
        raise ValueError("submission ids/order do not match sample_submission.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
