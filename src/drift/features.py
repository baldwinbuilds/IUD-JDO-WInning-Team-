"""Drift-robust feature construction for the 16-sensor x 8-descriptor array.

All transforms are row-wise except (a) train medians used to fill masked
sensor-state values and (b) scalers fitted by the caller.  Nothing here ever
looks at measurement_id or row order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEAT_COLS = [f"feat_{i}" for i in range(1, 129)]
TYPE_NAMES = ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]   # descriptor positions within each sensor block
# Data-derived structure (competition files only): the 128 features repeat with period 8 (16 sensors x 8 descriptors);
# d0 = large positive magnitude, d1 = a normalised magnitude (>= 1), d2-d4 = positive family with |d2|<|d3|<|d4|,
# d5-d7 = negative family with |d5|<|d6|<|d7|; all six co-vary with d0.
# sensor groups inferred from the log|d0| correlation structure (0-indexed sensors)
GROUPS = [[0, 1, 8, 9], [2, 3, 10, 11], [4, 5, 12, 13], [6, 7, 14, 15]]
ALL_BLOCKS = ("slog", "pattern", "logscale", "logconc", "state", "shape", "typemed")
DEFAULT_LINEAR = ("pattern", "logscale", "logconc", "state")
DEFAULT_NN = ("slog", "pattern", "logscale", "logconc", "state", "shape")
EPS = 1e-6


def to_cube(df: pd.DataFrame) -> np.ndarray:
    return df[FEAT_COLS].to_numpy(dtype=float).reshape(len(df), 16, 8)


def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def sensor_state(cube: np.ndarray) -> np.ndarray:
    """S = d0/(d1-1): a per-sensor quantity built from the two large descriptors; its batch-level median drifts
    with time and its log (raw and sensor-centred) helps the linear models on the drifted proxy (ablation-tested).

    Returns (n, 16) with NaN where undefined (d0 <= 0 or d1 <= 1.02).
    """
    dr, ratio = cube[:, :, 0], cube[:, :, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = dr / (ratio - 1.0)
    ok = (dr > 0) & (ratio > 1.02) & np.isfinite(s) & (s > 0)
    return np.where(ok, s, np.nan)


class FeatureBuilder:
    """fit() learns only per-sensor train medians of log S (used for masked values)."""

    def __init__(self, blocks=DEFAULT_LINEAR):
        unknown = set(blocks) - set(ALL_BLOCKS)
        if unknown:
            raise ValueError(f"unknown blocks {unknown}")
        self.blocks = tuple(blocks)
        self.state_fill_ = None
        self.names_ = []

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        cube = to_cube(df)
        with np.errstate(divide="ignore", invalid="ignore"):
            logS = np.log(sensor_state(cube))
        self.state_fill_ = np.nanmedian(logS, axis=0)
        self.names_ = self._names()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.state_fill_ is None:
            raise RuntimeError("call fit() first")
        cube = to_cube(df)
        n = len(df)
        parts = []
        absdr = np.abs(cube[:, :, 0])
        logdr = np.log1p(absdr)
        for b in self.blocks:
            if b == "slog":
                parts.append(signed_log(cube).reshape(n, -1))
            elif b == "pattern":
                l1 = np.abs(cube).sum(axis=1, keepdims=True) + EPS
                parts.append((cube / l1).reshape(n, -1))
            elif b == "logscale":
                parts.append(logdr.mean(axis=1, keepdims=True))
            elif b == "logconc":
                parts.append(np.log(df["concentration"].to_numpy(dtype=float))[:, None])
            elif b == "state":
                with np.errstate(divide="ignore", invalid="ignore"):
                    logS = np.log(sensor_state(cube))
                bad = ~np.isfinite(logS)
                logS = np.where(bad, np.broadcast_to(self.state_fill_, logS.shape), logS)
                parts.append(logS)
                parts.append(logS - logS.mean(axis=1, keepdims=True))
            elif b == "shape":
                denom = absdr[:, :, None] + EPS
                ratios = cube[:, :, 2:8] / denom                      # 6 small descriptors / |d0| (scale-free shape)
                parts.append(np.clip(signed_log(ratios), -20, 20).reshape(n, -1))
                asym = -cube[:, :, 5:8] / (cube[:, :, 2:5] + EPS)     # negative family / positive family, pairwise
                parts.append(np.clip(signed_log(asym), -20, 20).reshape(n, -1))
            elif b == "typemed":
                med = np.stack([np.median(logdr[:, g], axis=1) for g in GROUPS], axis=1)
                parts.append(med)
                dev = logdr.copy()
                for gi, g in enumerate(GROUPS):
                    dev[:, g] = logdr[:, g] - med[:, [gi]]
                parts.append(dev)
        X = np.concatenate(parts, axis=1)
        if not np.isfinite(X).all():
            raise ValueError("non-finite feature values")
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def _names(self) -> list:
        names = []
        for b in self.blocks:
            if b == "slog":
                names += [f"slog_s{s}_{t}" for s in range(16) for t in TYPE_NAMES]
            elif b == "pattern":
                names += [f"pat_s{s}_{t}" for s in range(16) for t in TYPE_NAMES]
            elif b == "logscale":
                names += ["logscale"]
            elif b == "logconc":
                names += ["logconc"]
            elif b == "state":
                names += [f"logS_s{s}" for s in range(16)] + [f"logSc_s{s}" for s in range(16)]
            elif b == "shape":
                names += [f"shape_s{s}_{t}" for s in range(16) for t in TYPE_NAMES[2:]]
                names += [f"asym_s{s}_a{a}" for s in range(16) for a in range(3)]
            elif b == "typemed":
                names += [f"typemed_g{g}" for g in range(4)] + [f"typedev_s{s}" for s in range(16)]
        return names
