"""Arrays and samplers for the NN track."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer

from drift.features import DEFAULT_NN, FeatureBuilder


def build_arrays(df_source, df_target, blocks=DEFAULT_NN, seed: int = 0, quantile: bool = True):
    """Fit features on the labelled source, scale with a quantile-normal transform fitted on
    source + unlabeled target pooled (legitimate transductive scaling). Returns dict of numpy arrays."""
    fb = FeatureBuilder(blocks).fit(df_source)
    Xs, Xt = fb.transform(df_source), fb.transform(df_target)
    if quantile:
        rng = np.random.default_rng(seed)
        pool = np.vstack([Xs, Xt])
        pool = pool + rng.normal(0, 1e-3, pool.shape) * (pool.std(0, keepdims=True) + 1e-9)
        qt = QuantileTransformer(n_quantiles=min(1000, len(pool)), output_distribution="normal",
                                 random_state=seed).fit(pool)
        Xs, Xt = qt.transform(Xs), qt.transform(Xt)
    ys = df_source["gas_class"].to_numpy(int) - 1            # 0..5 for torch
    bs = df_source["batch"].to_numpy(int) - 1                # 0..8
    bt = df_target["batch"].to_numpy(int) - 1                # 9 for the real test, k-1 for LOBO
    return {"Xs": Xs.astype(np.float32), "ys": ys, "bs": bs, "Xt": Xt.astype(np.float32), "bt": bt,
            "names": fb.names_}


def class_balanced_weights(y: np.ndarray, n_classes: int = 6) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = 1.0 / counts[y]
    return w / w.sum()


def make_source_loader(Xs, ys, bs, batch_size: int, seed: int, balanced: bool = True):
    g = torch.Generator().manual_seed(seed)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys).long(),
                                        torch.from_numpy(bs).long())
    if balanced:
        w = torch.from_numpy(class_balanced_weights(ys))
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(ys), replacement=True, generator=g)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)


def make_target_loader(Xt, bt, batch_size: int, seed: int):
    g = torch.Generator().manual_seed(seed + 1)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xt), torch.from_numpy(bt).long())
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)
