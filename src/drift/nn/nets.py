"""Network definitions: MLP / TabM-style BatchEnsemble MLP / 1D-CNN encoders,
gradient reversal, DANN and CDAN domain heads."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- gradient reversal
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


def dann_lambda(progress: float, lam_max: float = 1.0) -> float:
    """Standard DANN ramp 2/(1+exp(-10p)) - 1, scaled by lam_max."""
    return lam_max * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


# ----------------------------------------------------------------------- encoders
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden=(512, 512, 512), dropout: float = 0.15):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x):
        return self.net(x)


class BatchEnsembleLinear(nn.Module):
    """Shared weight W with per-member rank-1 adapters (TabM style). x: (B, k, in) -> (B, k, out)."""

    def __init__(self, in_dim: int, out_dim: int, k: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.r = nn.Parameter(torch.empty(k, in_dim))
        self.s = nn.Parameter(torch.empty(k, out_dim))
        self.b = nn.Parameter(torch.zeros(k, out_dim))
        with torch.no_grad():   # random +-1 init as in TabM
            self.r.copy_(torch.randint(0, 2, (k, in_dim)).float() * 2 - 1)
            self.s.copy_(torch.randint(0, 2, (k, out_dim)).float() * 2 - 1)

    def forward(self, x):
        return self.lin(x * self.r) * self.s + self.b


class TabMEncoder(nn.Module):
    """k implicit ensemble members sharing weights; returns (B, k, d)."""

    def __init__(self, in_dim: int, hidden=(512, 512, 512), dropout: float = 0.15, k: int = 32):
        super().__init__()
        self.k = k
        blocks, d = [], in_dim
        for h in hidden:
            blocks.append(nn.ModuleDict({"lin": BatchEnsembleLinear(d, h, k), "drop": nn.Dropout(dropout)}))
            d = h
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = d

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.k, -1)
        for blk in self.blocks:
            x = blk["drop"](F.relu(blk["lin"](x)))
        return x


class CNNEncoder(nn.Module):
    """1D conv over the 16 sensors; channels = the first `n_grid` descriptors per sensor
    (the caller places the slog and pattern blocks first: 16 sensors x (8+8) channels).
    Remaining scalar features are concatenated after global pooling."""

    def __init__(self, in_dim: int, n_grid: int = 256, hidden: int = 256, dropout: float = 0.15):
        super().__init__()
        assert n_grid % 128 == 0, "n_grid must be whole 16x8 sensor-major blocks"
        self.n_grid = n_grid
        c = n_grid // 16
        self.conv = nn.Sequential(
            nn.Conv1d(c, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        rest = in_dim - n_grid
        self.head = nn.Sequential(
            nn.Linear(128 * 2 + rest, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.out_dim = hidden

    def forward(self, x):
        B = len(x)
        nb = self.n_grid // 128                                    # number of sensor-major 16x8 blocks (slog, pattern)
        g = x[:, : self.n_grid].reshape(B, nb, 16, 8)              # (B, block, sensor, descriptor)
        g = g.permute(0, 1, 3, 2).reshape(B, nb * 8, 16)           # (B, channels=block*descriptor, 16 sensors)
        h = self.conv(g)
        pooled = torch.cat([h.mean(2), h.amax(2), x[:, self.n_grid:]], dim=1)
        return self.head(pooled)


# -------------------------------------------------------------------------- model
class DriftNet(nn.Module):
    """Encoder + label head + optional domain head (DANN or CDAN)."""

    def __init__(self, in_dim: int, n_classes: int = 6, n_domains: int = 10, arch: str = "mlp",
                 hidden=(512, 512, 512), dropout: float = 0.15, da: str = "none", k: int = 32,
                 disc_hidden: int = 256):
        super().__init__()
        self.arch, self.da, self.k = arch, da, k
        if arch == "mlp":
            self.enc = MLPEncoder(in_dim, hidden, dropout)
        elif arch == "tabm":
            self.enc = TabMEncoder(in_dim, hidden, dropout, k)
        elif arch == "cnn":
            self.enc = CNNEncoder(in_dim, hidden=hidden[0], dropout=dropout)
        else:
            raise ValueError(arch)
        d = self.enc.out_dim
        if arch == "tabm":
            self.head = BatchEnsembleLinear(d, n_classes, k)
        else:
            self.head = nn.Linear(d, n_classes)
        if da in ("dann", "cdan"):
            din = d * n_classes if da == "cdan" else d
            self.disc = nn.Sequential(nn.Linear(din, disc_hidden), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(disc_hidden, disc_hidden), nn.ReLU(),
                                      nn.Linear(disc_hidden, n_domains))
        else:
            self.disc = None

    def forward(self, x):
        f = self.enc(x)
        logits = self.head(f)               # (B, C) or (B, k, C) for tabm
        return logits, f

    def domain_logits(self, f, logits, lam: float):
        """Domain-classifier logits from (reversed-gradient) features."""
        if self.arch == "tabm":              # use member-averaged features/probs for the discriminator
            f = f.mean(1)
            logits = logits.mean(1)
        if self.da == "cdan":
            p = F.softmax(logits, dim=1).detach()   # condition on predictions; never train the head adversarially
            f = torch.bmm(f.unsqueeze(2), p.unsqueeze(1)).flatten(1)   # multilinear map (B, d*C)
        f = grad_reverse(f, lam)   # reversal wraps the whole discriminator input
        return self.disc(f)

    def probs(self, logits):
        if self.arch == "tabm":
            return F.softmax(logits, dim=-1).mean(1)
        return F.softmax(logits, dim=-1)

    def cls_loss(self, logits, y, label_smoothing: float = 0.1, weight=None):
        if self.arch == "tabm":              # mean of per-member losses (never the loss of the mean)
            B, k, C = logits.shape
            return F.cross_entropy(logits.reshape(B * k, C), y.repeat_interleave(k),
                                   label_smoothing=label_smoothing, weight=weight)
        return F.cross_entropy(logits, y, label_smoothing=label_smoothing, weight=weight)


# ------------------------------------------------------------------- test-time BN
@torch.no_grad()
def adabn(model: nn.Module, x_target: torch.Tensor, max_chunk: int = 8192) -> None:
    """Recompute BatchNorm running statistics on ONE target domain (AdaBN) in a single exact pass
    (domains here are <= 3,613 rows); only domains larger than max_chunk are split (cumulative average)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        raise ValueError("adabn(): model has no BatchNorm layers")
    for m in bns:
        m.reset_running_stats()
        m.momentum = None      # cumulative average
    was_training = model.training
    model.train()
    for m in model.modules():   # dropout off during the statistics pass
        if isinstance(m, nn.Dropout):
            m.eval()
    n = len(x_target)
    n_chunks = max(1, math.ceil(n / max_chunk))
    for idx in torch.tensor_split(torch.arange(n, device=x_target.device), n_chunks):
        model(x_target[idx])
    model.train(was_training)
