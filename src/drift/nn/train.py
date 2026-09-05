"""One training run = (config, labelled source, unlabeled target) -> probabilities on the target.

Design choices (see plan §6):
  * class-balanced source sampling (uniform training prior),
  * label smoothing (one-hot rows only), AdamW + one-cycle cosine schedule, SWA over the last
    `swa_frac` of epochs with BatchNorm buffers averaged (no post-hoc source refresh),
  * fixed epoch budget - NO early stopping on the held-out batch, so LOBO scores stay honest,
  * optional domain adaptation: DANN / CDAN (gradient reversal, batch-id domain labels, target rows
    unlabeled) or Deep-CORAL; optional AdaBN = BatchNorm statistics re-estimated PER TARGET DOMAIN,
  * optional pseudo-labelled target rows with soft targets and per-row weights (self-training),
  * index-batching on device-resident tensors (no per-sample DataLoader collation).
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from drift.features import DEFAULT_NN
from drift.nn.nets import DriftNet, adabn, dann_lambda


@dataclass
class RunConfig:
    variant: str = "a0"
    arch: str = "mlp"            # mlp | tabm | cnn
    da: str = "none"             # none | dann | cdan | coral
    lam_max: float = 0.3         # DANN/CDAN reversal strength
    coral_w: float = 1.0
    adabn: bool = False
    hidden: tuple = (512, 512, 512)
    dropout: float = 0.15
    k: int = 32                  # tabm members
    lr: float = 2e-3
    wd: float = 1e-4
    epochs: int = 150
    batch_size: int = 256
    label_smoothing: float = 0.1
    balanced: bool = True
    swa_frac: float = 0.3
    blocks: tuple = tuple(DEFAULT_NN)
    selftrain_rounds: tuple = ()          # e.g. (0.3, 0.5, 0.7); empty = none
    pseudo_weight: float = 0.5
    pseudo_T: float = 2.0
    pseudo_margin: float = 0.2
    sinkhorn: bool = False                # balanced pseudo-label assignment (real test domain only)
    seed: int = 0

    def to_dict(self):
        d = asdict(self)
        for key in ("hidden", "blocks", "selftrain_rounds"):
            d[key] = list(d[key])
        return d


def soft_ce(logits, q, w, eps: float):
    """Weighted soft-target cross-entropy; label smoothing applied to one-hot rows only."""
    if logits.dim() == 3:                      # tabm: (B, k, C) -> mean over members
        B, k, C = logits.shape
        q = q.unsqueeze(1).expand(-1, k, -1).reshape(B * k, C)
        w = w.unsqueeze(1).expand(-1, k).reshape(B * k)
        logits = logits.reshape(B * k, C)
    C = logits.shape[1]
    hard = (q.max(1, keepdim=True).values > 0.999).float()
    q = q * (1 - eps * hard) + (eps / C) * hard
    return -((q * F.log_softmax(logits, dim=1)).sum(1) * w).sum() / w.sum().clamp_min(1e-9)


def coral_loss(fs, ft):
    if fs.dim() == 3:
        fs, ft = fs.mean(1), ft.mean(1)
    d = fs.shape[1]
    return ((torch.cov(fs.T) - torch.cov(ft.T)) ** 2).sum() / (4 * d * d)


@torch.no_grad()
def predict_probs(model, X: torch.Tensor, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        logits, _ = model(X[i:i + batch_size])
        out.append(model.probs(logits).cpu().numpy())
    return np.vstack(out)


@torch.no_grad()
def adabn_predict(model, Xt: torch.Tensor, Bt: torch.Tensor) -> np.ndarray:
    """AdaBN per target domain: BN statistics from each domain's own rows, predictions for those rows."""
    out = np.zeros((len(Xt), 6), np.float32)
    for d in torch.unique(Bt).tolist():
        m = Bt == d
        net = copy.deepcopy(model)
        adabn(net, Xt[m])
        out[m.cpu().numpy()] = predict_probs(net, Xt[m])
    return out


def _batches(n: int, batch_size: int, g: torch.Generator, weights: torch.Tensor | None):
    """Index batches for one epoch: weighted sampling with replacement (class-balanced) or a permutation."""
    if weights is not None:
        idx = torch.multinomial(weights, n, replacement=True, generator=g)
    else:
        idx = torch.randperm(n, generator=g)
    nb = max(1, n // batch_size)
    return idx[: nb * batch_size].view(nb, batch_size)


def train_one(cfg: RunConfig, Xs, ys, bs, Xt, bt, n_domains: int = 10, device: str = "cpu",
              pseudo=None, y_eval=None, n_eval=None, verbose: bool = False) -> dict:
    """Train on (Xs, ys) [+ pseudo-labelled target rows] with unlabeled target Xt (batch ids bt).

    pseudo: optional (idx into Xt, soft targets (m,6), weights (m,)) appended to the source set.
    y_eval / n_eval: labels of the first n_eval target rows, for per-epoch diagnostics only.
    Returns dict(probs=final probs on Xt (SWA, +AdaBN if cfg.adabn), probs_last, probs_swa_raw,
    probs_last_adabn, history, seconds).
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device(device)
    C = 6
    Xs_t = torch.as_tensor(np.asarray(Xs, np.float32))
    Qs = F.one_hot(torch.as_tensor(np.asarray(ys)).long(), C).float()
    Ws = torch.ones(len(Xs))
    Bs = torch.as_tensor(np.asarray(bs)).long()
    Xt_t = torch.as_tensor(np.asarray(Xt, np.float32))
    Bt = torch.as_tensor(np.asarray(bt)).long()
    if pseudo is not None:
        idx, Qp, Wp = pseudo
        idx = torch.as_tensor(np.asarray(idx)).long()
        Xs_t = torch.cat([Xs_t, Xt_t[idx]])
        Qs = torch.cat([Qs, torch.as_tensor(np.asarray(Qp, np.float32))])
        Ws = torch.cat([Ws, torch.as_tensor(np.asarray(Wp, np.float32))])
        Bs = torch.cat([Bs, Bt[idx]])
    # class-balanced sampling weights (by hard/argmax label, pseudo rows included)
    hard = Qs.argmax(1).numpy()
    counts = np.bincount(hard, minlength=C).astype(float)
    sample_w = torch.as_tensor(1.0 / counts[hard]) if cfg.balanced else None
    Xs_t, Qs, Ws, Bs, Xt_t, Bt = (t.to(dev) for t in (Xs_t, Qs, Ws, Bs, Xt_t, Bt))
    g = torch.Generator().manual_seed(cfg.seed)

    use_da = cfg.da in ("dann", "cdan", "coral")
    model = DriftNet(Xs_t.shape[1], C, n_domains, arch=cfg.arch, hidden=tuple(cfg.hidden), dropout=cfg.dropout,
                     da=cfg.da if cfg.da in ("dann", "cdan") else "none", k=cfg.k).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    steps_per_epoch = max(1, len(Xs_t) // cfg.batch_size)
    total = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total, pct_start=0.1,
                                                anneal_strategy="cos", div_factor=20, final_div_factor=100)
    swa_start = min(int(round(cfg.epochs * (1 - cfg.swa_frac))), cfg.epochs - 1)
    swa_model = torch.optim.swa_utils.AveragedModel(model, use_buffers=True)   # BN buffers averaged too
    history = []
    step = 0
    t0 = time.time()
    n_t = len(Xt_t)
    for epoch in range(cfg.epochs):
        model.train()
        src_idx = _batches(len(Xs_t), cfg.batch_size, g, sample_w).to(dev)
        tgt_idx = torch.randint(n_t, (steps_per_epoch, cfg.batch_size), generator=g).to(dev) if use_da else None
        ep_loss = ep_dom = 0.0
        for i in range(steps_per_epoch):
            sb = src_idx[i]
            xb, qb, wb, bb = Xs_t[sb], Qs[sb], Ws[sb], Bs[sb]
            progress = step / max(1, total - 1)
            if use_da:
                tb = tgt_idx[i]
                xt, btb = Xt_t[tb], Bt[tb]
                logits_all, f_all = model(torch.cat([xb, xt]))
                logits_s = logits_all[: len(xb)]
                loss = soft_ce(logits_s, qb, wb, cfg.label_smoothing)
                if cfg.da in ("dann", "cdan"):
                    lam = dann_lambda(progress, cfg.lam_max)
                    d_logits = model.domain_logits(f_all, logits_all, lam)
                    dom = F.cross_entropy(d_logits, torch.cat([bb, btb]))
                    loss = loss + dom
                    ep_dom += dom.item()
                else:
                    loss = loss + cfg.coral_w * coral_loss(f_all[: len(xb)], f_all[len(xb):])
            else:
                logits_s, _ = model(xb)
                loss = soft_ce(logits_s, qb, wb, cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            ep_loss += loss.item()
        if epoch >= swa_start:
            swa_model.update_parameters(model)
        if y_eval is not None and (epoch % 10 == 9 or epoch == cfg.epochs - 1):
            from drift.metrics import macro_f1
            p = predict_probs(model, Xt_t[:n_eval]).argmax(1) + 1
            f1 = macro_f1(y_eval, p)
            history.append({"epoch": epoch + 1, "loss": ep_loss / steps_per_epoch, "dom": ep_dom / steps_per_epoch,
                            "f1_heldout": f1, "t": time.time() - t0})
            if verbose:
                print(f"    ep {epoch+1:3d} loss={ep_loss/steps_per_epoch:.3f} dom={ep_dom/steps_per_epoch:.3f} "
                      f"heldout_f1={f1:.3f} t={time.time()-t0:.0f}s", flush=True)
    assert int(swa_model.n_averaged) > 0, "SWA never updated"
    probs_last = predict_probs(model, Xt_t)
    swa = swa_model.module
    probs_swa_raw = predict_probs(swa, Xt_t)
    probs_last_adabn = adabn_predict(model, Xt_t, Bt) if cfg.adabn else None
    probs = adabn_predict(swa, Xt_t, Bt) if cfg.adabn else probs_swa_raw
    return {"probs": probs, "probs_last": probs_last, "probs_swa_raw": probs_swa_raw,
            "probs_last_adabn": probs_last_adabn, "history": history, "seconds": time.time() - t0}
