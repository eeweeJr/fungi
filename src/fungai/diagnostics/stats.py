"""Pure tensor statistics used by the diagnostics. Inputs are torch tensors; outputs
are plain floats so they can go straight to MLflow and JSON."""

from __future__ import annotations

import torch


def finite_counts(x: torch.Tensor) -> dict:
    return {"nan": int(torch.isnan(x).sum()), "inf": int(torch.isinf(x).sum())}


def basic(x: torch.Tensor) -> dict:
    """mean, std, rms over all elements, plus NaN/Inf counts."""
    x = x.detach().float()
    return {"mean": float(x.mean()), "std": float(x.std()) if x.numel() > 1 else 0.0,
            "rms": float(x.pow(2).mean().sqrt()), **finite_counts(x)}


def token_rms(x: torch.Tensor) -> torch.Tensor:
    """Per-token RMS over the feature dimension: [..., D] -> [...]."""
    return x.detach().float().pow(2).mean(-1).sqrt()


def mean_pairwise_cosine(x: torch.Tensor) -> float:
    """Mean cosine similarity between distinct rows of each [n, D] set in x [..., n, D].
    1 means the rows are identical (collapse / over-smoothing)."""
    x = torch.nn.functional.normalize(x.detach().float(), dim=-1)
    n = x.shape[-2]
    if n < 2:
        return float("nan")
    sim = x @ x.transpose(-1, -2)
    off = sim.sum((-1, -2)) - sim.diagonal(dim1=-2, dim2=-1).sum(-1)
    return float((off / (n * (n - 1))).mean())


def effective_rank(x: torch.Tensor) -> float:
    """exp(entropy of the normalised singular values) of a [n, D] matrix, after
    centring. Ranges from 1 (all rows on one line) to min(n, D)."""
    x = x.detach().float()
    x = x - x.mean(0, keepdim=True)
    if x.shape[0] < 2:
        return float("nan")
    s = torch.linalg.svdvals(x.cpu())
    p = s / s.sum().clamp_min(1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def normalised_entropy(weights: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
    """Entropy of attention rows divided by log(number of real keys).
    weights [..., L] (rows sum to 1, zeros on padding), n_valid broadcastable to [...].
    1 = uniform over the real tokens, 0 = all weight on one token."""
    w = weights.detach().float()
    ent = -(w * w.clamp_min(1e-30).log()).sum(-1)
    return ent / n_valid.float().clamp_min(2).log()


def prob_stats(logp: torch.Tensor) -> dict:
    """From log-probabilities [B, C]: mean max-probability and mean normalised entropy."""
    lp = logp.detach().float()
    p = lp.exp()
    ent = -(p * lp.clamp_min(-1e4)).sum(-1) / torch.log(torch.tensor(float(lp.shape[-1])))
    return {"max_prob": float(p.max(-1).values.mean()), "entropy": float(ent.mean())}


def logit_stats(logits: torch.Tensor) -> dict:
    lg = logits.detach().float()
    out = prob_stats(lg.log_softmax(-1))
    out["logit_norm"] = float(lg.norm(dim=-1).mean())
    return out
