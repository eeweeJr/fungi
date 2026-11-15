"""Perceiver-style fusion: learned queries attend over the patch tokens of all photos.

The fusion layer receives one pool of tokens per observation with a key-padding mask.
There is no photo-index or slot embedding, so its output does not depend on photo
order. Each photo's tokens keep only what the backbone put in them (including the
backbone's own within-photo position information).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    """Multi-head attention with an explicit path that can return the weights.

    key_mask: bool [B, Nk], True = a real token. Masked keys get exactly zero weight.
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q, k, v = self._split(self.q(x_q)), self._split(self.k(x_kv)), self._split(self.v(x_kv))
        p = self.dropout if self.training else 0.0
        weights = None
        if return_weights:
            scores = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
            if key_mask is not None:
                scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
            weights = scores.softmax(dim=-1)
            out = F.dropout(weights, p=p, training=self.training) @ v
        else:
            mask = key_mask[:, None, None, :] if key_mask is not None else None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        b, _, nq, _ = out.shape
        out = out.transpose(1, 2).reshape(b, nq, self.heads * self.head_dim)
        return self.proj(out), weights


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverBlock(nn.Module):
    """Pre-norm: cross-attention (queries -> tokens), MLP, self-attention among
    queries, MLP. Each sub-layer is residual."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm_q1, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross = Attention(dim, heads, dropout)
        self.norm_m1 = nn.LayerNorm(dim)
        self.mlp1 = MLP(dim, dropout=dropout)
        self.norm_q2 = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, heads, dropout)
        self.norm_m2 = nn.LayerNorm(dim)
        self.mlp2 = MLP(dim, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, tokens, key_mask, return_attention=False):
        a, w = self.cross(self.norm_q1(q), self.norm_kv(tokens), key_mask, return_attention)
        q = q + self.drop(a)
        q = q + self.mlp1(self.norm_m1(q))
        h = self.norm_q2(q)
        s, _ = self.self_attn(h, h)
        q = q + self.drop(s)
        q = q + self.mlp2(self.norm_m2(q))
        return q, w


class PerceiverFusion(nn.Module):
    """Fuse the patch tokens of up to P photos into one observation embedding.

    project(patches [N, T, in_dim]) -> [N, T, dim] is applied per real photo before
    padding. forward(tokens [B, L, dim], key_mask [B, L]) returns
    (obs_emb [B, dim], query_out [B, K, dim], attention list of [B, heads, K, L]).
    """

    def __init__(self, in_dim: int, dim: int = 512, n_queries: int = 16, n_blocks: int = 2,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, dim)
        self.queries = nn.Parameter(torch.empty(n_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.blocks = nn.ModuleList([PerceiverBlock(dim, n_heads, dropout) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(dim)
        self.dim, self.n_queries, self.n_heads = dim, n_queries, n_heads

    def project(self, patches: torch.Tensor) -> torch.Tensor:
        return self.in_proj(self.in_norm(patches))

    def forward(self, tokens: torch.Tensor, key_mask: torch.Tensor, return_attention: bool = False):
        q = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attention = []
        for block in self.blocks:
            q, w = block(q, tokens, key_mask, return_attention)
            if return_attention:
                attention.append(w)
        q = self.out_norm(q)
        return q.mean(dim=1), q, attention
