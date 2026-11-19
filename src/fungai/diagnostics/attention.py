"""Cross-attention statistics, part-attention measurements and attention figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fungai.diagnostics.stats import normalised_entropy

PARTS = ("fruiting_body", "cap", "gills", "stem")


def attention_stats(w: torch.Tensor, key_mask: torch.Tensor, image_mask: torch.Tensor,
                    tokens_per_photo: int) -> dict:
    """Statistics of one cross-attention block.

    w [B, heads, K, P*T]; key_mask [B, P*T]; image_mask [B, P].
    Per query, averaged over heads: normalised entropy, peak weight, mass per photo
    and the share of the top photo. Plus padding mass (must be 0) and query overlap
    (mean pairwise cosine between queries' attention maps).
    """
    w = w.detach().float()
    b, _, k, _ = w.shape
    p = image_mask.shape[1]
    wq = w.mean(1)                                                     # [B, K, L]
    n_valid = key_mask.sum(-1)[:, None]                                # [B, 1]
    ent = normalised_entropy(wq, n_valid)                              # [B, K]
    per_photo = wq.view(b, k, p, tokens_per_photo).sum(-1)             # [B, K, P]
    n_photos = image_mask.sum(-1)
    multi = n_photos >= 2
    top_share = per_photo.max(-1).values                               # [B, K]
    padding_mass = float(w.masked_fill(key_mask[:, None, None, :], 0).sum(-1).max())
    flat = torch.nn.functional.normalize(wq, dim=-1)
    sim = flat @ flat.transpose(1, 2)
    overlap = (sim.sum((-1, -2)) - sim.diagonal(dim1=-2, dim2=-1).sum(-1)) / max(k * (k - 1), 1)
    out = {
        "entropy": float(ent.mean()),
        "entropy_min": float(ent.min()),
        "entropy_max": float(ent.max()),
        "peak": float(wq.max(-1).values.mean()),
        "padding_mass": padding_mass,
        "query_overlap": float(overlap.mean()),
        "per_query_entropy": ent.mean(0).tolist(),
    }
    if multi.any():
        out["top_photo_share_multi"] = float(top_share[multi].mean())
        # uniform spread would give 1 / n_photos; report the ratio so 1.0 = even spread
        out["photo_spread_multi"] = float(((1.0 / n_photos[multi].float())[:, None]
                                           / top_share[multi]).mean())
    return out


def part_attention(w_last: torch.Tensor, image_mask: torch.Tensor, tokens_per_photo: int,
                   photo_paths: list[list[str]], part_grids: dict[str, dict[str, np.ndarray]]) -> dict:
    """Attention lift per part, and the query-by-part affinity matrix.

    lift = (share of a query's attention on a photo that falls on the part) /
           (share of the photo's patches covered by the part).
    Above 1 means the query looks at the part more than its size alone explains.
    Only photos with masks count. Uses the last block, averaged over heads.
    """
    w = w_last.detach().float().mean(1)                                # [B, K, L]
    b, k, _ = w.shape
    p = image_mask.shape[1]
    w = w.view(b, k, p, tokens_per_photo)
    sums = {part: np.zeros(k) for part in PARTS}
    counts = {part: np.zeros(k) for part in PARTS}
    for i in range(b):
        for j, path in enumerate(photo_paths[i]):
            grids = part_grids.get(path.rsplit("/", 1)[-1])
            if not grids:
                continue
            wp = w[i, :, j].cpu().numpy()                              # [K, T]
            mass = wp.sum(-1, keepdims=True).clip(1e-12)
            for part in PARTS:
                g = grids.get(part)
                if g is None or g.size != tokens_per_photo or g.mean() <= 0:
                    continue
                f = g.reshape(-1)
                lift = ((wp / mass) @ f) / f.mean()
                sums[part] += lift
                counts[part] += 1
    affinity, lift = {}, {}
    for part in PARTS:
        if counts[part].max() == 0:
            continue
        per_query = sums[part] / counts[part].clip(1)
        affinity[part] = per_query.tolist()
        lift[part] = float(per_query.mean())
    return {"lift": lift, "query_part_affinity": affinity,
            "n_photos": int(counts["fruiting_body"].max()) if "fruiting_body" in counts else 0}


def save_attention_figure(path: Path, images: torch.Tensor, image_mask: torch.Tensor,
                          w_last: torch.Tensor, tokens_per_photo: int, mean, std,
                          titles: list[str] | None = None) -> None:
    """Grid: one row per observation, one column per photo, with the last block's
    head- and query-averaged attention overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b, p = image_mask.shape
    g = int(round(tokens_per_photo ** 0.5))
    att = w_last.detach().float().mean((1, 2)).view(b, p, tokens_per_photo).cpu()
    mean_t, std_t = torch.tensor(mean)[:, None, None], torch.tensor(std)[:, None, None]
    fig, axes = plt.subplots(b, p, figsize=(2.6 * p, 2.6 * b), squeeze=False)
    for i in range(b):
        for j in range(p):
            ax = axes[i][j]
            ax.axis("off")
            if not image_mask[i, j]:
                continue
            img = (images[i, j].cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
            a = att[i, j].view(g, g).numpy()
            a = a / max(a.max(), 1e-12)
            ax.imshow(img)
            ax.imshow(a, cmap="magma", alpha=0.5, extent=(0, img.shape[1], img.shape[0], 0),
                      interpolation="bilinear")
            if j == 0 and titles:
                ax.set_title(titles[i], fontsize=7, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
