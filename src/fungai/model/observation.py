"""Observation-level classifier: 1 to P unlabelled, unordered photos -> one prediction.

    photos [B, P, 3, H, W] + image_mask [B, P]
      -> shared backbone on the real photos only      cls [N, D], patches [N, T, D]
      -> per-photo head on concat(cls, mean(patches)) photo_logits [N, S]   (auxiliary)
      -> fusion over all patch tokens of each obs     obs_emb [B, dim]
      -> genus / species-within-genus head            species_logp [B, S]

With fusion "none" the observation prediction is the average of the per-photo
probabilities, which is also the no-fusion baseline reported for every run.
"""

from __future__ import annotations

import torch
from torch import nn

from fungai.model.backbone import TokenBackbone
from fungai.model.fusion import PerceiverFusion
from fungai.model.hierarchy import build_head, flat_logp, membership


def photo_average_logp(photo_logits: torch.Tensor, photo_obs_index: torch.Tensor,
                       n_obs: int) -> torch.Tensor:
    """log of the mean per-photo species probability for each observation [B, S]."""
    probs = photo_logits.softmax(dim=-1)
    summed = torch.zeros(n_obs, probs.shape[1], device=probs.device, dtype=probs.dtype)
    summed.index_add_(0, photo_obs_index, probs)
    counts = torch.bincount(photo_obs_index, minlength=n_obs).clamp(min=1).to(probs.dtype)
    return (summed / counts[:, None]).clamp_min(1e-12).log()


class ObservationClassifier(nn.Module):
    def __init__(
        self,
        backbone: TokenBackbone,
        species_genus: torch.Tensor,
        n_genus: int,
        fusion: str = "perceiver",
        head: str = "hierarchical",
        fusion_dim: int = 512,
        n_queries: int = 16,
        fusion_blocks: int = 2,
        fusion_heads: int = 8,
        fusion_dropout: float = 0.1,
        photo_dropout: float = 0.1,
    ):
        super().__init__()
        if fusion not in ("perceiver", "none"):
            raise ValueError(f"unknown fusion {fusion!r}")
        self.backbone = backbone
        self.fusion_kind, self.head_kind = fusion, head
        self.n_species, self.n_genus = len(species_genus), n_genus
        self.register_buffer("species_genus", species_genus.clone(), persistent=False)
        d = backbone.embed_dim
        self.photo_drop = nn.Dropout(photo_dropout)
        self.photo_head = nn.Linear(2 * d, self.n_species)
        if fusion == "perceiver":
            self.fusion = PerceiverFusion(d, fusion_dim, n_queries, fusion_blocks, fusion_heads,
                                          fusion_dropout)
            self.head = build_head(head, fusion_dim, species_genus, n_genus)
        else:
            self.fusion, self.head = None, None
        self.backbone_frozen = False

    # ------------------------------------------------------------------ freezing

    def freeze_backbone(self, frozen: bool = True) -> None:
        """Frozen: no gradients, eval mode (no drop path) and no autograd graph."""
        self.backbone_frozen = frozen
        for p in self.backbone.parameters():
            p.requires_grad = not frozen
        self.backbone.train(self.training and not frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def new_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.named_parameters() if not n.startswith("backbone.")]

    # ------------------------------------------------------------------ forward

    def encode_photos(self, photos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backbone_frozen:
            with torch.no_grad():
                cls, patches = self.backbone(photos)
            return cls.detach(), patches.detach()
        return self.backbone(photos)

    def forward(self, images: torch.Tensor, image_mask: torch.Tensor,
                return_attention: bool = False) -> dict:
        b, p = image_mask.shape
        cls, patches = self.encode_photos(images[image_mask])         # [N, D], [N, T, D]
        photo_obs_index = image_mask.nonzero(as_tuple=False)[:, 0]
        photo_feat = torch.cat([cls, patches.mean(dim=1)], dim=-1)
        photo_logits = self.photo_head(self.photo_drop(photo_feat))
        out = {"photo_logits": photo_logits, "photo_obs_index": photo_obs_index,
               "photo_feat": photo_feat}

        if self.fusion is None:
            logp = photo_average_logp(photo_logits, photo_obs_index, b)
            out.update(flat_logp(logp, self.species_genus, self.n_genus))
            out["obs_emb"] = photo_feat.new_zeros(b, photo_feat.shape[1]).index_add_(
                0, photo_obs_index, photo_feat) / torch.bincount(
                photo_obs_index, minlength=b).clamp(min=1)[:, None].to(photo_feat.dtype)
            return out

        tokens = self.fusion.project(patches)                          # [N, T, dim]
        t, dim = tokens.shape[1], tokens.shape[2]
        padded = tokens.new_zeros(b, p, t, dim)
        padded[image_mask] = tokens
        key_mask = image_mask[:, :, None].expand(b, p, t).reshape(b, p * t)
        obs_emb, query_out, attention = self.fusion(padded.reshape(b, p * t, dim), key_mask,
                                                     return_attention)
        out.update(self.head(obs_emb))
        out.update({"obs_emb": obs_emb, "query_out": query_out, "tokens_per_photo": t,
                    "key_mask": key_mask})
        if return_attention:
            out["attn"] = attention
        return out

    @torch.no_grad()
    def photo_average(self, out: dict, n_obs: int) -> torch.Tensor:
        """No-fusion baseline from a forward's outputs: log mean photo probability."""
        return photo_average_logp(out["photo_logits"], out["photo_obs_index"], n_obs)

    def genus_membership(self) -> torch.Tensor:
        return membership(self.species_genus, self.n_genus)


def build_observation_model(cfg, species_genus: torch.Tensor, n_genus: int,
                            pretrained: bool | None = None) -> ObservationClassifier:
    """Build from an ExperimentConfig's model/data sections."""
    m = cfg.model
    backbone = TokenBackbone(
        m.backbone, pretrained=m.pretrained if pretrained is None else pretrained,
        drop_path=m.drop_path, image_size=cfg.data.image_size,
        grad_checkpointing=m.grad_checkpointing,
    )
    return ObservationClassifier(
        backbone, species_genus, n_genus, fusion=m.fusion, head=m.head,
        fusion_dim=m.fusion_dim, n_queries=m.n_queries, fusion_blocks=m.fusion_blocks,
        fusion_heads=m.fusion_heads, fusion_dropout=m.fusion_dropout, photo_dropout=m.dropout,
    )
