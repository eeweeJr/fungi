"""Pretrained ViT backbone that returns the class token and the patch tokens separately."""

from __future__ import annotations

import timm
import torch
from torch import nn


class TokenBackbone(nn.Module):
    """Wraps a timm ViT. forward(x [N, 3, H, W]) -> (cls [N, D], patches [N, T, D]).

    Prefix tokens (class token, plus register tokens on DINOv2-reg / DINOv3) are
    stripped from the patch tokens using the model's `num_prefix_tokens`, so the
    token layout of each family is never assumed. `forward_features` includes the
    backbone's final norm.
    """

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        drop_path: float = 0.0,
        image_size: int | None = None,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        kwargs = {"pretrained": pretrained, "num_classes": 0, "drop_path_rate": drop_path}
        self.model = self._create(name, kwargs, image_size)
        self.name = name
        self.num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
        self.embed_dim = int(getattr(self.model, "embed_dim", None) or self.model.num_features)
        ps = self.model.patch_embed.patch_size
        self.patch_size = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        self.n_blocks = len(self.model.blocks)
        if grad_checkpointing:
            self.model.set_grad_checkpointing(True)

    @staticmethod
    def _create(name: str, kwargs: dict, image_size: int | None) -> nn.Module:
        if image_size is None:
            return timm.create_model(name, **kwargs)
        # Fixed-position-embedding ViTs need the size at build time (timm resamples the
        # pretrained position embedding). RoPE models such as DINOv3 accept any size.
        try:
            return timm.create_model(name, img_size=image_size, **kwargs)
        except TypeError:
            return timm.create_model(name, **kwargs)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.model.blocks

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.model.forward_features(x)
        return feats[:, 0], feats[:, self.num_prefix_tokens:]

    def layer_groups(self) -> list[tuple[str, list[nn.Parameter]]]:
        """Parameters grouped bottom to top: embed, block_0 ... block_{L-1}, top.

        embed: everything below the first block (patch embedding, class/register
        tokens, position embedding). top: everything above the last block (final norms).
        """
        groups: dict[str, list[nn.Parameter]] = {"embed": []}
        for i in range(self.n_blocks):
            groups[f"block_{i}"] = []
        groups["top"] = []
        for pname, p in self.model.named_parameters():
            if pname.startswith("blocks."):
                groups[f"block_{int(pname.split('.')[1])}"].append(p)
            elif pname.split(".")[0] in ("norm", "fc_norm", "head", "head_drop", "attn_pool"):
                groups["top"].append(p)
            else:
                groups["embed"].append(p)
        return list(groups.items())


def layer_decay_scale(group_name: str, n_blocks: int, decay: float) -> float:
    """LR multiplier for a backbone group: 1 for the top block and final norm,
    decay ** k for the block k below the top, decay ** n_blocks for the embedding."""
    if group_name == "top":
        return 1.0
    if group_name == "embed":
        return decay ** n_blocks
    i = int(group_name.split("_")[1])
    return decay ** (n_blocks - 1 - i)
