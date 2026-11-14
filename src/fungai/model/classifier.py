"""Fungi classifier skeleton. Starting point for backbone experiments."""

import timm
import torch
from torch import nn


class FungiClassifier(nn.Module):
    """Image classifier with a configurable timm backbone."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            embed_dim = self.backbone(dummy).shape[-1]
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        return self.head(features)
