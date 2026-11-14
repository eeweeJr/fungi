"""Fungi classifier skeleton. Starting point for backbone experiments."""

import timm
import torch
from torch import nn


class FungiClassifier(nn.Module):
    """Image classifier with a configurable timm backbone."""

    def __init__(self, num_classes: int, backbone: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
