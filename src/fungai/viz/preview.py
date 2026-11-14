"""Quick visualization helpers for inspecting dataset samples."""

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def show_sample(image: torch.Tensor, label: int | str = "", ax: plt.Axes | None = None) -> None:
    """Display a single image tensor."""
    ax = ax or plt.gca()
    img = image.permute(1, 2, 0).clamp(0, 1)
    ax.imshow(img)
    ax.set_title(str(label))
    ax.axis("off")


def show_batch(images: torch.Tensor, labels: list | None = None, ncol: int = 8) -> None:
    """Display a grid of image tensors."""
    grid = make_grid(images, nrow=ncol, padding=2, normalize=True)
    plt.figure(figsize=(ncol * 2, (len(images) // ncol + 1) * 2))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    if labels:
        plt.title(f"Labels: {labels[:ncol]}...")
    plt.tight_layout()
    plt.show()
