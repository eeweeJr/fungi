"""Training loop skeleton with MLflow tracking."""

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-4))
    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run():
        mlflow.log_params(config)

        for epoch in range(config.get("epochs", 10)):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            mlflow.log_metric("train_loss", loss, step=epoch)
            print(f"Epoch {epoch}: loss={loss:.4f}")
