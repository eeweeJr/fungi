"""Zero-training baselines, all logged to MLflow via the same evaluation framework."""

from __future__ import annotations

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from fungai.config import ExperimentConfig
from fungai.eval.report import build_evaluation_report
from fungai.tracking.mlflow_utils import log_evaluation_report, setup_experiment


def _collect_labels(loader: DataLoader) -> tuple[np.ndarray, list[str], list[str]]:
    """Iterate loader once to collect labels, image_paths, observation_ids."""
    all_labels: list[int] = []
    all_paths: list[str] = []
    all_obs_ids: list[str] = []
    for batch in loader:
        all_labels.extend(batch["label"].numpy().tolist())
        if "image_path" in batch:
            all_paths.extend(batch["image_path"])
        if "observation_id" in batch:
            all_obs_ids.extend(batch["observation_id"])
    return np.array(all_labels), all_paths, all_obs_ids


def _train_label_counts(loader: DataLoader) -> dict[int, int]:
    counts: dict[int, int] = {}
    for batch in loader:
        for lbl in batch["label"].numpy():
            if lbl >= 0:
                counts[int(lbl)] = counts.get(int(lbl), 0) + 1
    return counts


def _run_baseline(
    strategy: str,
    labels: np.ndarray,
    image_paths: list[str],
    obs_ids: list[str],
    num_classes: int,
    train_counts: dict[int, int],
    species_map: dict[str, int],
    toxicity_map: dict[str, bool],
    species_to_genus: dict[int, str],
    genus_names: list[str],
    eval_config: dict,
    experiment_id: str,
    **kwargs,
) -> dict:
    n = len(labels)

    if strategy == "random":
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(num_classes), size=n)
    elif strategy == "majority":
        probs = np.zeros((n, num_classes))
        majority_class = max(train_counts, key=train_counts.get)
        probs[:, majority_class] = 1.0
    elif strategy == "class_prior":
        total = sum(train_counts.values())
        prior = np.zeros(num_classes)
        for cls, count in train_counts.items():
            if 0 <= cls < num_classes:
                prior[cls] = count / total
        prior = prior / prior.sum()
        probs = np.tile(prior, (n, 1))
    elif strategy == "knn_frozen":
        from sklearn.neighbors import KNeighborsClassifier

        train_feats = kwargs["train_features"]
        train_labels = kwargs["train_labels"]
        val_feats = kwargs["val_features"]
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(train_feats, train_labels)
        probs = knn.predict_proba(val_feats)
        if probs.shape[1] < num_classes:
            padded = np.zeros((n, num_classes))
            for i, cls in enumerate(knn.classes_):
                padded[:, cls] = probs[:, i]
            probs = padded
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy}")

    predictions = {
        "probs": probs,
        "labels": labels,
        "image_paths": image_paths,
        "observation_ids": obs_ids,
    }

    with mlflow.start_run(run_name=f"baseline_{strategy}", experiment_id=experiment_id):
        mlflow.log_param("strategy", strategy)
        report = build_evaluation_report(
            predictions, species_map, toxicity_map,
            species_to_genus, genus_names, train_counts, eval_config,
        )
        log_evaluation_report(report)
        for k, v in report["primary_metrics"].items():
            mlflow.log_metric(k, v)

    return report


def extract_frozen_features(
    model_name: str,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a frozen pretrained backbone on all images, return (features, labels)."""
    import timm

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device).eval()

    all_feats, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            feats = model(images)
            all_feats.append(feats.cpu().numpy())
            all_labels.extend(batch["label"].numpy().tolist())

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return np.concatenate(all_feats), np.array(all_labels)


def run_all_baselines(
    val_loader: DataLoader,
    train_loader: DataLoader,
    config: ExperimentConfig,
    species_map: dict[str, int],
    toxicity_map: dict[str, bool],
    species_to_genus: dict[int, str],
    genus_names: list[str],
    train_species_counts: dict[int, int],
    device: torch.device,
) -> dict[str, dict]:
    """Run all baselines, log each as an MLflow run under 'fungai-baselines'."""
    experiment_id = setup_experiment("fungai-baselines")
    num_classes = len(species_map)

    labels, image_paths, obs_ids = _collect_labels(val_loader)
    train_counts = train_species_counts or _train_label_counts(train_loader)

    eval_config = {
        "aggregate_observations": config.evaluation.aggregate_observations,
        "failure_gallery_n": config.evaluation.failure_gallery_n,
    }

    common = dict(
        labels=labels, image_paths=image_paths, obs_ids=obs_ids,
        num_classes=num_classes, train_counts=train_counts,
        species_map=species_map, toxicity_map=toxicity_map,
        species_to_genus=species_to_genus, genus_names=genus_names,
        eval_config=eval_config, experiment_id=experiment_id,
    )

    results: dict[str, dict] = {}
    for strategy in ["random", "majority", "class_prior"]:
        results[strategy] = _run_baseline(strategy, **common)

    print("Extracting frozen features for kNN baseline...")
    train_feats, train_labels = extract_frozen_features(
        config.model.backbone, train_loader, device
    )
    val_feats, _ = extract_frozen_features(
        config.model.backbone, val_loader, device
    )
    results["knn_frozen"] = _run_baseline(
        "knn_frozen", **common,
        train_features=train_feats, train_labels=train_labels,
        val_features=val_feats,
    )

    return results
