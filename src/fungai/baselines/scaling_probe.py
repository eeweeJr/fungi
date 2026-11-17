"""Backbone scaling probe: does a bigger frozen backbone separate these species better?

For each backbone in `evaluation.probe_backbones`, run it forward-only over the train
photos and the val evaluation photos, keep one vector per photo
(concat(class token, mean of patch tokens)), and fit:

  linear  standardised features, one linear layer, AdamW for `probe_epochs`, one fit
          per weight decay in `probe_weight_decays`, best kept by val macro F1
  knn     5 nearest neighbours, cosine (as the existing knn_frozen baseline)

Scores are per observation (photo probabilities averaged over the shared
`select_eval_photos` set), with 95% bootstrap intervals and paired-bootstrap
intervals for each step up in size. Features are cached under
data/processed/features/<backbone>_<image_size>/ as float16.

    uv run python -m fungai.baselines.scaling_probe --config configs/babys_first_model.yaml

Note: the weight decay is chosen on val and reported on val, the same way for every
backbone, so the comparison between backbones is fair but the absolute numbers are
slightly optimistic.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fungai.config import ExperimentConfig, load_config, seed_everything
from fungai.data.ingest import DATA_DIR

logger = logging.getLogger("fungai.scaling_probe")
FEATURE_DIR = DATA_DIR / "processed" / "features"
FORWARD_BATCH = 4  # photos per forward pass; measured to fit ViT-L at 448 px


def _collate(batch):
    return {"image": torch.stack([b["image"] for b in batch])}


@torch.no_grad()
def extract(backbone: str, df: pd.DataFrame, cfg: ExperimentConfig, device, out_dir: Path,
            split: str, monitor=None) -> np.ndarray:
    """[n_photos, 2D] float16 features for df's rows, cached on disk by row order.
    `monitor` (a RunMonitor) reports extraction progress live in MLflow."""
    from fungai.data.dataset import FungiDataset, backbone_normalization, build_eval_transform
    from fungai.model.backbone import TokenBackbone

    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path, index_path = out_dir / f"{split}.npy", out_dir / f"{split}_index.parquet"
    if feat_path.exists() and index_path.exists():
        idx = pd.read_parquet(index_path)
        if idx["filename"].tolist() == df["filename"].tolist():
            logger.info("cached features: %s", feat_path)
            return np.load(feat_path)
    mean, std = backbone_normalization(backbone)
    ds = FungiDataset(df, transform=build_eval_transform(cfg.data.image_size, mean, std))
    loader = DataLoader(ds, batch_size=FORWARD_BATCH, shuffle=False, num_workers=cfg.data.num_workers,
                        collate_fn=_collate)
    model = TokenBackbone(backbone, pretrained=True, image_size=cfg.data.image_size).to(device).eval()
    feats = np.empty((len(ds), 2 * model.embed_dim), dtype=np.float16)
    pos, t0 = 0, time.time()
    batches = monitor.track(loader, f"{split} features") if monitor is not None else loader
    for i, batch in enumerate(batches):
        cls, patches = model(batch["image"].to(device))
        v = torch.cat([cls, patches.mean(1)], dim=-1).float().cpu().numpy().astype(np.float16)
        feats[pos: pos + len(v)] = v
        pos += len(v)
        if i % 500 == 0:
            logger.info("%s %s: %d/%d photos (%.0fs)", backbone, split, pos, len(ds), time.time() - t0)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    np.save(feat_path, feats)
    df[["filename", "observation_id", "label", "genus_label"]].to_parquet(index_path, index=False)
    return feats


def fit_linear(xtr, ytr, xva, n_classes, weight_decay, epochs, device, seed) -> np.ndarray:
    """Train a linear softmax classifier; return val probabilities [n_val, C]."""
    torch.manual_seed(seed)
    xtr_t = torch.as_tensor(xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    layer = torch.nn.Linear(xtr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(ytr_t), generator=g).to(device)
        for start in range(0, len(perm), 1024):
            idx = perm[start: start + 1024]
            loss = torch.nn.functional.cross_entropy(layer(xtr_t[idx]), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        xva_t = torch.as_tensor(xva, dtype=torch.float32, device=device)
        return layer(xva_t).softmax(-1).cpu().numpy()


def fit_knn(xtr, ytr, xva, n_classes) -> np.ndarray:
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(xtr, ytr)
    p = knn.predict_proba(xva)
    out = np.zeros((len(xva), n_classes))
    out[:, knn.classes_] = p
    return out


def observation_probs(photo_probs: np.ndarray, val_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from fungai.eval.report import aggregate_by_observation

    obs_ids = val_df["observation_id"].astype(str).tolist()
    return aggregate_by_observation(photo_probs, val_df["label"].to_numpy(), obs_ids)


def run(cfg: ExperimentConfig, backbones: list[str], limit: int | None = None) -> dict:
    import mlflow

    from fungai.data.observations import select_eval_photos
    from fungai.data.snapshot import build_snapshot
    from fungai.eval.metrics import bootstrap_ci, macro_f1
    from fungai.model.train_observation import pick_device
    from fungai.safety.watchdog import MemoryWatchdog
    from fungai.tracking.mlflow_utils import get_git_hash, setup_experiment, start_monitored_run
    from fungai.tracking.monitor import RunMonitor

    seed_everything(cfg.seed)
    watchdog = MemoryWatchdog(cfg.safety.memory_limit_gb, cfg.safety.memory_poll_interval_s)
    watchdog.start()
    exp_id = setup_experiment(cfg.experiment_name)
    results, obs_level = {}, {}
    out = Path("outputs") / f"scaling_probe_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    # One parent run, watched live for the whole probe (progress, memory, console, errors);
    # each backbone's numbers go in a nested child run.
    start_monitored_run("scaling_probe", experiment_id=exp_id)
    mlflow.set_tags({"kind": "scaling_probe", "git_hash": get_git_hash(),
                     "backbones": ",".join(backbones), "limit": str(limit or "none")})
    try:
        with RunMonitor(out, len(backbones), watchdog=watchdog) as mon:
            mon.phase("loading data")
            device = pick_device()
            d, e = cfg.data, cfg.evaluation
            snapshot_dir, frames, protocol = build_snapshot(d.subset, d.protocol, d.split_seed,
                                                            d.n_holdout_species, d.n_holdout_toxic, size=d.size)
            train_df = frames["train"].reset_index(drop=True)
            val_df = select_eval_photos(frames["val"], d.max_photos)
            for name, df in (("train", train_df), ("val", val_df)):
                if "image_path" not in df.columns:
                    raise FileNotFoundError(f"no {d.size}p images for {name}; run the ingest for size {d.size}")
            if limit:  # quick pipeline check on a subset of observations
                keep_tr = train_df["observation_id"].drop_duplicates().sample(limit, random_state=0)
                n_va = min(limit, val_df["observation_id"].nunique())
                keep_va = val_df["observation_id"].drop_duplicates().sample(n_va, random_state=0)
                train_df = train_df[train_df["observation_id"].isin(keep_tr)].reset_index(drop=True)
                val_df = val_df[val_df["observation_id"].isin(keep_va)].reset_index(drop=True)
            n_classes = len(protocol.species_map)
            mlflow.set_tag("data.snapshot", snapshot_dir.name)
            for i_bb, bb in enumerate(backbones):
                mon.start_epoch(i_bb, label=f"{bb} ({i_bb + 1}/{len(backbones)})")
                tag = f"{bb}_{d.image_size}" + (f"_limit{limit}" if limit else "")
                t0 = time.time()
                xtr = extract(bb, train_df, cfg, device, FEATURE_DIR / tag, "train", mon).astype(np.float32)
                xva = extract(bb, val_df, cfg, device, FEATURE_DIR / tag, "val", mon).astype(np.float32)
                extract_s = time.time() - t0
                mon.phase(f"{bb} ({i_bb + 1}/{len(backbones)}) · fitting linear probe and kNN")
                mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
                xtr_s, xva_s = (xtr - mu) / sd, (xva - mu) / sd
                ytr = train_df["label"].to_numpy()
                fits = {}
                for wd in e.probe_weight_decays:
                    pp = fit_linear(xtr_s, ytr, xva_s, n_classes, wd, e.probe_epochs, device, cfg.seed)
                    op, ol = observation_probs(pp, val_df)
                    fits[wd] = (macro_f1(ol, op.argmax(1)), op, ol)
                best_wd = max(fits, key=lambda w: fits[w][0])
                lin_probs, labels = fits[best_wd][1], fits[best_wd][2]
                knn_probs, _ = observation_probs(fit_knn(xtr, ytr, xva, n_classes), val_df)
                known = labels >= 0
                ki = np.flatnonzero(known)
                y = labels[ki]
                lp, kp = lin_probs[ki].argmax(1), knn_probs[ki].argmax(1)
                n_s, seed = e.bootstrap_samples, e.bootstrap_seed
                res = {
                    "backbone": bb, "image_size": d.image_size, "best_weight_decay": best_wd,
                    "linear_macro_f1": bootstrap_ci(lambda i: macro_f1(y[i], lp[i]), len(ki), n_s, seed),
                    "linear_top_1": bootstrap_ci(lambda i: float((y[i] == lp[i]).mean()), len(ki), n_s, seed),
                    "knn_macro_f1": bootstrap_ci(lambda i: macro_f1(y[i], kp[i]), len(ki), n_s, seed),
                    "macro_f1_by_weight_decay": {str(w): f[0] for w, f in fits.items()},
                    "n_train_photos": len(train_df), "n_val_observations": len(labels),
                    "n_val_known_observations": int(known.sum()), "feature_dim": int(xtr.shape[1]),
                    "extract_seconds": round(extract_s, 1),
                }
                results[bb], obs_level[bb] = res, (y, lp)
                with mlflow.start_run(run_name=f"probe_{bb}", experiment_id=exp_id, nested=True):
                    mlflow.set_tags({"kind": "scaling_probe", "data.snapshot": snapshot_dir.name,
                                     "git_hash": get_git_hash()})
                    mlflow.log_params({"backbone": bb, "image_size": d.image_size, "best_weight_decay": best_wd,
                                       "probe_epochs": e.probe_epochs, "limit": limit or "none"})
                    mlflow.log_metrics({
                        "linear_macro_f1": res["linear_macro_f1"]["value"],
                        "linear_macro_f1_lo": res["linear_macro_f1"]["lo"],
                        "linear_macro_f1_hi": res["linear_macro_f1"]["hi"],
                        "linear_top_1": res["linear_top_1"]["value"],
                        "knn_macro_f1": res["knn_macro_f1"]["value"],
                        "peak_memory_gb": watchdog.peak_gb,
                    })
                logger.info("%s: linear macro F1 %.4f [%.4f, %.4f], kNN %.4f", bb,
                            res["linear_macro_f1"]["value"], res["linear_macro_f1"]["lo"],
                            res["linear_macro_f1"]["hi"], res["knn_macro_f1"]["value"])
                mon.end_epoch({f"{bb}/linear_macro_f1": res["linear_macro_f1"]["value"],
                               f"{bb}/knn_macro_f1": res["knn_macro_f1"]["value"]})

            diffs = {}
            for a, b in zip(backbones, backbones[1:]):
                (y, pa), (_, pb) = obs_level[a], obs_level[b]
                diffs[f"{b} - {a}"] = bootstrap_ci(
                    lambda i, y=y, pa=pa, pb=pb: macro_f1(y[i], pb[i]) - macro_f1(y[i], pa[i]),
                    len(y), e.bootstrap_samples, e.bootstrap_seed)
            summary = {"snapshot": snapshot_dir.name, "image_size": d.image_size, "limit": limit,
                       "results": results, "paired_differences_linear_macro_f1": diffs,
                       "peak_memory": watchdog.summary()}
            (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
            mlflow.log_artifact(str(out / "summary.json"))
            _print(summary)
            return summary
    finally:
        mlflow.end_run()  # no-op when the monitor already ended the run with its status
        print(f"Peak memory: {watchdog.summary()}", flush=True)
        watchdog.stop()


def _print(summary: dict) -> None:
    def f(ci):
        return f"{ci['value']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]" if ci and ci["value"] is not None else "n/a"

    print("\n=== Backbone scaling probe (frozen features, val, per observation) ===")
    for bb, r in summary["results"].items():
        print(f"  {bb:<40} linear macro F1 {f(r['linear_macro_f1'])}  kNN {r['knn_macro_f1']['value']:.4f}")
    for k, ci in summary["paired_differences_linear_macro_f1"].items():
        clear = ci["lo"] is not None and (ci["lo"] > 0 or ci["hi"] < 0)
        print(f"  {k}: {f(ci)}{'  (interval clear of zero)' if clear else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-feature backbone scaling probe")
    parser.add_argument("--config", default="configs/babys_first_model.yaml")
    parser.add_argument("--backbones", nargs="*", help="override evaluation.probe_backbones")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only this many train and val observations (pipeline check)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    run(cfg, args.backbones or cfg.evaluation.probe_backbones, args.limit)


if __name__ == "__main__":
    main()
