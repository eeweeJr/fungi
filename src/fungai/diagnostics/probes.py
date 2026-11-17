"""Instrumentation for the observation model.

Two cadences:
  step   every `every_n_steps` optimizer steps: losses, LR per group, gradient norm
         and update ratio per parameter group, pre-clip norm, clip rate, head
         confidence. Cheap.
  epoch  once per epoch in eval mode on a fixed probe set of val observations:
         activation taps T0..T8, cross-attention statistics, query and embedding
         collapse, backbone drift, part-attention lift, invariance checks, figures.

Metric names: act/<tap>/<stat>, attn/block<i>/<stat>, query/<stat>, emb/<stat>,
grad/<group>/norm, upd/<group>/ratio, lr/<group>, drift/block<i>/cos,
head/<name>/<stat>, loss/<term>, check/<name>, parts/<part>/lift.
Hooks are attached only for the probe pass, so training steps pay nothing for them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from fungai.diagnostics import stats as S
from fungai.diagnostics.attention import attention_stats, part_attention, save_attention_figure
from fungai.diagnostics.health import check_finite, health_warnings

DRIFT_TOKENS = 16  # fixed patch positions per photo kept for the drift measurement


# --------------------------------------------------------------------------- groups

def parameter_groups(model) -> dict[str, list[torch.nn.Parameter]]:
    """Named parameter groups for gradient and update tracking."""
    groups: dict[str, list] = {}
    n_blocks = model.backbone.n_blocks
    for name, p in model.named_parameters():
        if name.startswith("backbone."):
            parts = name.split(".")
            if "blocks" in parts:
                i = int(parts[parts.index("blocks") + 1])
                g = f"backbone_q{min(4 * i // n_blocks, 3)}"
            else:
                g = "backbone_other"
        elif name.startswith("fusion."):
            if ".cross." in name or ".norm_q1." in name or ".norm_kv." in name:
                g = "fusion_cross"
            elif ".self_attn." in name or ".norm_q2." in name:
                g = "fusion_self"
            elif name.startswith("fusion.queries"):
                g = "fusion_queries"
            elif name.startswith("fusion.in_"):
                g = "fusion_in_proj"
            else:
                g = "fusion_mlp"
        elif name.startswith("head.genus"):
            g = "head_genus"
        elif name.startswith("head.species"):
            g = "head_species"
        elif name.startswith("photo_head"):
            g = "head_photo"
        else:
            g = "other"
        groups.setdefault(g, []).append(p)
    return groups


def _group_norm(params, attr: str = "grad") -> float:
    sq = 0.0
    for p in params:
        t = p.grad if attr == "grad" else p
        if t is not None:
            sq += float(t.detach().float().pow(2).sum())
    return sq ** 0.5


# --------------------------------------------------------------------------- hooks

class _Taps:
    """Forward hooks on the backbone (patch embed, every k-th block) and the fusion
    input projection. Records statistics and drift features while `active`."""

    def __init__(self, model, every: int):
        self.model, self.every = model, every
        bb = model.backbone
        self.prefix = bb.num_prefix_tokens
        self.block_ids = sorted({*range(every - 1, bb.n_blocks, every), bb.n_blocks - 1})
        self.records: dict[str, torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        bb = self.model.backbone
        self.handles.append(bb.model.patch_embed.register_forward_hook(self._hook("patch_embed")))
        for i in self.block_ids:
            self.handles.append(bb.blocks[i].register_forward_hook(self._hook(f"block{i}")))
        if self.model.fusion is not None:
            self.handles.append(self.model.fusion.in_proj.register_forward_hook(self._hook("fusion_in")))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _hook(self, name):
        def fn(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self.records[name] = t.detach()
        return fn


# --------------------------------------------------------------------------- main

class Diagnostics:
    def __init__(self, model, cfg, probe_batch: dict, run_dir: Path, device: torch.device,
                 mean, std, part_grids: dict | None = None, log_fn=None, tag_fn=None):
        """probe_batch: a collated observation batch (CPU) of fixed val observations.
        part_grids: {filename: {part: patch-grid fraction array [g, g]}} for photos
        that have masks. log_fn(metrics, step) and tag_fn(key, value) default to MLflow."""
        self.model, self.cfg, self.device = model, cfg, device
        self.probe = probe_batch
        self.run_dir = Path(run_dir) / "diagnostics"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.mean, self.std = mean, std
        self.part_grids = part_grids or {}
        self.groups = parameter_groups(model)
        self._snapshot = None
        self._clip_events = 0
        self._clip_total = 0
        self._initial_drift: dict[str, torch.Tensor] | None = None
        self.warnings: dict[str, str] = {}
        if log_fn is None or tag_fn is None:
            import mlflow
            log_fn = log_fn or (lambda m, step: mlflow.log_metrics(
                {k: float(v) for k, v in m.items() if _is_number(v)}, step=step))
            tag_fn = tag_fn or mlflow.set_tag
        self.log_fn, self.tag_fn = log_fn, tag_fn

    # ------------------------------------------------------------------ step cadence

    def is_log_step(self, step: int) -> bool:
        return step % self.cfg.every_n_steps == 0

    def before_optimizer_step(self, step: int) -> None:
        """Call after backward/clip, before optimizer.step(). Snapshots weights on log steps."""
        if not self.is_log_step(step):
            self._snapshot = None
            return
        self._snapshot = {g: [p.detach().clone() for p in ps if p.requires_grad]
                          for g, ps in self.groups.items()}

    def after_optimizer_step(self, step: int, out: dict, loss_parts: dict, optimizer,
                             pre_clip_norm: float, clip_norm: float, grad_norms: dict) -> dict:
        """Call after optimizer.step(). grad_norms are from `grad_norms()` taken before step."""
        self._clip_total += 1
        if clip_norm > 0 and pre_clip_norm > clip_norm:
            self._clip_events += 1
        if not self.is_log_step(step):
            return {}
        m: dict[str, float] = {f"loss/{k}": float(v.detach()) for k, v in loss_parts.items()}
        m["grad/global_preclip/norm"] = float(pre_clip_norm)
        m["grad/clip_fraction"] = self._clip_events / max(self._clip_total, 1)
        for g, v in grad_norms.items():
            m[f"grad/{g}/norm"] = v
        if self._snapshot is not None:
            for g, before in self._snapshot.items():
                after = [p for p in self.groups[g] if p.requires_grad]
                if not before:
                    continue
                delta = sum(float((a.detach() - b).float().pow(2).sum()) for a, b in zip(after, before)) ** 0.5
                w = sum(float(b.float().pow(2).sum()) for b in before) ** 0.5
                m[f"upd/{g}/ratio"] = delta / max(w, 1e-12)
            self._snapshot = None
        for i, pg in enumerate(optimizer.param_groups):
            m[f"lr/{pg.get('name', i)}"] = float(pg["lr"])
        m.update(self._head_stats(out))
        check_finite(m)
        self.log_fn(m, step)
        return m

    def grad_norms(self) -> dict[str, float]:
        """Per-group gradient L2 norm; call after backward, before clipping, on log steps."""
        return {g: _group_norm([p for p in ps if p.requires_grad]) for g, ps in self.groups.items()
                if any(p.requires_grad for p in ps)}

    @staticmethod
    def _head_stats(out: dict) -> dict:
        m = {}
        for name, key in (("genus", "genus_logp"), ("species_cond", "species_cond_logp"),
                          ("species", "species_logp")):
            if key in out:
                lp = out[key]
                if name == "species_cond":
                    # within-genus distribution: exp sums to 1 per genus, not overall
                    m["head/species_cond/max_prob"] = float(lp.detach().exp().max(-1).values.mean())
                    continue
                for k, v in S.prob_stats(lp).items():
                    m[f"head/{name}/{k}"] = v
        if "photo_logits" in out:
            for k, v in S.logit_stats(out["photo_logits"]).items():
                m[f"head/photo/{k}"] = v
        return m

    # ------------------------------------------------------------------ epoch cadence

    def _probe_forward(self, images, mask, taps: bool, return_attention=True):
        if not taps:
            return self.model(images, mask, return_attention=return_attention), None
        with _Taps(self.model, self.cfg.backbone_tap_every) as tp:
            out = self.model(images, mask, return_attention=return_attention)
        return out, tp

    def _drift_features(self, tp: _Taps) -> dict[str, torch.Tensor]:
        feats = {}
        for name, t in tp.records.items():
            if not name.startswith("block"):
                continue
            patches = t[:, tp.prefix:]
            n = patches.shape[1]
            pos = torch.linspace(0, n - 1, DRIFT_TOKENS, device=t.device).long()
            feats[name] = torch.cat([t[:, :1], patches[:, pos]], dim=1).float().cpu()
        return feats

    @torch.no_grad()
    def capture_initial(self) -> None:
        """Record backbone features of the probe set before training, for drift."""
        was = self.model.training
        self.model.eval()
        images, mask = self.probe["images"].to(self.device), self.probe["image_mask"].to(self.device)
        _, tp = self._probe_forward(images, mask, taps=True, return_attention=False)
        self._initial_drift = self._drift_features(tp)
        self.model.train(was)

    @torch.no_grad()
    def on_epoch_end(self, epoch: int, step: int) -> dict:
        was = self.model.training
        self.model.eval()
        images, mask = self.probe["images"].to(self.device), self.probe["image_mask"].to(self.device)
        out, tp = self._probe_forward(images, mask, taps=True)
        m: dict[str, float] = {}
        extra: dict = {}

        # T0 input
        real = images[mask]
        for c in range(real.shape[1]):
            m[f"act/T0_input/ch{c}_mean"] = float(real[:, c].mean())
            m[f"act/T0_input/ch{c}_std"] = float(real[:, c].std())
        # T1 patch embed
        pe = tp.records["patch_embed"]
        pe = pe.reshape(pe.shape[0], -1, pe.shape[-1])
        for k, v in S.basic(pe).items():
            m[f"act/T1_patch_embed/{k}"] = v
        m["act/T1_patch_embed/token_rms"] = float(S.token_rms(pe).mean())
        # T2 backbone blocks
        for name in tp.block_ids:
            t = tp.records[f"block{name}"]
            cls, patches = t[:, 0], t[:, tp.prefix:]
            rms = S.token_rms(patches)
            m[f"act/T2_block{name}/token_rms"] = float(rms.mean())
            m[f"act/T2_block{name}/cls_patch_ratio"] = float(S.token_rms(cls).mean() / rms.mean().clamp_min(1e-12))
            sub = patches[:, torch.linspace(0, patches.shape[1] - 1, 64, device=t.device).long()]
            m[f"act/T2_block{name}/patch_pairwise_cos"] = S.mean_pairwise_cosine(sub)
            fc = S.finite_counts(t)
            m[f"act/T2_block{name}/nan"], m[f"act/T2_block{name}/inf"] = fc["nan"], fc["inf"]
        # T3 backbone output (the encoder output the heads see)
        photo_feat = out["photo_feat"]
        d = photo_feat.shape[1] // 2
        last = tp.records[f"block{tp.block_ids[-1]}"][:, tp.prefix:]
        prms = S.token_rms(last)
        m["act/T3_backbone_out/cls_rms"] = float(S.token_rms(photo_feat[:, :d]).mean())
        m["act/T3_backbone_out/patch_rms"] = float(prms.mean())
        med = prms.median(dim=1, keepdim=True).values
        m["act/T3_backbone_out/high_norm_patch_frac"] = float((prms > 3 * med).float().mean())
        # T4 fusion input
        if "fusion_in" in tp.records:
            for k, v in S.basic(tp.records["fusion_in"]).items():
                m[f"act/T4_fusion_in/{k}"] = v
        # T5 cross-attention
        if "attn" in out:
            t_per = out["tokens_per_photo"]
            for i, w in enumerate(out["attn"]):
                a = attention_stats(w, out["key_mask"], mask, t_per)
                extra[f"attn_block{i}"] = a
                for k, v in a.items():
                    if not isinstance(v, list):
                        m[f"attn/block{i}/{k}"] = v
            # part attention, last block
            if self.cfg.part_attention and self.part_grids:
                pa = part_attention(out["attn"][-1], mask, t_per, self.probe["image_paths"], self.part_grids)
                extra["part_attention"] = pa
                for part, v in pa["lift"].items():
                    m[f"parts/{part}/lift"] = v
        # T6 query outputs, T7 observation embedding
        if "query_out" in out:
            q = out["query_out"]
            m["query/rms"] = float(S.token_rms(q).mean())
            m["query/pairwise_cos"] = S.mean_pairwise_cosine(q)
            m["query/effective_rank"] = float(np.nanmean([S.effective_rank(q[i]) for i in range(len(q))]))
        emb = out["obs_emb"]
        m["emb/rms"] = float(S.token_rms(emb).mean())
        m["emb/dim_std_across_batch"] = float(emb.float().std(0).mean())
        m["emb/effective_rank"] = S.effective_rank(emb)
        # T8 heads
        m.update(self._head_stats(out))
        # drift
        if self._initial_drift is not None:
            now = self._drift_features(tp)
            for name, before in self._initial_drift.items():
                cos = torch.nn.functional.cosine_similarity(now[name], before, dim=-1)
                m[f"drift/{name}/cos"] = float(cos.mean())
        # invariance checks
        m.update(self._invariance(images, mask, out))

        check_finite(m)
        warnings = health_warnings(m, self.cfg)
        for name, msg in warnings.items():
            if self.warnings.get(name) != msg:
                # A warning, not a print, so the run monitor counts it and shows it in MLflow.
                logging.getLogger("fungai.health").warning("[health] %s: %s", name, msg)
                self.tag_fn(f"health.{name}", msg)
        self.warnings.update(warnings)
        m["epoch"] = epoch
        self.log_fn(m, step)
        (self.run_dir / f"epoch_{epoch}.json").write_text(json.dumps(
            {"epoch": epoch, "step": step, "metrics": m, "detail": extra, "warnings": warnings},
            indent=1, default=float))
        if self.cfg.save_attention_figures and "attn" in out:
            n = min(self.cfg.n_figure_observations, len(mask))
            save_attention_figure(
                self.run_dir / "attention" / f"epoch_{epoch}.png", images[:n], mask[:n],
                out["attn"][-1][:n], out["tokens_per_photo"], self.mean, self.std,
                titles=[str(s) for s in self.probe.get("species", self.probe["observation_id"])[:n]])
        self.model.train(was)
        return m

    def _invariance(self, images, mask, out) -> dict:
        ref = out["species_logp"]
        b, p = mask.shape
        g = torch.Generator().manual_seed(0)
        perm_images, perm_mask = images.clone(), mask.clone()
        for i in range(b):
            n = int(mask[i].sum())
            order = torch.randperm(n, generator=g)
            perm_images[i, :n] = images[i, :n][order.to(images.device)]
        perm = self.model(perm_images, perm_mask)["species_logp"]
        pad_images = torch.cat([images, torch.zeros_like(images[:, :1])], dim=1)
        pad_mask = torch.cat([mask, torch.zeros_like(mask[:, :1])], dim=1)
        pad = self.model(pad_images, pad_mask)["species_logp"]
        dup_images = pad_images.clone()
        dup_mask = pad_mask.clone()
        dup_images[:, p] = images[:, 0]
        dup_mask[:, p] = True
        dup = self.model(dup_images, dup_mask)["species_logp"]
        return {
            "check/permutation": float((perm - ref).abs().max()),
            "check/padding": float((pad - ref).abs().max()),
            "check/duplicate_conf_change": float((dup.exp().max(-1).values - ref.exp().max(-1).values).abs().mean()),
        }


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
