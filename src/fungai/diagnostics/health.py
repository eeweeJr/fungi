"""Health alarms over the diagnostic metrics. NaN/Inf stops the run; everything else
is a warning, logged as an MLflow tag `health.<name>` and printed."""

from __future__ import annotations

import math


class NonFiniteError(RuntimeError):
    pass


def check_finite(metrics: dict) -> None:
    """Raise if any NaN/Inf count is non-zero, or a loss is not finite."""
    bad = [k for k, v in metrics.items()
           if k.endswith(("/nan", "/inf")) and v]
    bad += [k for k, v in metrics.items()
            if k.startswith("loss/") and isinstance(v, float) and not math.isfinite(v)]
    if bad:
        raise NonFiniteError(f"non-finite values in: {sorted(bad)[:10]}")


def health_warnings(metrics: dict, cfg) -> dict[str, str]:
    """Name -> message for every alarm the metrics trip. cfg is a DiagnosticsConfig."""
    warn: dict[str, str] = {}

    def get(k):
        v = metrics.get(k)
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else v

    for k, v in metrics.items():
        if k.startswith("query/") and k.endswith("pairwise_cos") and v is not None \
                and v > cfg.alarm_query_cosine:
            warn["query_collapse"] = f"{k}={v:.3f} > {cfg.alarm_query_cosine}"
        if k.startswith("attn/") and k.endswith("/entropy") and get(k) is not None:
            if v < cfg.alarm_entropy_low:
                warn["attention_too_sharp"] = f"{k}={v:.3f} < {cfg.alarm_entropy_low}"
            if v > cfg.alarm_entropy_high:
                warn["attention_uniform"] = f"{k}={v:.3f} > {cfg.alarm_entropy_high}"
        if k.startswith("attn/") and k.endswith("/padding_mass") and v > cfg.alarm_padding_mass:
            warn["padding_attended"] = f"{k}={v:.2e} > {cfg.alarm_padding_mass}"
        if k.startswith("grad/") and k.endswith("/norm") and v == 0.0:
            warn[f"dead_gradient_{k.split('/')[1]}"] = f"{k} is exactly 0"
    for name in ("permutation", "padding"):
        v = get(f"check/{name}")
        if v is not None and v > cfg.invariance_tolerance:
            warn[f"invariance_{name}"] = f"check/{name}={v:.2e} > {cfg.invariance_tolerance}"
    return warn
