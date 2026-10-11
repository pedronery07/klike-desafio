"""Estatísticas de "lift" de atributos criativos sobre métricas de performance.

Usadas na EDA e reaproveitadas pela engine de recomendações, para que os números
mostrados ao time de marketing e os números das recomendações venham da mesma conta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["ctr", "engagement_rate", "watch_ratio", "conversion_rate", "klike_score"]

METRIC_LABELS = {
    "ctr": "CTR",
    "engagement_rate": "Engajamento",
    "watch_ratio": "Retenção (watch ratio)",
    "conversion_rate": "Taxa de conversão",
    "roas": "ROAS",
    "klike_score": "Klike Score",
}


def median_lift(
    df: pd.DataFrame,
    mask_treat: pd.Series,
    mask_ctrl: pd.Series,
    metric: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Lift relativo da mediana (tratamento vs controle) com IC bootstrap de 90%.

    Mediana em vez de média porque as métricas têm cauda longa e teto (CTR em 0,15).
    """
    t = df.loc[mask_treat, metric].dropna().to_numpy()
    c = df.loc[mask_ctrl, metric].dropna().to_numpy()
    out = {"n_treat": len(t), "n_ctrl": len(c), "lift": np.nan, "ci_low": np.nan, "ci_high": np.nan,
           "median_treat": np.nan, "median_ctrl": np.nan}
    if len(t) < 2 or len(c) < 2:
        return out
    mt, mc = np.median(t), np.median(c)
    out.update(median_treat=mt, median_ctrl=mc)
    if mc == 0:
        return out
    rng = np.random.default_rng(seed)
    bt = np.median(rng.choice(t, (n_boot, len(t))), axis=1)
    bc = np.median(rng.choice(c, (n_boot, len(c))), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = bt / bc - 1
    boots = boots[np.isfinite(boots)]
    out.update(lift=mt / mc - 1, ci_low=np.quantile(boots, 0.05), ci_high=np.quantile(boots, 0.95))
    return out


def boolean_lift_table(
    df: pd.DataFrame,
    attrs: list[str],
    metrics: list[str] = METRICS,
    by: str | None = None,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """Tabela longa com o lift de cada atributo booleano (True vs False), opcionalmente por segmento."""
    groups = [(None, df)] if by is None else list(df.groupby(by))
    rows = []
    for seg, g in groups:
        for attr in attrs:
            for m in metrics:
                is_true = g[attr].astype("boolean")
                stats = median_lift(g, is_true.eq(True).fillna(False), is_true.eq(False).fillna(False), m, n_boot=n_boot)
                rows.append({"segment": seg, "attribute": attr, "metric": m, **stats})
    return pd.DataFrame(rows)


def adjusted_log_effects(df: pd.DataFrame, metric: str, attrs: list[str], controls: list[str]) -> pd.Series:
    """Efeito % de cada atributo booleano sobre `metric`, controlando pelos demais atributos e
    pelo contexto (regressão linear em log(metric) com dummies). Checagem de robustez contra
    confounding: se o lift bruto e o ajustado concordam, o efeito não é só efeito de mix.
    """
    d = df[[metric] + attrs + controls].dropna()
    d = d[d[metric] > 0]
    X = pd.get_dummies(d[attrs + controls].astype({a: int for a in attrs}), columns=controls, drop_first=True, dtype=float)
    X.insert(0, "const", 1.0)
    y = np.log(d[metric].to_numpy())
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
    coefs = pd.Series(beta, index=X.columns)
    return (np.exp(coefs[attrs]) - 1).rename(metric)
