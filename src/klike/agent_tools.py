"""Ferramentas que o agente revisor pode chamar.

Todas são só de leitura e devolvem números já calculados pela engine, pelo modelo A
ou pela EDA: o agente interpreta, não calcula. Os tipos e docstrings seguem o padrão
que o SDK google-genai usa para gerar o schema de function calling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from klike.analysis import median_lift
from klike.data import PLATFORMS
from klike.features import build_features
from klike.model import coefficients
from klike.recommender import _reference

PERFORMANCE_METRICS = ["ctr", "engagement_rate", "watch_ratio", "conversion_rate", "roas"]
BOOLEAN_ATTRS = ["has_hook", "has_face", "has_cta", "has_subtitle"]
MODEL_MAE = 6.6  # erro médio do modelo A na validação cruzada (notebook 02)


def _coefs():
    model, df = _reference()
    return coefficients(model, list(build_features(df.head(1)).columns))


def perfil_campanha(campaign_id: str) -> dict:
    """Atributos do criativo, contexto, klike_score real e métricas de performance de uma
    campanha, ao lado da mediana das campanhas da mesma plataforma, para comparação.

    Args:
        campaign_id: identificador da campanha, no formato "KLK-0127".
    """
    _, df = _reference()
    match = df[df["campaign_id"] == campaign_id]
    if match.empty:
        return {"erro": f"campanha {campaign_id} não encontrada"}
    c = match.iloc[0]
    same = df[df["platform"] == c["platform"]]
    attrs = ["platform", "category", "objective", "target_audience_age", "is_retargeting", "video_duration_s",
             "format", "has_hook", "has_face", "has_cta", "has_subtitle", "text_density", "music_voice_ratio"]
    return {
        "campaign_id": campaign_id,
        "atributos": {a: _plain(c[a]) for a in attrs},
        "klike_score_real": _plain(c["klike_score"]),
        "klike_score_mediano_plataforma": round(float(same["klike_score"].median()), 1),
        "performance": {m: _round(c[m]) for m in PERFORMANCE_METRICS},
        "performance_mediana_plataforma": {m: _round(same[m].median()) for m in PERFORMANCE_METRICS},
    }


def pontos_por_atributo(plataforma: str) -> dict:
    """Quantos pontos de klike_score cada atributo vale numa plataforma, segundo o modelo A
    (regressão linear). Formato é relativo a vertical; densidade de texto, relativa a low.

    Args:
        plataforma: "Meta", "TikTok" ou "LinkedIn".
    """
    if plataforma not in PLATFORMS:
        return {"erro": f"plataforma deve ser uma de {PLATFORMS}"}
    coef = _coefs()

    def total(name: str) -> float:
        extra = coef.get(f"{plataforma}×{name}", 0.0) if plataforma != "Meta" else 0.0
        return round(float(coef[name] + extra), 1)

    pontos = {
        "hook": total("hook"), "rosto": total("face"), "cta": round(float(coef["cta"]), 1),
        "legenda": total("subtitle"),
        "texto_medium_vs_low": total("text_density=medium"), "texto_high_vs_low": total("text_density=high"),
        "formato_quadrado_vs_vertical": total("format=quadrado"),
        "musica_voz_de_0_para_1": round(float(coef["music_voice_ratio"]), 1),
    }
    if plataforma != "TikTok":
        pontos["formato_horizontal_vs_vertical"] = total("format=horizontal")
    return {"plataforma": plataforma, "pontos": pontos, "erro_medio_do_modelo_pts": MODEL_MAE}


def lift_metrica(atributo: str, plataforma: str, metrica: str) -> dict:
    """Diferença na mediana de uma métrica entre campanhas com e sem um atributo, na mesma
    plataforma, com intervalo de confiança bootstrap de 90% e o tamanho de cada grupo.
    Se o intervalo cruza o zero, a diferença não é clara.

    Args:
        atributo: "has_hook", "has_face", "has_cta" ou "has_subtitle".
        plataforma: "Meta", "TikTok", "LinkedIn" ou "todas".
        metrica: "ctr", "engagement_rate", "watch_ratio", "conversion_rate" ou "roas".
    """
    if atributo not in BOOLEAN_ATTRS or metrica not in PERFORMANCE_METRICS or plataforma not in PLATFORMS + ["todas"]:
        return {"erro": "argumento inválido; ver a descrição da ferramenta"}
    _, df = _reference()
    d = df if plataforma == "todas" else df[df["platform"] == plataforma]
    if metrica == "roas":
        d = d[d["has_revenue_objective"]]   # ROAS só faz sentido em objetivos com receita (D2)
    col = d[atributo]
    st = median_lift(d, col.eq(True).fillna(False), col.eq(False).fillna(False), metrica)
    out = {"atributo": atributo, "plataforma": plataforma, "metrica": metrica,
           "n_com": int(st["n_treat"]), "n_sem": int(st["n_ctrl"])}
    if np.isnan(st["lift"]):
        return {**out, "conclusao": "sem dados suficientes"}
    out.update({k: round(float(st[k]), 4) for k in ("median_treat", "median_ctrl", "lift", "ci_low", "ci_high")})
    out["conclusao"] = ("aumenta" if st["ci_low"] > 0 else "diminui" if st["ci_high"] < 0 else "diferença não é clara")
    return out


TOOLS = {f.__name__: f for f in (perfil_campanha, pontos_por_atributo, lift_metrica)}


def _plain(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    v = v.item() if hasattr(v, "item") else v
    return round(v, 4) if isinstance(v, float) else v


def _round(v):
    return None if pd.isna(v) else round(float(v), 4)
