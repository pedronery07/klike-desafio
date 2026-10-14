"""Feature engineering para os modelos do klike_score.

As categorias são fixas (não vêm de `pd.get_dummies` sobre os dados), para que uma
única campanha, ou uma versão modificada dela na engine de recomendações, gere
exatamente as mesmas colunas usadas no treino.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from klike.data import PLATFORMS

LEVELS = {
    "format": ["vertical", "quadrado", "horizontal"],
    "text_density": ["low", "medium", "high"],
    "platform": PLATFORMS,
    "category": ["E-commerce", "SaaS", "App Install", "Branding", "Lead Gen"],
    "objective": ["awareness", "traffic", "conversions", "engagement", "app_install"],
    "target_audience_age": ["18-24", "25-34", "35-44", "45+"],
}
# Primeiro nível de cada categoria fica de fora (referência), para evitar colinearidade.
REFERENCE = {col: levels[0] for col, levels in LEVELS.items()}

CREATIVE_FLAGS = {"has_hook": "hook", "has_face": "face", "has_cta": "cta", "has_subtitle": "subtitle"}

# Atributos cujo efeito pode mudar por plataforma (a EDA mostrou que mudam).
PLATFORM_INTERACTIONS = ["format", "text_density", "hook", "face", "subtitle"]


def build_features(df: pd.DataFrame, include_performance: bool = False) -> pd.DataFrame:
    """Matriz de features numéricas. NaN é mantido e tratado no pipeline do modelo.

    include_performance=False → modelo A ("pré-lançamento"): só criativo + contexto.
    include_performance=True  → modelo B ("completo"): + métricas observadas após veiculação.
    """
    X = pd.DataFrame(index=df.index)

    # Atributos do criativo
    for col, name in CREATIVE_FLAGS.items():
        X[name] = df[col].astype("Float64").astype(float)
    # Legenda faltante: preenche como "sem legenda" e sinaliza, em vez de imputar pela mediana (=True).
    X["subtitle_missing"] = X["subtitle"].isna().astype(float)
    X["subtitle"] = X["subtitle"].fillna(0.0)
    X["music_voice_ratio"] = df["music_voice_ratio"]
    X["log_duration"] = np.log(df["video_duration_s"])

    # Contexto
    X["retargeting"] = df["is_retargeting"].astype("Float64").astype(float)
    for col, levels in LEVELS.items():
        for lvl in levels[1:]:
            X[f"{col}={lvl}"] = (df[col] == lvl).astype(float)

    # Interações com a plataforma (Meta é a referência)
    for platform in PLATFORMS[1:]:
        on = (df["platform"] == platform).astype(float)
        for attr in PLATFORM_INTERACTIONS:
            if attr in LEVELS:
                for lvl in LEVELS[attr][1:]:
                    X[f"{platform}×{attr}={lvl}"] = on * X[f"{attr}={lvl}"]
            else:
                X[f"{platform}×{attr}"] = on * X[attr]

    if include_performance:
        X["ctr"] = df["ctr"]
        X["engagement_rate"] = df["engagement_rate"]
        X["watch_ratio"] = df["watch_ratio"]
        X["conversion_rate"] = df["conversion_rate"]
        X["log_roas"] = np.log1p(df["roas"])

    # O TikTok não tem campanhas horizontais: a interação seria uma coluna sempre zero.
    return X.drop(columns=["TikTok×format=horizontal"])
