"""Carga e limpeza do dataset do Klike.

Toda análise, modelo e a engine de recomendações partem de `load_clean()`,
para que as decisões de limpeza fiquem num lugar só (ver docs/decisions.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from klike.paths import DATA

RAW_PATH = DATA / "klike_challenge_dataset.csv"

TARGET = "klike_score"

PLATFORMS = ["Meta", "TikTok", "LinkedIn"]
CATEGORICAL = ["platform", "category", "objective", "target_audience_age", "format", "text_density"]
BOOLEAN = ["is_retargeting", "has_subtitle", "has_cta", "has_hook", "has_face"]

# Atributos que o anunciante controla ao produzir o criativo.
CREATIVE = [
    "video_duration_s", "format", "has_subtitle", "has_cta", "has_hook",
    "has_face", "text_density", "music_voice_ratio",
]
# Contexto da campanha: não é "melhorável" pelo criativo, mas condiciona o efeito dele.
CONTEXT = ["platform", "category", "objective", "target_audience_age", "is_retargeting"]
# Resultados observados depois da veiculação.
PERFORMANCE = [
    "impressions", "clicks", "ctr", "cpc", "spend", "conversions",
    "revenue", "roas", "avg_watch_time_s", "engagement_rate",
]

# Objetivos em que receita/ROAS fazem sentido (awareness/engagement têm receita 0 por construção).
REVENUE_OBJECTIVES = ["traffic", "conversions", "app_install"]

_CTR_TOL = 0.002   # tolerância de arredondamento entre clicks/impressions e ctr
_CPC_TOL = 0.05    # tolerância em R$ entre spend/clicks e cpc


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    # has_subtitle tem faltantes, então vem como object; usamos o tipo booleano anulável.
    for col in BOOLEAN:
        df[col] = df[col].map({True: True, False: False, "True": True, "False": False}).astype("boolean")
    return df


def load_clean(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Dataset com as correções de consistência aplicadas e colunas derivadas.

    Não imputa os atributos criativos: o modelo (HistGradientBoosting) lida com NaN
    nativamente, e a EDA precisa enxergar o faltante como faltante. As flags
    `*_missing` preservam a informação para quem quiser imputar depois.
    """
    df = load_raw(path)
    df["data_issue"] = ""

    # 1) Impressões infladas: 4 linhas com 12-14M impressões cujo clicks/impressions
    #    não bate com o CTR registrado. CTR e cliques concordam com spend/cpc, então
    #    o campo errado é impressions; reconstruímos a partir de clicks/ctr.
    bad_impr = (df["clicks"] / df["impressions"] - df["ctr"]).abs() > _CTR_TOL
    df.loc[bad_impr, "impressions"] = (df.loc[bad_impr, "clicks"] / df.loc[bad_impr, "ctr"]).round()
    _flag(df, bad_impr, "impressions_fixed")

    # 2) spend x cpc inconsistentes (6 linhas). Em todos os casos um dos dois está
    #    inflado em várias vezes. Onde dá para checar, o ROAS confirma que o valor
    #    errado é o maior (ex.: KLK-0109 e KLK-0259: revenue/roas bate com cpc*clicks).
    #    Regra: o spend correto é o menor entre spend e cpc*clicks.
    implied_spend = df["cpc"] * df["clicks"]
    mismatch = df["cpc"].notna() & ((df["spend"] / df["clicks"] - df["cpc"]).abs() > _CPC_TOL)
    spend_inflated = mismatch & (df["spend"] > implied_spend)
    df.loc[spend_inflated, "spend"] = implied_spend[spend_inflated].round(2)
    _flag(df, spend_inflated, "spend_fixed")
    _flag(df, mismatch & ~spend_inflated, "cpc_fixed")

    # 3) Métricas derivadas recalculadas a partir das colunas base: corrige os cpc
    #    inflados e preenche cpc/revenue faltantes (são deriváveis, não "faltam").
    df["cpc_missing"] = df["cpc"].isna()
    df["revenue_missing"] = df["revenue"].isna()
    df["cpc"] = (df["spend"] / df["clicks"]).round(2)
    df["revenue"] = df["revenue"].fillna((df["roas"] * df["spend"]).round(2))
    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], 0.0).round(2)

    # 4) Flags dos faltantes que não dá para reconstruir.
    for col in ["video_duration_s", "music_voice_ratio", "has_subtitle", "engagement_rate", "avg_watch_time_s"]:
        df[f"{col}_missing"] = df[col].isna()

    # 5) Derivadas de negócio.
    df["watch_ratio"] = (df["avg_watch_time_s"] / df["video_duration_s"]).clip(upper=1.0)
    df["conversion_rate"] = np.where(df["clicks"] > 0, df["conversions"] / df["clicks"], np.nan)
    df["cpa"] = np.where(df["conversions"] > 0, df["spend"] / df["conversions"], np.nan)
    df["has_revenue_objective"] = df["objective"].isin(REVENUE_OBJECTIVES)
    df["month"] = df["date"].dt.month

    df["data_issue"] = df["data_issue"].str.strip(";")
    return df


def _flag(df: pd.DataFrame, mask: pd.Series, label: str) -> None:
    df.loc[mask, "data_issue"] = df.loc[mask, "data_issue"] + label + ";"
