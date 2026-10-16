"""Confronto engine × agente: amostra de campanhas e métricas a partir das revisões salvas.

As revisões são geradas por `python scripts/run_agent_comparison.py` (precisa de chave) e
ficam no cache; tudo aqui lê o cache e roda sem chave (notebook 04).
"""
from __future__ import annotations

import json

import pandas as pd
from scipy.stats import kendalltau

from klike.agent import CACHE_DIR, review
from klike.data import PLATFORMS
from klike.recommender import recommend

PER_PLATFORM = 6
SEED = 0
# Campanhas usadas como exemplo no notebook 03 e na UI: revisadas também, fora da amostra.
SHOWCASE = ["KLK-0127", "KLK-0262", "KLK-0348"]

VIOLATION_TYPES = {
    "números que não aparecem": "número não rastreável",
    "execucao_criativa não pode conter números": "número na execução criativa",
    "evidencia deve ser": "evidência trocada",
    "hipotese": "hipótese fora do fim",
    "afirma efeito": "promessa sem evidência",
    "JSON": "JSON inválido",
}


def sample_campaigns(df: pd.DataFrame) -> list[str]:
    """6 campanhas por plataforma, entre as que têm ao menos 3 recomendações (há o que revisar)."""
    eligible = df[[recommend(row)["n_available"] >= 3 for _, row in df.iterrows()]]
    return [cid for p in PLATFORMS
            for cid in eligible[eligible["platform"] == p].sample(PER_PLATFORM, random_state=SEED)["campaign_id"]]


def load_reviews(df: pd.DataFrame, ids: list[str], version: str | None = None) -> dict[str, dict]:
    """Revisões salvas das campanhas `ids`. Sem `version`, usa o prompt atual; com `version`
    (ex.: "v4"), lê os arquivos daquela versão, para comparar iterações do prompt."""
    if version is None:
        reviews = (review(df[df["campaign_id"] == cid].iloc[0], allow_live=False) for cid in ids)
        return {r["campaign_id"]: r for r in reviews if r is not None}
    out = {}
    for path in sorted(CACHE_DIR.glob("KLK-*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if r["prompt_version"] == version and r["campaign_id"] in ids:
            out[r["campaign_id"]] = r
    return {cid: out[cid] for cid in ids if cid in out}


def summary_table(reviews: dict[str, dict]) -> pd.DataFrame:
    """Uma linha por campanha: validação, custo, uso de ferramentas e concordância de ordem."""
    rows = []
    for cid, r in reviews.items():
        eng = r["engine"]
        engine_order = [x["key"] for x in eng["recomendacoes"]]
        evid = {x["key"]: x["evidencia"] for x in eng["recomendacoes"]}
        row = {
            "campanha": cid, "plataforma": eng["platform"], "objetivo": eng["objective"],
            "aprovado": r["aprovado"], "correções": len(r["tentativas"]) - 1,
            "consultas": len(r["ferramentas_chamadas"]), "tokens": r["tokens"], "latência (s)": r["latencia_s"],
            "top1 engine": engine_order[0], "evid. top1 engine": evid[engine_order[0]],
        }
        if r["aprovado"]:
            agent_order = r["revisao"]["ordem_sugerida"]
            tau = kendalltau(range(len(engine_order)), [agent_order.index(k) for k in engine_order]).statistic
            row.update({
                "top1 agente": agent_order[0], "evid. top1 agente": evid[agent_order[0]],
                "mesmo top1": agent_order[0] == engine_order[0], "mesma ordem": agent_order == engine_order,
                "kendall τ": round(float(tau), 2) if len(engine_order) > 1 else 1.0,
                "alertas": len(r["revisao"]["alertas"]),
            })
        rows.append(row)
    return pd.DataFrame(rows).set_index("campanha")


def violations_table(reviews: dict[str, dict]) -> pd.DataFrame:
    """Violações por tipo em todas as tentativas (inclusive as que foram corrigidas)."""
    rows = []
    for cid, r in reviews.items():
        for attempt, t in enumerate(r["tentativas"], 1):
            for v in t["violacoes"]:
                kind = next((label for key, label in VIOLATION_TYPES.items() if key in v), "estrutura (ações/ordem)")
                rows.append({"campanha": cid, "tentativa": attempt, "tipo": kind, "violação": v})
    return pd.DataFrame(rows, columns=["campanha", "tentativa", "tipo", "violação"])


def tool_usage(reviews: dict[str, dict]) -> pd.DataFrame:
    rows = [{"campanha": cid, "ferramenta": t["ferramenta"], "métrica": t["args"].get("metrica"),
             "atributo": t["args"].get("atributo")} for cid, r in reviews.items() for t in r["ferramentas_chamadas"]]
    return pd.DataFrame(rows, columns=["campanha", "ferramenta", "métrica", "atributo"])
