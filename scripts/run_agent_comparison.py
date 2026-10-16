"""Roda o agente revisor na amostra do confronto e salva as revisões no cache.

Uso: python scripts/run_agent_comparison.py   (precisa de LLM_API_KEY no .env)
O notebook 04 lê os resultados do cache (reports/agent/cache), sem precisar de chave.
"""
from klike.agent import review
from klike.agent_eval import SHOWCASE, sample_campaigns
from klike.data import load_clean

if __name__ == "__main__":
    df = load_clean()
    ids = sample_campaigns(df) + SHOWCASE
    for i, cid in enumerate(ids, 1):
        r = review(df[df["campaign_id"] == cid].iloc[0])
        status = "cache" if r["cache_hit"] else f"{r['latencia_s']}s"
        print(f"[{i:2}/{len(ids)}] {cid} aprovado={r['aprovado']} tentativas={len(r['tentativas'])} ({status})", flush=True)
