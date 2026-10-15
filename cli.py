"""Recomendações para uma campanha do dataset.

Uso:
    python cli.py --campaign-id KLK-0127
    python cli.py --campaign-id KLK-0127 --top 3 --json
"""
from __future__ import annotations

import argparse
import json
import sys

from klike.data import load_clean
from klike.recommender import format_report, recommend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recomendações de criativo para uma campanha do Klike.")
    parser.add_argument("--campaign-id", required=True, help="ex.: KLK-0127")
    parser.add_argument("--top", type=int, default=5, help="número máximo de recomendações (padrão: 5)")
    parser.add_argument("--json", action="store_true", help="saída em JSON em vez de texto")
    args = parser.parse_args(argv)

    df = load_clean()
    match = df[df["campaign_id"] == args.campaign_id]
    if match.empty:
        print(f"Campanha {args.campaign_id} não encontrada.", file=sys.stderr)
        return 1

    result = recommend(match.iloc[0], top_k=args.top)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str) if args.json else format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
