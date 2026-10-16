"""Agente revisor: uma camada de LLM por cima da engine de recomendações.

Papel: revisor sênior de criativos. Recebe a saída da engine para uma campanha e pode
consultar os dados com as ferramentas de `agent_tools.py` (só leitura). Devolve:
  1. se mantém ou muda a ordem das ações, e por quê;
  2. para cada ação, uma execução criativa concreta (o "como" que dados tabulares não dão);
  3. alertas sobre a campanha e um resumo para o anunciante.

O agente não calcula nem cria evidência: a engine continua sendo a fonte dos números.
Os guardrails são código, não prompt: a resposta passa por validações automáticas
(schema, ações existentes, classificação da evidência, números rastreáveis). Se falhar,
o agente recebe a lista de violações e tem uma chance de corrigir; se falhar de novo, a
revisão é descartada e vale só a saída da engine.

Usa function calling do SDK google-genai em loop manual (para registrar tokens, latência
e as ferramentas chamadas) e cache em disco, para que os resultados possam ser lidos e
reproduzidos sem chave de API.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Literal

from pydantic import BaseModel, ValidationError

from klike.agent_tools import TOOLS
from klike.paths import ROOT
from klike.recommender import evidence_class, recommend

PROMPT_VERSION = "v5"
CACHE_DIR = ROOT / "reports" / "agent" / "cache"
MAX_TOOL_ROUNDS = 6
MAX_CORRECTIONS = 1

Evidence = Literal["forte", "nao_clara", "contraria", "sem_dados", "hipotese"]


class ReviewedAction(BaseModel):
    key: str
    evidencia: Evidence
    comentario: str
    execucao_criativa: str


class Review(BaseModel):
    ordem_sugerida: list[str]
    mantem_ordem_da_engine: bool
    justificativa_ordem: str
    acoes: list[ReviewedAction]
    alertas: list[str]
    resumo_para_o_anunciante: str


SYSTEM_PROMPT = """Você é um revisor sênior de criativos em vídeo para anúncios (Meta, TikTok, LinkedIn) \
na plataforma Klike. O klike_score (0–100) mede a qualidade de um criativo. Uma engine baseada em dados \
já analisou a campanha e propôs ações; seu trabalho é revisar essa proposta como um especialista faria.

O que você recebe: o JSON da engine. Para cada ação, `score_gain` é o ganho previsto no klike_score \
(modelo linear treinado em 500 campanhas, erro médio de 6,6 pts), `evidence` é a diferença na mediana \
da métrica de negócio (CTR ou engajamento) entre campanhas com e sem aquela característica na mesma \
plataforma, com IC de 90%, e `evidencia` é a classificação dessa evidência. `priority` = ganho ÷ esforço, \
e o esforço é uma estimativa da engine, não um dado.

Ferramentas (só leitura): perfil_campanha (atributos, score real e performance da campanha contra a \
mediana da plataforma), pontos_por_atributo (pontos que cada atributo vale na plataforma) e lift_metrica \
(efeito de has_hook, has_face, has_cta ou has_subtitle em qualquer métrica de performance).

Como investigar antes de responder:
- Se houver campaign_id, chame perfil_campanha para ver onde a campanha está fraca em relação à mediana \
da plataforma. Se campaign_id for null, é um criativo ainda não lançado: não há performance real, então \
use pontos_por_atributo e lift_metrica para embasar a revisão.
- A engine só olha CTR ou engajamento. Para cada ação de hook, rosto, CTA ou legenda, chame lift_metrica \
na métrica mais próxima do objetivo que a engine NÃO usou: conversion_rate (objetivos conversions e \
app_install), watch_ratio (awareness) ou ctr (engagement). Use o resultado para confirmar ou questionar a ordem.
- Reordene quando houver motivo concreto: por exemplo, a campanha é de conversão e só uma ação tem \
evidência forte, ou a métrica mais fraca da campanha é justamente a que uma ação melhora. Se a ordem da \
engine já faz sentido, mantenha e diga por quê.

Regras obrigatórias (a resposta é verificada automaticamente e rejeitada se violar alguma):
1. Você não calcula nada. Todo número que escrever deve aparecer no JSON da engine ou no resultado de \
uma ferramenta (frações podem ser escritas como %, por exemplo 0.7854 → 79%). Não faça contas, \
diferenças ou médias.
2. Em `acoes`, inclua todas e somente as ações da engine (mesmas `key`), e copie em `evidencia` a \
classificação da engine para aquela ação.
3. Se a evidência de uma ação for nao_clara, sem_dados ou hipotese, não afirme que ela aumenta CTR, \
engajamento ou conversão; diga que o ganho é no score ou que precisa de teste.
4. `ordem_sugerida` contém todas as `key` da engine, cada uma uma vez. Ações de evidência "hipotese" \
ficam sempre no fim. Você pode reordenar as demais considerando o objetivo da campanha, a força da \
evidência e o esforço real; explique em `justificativa_ordem`. `mantem_ordem_da_engine` deve ser true \
só se a ordem for idêntica à da engine.
5. `execucao_criativa`: um exemplo concreto do que gravar ou escrever, adaptado à plataforma, à \
categoria e ao público (ex.: a primeira frase do hook, o texto do CTA). Não use porcentagens, valores \
em R$, estatísticas ou promessas de resultado; números só se vierem da própria ação (ex.: "11 a 15 \
segundos", "1:1"). Para dados do anunciante, use marcadores como [produto] ou [benefício].
6. `comentario`: 1 a 2 frases sobre por que a ação importa PARA ESTA campanha (objetivo, plataforma, \
ponto fraco), citando o ganho em pontos. Se você chamou lift_metrica para essa ação, diga o que ela \
mostrou (inclusive quando a diferença não é clara). Não repita só a classificação da evidência.
7. `alertas`: o que você viu nos dados e a engine não destacou, citando os números das ferramentas \
(ex.: uma métrica muito abaixo da mediana da plataforma; score real muito diferente do previsto). Pode ser vazio.
8. Escreva em português do Brasil, com acentuação correta e vírgula como separador decimal (16,9 pts; \
CTR de 0,0133), de forma direta, para um time de marketing.

Primeiro investigue com as ferramentas; quando terminar, você será instruído a escrever a revisão \
final. Ela deve ser um JSON válido, sem markdown, no formato:
{"ordem_sugerida": ["key", ...], "mantem_ordem_da_engine": true|false, "justificativa_ordem": "...",
 "acoes": [{"key": "...", "evidencia": "...", "comentario": "...", "execucao_criativa": "..."}],
 "alertas": ["..."], "resumo_para_o_anunciante": "2 a 3 frases"}"""


# --------------------------------------------------------------------------- entrada

def engine_payload(campaign) -> dict:
    """Saída da engine, enxuta, com a classificação da evidência de cada ação."""
    result = recommend(campaign)
    recs = []
    for r in result["recommendations"]:
        ev = {k: r["evidence"].get(k) for k in ("metric", "context", "lift", "ci_low", "ci_high", "n_treat", "n_ctrl")}
        recs.append({
            "key": r["key"], "titulo": r["title"], "score_gain": r["score_gain"], "score_after": r["score_after"],
            "effort": r["effort"], "priority": r["priority"], "evidence": ev,
            "evidencia": evidence_class(r["evidence"], r["hypothesis"]),
        })
    payload = {k: result[k] for k in ("campaign_id", "platform", "category", "objective", "score_predicted", "score_actual")}
    if result["score_actual"] is not None:
        payload["real_menos_previsto"] = round(result["score_actual"] - result["score_predicted"], 1)
    payload["recomendacoes"] = recs
    payload["plano_combinado"] = result["combined_plan"]
    return payload


# --------------------------------------------------------------------------- guardrails

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_RATIO = re.compile(r"\b\d{1,2}:\d{1,2}\b")
_CLAIM = re.compile(r"(aument|elev|melhor|sob[ea]|impulsion|dobr)\w*\s+(\w+\s+){0,3}(o\s+|a\s+)?"
                    r"(ctr|cliques?|engajamento|convers)", re.IGNORECASE)
_NEGATION = re.compile(r"não (é|está) clar|sem evidência|não há evidência|precisa de teste|teste a/b", re.IGNORECASE)


def _numbers_in(obj) -> list[float]:
    """Todos os números de um objeto JSON: valores numéricos e números escritos em strings."""
    out = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        return [abs(float(obj))]
    if isinstance(obj, str):
        return [float(m.replace(",", ".")) for m in _NUMBER.findall(obj)]
    items = obj.values() if isinstance(obj, dict) else obj
    for v in items:
        out += _numbers_in(v)
    return out


def _is_traceable(x: float, allowed: list[float]) -> bool:
    if x == int(x) and x <= 10:
        return True   # contagens e ordinais ("3 primeiros segundos", "2 ações")
    tol = 0.5 if x == int(x) else 0.051
    return any(abs(x - a) <= tol or abs(x - a * 100) <= tol for a in allowed)


def check_review(review: Review, payload: dict, context: list) -> list[str]:
    """Lista de violações; vazia quando a revisão passa em todos os guardrails."""
    violations = []
    engine_keys = [r["key"] for r in payload["recomendacoes"]]
    expected = {r["key"]: r["evidencia"] for r in payload["recomendacoes"]}

    if sorted(review.ordem_sugerida) != sorted(engine_keys):
        violations.append(f"ordem_sugerida deve conter exatamente as keys {engine_keys}")
    if sorted(a.key for a in review.acoes) != sorted(engine_keys):
        violations.append(f"acoes deve conter exatamente as keys {engine_keys}")
    for a in review.acoes:
        if a.key in expected and a.evidencia != expected[a.key]:
            violations.append(f"acao {a.key}: evidencia deve ser '{expected[a.key]}', não '{a.evidencia}'")
    hypotheses = [k for k in engine_keys if expected[k] == "hipotese"]
    if hypotheses and review.ordem_sugerida[-len(hypotheses):] != [k for k in review.ordem_sugerida if k in hypotheses]:
        violations.append("ações de evidência 'hipotese' devem ficar no fim de ordem_sugerida")
    if review.mantem_ordem_da_engine != (review.ordem_sugerida == engine_keys):
        violations.append("mantem_ordem_da_engine não corresponde à ordem sugerida")
    for a in review.acoes:
        if re.search(r"%|R\$", a.execucao_criativa):
            violations.append(f"acao {a.key}: execucao_criativa não pode conter porcentagens nem valores em R$")
        # Heurística para a regra 3: promessa de efeito em métrica sem evidência forte.
        if expected.get(a.key) != "forte" and _CLAIM.search(a.comentario) and not _NEGATION.search(a.comentario):
            violations.append(f"acao {a.key}: afirma efeito em métrica de negócio, mas a evidência é '{expected.get(a.key)}'")

    allowed = _numbers_in(context)
    texts = [review.justificativa_ordem, review.resumo_para_o_anunciante, *review.alertas,
             *(a.comentario for a in review.acoes), *(a.execucao_criativa for a in review.acoes)]
    # Proporções de tela ("16:9", "1:1") são vocabulário do formato, não dados: ficam fora da checagem.
    texts = [_RATIO.sub("", t) for t in texts]
    untraceable = sorted({x for t in texts for x in _numbers_in(t) if not _is_traceable(x, allowed)})
    if untraceable:
        violations.append(f"números que não aparecem na engine nem nas ferramentas: {untraceable}")
    return violations


def _tokens(response) -> int:
    return (response.usage_metadata.total_token_count or 0) if response.usage_metadata else 0


def _parse(text: str) -> Review:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return Review(**json.loads(text))


# --------------------------------------------------------------------------- execução

def _client():
    from dotenv import load_dotenv
    from google import genai

    load_dotenv(ROOT / ".env")
    if not os.environ.get("LLM_API_KEY"):
        raise RuntimeError("Defina LLM_API_KEY no arquivo .env (ver .env.example) para rodar o agente.")
    return genai.Client(api_key=os.environ["LLM_API_KEY"])


def has_api_key() -> bool:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return bool(os.environ.get("LLM_API_KEY"))


def model_name() -> str:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")


def _cache_path(payload: dict, model: str):
    key = json.dumps(payload, sort_keys=True, ensure_ascii=False) + model + PROMPT_VERSION + SYSTEM_PROMPT
    return CACHE_DIR / f"{payload['campaign_id'] or 'nova'}_{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"


def _generate(client, model, contents, config, attempts: int = 5):
    """Chamada à API com retry para limite de requisições (429) e sobrecarga (503)."""
    from google.genai import errors

    for attempt in range(1, attempts + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except (errors.ClientError, errors.ServerError) as e:
            if e.code not in (429, 503) or attempt == attempts:
                raise
            time.sleep(10 * attempt)


def review(campaign, use_cache: bool = True, allow_live: bool = True) -> dict | None:
    """Roda o agente revisor para uma campanha (linha do dataset ou dict).

    Retorna a saída da engine, a revisão validada (ou None, se reprovada nos guardrails),
    as violações encontradas em cada tentativa, as ferramentas chamadas, tokens e latência.
    Com allow_live=False, só consulta o cache e devolve None se a campanha não estiver lá.
    """
    payload = engine_payload(campaign)
    model = model_name()
    path = _cache_path(payload, model)
    if use_cache and path.exists():
        return {**json.loads(path.read_text(encoding="utf-8")), "cache_hit": True}
    if not allow_live:
        return None

    from google.genai import types

    client = _client()
    # Fase de investigação: o modelo decide quais ferramentas chamar.
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0,
        tools=list(TOOLS.values()),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    # Fase de resposta: sem ferramentas e com o schema imposto pela API (JSON sempre completo).
    answer_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0,
        response_mime_type="application/json",
        response_schema=Review,
    )
    user_msg = "Saída da engine para revisar:\n" + json.dumps(payload, ensure_ascii=False, indent=1)
    contents = [types.Content(role="user", parts=[types.Part(text=user_msg)])]
    context: list = [payload]          # tudo que o agente viu: base da checagem de números
    tool_calls, attempts = [], []
    tokens, start = 0, time.time()
    accepted = None

    for _ in range(MAX_TOOL_ROUNDS):
        response = _generate(client, model, contents, config)
        tokens += _tokens(response)
        candidate = response.candidates[0].content
        calls = [p.function_call for p in candidate.parts or [] if p.function_call]
        if not calls:
            break   # terminou de investigar; o texto livre desta fase é descartado
        contents.append(candidate)
        parts = []
        for fc in calls:
            args = dict(fc.args or {})
            try:
                result = TOOLS[fc.name](**args)
            except Exception as e:   # nome ou argumentos inválidos
                result = {"erro": str(e)}
            tool_calls.append({"ferramenta": fc.name, "args": args})
            context.append(result)
            parts.append(types.Part.from_function_response(name=fc.name, response={"resultado": result}))
        contents.append(types.Content(role="user", parts=parts))

    closing = "Investigação encerrada. Escreva agora a revisão final em JSON, seguindo todas as regras."
    contents.append(types.Content(role="user", parts=[types.Part(text=closing)]))
    for _ in range(MAX_CORRECTIONS + 1):
        response = _generate(client, model, contents, answer_config)
        tokens += _tokens(response)
        text = response.text or ""
        contents.append(response.candidates[0].content)
        try:
            parsed = _parse(text)
            violations = check_review(parsed, payload, context)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            parsed, violations = None, [f"JSON inválido ou fora do schema: {str(e)[:300]}"]
        attempts.append({"violacoes": violations, "texto_bruto": text})
        if not violations:
            accepted = parsed
            break
        feedback = ("Sua resposta foi rejeitada pelas validações automáticas:\n- " + "\n- ".join(violations)
                    + "\nCorrija e responda de novo só com o JSON completo.")
        contents.append(types.Content(role="user", parts=[types.Part(text=feedback)]))

    result = {
        "campaign_id": payload["campaign_id"],
        "modelo": model,
        "prompt_version": PROMPT_VERSION,
        "engine": payload,
        "revisao": accepted.model_dump() if accepted else None,
        "aprovado": accepted is not None,
        "tentativas": attempts,
        "ferramentas_chamadas": tool_calls,
        "tokens": tokens,
        "latencia_s": round(time.time() - start, 1),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {**result, "cache_hit": False}


def format_review(result: dict) -> str:
    """Revisão do agente em texto, ao lado da ordem original da engine."""
    engine = result["engine"]
    titles = {r["key"]: r["titulo"] for r in engine["recomendacoes"]}
    lines = [f"Revisão do agente · {result['campaign_id']} · {engine['platform']} · {engine['objective']}"]
    if not result["aprovado"]:
        lines.append("Revisão reprovada nos guardrails; vale só a saída da engine. Violações:")
        lines += [f"  - {v}" for v in result["tentativas"][-1]["violacoes"]]
        return "\n".join(lines)
    rv = result["revisao"]
    lines += [
        "Ordem da engine:  " + " → ".join(r["key"] for r in engine["recomendacoes"]),
        "Ordem do agente:  " + " → ".join(rv["ordem_sugerida"]) + ("" if rv["mantem_ordem_da_engine"] else "  (reordenada)"),
        f"Por quê: {rv['justificativa_ordem']}", "",
    ]
    by_key = {a["key"]: a for a in rv["acoes"]}
    for i, key in enumerate(rv["ordem_sugerida"], 1):
        a = by_key[key]
        lines += [f"{i}. {titles[key]} [evidência: {a['evidencia']}]", f"   {a['comentario']}",
                  f"   Execução: {a['execucao_criativa']}"]
    if rv["alertas"]:
        lines += ["", "Alertas:"] + [f"  - {x}" for x in rv["alertas"]]
    lines += ["", f"Resumo: {rv['resumo_para_o_anunciante']}"]
    return "\n".join(lines)
