"""Engine de recomendações para criativos em vídeo.

Para uma campanha (uma linha do dataset ou um dict com os mesmos campos), a engine:

1. lista as ações que o anunciante controla e que a campanha ainda não cumpre;
2. estima o ganho de cada uma no klike_score com o modelo A (contrafactual: muda só
   aquele atributo, mantém o resto da campanha e compara as duas previsões);
3. busca evidência empírica do efeito na métrica que importa para o objetivo da
   campanha (CTR ou engajamento), na mesma plataforma (ou na base inteira, se faltar n);
4. prioriza por impacto ÷ esforço e monta um plano combinado com as melhores ações.

Os efeitos são associações observadas em 500 campanhas, não causalidade: são boas
apostas para testar, não garantias (ver notebook 03, limitações).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from klike.analysis import METRIC_LABELS, median_lift
from klike.data import BOOLEAN, TARGET, load_clean
from klike.features import build_features
from klike.model import load

# Métrica de negócio usada como evidência, conforme o objetivo da campanha.
OBJECTIVE_METRIC = {
    "traffic": "ctr",
    "conversions": "ctr",
    "app_install": "ctr",
    "awareness": "engagement_rate",
    "engagement": "engagement_rate",
}

EFFORT_LABEL = {1: "baixo", 2: "médio", 3: "alto"}

# Faixas de duração com melhor CTR na EDA. São hipóteses (D5): viram "teste sugerido".
DURATION_WINDOWS = {"Meta": (11, 15), "TikTok": (16, 30)}

# Proporção música/voz acima da qual sugerimos mais voz (o modelo penaliza música dominante).
MUSIC_VOICE_MAX = 0.6
MUSIC_VOICE_TARGET = 0.4

MIN_SCORE_GAIN = 0.5   # abaixo disso, a ação não vale a recomendação
MIN_GROUP_N = 15       # mínimo de campanhas em cada grupo para usar um contexto na evidência


@dataclass
class Action:
    """Uma mudança possível no criativo."""
    key: str
    title: str
    attribute: str
    new_value: object
    effort: int
    how: str
    hypothesis: bool = False


@dataclass
class Recommendation:
    key: str
    title: str
    how: str
    change: dict
    score_gain: float
    score_after: float
    effort: str
    priority: float
    evidence: dict = field(default_factory=dict)
    hypothesis: bool = False
    text: str = ""


# --------------------------------------------------------------------------- dados

_CACHE: dict = {}


def _reference() -> tuple[object, pd.DataFrame]:
    """Modelo A e base limpa, carregados uma vez por processo."""
    if not _CACHE:
        _CACHE["model"] = load()
        _CACHE["df"] = load_clean()
    return _CACHE["model"], _CACHE["df"]


def _as_frame(campaign: pd.Series | dict) -> pd.DataFrame:
    row = pd.DataFrame([dict(campaign)])
    for col in BOOLEAN:
        if col in row:
            row[col] = pd.array([None if pd.isna(v) else bool(v) for v in row[col]], dtype="boolean")
    return row


def predict_score(model, campaign: pd.DataFrame) -> float:
    return float(np.clip(model.predict(build_features(campaign))[0], 0, 100))


# --------------------------------------------------------------------------- ações

def candidate_actions(c: pd.Series) -> list[Action]:
    """Ações aplicáveis à campanha. O que ela já cumpre fica de fora."""
    actions = []
    if not _is_true(c["has_hook"]):
        actions.append(Action("hook", "Adicionar um hook nos 3 primeiros segundos", "has_hook", True, 2,
                              "Abrir com a promessa, o problema ou um movimento forte, antes de qualquer logo."))
    if not _is_true(c["has_face"]):
        actions.append(Action("face", "Mostrar um rosto humano", "has_face", True, 3,
                              "Incluir uma pessoa falando para a câmera, idealmente já na abertura."))
    if not _is_true(c["has_cta"]):
        actions.append(Action("cta", "Incluir um CTA explícito", "has_cta", True, 1,
                              "Dizer e escrever o próximo passo (\"Baixe agora\", \"Peça uma demo\")."))
    if not _is_true(c["has_subtitle"]):
        actions.append(Action("subtitle", "Adicionar legendas", "has_subtitle", True, 1,
                              "Legendar a fala: boa parte do público assiste sem som."))

    if c["text_density"] != "medium":
        verb = "Reduzir" if c["text_density"] == "high" else "Aumentar um pouco"
        actions.append(Action("text", f"{verb} o texto na tela (densidade média)", "text_density", "medium", 1,
                              "Manter só a mensagem principal e o CTA em texto; o resto vai na fala ou na legenda."
                              if c["text_density"] == "high" else
                              "Reforçar a mensagem principal e o CTA com texto curto na tela."))

    formats = ["vertical", "quadrado", "horizontal"]
    if c["platform"] == "TikTok":
        formats.remove("horizontal")   # não existe na base (nem faz sentido no feed do TikTok)
    for fmt in formats:
        if fmt != c["format"]:
            actions.append(Action(f"format_{fmt}", f"Exportar em formato {fmt}", "format", fmt, 2,
                                  "Reenquadrar o vídeo para o formato nativo do feed desta plataforma."))

    mv = c.get("music_voice_ratio")
    if pd.notna(mv) and mv > MUSIC_VOICE_MAX:
        actions.append(Action("voice", "Dar mais espaço à voz no áudio", "music_voice_ratio", MUSIC_VOICE_TARGET, 2,
                              f"Baixar a música e aumentar a narração (proporção música/voz de {mv:.2f} para ~{MUSIC_VOICE_TARGET})."))

    window = DURATION_WINDOWS.get(c["platform"])
    dur = c.get("video_duration_s")
    if window and pd.notna(dur) and not window[0] <= dur <= window[1]:
        target = window[0] if dur < window[0] else window[1]
        actions.append(Action("duration", f"Testar uma versão de {window[0]}–{window[1]}s", "video_duration_s", target, 2,
                              f"Hoje o vídeo tem {dur:.0f}s. Faixa com melhor CTR no {c['platform']} na base; vale um teste A/B.",
                              hypothesis=True))
    return actions


def _is_true(v) -> bool:
    """True só quando o atributo está presente; faltante (NA) conta como ausente."""
    return not pd.isna(v) and bool(v)


def _with_change(campaign: pd.DataFrame, attribute: str, value) -> pd.DataFrame:
    changed = campaign.copy()
    if attribute in BOOLEAN:
        changed[attribute] = pd.array([value], dtype="boolean")
    else:
        changed[attribute] = value
    return changed


# --------------------------------------------------------------------------- evidência empírica

def _groups(df: pd.DataFrame, action: Action, c: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Máscaras de tratamento (já tem o atributo-alvo) e controle (está como a campanha)."""
    col = df[action.attribute]
    if action.attribute in BOOLEAN:
        return col.eq(True).fillna(False), col.eq(False).fillna(False)
    if action.key == "voice":
        return col <= 0.5, col > MUSIC_VOICE_MAX
    if action.key == "duration":
        lo, hi = DURATION_WINDOWS[c["platform"]]
        return col.between(lo, hi), col.notna() & ~col.between(lo, hi)
    return col.eq(action.new_value), col.eq(c[action.attribute])


def evidence(df: pd.DataFrame, action: Action, c: pd.Series) -> dict:
    """Lift mediano da métrica-chave na plataforma da campanha, com IC bootstrap de 90%."""
    # A faixa de duração foi escolhida pelo CTR na EDA; as demais ações seguem o objetivo.
    metric = "ctr" if action.key == "duration" else OBJECTIVE_METRIC.get(c["objective"], "ctr")
    key = (action.key, _plain(c[action.attribute]) if action.attribute in ("format", "text_density") else None,
           c["platform"], metric)
    if key not in _EVIDENCE:
        _EVIDENCE[key] = _compute_evidence(df, action, c, metric)
    return dict(_EVIDENCE[key])


_EVIDENCE: dict = {}


def _compute_evidence(df: pd.DataFrame, action: Action, c: pd.Series, metric: str) -> dict:
    treat, ctrl = _groups(df, action, c)
    # Não descemos ao nível plataforma × categoria: com ~30 campanhas por célula, divididas
    # em dois grupos, os intervalos ficam largos demais (ex.: −44% a +232%) para orientar
    # uma decisão. A categoria entra no score previsto, pelo modelo.
    contexts = [
        (f"no {c['platform']}", df["platform"] == c["platform"]),
        ("na base inteira", pd.Series(True, index=df.index)),
    ]
    if action.attribute == "format":
        contexts = contexts[:1]   # o efeito do formato muda de sinal entre plataformas: média global engana
    for label, scope in contexts:
        n_t = (scope & treat & df[metric].notna()).sum()
        n_c = (scope & ctrl & df[metric].notna()).sum()
        if n_t >= MIN_GROUP_N and n_c >= MIN_GROUP_N:
            st = median_lift(df, scope & treat, scope & ctrl, metric)
            return {"metric": metric, "context": label, "n_treat": int(st["n_treat"]), "n_ctrl": int(st["n_ctrl"]),
                    **{k: round(float(st[k]), 4) for k in ("lift", "ci_low", "ci_high", "median_treat", "median_ctrl")}}
    return {"metric": metric, "context": None}


# --------------------------------------------------------------------------- engine

def recommend(campaign: pd.Series | dict, top_k: int = 5) -> dict:
    """Recomendações priorizadas para uma campanha.

    Retorna um dict com o score previsto atual, as recomendações (ordenadas por
    impacto ÷ esforço) e o plano combinado com todas elas aplicadas juntas.
    """
    model, df = _reference()
    frame = _as_frame(campaign)
    c = frame.iloc[0]
    base = predict_score(model, frame)

    recs = []
    for action in candidate_actions(c):
        after = predict_score(model, _with_change(frame, action.attribute, action.new_value))
        gain = after - base
        if gain < MIN_SCORE_GAIN and not action.hypothesis:
            continue
        recs.append(Recommendation(
            key=action.key,
            title=action.title,
            how=action.how,
            change={"attribute": action.attribute, "from": _plain(c[action.attribute]), "to": action.new_value},
            score_gain=round(gain, 1),
            score_after=round(after, 1),
            effort=EFFORT_LABEL[action.effort],
            priority=round(max(gain, 0) / action.effort, 2),
            evidence=evidence(df, action, c),
            hypothesis=action.hypothesis,
        ))

    # Só o melhor formato entra (as alternativas de formato são excludentes).
    formats = [r for r in recs if r.key.startswith("format_")]
    if len(formats) > 1:
        best = max(formats, key=lambda r: r.score_gain)
        recs = [r for r in recs if not r.key.startswith("format_") or r is best]

    # Hipóteses (duração) vão sempre para o fim: são testes, não recomendações firmes.
    recs.sort(key=lambda r: (r.hypothesis, -r.priority, -r.score_gain))
    n_available = len(recs)
    recs = recs[:top_k]
    for r in recs:
        r.text = _describe(r, c, base)

    plan = _combined_plan(model, frame, [r for r in recs if not r.hypothesis], base)
    return {
        "campaign_id": c.get("campaign_id"),
        "platform": c["platform"],
        "category": c["category"],
        "objective": c["objective"],
        "score_actual": _plain(c.get(TARGET)),
        "score_predicted": round(base, 1),
        "n_available": n_available,
        "recommendations": [asdict(r) for r in recs],
        "combined_plan": plan,
    }


def _combined_plan(model, frame: pd.DataFrame, recs: list[Recommendation], base: float) -> dict:
    changed = frame
    for r in recs:
        changed = _with_change(changed, r.change["attribute"], r.change["to"])
    after = predict_score(model, changed)
    return {"actions": [r.key for r in recs], "score_before": round(base, 1), "score_after": round(after, 1),
            "score_gain": round(after - base, 1)}


# --------------------------------------------------------------------------- texto

def _plain(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    return v.item() if hasattr(v, "item") else v


def _num(x: float, digits: int = 1, sign: bool = False) -> str:
    return f"{x:{'+' if sign else ''}.{digits}f}".replace(".", ",").replace("-", "−")


def _pct(x: float) -> str:
    return f"{x * 100:+.0f}%".replace("-", "−")


def evidence_class(ev: dict, hypothesis: bool = False) -> str:
    """Resumo categórico da evidência: forte, nao_clara, contraria, sem_dados ou hipotese."""
    if hypothesis:
        return "hipotese"
    if ev.get("context") is None or np.isnan(ev.get("lift", np.nan)):
        return "sem_dados"
    if ev["ci_low"] > 0:
        return "forte"
    if ev["ci_high"] < 0:
        return "contraria"
    return "nao_clara"


def _upper_first(s: str) -> str:
    return s[:1].upper() + s[1:]


def _describe(r: Recommendation, c: pd.Series, base: float) -> str:
    if r.hypothesis:
        head = f"{r.title} (hipótese a testar): {_num(r.score_gain, sign=True)} pts no score previsto."
    else:
        head = f"{r.title}: {_num(r.score_gain, sign=True)} pts no klike_score previsto ({_num(base)} → {_num(r.score_after)})."
    ev = r.evidence
    label = METRIC_LABELS[ev["metric"]]
    kind = evidence_class(ev)
    if kind == "sem_dados":
        proof = f"Sem campanhas suficientes na base para medir o efeito no {label}."
    else:
        ci = f"IC 90%: {_pct(ev['ci_low'])} a {_pct(ev['ci_high'])}; n={ev['n_treat']} vs {ev['n_ctrl']}"
        if kind == "forte":
            proof = f"{_upper_first(ev['context'])}, campanhas assim têm {label} mediano {_pct(ev['lift'])} ({ci})."
        elif kind == "contraria":
            proof = f"Atenção: {ev['context']}, o {label} mediano é {_pct(ev['lift'])} ({ci}); o ganho é só no score."
        else:
            proof = f"{_upper_first(ev['context'])}, a diferença no {label} não é clara ({_pct(ev['lift'])}; {ci})."
    return f"{head} {proof} Como: {r.how} Esforço: {r.effort}."


def format_report(result: dict) -> str:
    """Relatório em texto para o terminal ou o notebook."""
    lines = [
        f"Campanha {result['campaign_id'] or '(nova)'} · {result['platform']} · {result['category']} · objetivo {result['objective']}",
        f"Klike score previsto: {_num(result['score_predicted'])}"
        + (f" (real: {_num(result['score_actual'])})" if result["score_actual"] is not None else ""),
        "",
    ]
    if not result["recommendations"]:
        lines.append("Nenhuma ação com ganho relevante: o criativo já cumpre as boas práticas mapeadas.")
    elif result["n_available"] < 3:
        lines += ["O criativo já cumpre a maior parte das boas práticas mapeadas; restam poucas melhorias.", ""]
    for i, r in enumerate(result["recommendations"], 1):
        lines.append(f"{i}. {r['text']}")
    plan = result["combined_plan"]
    if len(plan["actions"]) > 1:
        lines += ["", f"Plano combinado ({len(plan['actions'])} ações): score previsto "
                      f"{_num(plan['score_before'])} → {_num(plan['score_after'])} ({_num(plan['score_gain'], sign=True)} pts)."]
    return "\n".join(lines)
