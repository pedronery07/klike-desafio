"""Simulador de criativos: interface para a engine de recomendações e o agente revisor.

Uso: streamlit run app.py
"""
from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

from klike import agent
from klike.data import PLATFORMS, load_clean
from klike.features import LEVELS, build_features
from klike.model import coefficients
from klike.recommender import METRIC_LABELS, evidence_class, recommend
from klike.recommender import _reference as reference

st.set_page_config(page_title="Klike · Simulador de criativos", page_icon="🎬", layout="wide")

# No Streamlit Cloud a chave vem dos secrets; localmente, do .env (ver .env.example).
try:
    if "LLM_API_KEY" in st.secrets:
        os.environ.setdefault("LLM_API_KEY", st.secrets["LLM_API_KEY"])
except FileNotFoundError:
    pass

LIVE_LIMIT = 5   # chamadas ao vivo ao agente por sessão (cota gratuita da API)
MAE = 6.6
EVIDENCE_BADGE = {
    "forte": "🟢 evidência forte",
    "nao_clara": "🟡 diferença não clara",
    "contraria": "🔴 efeito contrário",
    "sem_dados": "⚪ sem dados suficientes",
    "hipotese": "🧪 hipótese a testar",
}
def br(x: float, sign: bool = False) -> str:
    """Número com vírgula decimal (1 casa)."""
    return f"{x:{'+' if sign else ''}.1f}".replace(".", ",")


def br_pct(x: float) -> str:
    return f"{x:+.0%}".replace("-", "−")


FIELDS = ["platform", "category", "objective", "target_audience_age", "is_retargeting", "video_duration_s",
          "format", "has_hook", "has_face", "has_cta", "has_subtitle", "text_density", "music_voice_ratio"]


@st.cache_resource
def data() -> pd.DataFrame:
    reference()   # carrega (ou treina) o modelo A uma vez
    return load_clean()


def breakdown(campaign: dict) -> pd.DataFrame:
    """Contribuição de cada grupo de atributos para o score previsto (modelo linear)."""
    model, df = reference()
    X = build_features(pd.DataFrame([campaign]))
    X = pd.DataFrame(model[:-1].transform(X), columns=X.columns)
    contrib = coefficients(model, list(X.columns)) * X.iloc[0]
    groups = {"hook": "Hook", "face": "Rosto", "cta": "CTA", "subtitle": "Legenda", "text_density": "Texto",
              "format": "Formato", "music": "Áudio", "duration": "Duração"}

    def group(name: str) -> str:
        return next((label for key, label in groups.items() if key in name), "Contexto")

    out = contrib.groupby(contrib.index.map(group)).sum().rename_axis("grupo").rename("pontos").reset_index()
    return out[out["pontos"].abs() >= 0.05]


def campaign_form(base: dict, prefix: str) -> dict:
    """Formulário preenchido com `base`. Os widgets usam chaves com `prefix`: trocar o prefixo
    recria os widgets com os valores de `base` (é assim que o botão de restaurar funciona)."""
    c = dict(base)

    def k(name: str) -> str:
        return f"{prefix}-{name}"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Contexto**")
        c["platform"] = st.selectbox("Plataforma", PLATFORMS, index=PLATFORMS.index(base["platform"]), key=k("platform"))
        c["category"] = st.selectbox("Categoria", LEVELS["category"], index=LEVELS["category"].index(base["category"]), key=k("category"))
        c["objective"] = st.selectbox("Objetivo", LEVELS["objective"], index=LEVELS["objective"].index(base["objective"]), key=k("objective"))
        ages = LEVELS["target_audience_age"]
        c["target_audience_age"] = st.selectbox("Público", ages, index=ages.index(base["target_audience_age"]), key=k("age"))
        c["is_retargeting"] = st.checkbox("Retargeting", value=bool(base["is_retargeting"]), key=k("retargeting"))
    with col2:
        st.markdown("**Criativo**")
        c["has_hook"] = st.checkbox("Hook nos 3 primeiros segundos", value=base["has_hook"] is True, key=k("hook"))
        c["has_face"] = st.checkbox("Rosto humano", value=base["has_face"] is True, key=k("face"))
        c["has_cta"] = st.checkbox("CTA explícito", value=base["has_cta"] is True, key=k("cta"))
        c["has_subtitle"] = st.checkbox("Legendas", value=base["has_subtitle"] is True, key=k("subtitle"))
        formats = [f for f in LEVELS["format"] if not (c["platform"] == "TikTok" and f == "horizontal")]
        fmt = base["format"] if base["format"] in formats else formats[0]
        c["format"] = st.selectbox("Formato", formats, index=formats.index(fmt), key=k("format"))
        dens = LEVELS["text_density"]
        c["text_density"] = st.select_slider("Texto na tela", dens, value=base["text_density"], key=k("text"))
        dur = base["video_duration_s"]
        c["video_duration_s"] = float(st.slider("Duração (s)", 5, 60, int(dur) if pd.notna(dur) else 20, key=k("duration")))
        mv = base["music_voice_ratio"]
        c["music_voice_ratio"] = st.slider("Proporção música/voz", 0.0, 1.0, float(mv) if pd.notna(mv) else 0.4, 0.05,
                                           help="0 = só voz, 1 = só música", key=k("music"))
    return c


def show_recommendations(result: dict) -> None:
    if not result["recommendations"]:
        st.success("O criativo já cumpre as boas práticas mapeadas: nenhuma ação com ganho relevante.")
        return
    if result["n_available"] < 3:
        st.info("O criativo já cumpre a maior parte das boas práticas mapeadas; restam poucas melhorias.")
    for i, r in enumerate(result["recommendations"], 1):
        kind = evidence_class(r["evidence"], r["hypothesis"])
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f"**{i}. {r['title']}**  \n{EVIDENCE_BADGE[kind]} · esforço {r['effort']}")
            right.metric("score", br(r["score_after"]), f"{br(r['score_gain'], sign=True)} pts")
            ev = r["evidence"]
            if kind not in ("sem_dados", "hipotese") or ev.get("context"):
                if ev.get("context") and pd.notna(ev.get("lift")):
                    st.caption(f"{METRIC_LABELS[ev['metric']]} {ev['context']}: mediana {br_pct(ev['lift'])} "
                               f"(IC 90%: {br_pct(ev['ci_low'])} a {br_pct(ev['ci_high'])}; n={ev['n_treat']} vs {ev['n_ctrl']})")
            st.markdown(f"*Como:* {r['how']}")
    plan = result["combined_plan"]
    if len(plan["actions"]) > 1:
        st.markdown(f"**Plano combinado** ({len(plan['actions'])} ações): score previsto "
                    f"{br(plan['score_before'])} → **{br(plan['score_after'])}** ({br(plan['score_gain'], sign=True)} pts)")


TOOL_LABELS = {"perfil_campanha": "perfil da campanha", "pontos_por_atributo": "pontos por atributo",
               "lift_metrica": "efeito em métricas"}


def _review_header(res: dict) -> None:
    """Faixa de metadados: validação, dados consultados e custo."""
    fixed = len(res["tentativas"]) - 1
    tools = sorted({TOOL_LABELS.get(t["ferramenta"], t["ferramenta"]) for t in res["ferramentas_chamadas"]})
    n_calls = len(res["ferramentas_chamadas"])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(":green-badge[✅ Aprovada]")
        st.caption("sem ressalvas em relação à engine" if not fixed else
                   f"após {fixed} correção(ões) automática(s): a 1ª resposta violou uma regra e o agente corrigiu")
    with c2:
        st.markdown(f":blue-badge[🔎 {n_calls} consulta{'s' if n_calls != 1 else ''} aos dados]")
        st.caption("Ferramentas usadas: " + ", ".join(tools) if tools else "respondeu só com a saída da engine")
    with c3:
        st.markdown(f":gray-badge[{res['tokens']:,} tokens]".replace(",", "."))
        st.caption(res["modelo"] + (" · resposta salva (cache)" if res.get("cache_hit") else " · ao vivo"))
    if fixed:
        with st.expander("O que foi corrigido"):
            for v in res["tentativas"][0]["violacoes"]:
                st.markdown(f"- {v}")


def show_review(res: dict) -> None:
    if not res["aprovado"]:
        st.error("A revisão foi reprovada nos guardrails e descartada; vale só a saída da engine.")
        for v in res["tentativas"][-1]["violacoes"]:
            st.markdown(f"- {v}")
        return
    rv = res["revisao"]
    titles = {r["key"]: r["titulo"] for r in res["engine"]["recomendacoes"]}
    _review_header(res)

    engine_order = [r["key"] for r in res["engine"]["recomendacoes"]]
    st.markdown("**Prioridade: engine × agente**")
    st.caption("A engine ordena por ganho no score ÷ esforço. O agente pode manter essa ordem ou propor "
               "outra (por exemplo, pôr primeiro a única ação com evidência forte numa campanha de conversão).")
    order = pd.DataFrame({
        "#": range(1, len(engine_order) + 1),
        "Engine (ganho ÷ esforço)": [titles[k] for k in engine_order],
        "Agente": [titles[k] + ("" if k == engine_order[i] else "  ⬅ mudou")
                   for i, k in enumerate(rv["ordem_sugerida"])],
    })
    st.dataframe(order, hide_index=True, width="stretch")
    verdict = "Manteve a ordem da engine." if rv["mantem_ordem_da_engine"] else "Reordenou."
    st.markdown(f"**{verdict}** {rv['justificativa_ordem']}")
    by_key = {a["key"]: a for a in rv["acoes"]}
    for i, key in enumerate(rv["ordem_sugerida"], 1):
        a = by_key[key]
        with st.container(border=True):
            st.markdown(f"**{i}. {titles[key]}** · {EVIDENCE_BADGE[a['evidencia']]}  \n{a['comentario']}")
            st.markdown(f"🎬 *Execução:* {a['execucao_criativa']}")
    if rv["alertas"]:
        st.markdown("**Alertas**")
        for x in rv["alertas"]:
            st.warning(x)
    st.markdown(f"**Resumo:** {rv['resumo_para_o_anunciante']}")


# --------------------------------------------------------------------------- página

df = data()
st.title("🎬 Klike · Simulador de criativos")
st.markdown(
    "Descreva um criativo (ou parta de uma campanha da base), veja o **klike_score previsto** e as "
    "**recomendações** da engine, e peça uma revisão ao **agente especialista**. "
    "Detalhes em [README](https://github.com/pedronery07/klike-desafio) e nos notebooks."
)

ids = ["(criativo em branco)"] + df["campaign_id"].tolist()
start = st.sidebar.selectbox("Partir de uma campanha da base", ids, index=ids.index("KLK-0127"))
top_k = st.sidebar.slider("Máximo de recomendações", 3, 8, 5)
st.sidebar.caption("Modelo A: regressão linear, erro médio de 6,6 pts na validação cruzada.")
if start == "(criativo em branco)":
    base = {"platform": "Meta", "category": "E-commerce", "objective": "traffic", "target_audience_age": "25-34",
            "is_retargeting": False, "video_duration_s": 20.0, "format": "vertical", "has_hook": False,
            "has_face": False, "has_cta": False, "has_subtitle": False, "text_density": "medium",
            "music_voice_ratio": 0.4, "campaign_id": None, "klike_score": None}
else:
    row = df[df["campaign_id"] == start].iloc[0]
    base = {k: (None if pd.isna(row[k]) else (row[k].item() if hasattr(row[k], "item") else row[k]))
            for k in FIELDS + ["campaign_id", "klike_score"]}

form_col, result_col = st.columns([2, 3], gap="large")
with form_col:
    version = st.session_state.setdefault("form_version", 0)
    campaign = campaign_form(base, prefix=f"{start}-{version}")
    # Campos faltantes na base (None) aparecem no formulário com um valor padrão; só conta
    # como edição o que o usuário mudou de fato.
    defaults = {"has_subtitle": False, "video_duration_s": 20.0, "music_voice_ratio": 0.4}
    edited = any(campaign[k] != (defaults.get(k) if base[k] is None else base[k]) for k in FIELDS)
    if edited:
        campaign["campaign_id"], campaign["klike_score"] = None, None
        if base["campaign_id"]:
            st.caption(f"Versão modificada de {base['campaign_id']}: o score real não se aplica.")
        label = f"↩ Voltar à campanha original ({base['campaign_id']})" if base["campaign_id"] else "↩ Desfazer alterações"
        st.button(label, on_click=lambda: st.session_state.update(form_version=version + 1))
    else:
        campaign = base   # mantém os faltantes como faltantes, igual à CLI e ao cache do agente

result = recommend(campaign, top_k=top_k)
with result_col:
    m1, m2 = st.columns(2)
    m1.metric("Klike score previsto", br(result["score_predicted"]), help=f"± {br(MAE)} pts de erro médio do modelo")
    if result["score_actual"] is not None:
        m2.metric("Score real", br(result["score_actual"]))

    chart = breakdown(campaign)
    st.altair_chart(
        alt.Chart(chart).mark_bar().encode(
            x=alt.X("pontos:Q", title="pontos no score previsto"),
            y=alt.Y("grupo:N", sort="-x", title=None),
            color=alt.condition("datum.pontos > 0", alt.value("#2a9d8f"), alt.value("#e76f51")),
            tooltip=["grupo", alt.Tooltip("pontos:Q", format="+.1f")],
        ).properties(height=220, title="De onde vem o score (contribuição de cada atributo)"),
        width="stretch",
    )

    tab_engine, tab_agent = st.tabs(["Recomendações da engine", "Revisão do agente (LLM)"])
    with tab_engine:
        show_recommendations(result)
    with tab_agent:
        st.caption("Um LLM revisa a saída da engine e pode consultar os dados por ferramentas de leitura. "
                   "Ele não calcula nada: a resposta passa por validações automáticas (ações, evidência e "
                   "números rastreáveis) e é descartada se falhar.")
        if not result["recommendations"]:
            st.info("Sem recomendações para revisar.")
        else:
            cached = agent.review(campaign, allow_live=False)
            if cached:
                show_review(cached)
            elif not agent.has_api_key():
                st.info("Revisão ao vivo indisponível: defina LLM_API_KEY no .env (ver .env.example). "
                        "As campanhas do notebook 04 têm revisões salvas e aparecem aqui sem chave.")
            else:
                used = st.session_state.setdefault("live_calls", 0)
                if used >= LIVE_LIMIT:
                    st.info(f"Limite de {LIVE_LIMIT} revisões ao vivo por sessão atingido.")
                elif st.button(f"Pedir revisão ao agente ({LIVE_LIMIT - used} restantes nesta sessão)"):
                    st.session_state["live_calls"] = used + 1
                    with st.spinner("O agente está revisando…"):
                        try:
                            show_review(agent.review(campaign))
                        except Exception as e:   # cota, rede etc.: a engine segue funcionando
                            st.error(f"Não foi possível consultar o agente agora ({type(e).__name__}).")
