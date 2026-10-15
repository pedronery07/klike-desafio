# Klike Data Science Challenge

Análise de 500 campanhas de vídeo (Meta, TikTok, LinkedIn), um modelo que prevê o `klike_score` antes do lançamento e uma engine que recomenda o que mudar no criativo.

> 🚧 Falta a Parte 4 (visão de produto) e a revisão final.

## Resumo de descobertas

- **Hook nos 3 primeiros segundos é a alavanca mais forte:** vale +16 a +19 pontos de score e +50–70% de CTR, nas três plataformas.
- **O que funciona depende da plataforma.** Rosto gera clique no TikTok (+75% CTR) e interação no Meta (+54% engajamento); vertical é o melhor formato no TikTok e o pior no LinkedIn.
- **O score parece um checklist de boas práticas + performance + ruído.** Uma regressão linear explica 71% dele só com atributos do criativo e contexto, e venceu o gradient boosting. Por ser linear, cada coeficiente é "quantos pontos o atributo vale".
- **Cuidado com conclusões brutas:** campanhas com hook têm ROAS 21% menor, mas a diferença some (−2%) ao controlar pela plataforma. Por isso a engine compara campanhas dentro da mesma plataforma.
- **A engine** devolve ações quantificadas (pontos de score + efeito em CTR/engajamento com intervalo de confiança), priorizadas por ganho ÷ esforço, e um plano combinado.

![Pontos por atributo](reports/figures/modelagem/04_pontos_por_atributo.png)

**Exemplo** (`python cli.py --campaign-id KLK-0262 --top 2`, resumido):

```
Campanha KLK-0262 · LinkedIn · SaaS · objetivo conversions
Klike score previsto: 47,6 (real: 46,7)

1. Adicionar um hook nos 3 primeiros segundos: +19,4 pts no klike_score previsto (47,6 → 66,9).
   No LinkedIn, campanhas assim têm CTR mediano +119% (IC 90%: +29% a +239%; n=45 vs 38). Esforço: médio.
2. Aumentar um pouco o texto na tela (densidade média): +8,7 pts (47,6 → 56,2).
   No LinkedIn, a diferença no CTR não é clara (+3%; IC 90%: −29% a +90%). Esforço: baixo.

Plano combinado (4 ações): score previsto 47,6 → 91,9 (+44,3 pts).
```

## Como rodar

Requer Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                                  # instala o pacote local `klike` (src/klike)

python cli.py --campaign-id KLK-0127              # recomendações (--top N, --json)
pytest                                            # testes da engine

python -m ipykernel install --user --name klike --display-name "Python (klike)"
jupyter lab notebooks/                            # use o kernel "Python (klike)"
```

Os notebooks estão salvos com os resultados: dá para lê-los no GitHub sem executar nada.

## Como pensei

Cada decisão está em [`docs/decisions.md`](docs/decisions.md) no formato "o que vi → o que decidi → por quê → o que descartei" (D1–D18). Os conceitos estatísticos estão explicados em [`docs/conceitos.md`](docs/conceitos.md).

**Limpeza** · `src/klike/data.py`
- 10 campanhas tinham impressões, spend ou CPC incoerentes com as outras colunas. Corrigi o campo que contradizia os demais, confirmando com o ROAS quando possível, e marquei na coluna `data_issue`. Nenhuma linha foi descartada (D1–D2).
- Receita zero em awareness/engagement não é faltante, é estrutural: o ROAS só é analisado nos objetivos de conversão (D5).

**Parte 1 · EDA** · [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)
- Comparei medianas com intervalo de confiança bootstrap, porque as métricas têm cauda longa e n pequeno por grupo; e usei taxas (CTR, engajamento), não volumes, que refletem orçamento (D7–D8).
- Todo efeito foi checado por plataforma e com regressão ajustada pelo contexto, para separar efeito real de efeito de mix (D9).
- Duração tem correlação fraca com tudo (|ρ| < 0,15): virou hipótese a testar, não regra (D10).

**Parte 2 · Modelo** · [`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb)
- **Dois modelos, por causa de vazamento de dados.** Os dois preveem o score; o que muda é o que eles podem usar. O **A** usa só o que existe antes de veicular (criativo + contexto) e é o que a engine usa. O **B** recebe também CTR, engajamento, retenção e ROAS. Ele não serve para prever (essas métricas não existem antes do lançamento), mas a diferença entre os dois mede quanto do score vem da performance (D11).
- Validação cruzada 5-fold × 3: o linear (R² 0,71, erro médio de 6,6 pts) venceu o gradient boosting (0,62) e o baseline (erro de 12,6). Escolhi o linear também pela interpretabilidade (D12, D14).
- 86–95% do valor de hook, rosto, CTA e legenda é efeito direto no score, não via CTR; a engine usa o efeito total (D15).

**Parte 3 · Engine** · `src/klike/recommender.py`, [`notebooks/03_recommendations_demo.ipynb`](notebooks/03_recommendations_demo.ipynb)
- Para cada ação que a campanha ainda não cumpre, muda só aquele atributo e repreve com o modelo A (ganho em pontos). Ao lado, mostra o lift de CTR ou engajamento (conforme o objetivo) na mesma plataforma, com IC e n. Quando o IC cruza o zero, diz que a diferença não é clara (D16).
- Prioriza por ganho ÷ esforço: um CTA (esforço baixo) vem antes de regravar com uma pessoa (alto), mesmo valendo menos pontos (D17).
- Não recomenda o que a campanha já tem nem horizontal no TikTok; duração aparece como teste A/B; um criativo que já cumpre as boas práticas recebe menos recomendações, com aviso, em vez de sugestões inventadas (D18).

## Limitações

- **Associação, não causa.** São 500 campanhas observadas, não experimentos; cada recomendação deveria ser validada com um teste A/B.
- **Erro de ~6,6 pts** no score previsto: ganhos de 1–2 pts estão dentro do ruído.
- **Amostras pequenas em alguns cortes** (ex.: 11 campanhas verticais no LinkedIn); a engine mostra o n e evita conclusões nesses casos.
- **O plano combinado soma os efeitos**; na prática deve haver retorno decrescente.

## Parte 4 · Visão de produto

*(em construção)*

## Estrutura

```
data/                  dataset original
docs/                  decisions.md (log de decisões) · conceitos.md (glossário estatístico)
notebooks/             01 EDA · 02 modelagem · 03 demo da engine
src/klike/             data.py (limpeza) · analysis.py (lift + IC) · features.py · model.py · recommender.py
cli.py                 interface de linha de comando da engine
reports/figures/       figuras da EDA (eda/) e da modelagem (modelagem/)
tests/                 testes da engine
models/                modelo treinado (gerado na primeira execução, fora do git)
```
