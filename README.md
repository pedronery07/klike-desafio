# Klike Data Science Challenge

> 🚧 Em construção: EDA, modelagem do `klike_score` e um recommendations engine para criativos em vídeo.

## Status

| Parte | Status | Onde |
|---|---|---|
| Limpeza dos dados | ✅ | `src/klike/data.py` |
| 1 · Análise exploratória | ✅ | `notebooks/01_eda.ipynb` |
| 2 · Modelagem do `klike_score` | ⏳ | `notebooks/02_modeling.ipynb` |
| 3 · Recommendations engine | ⏳ | `src/klike/recommender.py`, `cli.py` |
| 4 · Visão de produto | ⏳ | este README |

## Como rodar

Requer Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                      # instala o pacote local `klike` (src/klike)
python -m ipykernel install --user --name klike --display-name "Python (klike)"
jupyter lab notebooks/01_eda.ipynb   # use o kernel "Python (klike)"
```

Os notebooks estão salvos com os resultados, então dá para lê-los direto no GitHub sem executar nada.

## Estrutura

```
data/                   dataset original
docs/decisions.md       log de decisões: o que vi, o que decidi, por quê e o que descartei
docs/conceitos.md       glossário dos conceitos estatísticos usados (bootstrap, confounding etc.)
notebooks/              01 EDA · 02 modelagem · 03 demo do engine
src/klike/              código reutilizável
  data.py               carga e limpeza (correção de inconsistências)
  analysis.py           lift com IC bootstrap e efeito ajustado por regressão
  paths.py              caminhos do projeto
reports/figures/        figuras usadas na análise
tests/                  testes da engine
pyproject.toml          define o pacote `klike` para que notebooks e scripts importem `src/klike`
```

## Destaques da EDA (preliminar)

- **Qualidade dos dados:** 10 das 500 campanhas tinham erros de registro (impressões, spend ou CPC incoerentes com as outras colunas). Foram corrigidas por consistência interna e marcadas na coluna `data_issue`. Nenhuma linha foi descartada.
- **Hook nos 3 primeiros segundos é a alavanca mais forte:** +50–70% de CTR e score mediano de 70 contra 52 sem hook, nas três plataformas.
- **Rosto humano tem papel diferente por plataforma:** no TikTok gera clique (+75% CTR); no Meta, interação (+54% engajamento).
- **Texto demais na tela** derruba o CTR, principalmente no Meta (−48% em relação a "medium").
- **O formato depende da plataforma:** vertical é o melhor no TikTok e o pior no LinkedIn.
- **As boas práticas se somam:** o score mediano vai de 36 (nenhuma) a 81 (hook + rosto + CTA + legenda).
- **Cuidado com confounding:** no bruto, campanhas com hook têm ROAS 21% menor. Ajustando pelo contexto (principalmente a plataforma), o efeito é ≈ −2%, ou seja, nulo. Por isso a engine vai usar efeitos condicionados ao contexto, não médias globais.

![Lift por plataforma](reports/figures/01_lift_por_plataforma.png)
![Boas práticas e score](reports/figures/02_escada_boas_praticas.png)

Detalhes, gráficos e interpretações em `notebooks/01_eda.ipynb`; justificativas em `docs/decisions.md`.
