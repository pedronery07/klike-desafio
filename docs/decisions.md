# Log de decisões

Registro cronológico do que vi nos dados, o que decidi, por quê e o que descartei.
Alimenta a seção "Processo de pensamento" do README.

## Qualidade dos dados (`src/klike/data.py`)

### D1. Impressões infladas em 4 linhas
- **Vi:** KLK-0148, 0232, 0258 e 0413 têm 12–14M impressões (a próxima maior tem 1,6M), e `clicks/impressions` não bate com o `ctr` registrado.
- **Decidi:** reconstruir `impressions = clicks / ctr`.
- **Por quê:** cliques, CTR, spend e CPC são coerentes entre si nessas linhas; só impressões destoa. Corrigir preserva as 4 campanhas (que têm atributos criativos válidos) em vez de descartá-las.
- **Descartei:** remover as linhas (perde informação boa por causa de um campo) e winsorizar (esconderia um erro de registro como se fosse um outlier legítimo).

### D2. Spend e CPC inconsistentes em 6 linhas
- **Vi:** `spend/clicks ≠ cpc`, e em todos os casos um dos dois está inflado em várias vezes. Em 3 linhas o inflado é o spend (KLK-0109, 0224, 0259) e em 3 é o CPC (KLK-0100, 0240, 0416, com CPC entre 20 e 26 contra uma mediana de ~0,8).
- **Decidi:** o spend correto é o menor entre `spend` e `cpc × clicks`; depois o CPC é recalculado como `spend/clicks`.
- **Por quê:** onde há receita para checar (KLK-0109 e KLK-0259), `revenue/roas` bate exatamente com o spend menor, o que confirma a regra.

### D3. CPC e receita faltantes são deriváveis
- **Vi:** 28 CPCs e 25 receitas faltando, mas `spend`, `clicks` e `roas` estão sempre presentes.
- **Decidi:** recalcular `cpc = spend/clicks` e `revenue = roas × spend`, mantendo flags `*_missing`.
- **Por quê:** não é dado ausente de verdade, é uma coluna redundante que não foi preenchida.

### D4. Faltantes nos atributos criativos ficam como NaN
- **Vi:** `has_subtitle` (46), `video_duration_s` (39), `music_voice_ratio` (38), `engagement_rate` (32) e `avg_watch_time_s` (27) faltando, em proporção parecida entre as plataformas (~5–13%).
- **Decidi:** não imputar na base limpa. A EDA analisa com os faltantes excluídos por métrica, o modelo usa o suporte nativo a NaN do HistGradientBoosting, e há flags `*_missing` como features.
- **Descartei:** imputar pela mediana logo de saída, o que achataria distribuições e criaria associações artificiais na EDA.

### D5. Receita zero não é faltante
- **Vi:** `revenue = 0` concentrado em awareness/engagement.
- **Decidi:** ROAS e receita são analisados só nos objetivos traffic, conversions e app_install.

### D6. Tetos no CTR e no engagement (observação)
- **Vi:** 55 campanhas com CTR exatamente 0,15 e 11 com engagement exatamente 0,10, o que indica clipping na geração dos dados.
- **Implicação:** os efeitos no topo da distribuição ficam subestimados. Medianas são mais robustas a isso do que médias.
