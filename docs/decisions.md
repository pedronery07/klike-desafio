# Log de decisões

O que vi nos dados, o que decidi, por quê e o que descartei. Os números de apoio estão nos notebooks.

## Dados (`src/klike/data.py`)

### D1. Corrigir colunas incoerentes em vez de descartar linhas
- **Vi:** 4 campanhas com 12–14M impressões (a próxima tem 1,6M) e CTR incoerente com cliques/impressões; 6 com `spend/clicks ≠ cpc`, um dos dois inflado várias vezes.
- **Decidi:** recalcular o campo que contradiz os demais (`impressions = clicks/ctr`; spend = o menor entre `spend` e `cpc × clicks`), usando o ROAS como desempate, que confirma 5 dos 6 casos. As 10 linhas ficam marcadas em `data_issue`. CPC e receita faltantes são recalculados a partir de spend, clicks e ROAS.
- **Descartei:** remover as linhas, perdendo atributos criativos válidos por causa de um campo.

### D2. Faltantes, zeros e tetos
- Faltantes nos atributos (5–13% por coluna) ficam como NaN, sem imputação na EDA, para não criar associações artificiais.
- Receita zero em awareness/engagement é estrutural: ROAS só é analisado em traffic, conversions e app_install.
- CTR e engajamento têm teto artificial (55 campanhas com CTR exatamente 0,15), o que favorece medianas.

## EDA (`notebooks/01_eda.ipynb`, `src/klike/analysis.py`)

### D3. Medianas com IC bootstrap, sobre taxas
- **Por quê:** as métricas têm cauda longa e teto, e o n por grupo é pequeno; o bootstrap não exige normalidade. Uso taxas (CTR, engajamento, retenção, conversão, ROAS), porque volumes refletem orçamento, não o criativo.

### D4. Checar confounding antes de concluir
- **Vi:** no bruto, campanhas com hook têm ROAS 21% menor.
- **Decidi:** comparar cada efeito bruto com o ajustado por regressão em log(métrica) com o contexto como controle. O efeito do hook no ROAS cai para −2%, e quase toda a diferença vem da plataforma (o LinkedIn tem mais hooks e ROAS baixo).
- **Implicação:** a engine compara campanhas da mesma plataforma, nunca médias globais.

### D5. Duração é hipótese, não regra
- **Vi:** |ρ| < 0,15 entre duração e as métricas.
- **Decidi:** reportar as faixas com melhor CTR (Meta 11–15s, TikTok 16–30s) apenas como teste A/B sugerido.

## Modelagem (`notebooks/02_modeling.ipynb`, `src/klike/model.py`)

### D6. Um modelo para prever (A), outro para diagnosticar (B)
- **Decidi:** o modelo A usa só o que existe antes de veicular (criativo + contexto) e é o da engine. O B inclui CTR, engajamento e ROAS e serve só de diagnóstico.
- **Por quê:** o Klike apoia decisões antes de investir; usar CTR para prever o score seria usar uma informação que ainda não existe.

### D7. Regressão linear em vez de gradient boosting
- **Vi:** na validação cruzada, o linear teve R² 0,71 contra 0,62 do gradient boosting no modelo A.
- **Por quê:** além de melhor, sem padronizar as features cada coeficiente vira "pontos de score por atributo", que é a base das recomendações.

### D8. Features e validação
- Interações plataforma × (formato, texto, hook, rosto, legenda), porque a EDA mostrou efeitos que mudam por plataforma; descartei as features que não melhoraram a validação.
- Validação cruzada 5-fold × 3: com 500 linhas, um único corte treino/teste dependeria da sorte.

### D9. A engine usa o efeito total
- **Vi:** 86–95% do valor de hook, rosto, CTA e legenda é direto no score, não via CTR.
- **Decidi:** estimar o ganho pelo modelo A, que captura o efeito direto e o que vem pela performance.

## Engine (`src/klike/recommender.py`, `cli.py`)

### D10. Duas medidas por recomendação
- **Decidi:** (1) o ganho no score, mudando só aquele atributo e repredizendo com o modelo A; (2) o lift da métrica ligada ao objetivo (CTR ou engajamento) na mesma plataforma, com IC 90% e n. Se o IC cruza o zero, a engine diz que a diferença não é clara.
- **Descartei:** evidência por plataforma × categoria; com ~30 campanhas por célula, os ICs ficam largos demais.

### D11. Prioridade = ganho ÷ esforço
- **Por quê:** priorizar só por impacto leva sempre a "regrave com uma pessoa". Esforço em 3 níveis (CTA e legenda baixo; hook e formato médio; rosto alto).
- **Limite:** os pesos de esforço são julgamento meu; num produto, viriam do time de criação.

### D12. Regras de segurança
- Não recomenda o que a campanha já tem, nem horizontal no TikTok, nem ganhos < 0,5 pt. Duração vai sempre no fim, como hipótese.
- Um criativo que já cumpre as boas práticas recebe menos recomendações, com aviso, em vez de sugestões inventadas.

## Agente e simulador (`src/klike/agent.py`, `app.py`, `notebooks/04_agent_comparison.ipynb`)

### D13. O LLM revisa a engine, com guardrails em código
- **Decidi:** o agente lê a saída da engine, tem acesso a 3 ferramentas só de leitura (decide quais chamar e quantas vezes) e pode reordenar, comentar, sugerir a execução criativa e dar alertas. Não cria nem remove ações, não muda a evidência e não calcula. A resposta é validada em código (schema, mesmas ações, números rastreáveis); se falhar duas vezes, vale só a engine.
- **Por quê:** a engine garante números e determinismo; o LLM traz julgamento e linguagem. "Não invente números" no prompt é um pedido; no código é uma verificação mensurável.
- **Descartei:** um agente que analisa os dados livremente.

### D14. Uma iteração no confronto, e parar
- **Vi:** na 1ª versão, 5 de 18 revisões foram rejeitadas, por falsos positivos das minhas regras e problemas de formato.
- **Decidi:** separar investigação de resposta (com schema imposto pela API) e afrouxar a regra da execução criativa. Foi para 1 rejeição. Parei aí: continuar ajustando sobre as mesmas 18 campanhas seria sobreajuste.

### D15. Simulador com o agente lido do cache
- **Por quê:** o avaliador testa "e se…" sem chave de API; ao vivo, o limite é de 5 chamadas por sessão, por causa da cota gratuita.
