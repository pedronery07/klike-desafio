# Log de decisões

Registro cronológico do que vi nos dados, o que decidi, por quê e o que descartei.
Alimenta a seção "Processo de pensamento" do README.

## Qualidade dos dados (`src/klike/data.py`)

### D1. Impressões infladas em 4 linhas
- **Vi:** KLK-0148, 0232, 0258 e 0413 têm 12–14M impressões (a próxima maior tem 1,6M), e `clicks/impressions` não bate com o `ctr` registrado.
- **Decidi:** reconstruir `impressions = clicks / ctr`.
- **Por quê:** cliques, CTR, spend e CPC são coerentes entre si nessas linhas; só impressões destoa. O custo por mil impressões confirma: com 12–14M impressões, o CPM seria de R$ 0,01–0,66, contra mínimo de R$ 1,31 nas outras 496 campanhas (mediana de R$ 27 no TikTok). Com as impressões recalculadas, os CPMs ficam normais (R$ 8–128). Corrigir preserva as 4 campanhas, que têm atributos criativos válidos, em vez de descartá-las.
- **Descartei:** remover as linhas (perde informação boa por causa de um campo).

### D2. Spend e CPC inconsistentes em 6 linhas
- **Vi:** `spend/clicks ≠ cpc`, e em todos os casos um dos dois está inflado em várias vezes. Em 3 linhas o inflado é o spend (KLK-0109, 0224, 0259) e em 3 é o CPC (KLK-0100, 0240, 0416, com CPC entre 20 e 26 contra uma mediana de ~0,8).
- **Decidi:** o spend correto é o menor entre `spend` e `cpc × clicks`; depois o CPC é recalculado como `spend/clicks`.
- **Por quê:** o ROAS funciona como terceira coluna de desempate e confirma a regra em **5 dos 6** casos. Em KLK-0109 e 0259, `revenue/roas` reproduz exatamente `cpc × clicks`, então o spend é que estava errado. Em KLK-0100, 0240 e 0416, `revenue/spend` reproduz exatamente o ROAS registrado, então o spend está certo e o CPC estava errado. O KLK-0224 tem receita zero e não permite a checagem; ali a escolha se apoia no CPC registrado (R$ 1,22) ser plausível para o Meta.
- **Ressalva:** é uma inferência, não uma certeza. A correção escolhida é a única que torna a linha inteira coerente. As 10 linhas corrigidas (2% da base) ficam marcadas na coluna `data_issue`.

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

## EDA (`notebooks/01_eda.ipynb`, `src/klike/analysis.py`)

### D7. Medianas com IC bootstrap em vez de médias com teste t
- **Por quê:** as métricas têm cauda longa e teto artificial (D6), e n é pequeno por segmento. A mediana é robusta, e o bootstrap não exige normalidade. O IC de 90% é mostrado para deixar explícita a incerteza (a engine também mostra o n).

### D8. Métricas normalizadas em vez de volumes
- **Decidi:** analisar CTR, engajamento, retenção (`avg_watch_time / duração`), taxa de conversão (`conversões / cliques`) e ROAS.
- **Por quê:** volumes refletem orçamento, não a qualidade do criativo.

### D9. Checagem de confounding com regressão em log(métrica)
- **Vi:** no bruto, campanhas com hook têm ROAS 21% menor (mediana).
- **Decidi:** comparar o lift bruto com o efeito ajustado (regressão com plataforma, objetivo, categoria, formato, densidade de texto, idade e retargeting como controles) e decompor a diferença, adicionando os controles um a um.
- **Resultado:** o efeito ajustado do hook no ROAS é ≈ −2%, ou seja, nulo. A decomposição mostra o efeito encolhendo de −17% para −2%, com o maior salto vindo da **plataforma**: o LinkedIn tem 54% de campanhas com hook e ROAS mediano de 0,53, o TikTok 44% e 2,96. O retargeting contribui menos (18% das campanhas com hook contra 25% das sem hook, e ele quase dobra o ROAS). Para CTR, engajamento e score, bruto e ajustado concordam em sinal e ordem de grandeza.
- **Implicação para a engine:** usar efeitos condicionados ao contexto (por plataforma e pelo modelo), nunca médias brutas globais.

### D10. A duração é tratada como hipótese, não como regra
- **Vi:** |ρ| < 0,15 entre duração e as métricas; as medianas por faixa oscilam bastante com n pequeno.
- **Decidi:** reportar as faixas com melhor CTR por plataforma (Meta 11–15s, TikTok 16–30s), mas com peso baixo na engine e linguagem de "teste sugerido".

## Modelagem (`notebooks/02_modeling.ipynb`, `src/klike/features.py`, `src/klike/model.py`)

### D11. Dois modelos: pré-lançamento (A) e completo (B)
- **Vi:** o dataset mistura informação disponível **antes** de veicular (criativo, contexto) com informação que só existe **depois** (CTR, engajamento, ROAS).
- **Decidi:** o modelo principal (A) usa só criativo + contexto. O modelo B inclui as métricas de performance e serve apenas de diagnóstico.
- **Por quê:** o Klike existe para apoiar decisões antes de investir, e a engine recomenda mudanças em criativos. Usar CTR para prever o score seria vazamento de dados: no momento da decisão, ele não existe.

### D12. Regressão linear (Ridge) em vez de gradient boosting
- **Vi:** na validação cruzada (5-fold × 3), o linear tem R² 0,71 contra 0,62 do gradient boosting no modelo A; no modelo B, os dois empatam (0,82 × 0,80).
- **Decidi:** usar o linear como modelo principal, sem padronizar as features.
- **Por quê:** é melhor nos dados, estável com 500 linhas e interpretável. Sem padronização, cada coeficiente é "pontos de score por atributo", o que vira a engenharia reversa do score e a quantificação das recomendações.
- **Descartei:** o gradient boosting como modelo principal. Ele perderia a interpretabilidade sem ganho de desempenho.

### D13. Interações com a plataforma; features descartadas
- **Decidi:** incluir interações plataforma × (formato, texto, hook, rosto, legenda), porque a EDA mostrou efeitos que mudam por plataforma. Legenda faltante vira "sem legenda" + flag, e a duração entra em log.
- **Descartei** (não melhoraram a validação cruzada): desvio da duração em relação à mediana da plataforma e rosto × público jovem.

### D14. Validação cruzada repetida em vez de um único corte treino/teste
- **Por quê:** com 500 linhas, um único corte depende muito da sorte de quais campanhas caem no teste. 15 rodadas dão uma média estável e um desvio-padrão, que mostra quanto o resultado varia.

### D15. A engine usa o efeito total (modelo A)
- **Vi:** comparando A e B, 86–95% do valor de hook, rosto, CTA e legenda é direto; o texto "high" tem 36% da penalidade via performance (pior CTR).
- **Decidi:** estimar o ganho de cada recomendação pelo modelo A.
- **Por quê:** ao adicionar um hook, o anunciante ganha tanto o ponto direto quanto o que vem da melhora de CTR e retenção. O efeito total é o que importa para a decisão.

## Recommendations engine (`src/klike/recommender.py`, `cli.py`, `notebooks/03_recommendations_demo.ipynb`)

### D16. Duas medidas por recomendação: score previsto e evidência de negócio
- **Decidi:** cada recomendação mostra (1) o ganho no klike_score pelo contrafactual do modelo A e (2) o lift mediano da métrica ligada ao objetivo (CTR para traffic/conversions/app_install; engajamento para awareness/engagement), na mesma plataforma, com IC 90% e n.
- **Por quê:** o score é a "nota" do Klike, mas o anunciante paga por cliques e engajamento. Quando o IC cruza o zero, a engine diz que a diferença não é clara, em vez de prometer um ganho de CTR que a base não sustenta.
- **Descartei:** evidência no nível plataforma × categoria. Com ~30 campanhas por célula, os ICs ficam largos demais (ex.: −44% a +232%). A categoria entra pelo modelo. No formato, não há fallback para a base inteira, porque o efeito muda de sinal entre plataformas.

### D17. Prioridade = ganho ÷ esforço
- **Decidi:** esforço em 3 níveis (baixo = CTA, legenda, texto; médio = hook, formato, áudio, duração; alto = rosto, que exige regravar) e ordenar por ganho ÷ esforço. O ganho em pontos continua visível em cada item.
- **Por quê:** "priorizar por impacto" sem custo leva sempre a "regrave o vídeo com uma pessoa". Um time de marketing começa pelo que dá mais retorno por hora de trabalho.
- **Limite:** os pesos de esforço são julgamento meu, não vêm dos dados; num produto real, viriam do time de criação.

### D18. Regras de segurança da engine
- Não recomenda o que a campanha já tem; não sugere horizontal no TikTok (não existe na base); só sugere ações com ganho ≥ 0,5 pt.
- Duração é sempre "hipótese a testar" e vai para o fim da lista (D10).
- Um criativo que já cumpre as boas práticas recebe menos de 3 recomendações, com aviso explícito, em vez de sugestões inventadas para completar a cota.
- O plano combinado soma os efeitos (modelo linear); na prática pode haver retorno decrescente, o que a base não permite estimar.

## Agente revisor e simulador (`src/klike/agent.py`, `app.py`, `notebooks/04_agent_comparison.ipynb`)

### D19. O LLM revisa a engine, não a substitui
- **Decidi:** o agente recebe a saída da engine e pode consultar 3 ferramentas só de leitura (perfil da campanha contra a mediana da plataforma, pontos por atributo, lift em qualquer métrica). Ele pode reordenar, comentar, sugerir a execução criativa e dar alertas. Não pode criar nem remover ações, mudar a classificação da evidência ou calcular.
- **Por quê:** a engine tem o que um LLM não garante (números corretos, determinismo, auditoria); o LLM tem o que a engine não tem (julgamento contextual e linguagem criativa). Duas fontes de verdade competindo deixariam o resultado confuso.
- **Descartei:** um agente que analisa os dados livremente e um "veredito por ação" (recomendo / não recomendo). Mais liberdade aumentaria o risco de alucinação sem um ganho claro.

### D20. Guardrails em código, com uma chance de correção
- **Decidi:** validar a resposta em código (schema, mesmas ações, evidência copiada, hipóteses no fim, números rastreáveis até a engine ou as ferramentas, sem % nem R$ na execução criativa, heurística contra promessa de efeito sem evidência forte). Se falhar, o agente recebe as violações e tenta mais uma vez; se falhar de novo, a revisão é descartada e vale só a engine.
- **Por quê:** "não invente números" no prompt é um pedido; no código é uma verificação, que pode ser medida (taxa de rejeição).
- **Limite:** a checagem de promessas é por regex e não pega frases indiretas.

### D21. Uma iteração no confronto, e parar
- **Vi:** na v4, 5 de 18 revisões foram rejeitadas: falsos positivos da minha regra ("1:1", "11 a 15 segundos" na execução) e problemas de formato (resposta vazia, JSON cortado, campo com nome errado).
- **Decidi:** separar investigação (com ferramentas) de resposta (sem ferramentas, com o schema imposto pela API) e trocar a regra da execução por "sem % ou R$, números rastreáveis". A v5 teve 1 rejeição, outro falso positivo ("16:9"), corrigido no código sem rodar de novo.
- **Por quê:** continuar ajustando regras em cima das mesmas 18 campanhas seria sobreajuste; uma rodada nova deveria usar outra amostra.

### D22. Simulador em Streamlit; agente do cache por padrão
- **Decidi:** uma página que recalcula score, decomposição e recomendações a cada mudança, a partir de uma campanha da base ou de um criativo em branco. O agente aparece do cache quando a revisão existe; ao vivo, só com chave e no máximo 5 chamadas por sessão.
- **Por quê:** o avaliador pode testar "e se…" sem instalar nada além do requirements e sem chave; a cota gratuita da API não suportaria uso aberto.
- **Descartei:** reproduzir na UI os gráficos da EDA e da modelagem (já estão nos notebooks).
