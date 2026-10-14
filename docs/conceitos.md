# Conceitos estatísticos usados na análise

Glossário de apoio ao notebook de EDA e ao `docs/decisions.md`. Cada conceito tem: **o que é**, **onde aparece no projeto** e um **resumo** em linguagem simples.

---

## 1. Distribuição, cauda longa e escala log

**O que é.** A *distribuição* descreve como os valores se espalham (o histograma). *Cauda longa* é quando há muitos valores pequenos e poucos enormes: impressões vão de 1 mil a 1,5 milhão.

**Escala log.** Transforma multiplicações em somas: 10 → 1, 100 → 2, 1.000 → 3. Dados de cauda longa ficam com cara de sino em escala log (distribuição *log-normal*), o que facilita visualizar e modelar.

**No projeto.** Os histogramas de volume estão em log10. A regressão de confounding usa log(métrica), e por isso os coeficientes viram efeitos **percentuais** (ver §8).

> **Resumo:** "Volumes de mídia variam em várias ordens de grandeza. Em log, uma campanha 10× maior fica a uma distância fixa, e é a variação relativa que importa."

---

## 2. Dados faltantes: MCAR, MAR, MNAR

| Tipo | Significado | Exemplo | Risco |
|---|---|---|---|
| **MCAR**: Missing Completely At Random | a ausência é puro acaso | o sistema falhou em salvar | baixo: ignorar as linhas não enviesa |
| **MAR**: Missing At Random | a ausência depende de **outra** variável observada | o LinkedIn registra menos a proporção música/voz | médio: controlável usando a variável |
| **MNAR**: Missing Not At Random | a ausência depende do **próprio valor** que falta | campanhas ruins não reportam o tempo de visualização | alto: viés difícil de corrigir |

**Imputar × marcar.** Imputar é preencher o faltante (ex.: com a mediana), o que é simples mas achata a distribuição e cria associações falsas. Marcar é criar uma coluna `*_missing` e deixar o modelo lidar com o NaN.

**No projeto.** As taxas de ausência são parecidas entre plataformas (5–13%), e o score de quem tem e de quem não tem o dado é parecido. Isso é compatível com MCAR. CPC e receita foram **recalculados** (são contas). O resto ficou como NaN + flag.

> **Resumo:** "Verifiquei se a ausência tinha padrão, por plataforma ou associada ao score, e não tinha. Então não imputei na análise; o modelo lida com o faltante nativamente."

---

## 3. Lift

**O que é.** A variação percentual de uma métrica entre dois grupos: `lift = mediana(com atributo) / mediana(sem atributo) − 1`.

**No projeto.** "Hook tem lift de +68% no CTR" significa que o CTR mediano das campanhas com hook é 68% maior que o das sem hook.

**Cuidado.** O lift é uma **comparação**, não um efeito causal (ver §7 e §8).

---

## 4. Correlação: Pearson × Spearman

**O que é.** Um número de −1 a +1 que diz o quanto duas variáveis andam juntas. Como referência: |ρ| ≈ 0,1 é fraco, 0,3 moderado e 0,5 forte.

| | Pearson | Spearman |
|---|---|---|
| Mede | relação **linear** | relação **monotônica** (sempre sobe ou sempre desce) |
| Usa | os valores | a **ordem** (ranking) dos valores |
| Outliers | sensível | robusto |

**Limitação importante.** Nenhuma das duas capta relações em **U**. A densidade de texto tem ρ ≈ −0,07 com o score, mas importa muito: "medium" é melhor que "low" e que "high". Correlação perto de zero **não** significa "sem relação".

**No projeto.** Heatmap de Spearman na seção 2. Hook tem ρ = 0,56 com o score.

> **Resumo:** "Usei Spearman porque as métricas têm outliers e relações não lineares. Mas correlação só capta relações que sempre sobem ou descem; por isso analisei a densidade de texto por categoria, onde o padrão em U aparece."

---

## 5. Amostra, população e incerteza amostral

**O que é.** A *população* é o universo de todas as campanhas possíveis. A *amostra* são as 500 que temos. Qualquer número calculado na amostra seria um pouco diferente em outra amostra. Essa variação é a **incerteza amostral**.

**Regra prática.** Quanto menor o grupo, maior a incerteza. O LinkedIn (n=83) e as campanhas verticais no LinkedIn (n=11) têm estimativas bem menos confiáveis que as do Meta (n=241).

---

## 6. Intervalo de confiança e bootstrap

**Intervalo de confiança (IC).** É uma faixa de valores plausíveis para o número real. "+68% [+43%, +90%]" significa que o efeito estimado é +68% e é plausível que o real esteja entre +43% e +90%.

**Leitura prática.**
- O IC inteiro do mesmo lado do zero indica um efeito **sólido** (hook → CTR).
- Um IC que **cruza o zero** é indistinguível de ruído (CTA → CTR: +14% [−11%, +39%]).

**Bootstrap.** É a forma de calcular o IC sem fórmulas:
1. sortear, **com reposição**, o mesmo número de campanhas de cada grupo;
2. recalcular a estatística (aqui, o lift das medianas);
3. repetir 1.000 vezes e pegar os percentis 5% e 95%.

**Por que não um teste t?** O teste t é feito para **médias** e supõe distribuição normal. Aqui comparo uma **razão de medianas** em dados assimétricos. O bootstrap funciona para qualquer estatística.

**90% × 95%.** É o nível de exigência. 95% é o padrão acadêmico; 90% é razoável para decisões de negócio exploratórias. Conferi as duas versões: em 23 das 24 combinações a conclusão é a mesma. A exceção (hook → ROAS) é justamente o efeito que se revela confounding.

> **Resumo:** "O bootstrap simula outras amostras reamostrando a própria base. Uso para saber se um efeito é sólido ou pode ser ruído, sem precisar supor que os dados são normais."

---

## 7. Correlação não é causalidade

**O que é.** Duas coisas andarem juntas não significa que uma causa a outra. Pode haver:
- uma **terceira variável** causando ambas (confounding, §8);
- **causalidade reversa**, em que B causa A;
- **acaso**, sobretudo com amostras pequenas.

**Por que importa aqui.** O desafio pede frases como "adicionar X **aumentaria** Y em Z%". Isso é uma afirmação **causal** sobre uma intervenção. Dados observacionais (campanhas que simplesmente aconteceram) só mostram associação.

**O padrão-ouro é o teste A/B**: sortear quem recebe a mudança. O sorteio equilibra todas as outras variáveis entre os grupos, e a diferença vira efeito causal.

> **Resumo:** "As recomendações são baseadas em associações ajustadas pelo contexto. São a melhor estimativa possível com dados observacionais, mas para confirmar o efeito eu rodaria um A/B test, e isso entra na proposta de produção."

---

## 8. Confounding (variável de confusão)

**O que é.** Uma terceira variável que influencia ao mesmo tempo o "tratamento" e o "resultado" e cria uma associação enganosa entre eles.

**Exemplo clássico.** Vendas de sorvete e afogamentos sobem juntos. O **calor** causa os dois. Comparando dias de mesma temperatura, a associação some.

**No projeto: hook × ROAS.**
- **No bruto**, campanhas com hook têm ROAS ~17–21% menor.
- **Por quê:** campanhas com hook estão concentradas onde o ROAS é naturalmente baixo. O LinkedIn tem 54% de campanhas com hook e ROAS mediano de 0,53. O TikTok tem 44% com hook e ROAS de 2,96. Também há menos retargeting entre as campanhas com hook (18% contra 25%), e retargeting quase dobra o ROAS.
- **Ajustando pelo contexto**, o efeito cai para ≈ −2%, ou seja, nulo.

| Controlando por | Efeito do hook no ROAS |
|---|---|
| nada | −17% |
| + retargeting | −14% |
| + objetivo | −11% |
| + categoria | −10% |
| + plataforma | −4% |
| + formato, texto, idade | −2% |

O maior confounder é a **plataforma**.

**Como se controla.**
- **Estratificação:** comparar dentro de grupos homogêneos (por plataforma, por exemplo). Simples, mas os grupos encolhem rápido.
- **Regressão com controles:** colocar o tratamento e os confounders juntos no modelo. O coeficiente do tratamento é o efeito "mantendo o resto constante".

**Ler um coeficiente de regressão em log.** Com `log(ROAS) = β₀ + β₁·hook + controles`, o efeito percentual do hook é `exp(β₁) − 1`.

**Limite.** A regressão só controla confounders **observados**. Se houver um que não está no dataset (ex.: qualidade da oferta), ele continua atrapalhando. Por isso, o A/B test.

> **Resumo:** "No bruto parecia que o hook piorava o ROAS. Mas as campanhas com hook estavam concentradas em plataformas e contextos de ROAS baixo. Controlando por isso numa regressão, o efeito some. Por isso a engine usa efeitos condicionados à plataforma e ao contexto, não médias globais."

---

## 9. Paradoxo de Simpson (parente do confounding)

**O que é.** O caso extremo de confounding: uma tendência aparece em todos os subgrupos, mas **se inverte** quando os grupos são juntados.

**Relação com o projeto.** No nosso caso não houve inversão completa, só o efeito sumindo. Mas é o mesmo mecanismo: misturar grupos com composições diferentes distorce a comparação. Vale citar se perguntarem sobre confounding.

---

## 10. Dados com teto (clipping)

**O que é.** Quando uma variável é cortada num valor máximo. Aqui, CTR em 0,15 (55 campanhas) e engajamento em 0,10 (11).

**Consequência.** Campanhas muito boas aparecem como "apenas 0,15", então os efeitos positivos no topo ficam **subestimados**. A mediana sofre menos com isso que a média.

---

## 11. Tamanho de amostra e segmentação

Quanto mais se quebram os dados (plataforma × categoria × idade), menor fica cada grupo e mais ruidosa cada estimativa. A engine vai usar **fallback hierárquico**: tenta o segmento específico e, se o n for pequeno, sobe para a plataforma e depois para o global. Cada recomendação mostra o n.

---

# Próximas partes (modelagem)

## 12. Treino, teste e validação cruzada

**O problema.** Avaliar o modelo nos mesmos dados em que ele treinou mede memorização, não aprendizado.

**Validação cruzada (k-fold).** Divide os dados em k partes (ex.: 5). Treina em 4, testa na 5ª e roda 5 vezes, para que cada parte seja teste uma vez. A média dos 5 resultados estima o desempenho em dados novos, e o desvio entre as rodadas mostra a estabilidade.

## 13. Métricas de regressão

| Métrica | O que é | Leitura |
|---|---|---|
| **MAE**: erro absoluto médio | média de \|previsto − real\| | "erramos em média X pontos de score" |
| **RMSE**: raiz do erro quadrático médio | parecido, mas pune mais os erros grandes | RMSE ≫ MAE indica alguns erros grandes |
| **R²** | fração da variação do score explicada pelo modelo (0 a 1) | 0,8 = explica 80% da variação |

**Baseline.** Sempre comparar com um modelo trivial (ex.: prever sempre a média). Um modelo só é útil se for melhor que isso.

## 14. Vazamento de dados (data leakage)

Usar no modelo uma informação que **não estaria disponível** no momento da previsão. Se o objetivo é recomendar melhorias **antes** de veicular, usar CTR ou ROAS como feature é vazamento: eles só existem depois. Por isso o plano tem dois modelos:
- **A, "pré-lançamento":** só atributos do criativo e contexto. É o que a engine usa.
- **B, "completo":** inclui as métricas, para medir quanto do score a performance explica.

## 15. Gradient Boosting

Um conjunto de muitas árvores de decisão pequenas, em que cada nova árvore corrige os erros das anteriores. Captura interações (ex.: "rosto ajuda mais no TikTok") e relações não lineares sem precisar especificá-las. O `HistGradientBoostingRegressor` do scikit-learn aceita NaN nativamente.

## 16. Importância de features e SHAP

- **Permutation importance:** embaralha uma coluna e mede o quanto o erro do modelo piora. Quanto mais piora, mais importante a coluna.
- **SHAP:** decompõe cada previsão em contribuições por feature. Ex.: "este vídeo tem score previsto de 72 = 60 (média) + 9 (hook) + 5 (rosto) − 2 (texto high)…". Serve para explicar previsões individuais, o que é útil para justificar as recomendações.

**Cuidado.** Importância mede o quanto o **modelo usa** a feature, não causalidade.

---

## 17. Regressão linear e regularização (Ridge)

**Regressão linear.** Modela o score como uma soma: `score = base + β₁·hook + β₂·rosto + …`. Cada coeficiente β é quanto o score muda quando aquela feature aumenta 1 unidade, mantendo as outras constantes. Com features 0/1 (tem ou não tem hook), β é diretamente "quantos pontos o hook vale".

**Regularização (Ridge).** Com poucas linhas e muitas features, a regressão comum pode exagerar coeficientes para se ajustar ao ruído (overfitting). O Ridge adiciona uma penalidade ao tamanho dos coeficientes, o que os "puxa" levemente para zero. A força da penalidade (α) é escolhida por validação cruzada.

**No projeto.** O modelo A é um Ridge sem padronização das features, para os coeficientes ficarem em pontos de score. Ele venceu o gradient boosting (R² 0,71 contra 0,62), o que sugere que o score é aditivo.

> **Resumo:** "Um modelo linear simples bastou porque o score se comporta como uma soma de pontos por boa prática. Isso deixa cada recomendação quantificável e explicável."

---

## 18. Efeito total × efeito direto (mediação)

**O que é.** Um atributo pode afetar o resultado por dois caminhos: diretamente, e **por meio** de uma variável intermediária (o *mediador*). O hook pode subir o score por si só e também por melhorar o CTR, que por sua vez sobe o score.

- **Efeito total:** a soma dos dois caminhos. É o que se mede num modelo **sem** o mediador.
- **Efeito direto:** só o primeiro caminho. É o que se mede num modelo **com** o mediador.

**No projeto.** Modelo A (sem métricas) = efeito total; modelo B (com CTR, retenção etc.) = efeito direto. Para hook, rosto, CTA e legenda, 86–95% do efeito é direto. Para texto "high", 36% da penalidade passa pela performance (pior CTR).

**Qual usar para recomendar?** O **total**: quem adiciona um hook ganha os dois caminhos.

> **Resumo:** "Comparar um modelo com e sem as métricas de performance mostra quanto do valor de cada atributo passa pelo resultado da campanha. Para recomendar, uso o efeito total."
