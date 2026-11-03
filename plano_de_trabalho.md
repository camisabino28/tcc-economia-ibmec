# Plano de Trabalho do TCC CAMILA

### 1. Introdução e objetivos — Capítulo 1

- Reduzir afirmações sobre a superioridade do *machine learning*.

### 2. Revisão da literatura — Capítulo 2

Melhorar a conexão entre os estudos anteriores, a pergunta de pesquisa e os resultados encontrados. Organizar a argumentação nesta sequência:

1. Relevância da volatilidade do USD/BRL: Campani e Durães (2018) e Banco Central do Brasil (2021).
2. Fundamentação dos modelos GARCH e EGARCH: Engle (1982), Bollerslev (1986) e Nelson (1991).
3. Fundamentação do Random Forest: Breiman (2001).
4. Comparação entre modelos tradicionais e *machine learning*: Hansen e Lunde (2005).
5. Avaliação das previsões: Patton (2011) e Diebold e Mariano (1995).

### 3. Metodologia — Capítulo 3

- OK.

### 4. Base de dados — Capítulo 4

- OK.

### 5. Resultados — Capítulo 5

Os resultados estão bem organizados.

#### 5.1. Comparação dos modelos

- Na janela expansiva, o Random Forest apresentou os menores valores de MSE e QLIKE, enquanto o EGARCH apresentou o menor MAE.
- O teste Diebold–Mariano não identificou diferenças estatisticamente significativas entre os três modelos.
- Evitar afirmar que o Random Forest foi estatisticamente superior. O correto é dizer que apresentou vantagem numérica, sem superioridade estatisticamente significativa.
- Corrigir a frase sobre o EGARCH: ele apresentou desempenho ligeiramente superior ao GARCH em MSE, RMSE, MAE e QLIKE, embora as diferenças sejam pequenas.

#### 5.2. Teste de robustez

- Manter apenas GARCH e Random Forest na análise com janela rolante de cinco anos.
- Apresentar a exclusão do EGARCH como uma limitação da análise complementar, e não como evidência de inferioridade desse modelo.
- Manter os diagnósticos de convergência e as estimações problemáticas do EGARCH no apêndice.
- Destacar que, na janela expansiva, o RF apresentou vantagem numérica, mas sem significância estatística.
- Na janela rolante de cinco anos, o GARCH apresentou desempenho superior ao Random Forest, com diferenças estatisticamente significativas em MSE e QLIKE.
- Os resultados sugerem que o desempenho relativo dos modelos é sensível à estratégia de estimação.

### 6. Conclusão — Capítulo 6

Responder diretamente à pergunta de pesquisa e conectar os resultados com a literatura. Organizar a argumentação destacando:

- O Random Forest apresentou os menores valores de MSE e QLIKE na janela expansiva, mas sem diferenças estatisticamente significativas em relação ao GARCH e EGARCH.
- Na janela rolante de cinco anos, o GARCH apresentou desempenho superior ao Random Forest, com diferenças estatisticamente significativas em MSE e QLIKE.
- Os resultados mostram que modelos econométricos tradicionais continuam competitivos e que a maior flexibilidade do *machine learning* não garante melhor desempenho preditivo.
- A análise de robustez ficou restrita ao GARCH e Random Forest, devido aos problemas de estimação do EGARCH, limitando o alcance das conclusões dessa análise complementar.
- Relacionar essas conclusões principalmente com Hansen e Lunde (2005) e Campani e Durães (2018).

### 7. Tabelas, gráficos e apêndices

A organização atual está adequada.

- **Manter no corpo do trabalho:** tabela comparativa com MSE, RMSE, MAE e QLIKE; resultados principais do teste Diebold–Mariano; gráficos de comparação entre variância observada e prevista; e resultados do teste de robustez.
- **Manter nos apêndices:** estimativas completas dos parâmetros dos modelos; diagnósticos de convergência do EGARCH; importância das variáveis do Random Forest; e resultados estatísticos complementares.

### 8. Referências e apresentação final

- Conferir se todas as obras citadas estão na lista de referências.
- Finalizar título, resumo, *abstract* e formatação conforme as normas do IBMEC.
