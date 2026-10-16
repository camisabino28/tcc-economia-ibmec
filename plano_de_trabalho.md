# Plano de Trabalho do TCC CAMILA

## Escopo

- Dados: USD/BRL, série DEXBZUS do FRED, a partir de janeiro de 2000.
- Modelos principais: GARCH(1,1) e EGARCH.
- Comparação complementar: Random Forest (RF).
- Treinamento inicial: janeiro de 2000 a dezembro de 2019.
- Teste: janeiro de 2020 a maio de 2026.

## 1. Dados

- Baixar as cotações do FRED e salvar o arquivo original em `data/raw/cambio_usdbrl.csv`, preservando os valores ausentes.
- Registrar a fonte, a série e a data do download.
- Retirar a interpolação linear de `SCRIPTS/data.py`.
- Identificar e registrar as observações ausentes, incluindo dias sem cotação, como feriados.
- Excluir as datas sem cotação válida e calcular os retornos entre observações consecutivas disponíveis.
- Registrar que esses retornos representam intervalos entre observações disponíveis, que nem sempre correspondem a um único dia calendário.
- Conferir a ordem das datas e possíveis duplicidades antes de calcular os retornos.
- Fixar a data final da amostra em maio de 2026 e registrar a última observação disponível.
- Salvar a versão tratada em `data/processed/cambio_usdbrl.csv`, sem sobrescrever o arquivo original.
- Fazer os modelos lerem o CSV tratado, sem baixar novamente as cotações a cada execução, para repetir a análise com os mesmos dados.


## 2. Modelos e previsões

- Retirar o XGBoost.
- Manter apenas GARCH, EGARCH e RF no pipeline principal.
- Iniciar as previsões em janeiro de 2020, usando somente os dados anteriores para treinamento.

  Isso significa separar os dados usados para aprender dos dados usados para avaliar as previsões. Os dados de 2000 a 2019 servem para o treinamento inicial. A partir de janeiro de 2020, as previsões são comparadas com os valores realmente observados.

  Por exemplo, para prever a variância do primeiro dia disponível de 2020, o modelo só pode usar informações disponíveis até o fim de 2019. Depois que o valor desse dia for observado, ele pode entrar no histórico das previsões seguintes, conforme a frequência de atualização de cada modelo. Assim, a avaliação simula o uso na prática, sem conhecer antecipadamente o valor que se tenta prever.




## 3. Avaliação e saídas

- Verificar se as previsões de variância dos três modelos são estritamente positivas, como exige o QLIKE.
- Se houver previsões iguais a zero, aplicar o mesmo piso positivo aos três modelos no cálculo do QLIKE e registrar os casos. Previsões negativas devem ser investigadas e corrigidas.
- Calcular MSE e QLIKE com os mesmos critérios para GARCH, EGARCH e RF.

  1. **MSE:** mede o erro e penaliza mais os erros grandes.
  2. **QLIKE:** complementa a avaliação por ser uma função de perda específica para previsões de variância.

- Executar o teste Diebold–Mariano para avaliar se as diferenças de desempenho entre os modelos são estatisticamente significativas.
- Priorizar MSE, QLIKE e Diebold–Mariano na comparação principal.

## 4. Resultados finais

- Produzir uma tabela comparativa de GARCH, EGARCH e RF com MSE e QLIKE.
- Apresentar os resultados do teste Diebold–Mariano, com os p-valores.
- Elaborar um gráfico simples comparando o retorno ao quadrado (proxy da variância) e as previsões dos três modelos.
- Escrever um parágrafo interpretando qual modelo apresentou melhor desempenho e se as diferenças foram estatisticamente significativas, sem presumir um vencedor.


## Referências de apoio

- [Patton (2011)](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf): referência para o uso do retorno diário ao quadrado (`r_t²`) como proxy da variância e para a avaliação com MSE e QLIKE, sob as condições discutidas no artigo. A seção 4 e a Tabela 3 apresentam uma aplicação com essa proxy.
- [Montali (2026)](https://run.unl.pt/entities/publication/8aca5cab-9de3-4d88-9d7d-53de65b9a8f2): referência metodológica, adaptada aos três modelos.
- [Campani e Durães (2018)](https://www.scielo.br/j/ee/a/4BHzQbVFTqKjLgQGBGqj49z/?lang=en): referência sobre USD/BRL.
