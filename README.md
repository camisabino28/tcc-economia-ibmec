<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a0a0a,50:3d1515,100:6b2020&height=200&section=header&text=TCC%20—%20Ciências%20Econômicas&fontSize=36&fontColor=e8f4f8&fontAlignY=38&desc=Ibmec-DF%20·%20Camila%20Sabino%20Ribeiro&descAlignY=58&descSize=16&descColor=f5c0c0" width="100%" />

<div align="center">

[![Last Commit](https://img.shields.io/github/last-commit/camisabino28/tcc-economia-ibmec?style=flat-square&color=6b2020&label=último%20commit)](https://github.com/camisabino28/tcc-economia-ibmec/commits)
[![Repo Size](https://img.shields.io/github/repo-size/camisabino28/tcc-economia-ibmec?style=flat-square&color=3d1515&label=tamanho)](https://github.com/camisabino28/tcc-economia-ibmec)
[![Top Language](https://img.shields.io/github/languages/top/camisabino28/tcc-economia-ibmec?style=flat-square&color=1a0a0a)](https://github.com/camisabino28/tcc-economia-ibmec)
[![Status](https://img.shields.io/badge/status-concluído-2e8b57?style=flat-square)](https://github.com/camisabino28/tcc-economia-ibmec)

</div>

<br>

> Repositório de pesquisa acadêmica desenvolvido no âmbito do Trabalho de Conclusão de Curso em Ciências Econômicas — Ibmec Brasília.

---

## 📌 Sobre a Pesquisa

Este trabalho investiga o desempenho preditivo de diferentes abordagens na previsão da volatilidade cambial, utilizando como objeto de estudo a taxa de câmbio USD/BRL.

A pesquisa compara duas perspectivas metodológicas: modelos econométricos tradicionais, baseados em estruturas paramétricas de volatilidade condicional (*model-based*), e métodos de aprendizado de máquina (*data-driven*).

A análise empírica compara o desempenho preditivo das diferentes abordagens fora da amostra (*out-of-sample*), considerando diferentes procedimentos de estimação e atualização dos modelos, sem pressupor a superioridade de uma classe de modelos sobre a outra.

<table>
<tr>
<td><b>Autora</b></td>
<td>Camila Sabino Ribeiro</td>
</tr>
<tr>
<td><b>Orientador</b></td>
<td>Laerte Jun Takeuti</td>
</tr>
<tr>
<td><b>Instituição</b></td>
<td>Ibmec Brasília</td>
</tr>
<tr>
<td><b>Curso</b></td>
<td>Ciências Econômicas</td>
</tr>
<tr>
<td><b>Área</b></td>
<td>Econometria Financeira · Séries Temporais · Machine Learning</td>
</tr>
<tr>
<td><b>Objeto de estudo</b></td>
<td>Volatilidade da taxa de câmbio USD/BRL</td>
</tr>
<tr>
<td><b>Período amostral</b></td>
<td>Janeiro de 2000 a maio de 2026</td>
</tr>
</table>

## 🔬 Escopo Metodológico

A estratégia empírica está estruturada na comparação entre abordagens *model-based* e *data-driven* para a previsão da volatilidade cambial.

### Modelos estimados

| Modelo            | Abordagem   | Aplicação                                                                        |
| :---------------- | :---------- | :------------------------------------------------------------------------------- |
| **GARCH(1,1)**    | Model-based | Modelagem da volatilidade condicional                                            |
| **EGARCH(1,1)**   | Model-based | Modelagem da assimetria nos efeitos dos choques sobre a volatilidade             |
| **Random Forest** | Data-driven | Aprendizado de relações não lineares entre variáveis explicativas e volatilidade |

### Estratégia de avaliação

A avaliação considera previsões fora da amostra (*out-of-sample*), utilizando diferentes procedimentos de estimação e atualização dos modelos.

As métricas de desempenho incluem:

- **MSE** — Mean Squared Error;
- **RMSE** — Root Mean Squared Error;
- **MAE** — Mean Absolute Error;
- **QLIKE** — Quasi-Likelihood Loss.

A comparação estatística entre as previsões é realizada por meio do teste de Diebold-Mariano.

Também é conduzida uma análise de robustez com janela móvel (*rolling window*) de cinco anos.

Na análise de robustez, a comparação considera GARCH(1,1) e Random Forest, uma vez que o EGARCH apresentou problemas de convergência durante as estimações com janela móvel.

### Fonte de dados

**FRED — Federal Reserve Economic Data**

- Série: `DEXBZUS`;
- Variável: taxa de câmbio entre o dólar americano e o real brasileiro;
- Frequência: diária;
- Período: janeiro de 2000 a maio de 2026.

---

## 📁 Dados da Pesquisa

Os dados utilizados na análise empírica estão organizados no diretório `DATA/`, dividido entre dados originais (`RAW`) e dados processados (`PROCESSED`).

| Arquivo | Finalidade |
| :--- | :--- |
| `cambio_usdbrl.csv` (RAW) | Base original obtida da fonte de dados |
| `cambio_usdbrl_metadados.txt` | Informações sobre a fonte, série e período |
| `datas_sem_cotacao.csv` | Registro de datas sem cotação |
| `cambio_usdbrl.csv` (PROCESSED) | Base tratada utilizada na pesquisa |

**Fonte:** Federal Reserve Economic Data (FRED), série `DEXBZUS`.

---

## 🛠️ Stack Tecnológico

<div align="center">

![Python](https://img.shields.io/badge/Python-1a0a0a?style=for-the-badge&logo=python&logoColor=ffdd54)
![Git](https://img.shields.io/badge/Git-1a0a0a?style=for-the-badge&logo=git&logoColor=F05032)
![LaTeX](https://img.shields.io/badge/LaTeX-6b2020?style=for-the-badge&logo=latex&logoColor=white)

</div>

**Python** — Tratamento de dados, construção de variáveis, estimação dos modelos e avaliação preditiva (`pandas`, `numpy`, `scikit-learn`, `arch`, entre outras bibliotecas).

**LaTeX** — Redação, organização e formatação do trabalho acadêmico.

**Git/GitHub** — Controle de versão, documentação e disponibilização dos códigos e dados da pesquisa.

---

## 🗂️ Estrutura do Repositório

```text
📦 tcc-economia-ibmec/
│
├── 📁 DATA/
│   ├── 📁 RAW/
│   │   ├── cambio_usdbrl.csv
│   │   ├── cambio_usdbrl_metadados.txt
│   │   └── datas_sem_cotacao.csv
│   │
│   └── 📁 PROCESSED/
│       └── cambio_usdbrl.csv
│
├── 📁 DOCS/
│   ├── 📁 manuscript_latex/
│   │   ├── 📁 capitulos/
│   │   ├── tcc_camila.tex
│   │   └── referencias.bib
│
├── 📁 OUTPUTS/
│   ├── 📁 figures/
│   ├── 📁 TABLES/
│
├── 📁 SCRIPTS/
│   ├── 📁 Backtesting/
│   |     ├── backtesting.py
│   |     ├── diebold_mariano.py
│   ├── main.py
│   ├── data.py
│   ├── features.py
│   ├── garch_model.py
│   ├── random_forest.py
│   ├── avaliar_resultados.py
│   ├── testes_estatisticos.py
│   └── visualizacoes.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── main.py
```

### Organização dos diretórios

| Diretório          | Descrição                                                 |
| :----------------- | :-------------------------------------------------------- |
| `DATA/RAW/`        | Dados originais obtidos na fonte e respectivos metadados  |
| `DATA/PROCESSED/`  | Dados tratados e preparados para a análise empírica       |
| `DOCS/`            | Manuscrito, documentação e referências bibliográficas     |
| `OUTPUTS/figures/` | Figuras e gráficos produzidos pela análise                |
| `OUTPUTS/TABLES/`  | Tabelas geradas para apresentação dos resultados          |
| `OUTPUTS/results/` | Resultados das estimações e métricas de desempenho        |
| `SCRIPTS/`         | Códigos de tratamento, modelagem, backtesting e avaliação |

---

## 📊 Resultados e Reprodutibilidade

Os resultados produzidos durante a pesquisa estão organizados no diretório `OUTPUTS/`.

Os arquivos incluem:

- Métricas de desempenho preditivo dos modelos;
- Previsões geradas pelas estimações;
- Resultados das especificações com janela expansiva e janela móvel;
- Figuras e tabelas utilizadas na análise empírica.

Os códigos responsáveis pelos procedimentos estão disponíveis em `SCRIPTS/`, permitindo consultar as etapas de preparação dos dados, construção das variáveis, estimação e avaliação dos modelos.

As dependências necessárias para execução dos códigos são documentadas em `requirements.txt`.

---

## 📚 Referências Bibliográficas

As referências utilizadas na pesquisa estão organizadas no arquivo `referencias.bib`, localizado em:

`DOCS/manuscript_latex/referencias.bib`

A base bibliográfica contempla trabalhos sobre modelagem da volatilidade, econometria financeira, aprendizado de máquina e avaliação de previsões.

---

## 📬 Contato

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Camila%20Sabino-1a0a0a?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/camila-sabino-r-a8919830a/)
[![Gmail](https://img.shields.io/badge/Gmail-camilasab2806%40gmail.com-3d1515?style=for-the-badge&logo=gmail&logoColor=white)](mailto:camilasab2806@gmail.com)

</div>

<br>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6b2020,50:3d1515,100:1a0a0a&height=100&section=footer" width="100%" />
