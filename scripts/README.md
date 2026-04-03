# SCRIPTS

Esta pasta concentra rotinas reprodutiveis do projeto.

## Convencoes

- `coleta_*.py` ou `coleta_*.R`: download e captura de bases.
- `limpeza_*.py` ou `limpeza_*.R`: tratamento e construcao das variaveis.
- `analise_*.ipynb`: exploracao e validacao inicial.
- `modelo_*.py` ou `modelo_*.R`: estimacoes finais.

## Regra pratica

Todo script deve ler dados de `data/raw` ou `data/processed` e gravar resultados em `data/processed` ou `outputs`.
