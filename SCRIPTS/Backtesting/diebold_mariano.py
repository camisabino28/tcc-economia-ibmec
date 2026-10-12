# TESTE DIEBOLD-MARIANO (DM)
# Comparação pareada das previsões de variância

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

"""
Compara as previsões dos modelos pelo teste Diebold-Mariano.

Parâmetros
----------
resultados : dict
     Dicionário com nome do modelo e DataFrame correspondente.
    Cada DataFrame deve conter 'actual' e uma coluna 'pred_*'.

perda : str
    Função de perda: 'MSE' ou 'QLIKE'.

h : int
    Horizonte de previsão. Para este TCC, h=1.

 maxlags : int ou None
    Número de defasagens da estimativa HAC.
    Se None, utiliza uma regra automática.

Retorna
-------
pandas.DataFrame
    Resultados das comparações pareadas entre os modelos.
"""


def teste_diebold_mariano(
    resultados,
    perda="QLIKE",
    h=1,
    maxlags=None
):
    # 1. Validação dos argumentos

    perda = perda.upper()

    if perda not in {"MSE", "QLIKE"}:
        raise ValueError("A perda deve ser 'MSE' ou 'QLIKE'.")

    if h < 1:
        raise ValueError("O horizonte h deve ser pelo menos 1.")

    if len(resultados) < 2:
        raise ValueError("São necessários pelo menos dois modelos.")

    nomes = list(resultados)
    primeiro = resultados[nomes[0]]

    if "actual" not in primeiro.columns:
        raise ValueError("A coluna 'actual' não foi encontrada.")

    indice_base = primeiro.index
    actual_base = primeiro["actual"].to_numpy(dtype=float)

    # Verifica se as datas e os valores observados coincidem.
    for nome, df in resultados.items():

        colunas_pred = [
            c for c in df.columns if c.startswith("pred_")
        ]

        if "actual" not in df.columns or len(colunas_pred) != 1:
            raise ValueError(
                f"{nome}: esperado 'actual' e uma coluna 'pred_*'."
            )

        if not df.index.equals(indice_base):
            raise ValueError(
                f"As datas do modelo {nome} são diferentes."
            )

        if not np.allclose(
            actual_base,
            df["actual"].to_numpy(dtype=float),
            rtol=1e-7,
            atol=1e-12
        ):
            raise ValueError(
                f"A coluna 'actual' diverge no modelo {nome}."
            )

    n = len(actual_base)

    if n < 10:
        raise ValueError("A amostra é muito pequena para este teste.")

    # 2. Definição do piso para o QLIKE

    # O piso evita logaritmo de zero e divisão por zero.
    # Ele é aplicado somente ao cálculo do QLIKE.
    piso = max(
        np.mean(actual_base) * 0.01,
        np.finfo(float).tiny
    )

    # 3. Cálculo das perdas individuais

    perdas_modelos = {}

    for nome, df in resultados.items():

        col_pred = next(
            c for c in df.columns if c.startswith("pred_")
        )

        actual = df["actual"].to_numpy(dtype=float)
        pred = df[col_pred].to_numpy(dtype=float)

        if not np.isfinite(actual).all():
            raise ValueError(f"Valores não finitos em actual: {nome}.")

        if not np.isfinite(pred).all():
            raise ValueError(f"Previsões não finitas: {nome}.")

        if np.any(actual < 0):
            raise ValueError("A variância observada não pode ser negativa.")

        if perda == "MSE":
            perdas_modelos[nome] = (actual - pred) ** 2

        else:
            if np.any(pred < 0):
                raise ValueError(
                    f"Previsões negativas encontradas em {nome}."
                )

            pred_segura = np.maximum(pred, piso)

            perdas_modelos[nome] = (
                np.log(pred_segura) + actual / pred_segura
            )

    # 4. Configuração da estimativa HAC

    if maxlags is None:
        maxlags = min(
            int(np.floor(4 * (n / 100) ** (2 / 9))),
            n - 1
        )

    if not isinstance(maxlags, int) or not 0 <= maxlags < n:
        raise ValueError("maxlags deve ser um inteiro entre 0 e n-1.")

    # 5. Testes DM para todos os pares de modelos

    linhas = []

    for modelo_a, modelo_b in combinations(nomes, 2):

        # Diferença positiva indica maior perda média para A.
        d = perdas_modelos[modelo_a] - perdas_modelos[modelo_b]

        media_d = np.mean(d)
        d_centrada = d - media_d

        # Variância de longo prazo estimada por Newey-West.
        variancia_longo_prazo = np.dot(
            d_centrada, d_centrada
        ) / n

        for lag in range(1, maxlags + 1):

            covariancia = np.dot(
                d_centrada[lag:],
                d_centrada[:-lag]
            ) / n

            peso = 1 - lag / (maxlags + 1)

            variancia_longo_prazo += 2 * peso * covariancia

        if (
            not np.isfinite(variancia_longo_prazo)
            or variancia_longo_prazo <= 0
        ):
            raise ValueError(
                f"Variância de longo prazo inválida: "
                f"{modelo_a} vs. {modelo_b}."
            )

        erro_padrao = np.sqrt(variancia_longo_prazo / n)
        dm_original = media_d / erro_padrao

        # Correção Harvey-Leybourne-Newbold (HLN).
        fator_hln_quadrado = (
            n + 1 - 2 * h + h * (h - 1) / n
        ) / n

        if fator_hln_quadrado <= 0:
            raise ValueError(
                "A combinação de amostra e horizonte não permite "
                "aplicar a correção HLN."
            )

        dm_corrigido = dm_original * np.sqrt(fator_hln_quadrado)

        # Teste bilateral com distribuição t.
        p_valor = 2 * stats.t.sf(
            abs(dm_corrigido),
            df=n - 1
        )

        linhas.append({
            "Modelo A": modelo_a,
            "Modelo B": modelo_b,
            "Perda": perda,
            "Perda média A": np.mean(perdas_modelos[modelo_a]),
            "Perda média B": np.mean(perdas_modelos[modelo_b]),
            "Diferença média A-B": media_d,
            "DM-HLN": dm_corrigido,
            "p-valor": p_valor,
            "N": n
        })

    return pd.DataFrame(linhas)



if __name__ == "__main__":

    from pathlib import Path

    # Localiza a pasta results a partir do diretório atual
    # ou de algum diretório pai.
    BASE_DIR = Path.cwd()

    while (
        not (BASE_DIR / "results").is_dir()
        and BASE_DIR != BASE_DIR.parent
    ):
        BASE_DIR = BASE_DIR.parent

    RESULTS_DIR = BASE_DIR / "results"

    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            "Pasta 'results' não encontrada."
        )

    # Carrega as previsões dos quatro modelos.
    resultados = {
        "RF": pd.read_parquet(RESULTS_DIR / "pred_rf.parquet"),
        "XGBoost": pd.read_parquet(RESULTS_DIR / "pred_xgb.parquet"),
        "GARCH": pd.read_parquet(RESULTS_DIR / "pred_garch.parquet"),
        "EGARCH": pd.read_parquet(RESULTS_DIR / "pred_egarch.parquet"),
    }

    # Executa o teste DM com MSE.
    dm_mse = teste_diebold_mariano(
        resultados,
        perda="MSE",
        h=1
    )

    # Executa o teste DM com QLIKE.
    dm_qlike = teste_diebold_mariano(
        resultados,
        perda="QLIKE",
        h=1
    )

    # Exibe os resultados no terminal.
    print("\n=== DIEBOLD-MARIANO: MSE ===\n")
    print(dm_mse.round(8).to_string(index=False))

    print("\n=== DIEBOLD-MARIANO: QLIKE ===\n")
    print(dm_qlike.round(8).to_string(index=False))


