# TESTE DE MINCER-ZARNOWITZ
# Avaliação da calibração das previsões de variância


import numpy as np
import pandas as pd
import statsmodels.api as sm

"""
Executa o teste de Mincer-Zarnowitz para cada modelo.

Parâmetros
----------
resultados : dict
    Dicionário com os resultados de cada modelo.
    Cada DataFrame deve conter:
    - actual: variância realizada observada
    - pred_<modelo>: variância prevista

maxlags : int
    Número de defasagens utilizado na matriz de covariância HAC.

Retorna
-------
pandas.DataFrame
    Coeficientes, p-valores, R² e teste conjunto.
"""

def teste_mincer_zarnowitz(resultados, maxlags=5):

    linhas = []

    for nome, df in resultados.items():

        # Identifica a coluna de previsões.
        colunas_pred = [
            coluna
            for coluna in df.columns
            if coluna.startswith("pred_")
        ]

        if "actual" not in df.columns:
            raise ValueError(
                f"{nome}: coluna 'actual' não encontrada."
            )

        if len(colunas_pred) != 1:
            raise ValueError(
                f"{nome}: esperado exatamente uma coluna 'pred_*'."
            )

        # Seleciona os valores observados e previstos.
        dados = df[["actual", colunas_pred[0]]].copy()
        dados.columns = ["actual", "pred"]

        dados = dados.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        # Define a variável dependente e a explicativa.
        y = dados["actual"]
        X = sm.add_constant(dados["pred"])

        # Estima a regressão com erros-padrão HAC.
        modelo = sm.OLS(y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": maxlags}
        )

        # H0 conjunta: intercepto = 0 e inclinação = 1.
        teste = modelo.f_test(
            "const = 0, pred = 1"
        )

        linhas.append({
            "Modelo": nome,
            "Intercepto": modelo.params["const"],
            "p_intercepto": modelo.pvalues["const"],
            "Inclinacao": modelo.params["pred"],
            "p_inclinacao": modelo.pvalues["pred"],
            "R2": modelo.rsquared,
            "F_conjunto": float(np.asarray(teste.fvalue).item()),
            "p_conjunto": float(np.asarray(teste.pvalue).item()),
            "N": int(modelo.nobs)
        })

    return pd.DataFrame(linhas).set_index("Modelo")



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
            "Não foi encontrada a pasta 'results'. "
            "Execute o script a partir da pasta do projeto "
            "ou de uma de suas subpastas."
        )

    # Carrega as previsões salvas.
    resultados = {
        "RF": pd.read_parquet(RESULTS_DIR / "pred_rf.parquet"),
        "XGBoost": pd.read_parquet(RESULTS_DIR / "pred_xgb.parquet"),
        "GARCH": pd.read_parquet(RESULTS_DIR / "pred_garch.parquet"),
        "EGARCH": pd.read_parquet(RESULTS_DIR / "pred_egarch.parquet"),
    }

    # Verifica se todos os modelos usam as mesmas datas.
    indices = [df.index for df in resultados.values()]

    if not all(indices[0].equals(indice) for indice in indices[1:]):
        raise ValueError(
            "Os modelos não possuem as mesmas datas de previsão."
        )

    # Verifica se a variável observada é idêntica nos modelos.
    actual_base = resultados["RF"]["actual"]

    for nome, df in resultados.items():
        if not np.allclose(
            actual_base.to_numpy(),
            df["actual"].to_numpy(),
            rtol=1e-7,
            atol=1e-12
        ):
            raise ValueError(
                f"A coluna 'actual' diverge no modelo {nome}."
            )

    # Executa o teste.
    resultado_mz = teste_mincer_zarnowitz(
        resultados,
        maxlags=5
    )

    print("\n=== TESTE DE MINCER-ZARNOWITZ ===\n")
    print(resultado_mz.round(6).to_string())
