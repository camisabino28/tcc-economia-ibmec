
from pathlib import Path

import numpy as np
import pandas as pd
from arch.bootstrap import MCS


def calcular_perdas(resultados, perda="QLIKE"):
    """
    Constrói uma matriz de perdas:
    linhas = datas; colunas = modelos.
    """

    if perda not in {"MSE", "QLIKE"}:
        raise ValueError("perda deve ser 'MSE' ou 'QLIKE'.")

    if len(resultados) < 2:
        raise ValueError("O MCS exige pelo menos dois modelos.")

    matriz = {}
    indice_base = None
    actual_base = None

    for nome, df in resultados.items():
        colunas_pred = [
            c for c in df.columns if c.startswith("pred_")
        ]

        if "actual" not in df.columns:
            raise ValueError(f"{nome}: coluna 'actual' ausente.")

        if len(colunas_pred) != 1:
            raise ValueError(
                f"{nome}: esperado exatamente uma coluna pred_."
            )

        dados = df[["actual", colunas_pred[0]]].copy()
        dados = dados.replace([np.inf, -np.inf], np.nan)

        if dados.isna().any().any():
            raise ValueError(
                f"{nome}: existem valores ausentes ou infinitos."
            )

        if indice_base is None:
            indice_base = dados.index
            actual_base = dados["actual"].to_numpy()
        else:
            if not dados.index.equals(indice_base):
                raise ValueError("Os índices dos modelos diferem.")

            if not np.allclose(
                dados["actual"].to_numpy(),
                actual_base,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    "A coluna actual difere entre os modelos."
                )

        actual = dados["actual"].to_numpy(dtype=float)
        pred = dados[colunas_pred[0]].to_numpy(dtype=float)

        if np.any(actual < 0):
            raise ValueError("A variância realizada não pode ser negativa.")

        if perda == "MSE":
            perdas = (actual - pred) ** 2

        else:
            if np.any(pred < 0):
                raise ValueError(
                    f"{nome}: previsões negativas de variância."
                )

            piso = max(
                float(np.mean(actual)) * 0.01,
                np.finfo(float).tiny,
            )
            pred_segura = np.maximum(pred, piso)

            perdas = np.log(pred_segura) + actual / pred_segura

        matriz[nome] = perdas

    return pd.DataFrame(matriz, index=indice_base)


def executar_mcs(
    resultados,
    perda="QLIKE",
    alpha=0.05,
    reps=1000,
    seed=12345,
):
    """
    Executa o Model Confidence Set usando a estatística
    R (range), com bootstrap estacionário.
    """

    perdas = calcular_perdas(resultados, perda=perda)

    mcs = MCS(
        losses=perdas,
        size=alpha,
        reps=reps,
        block_size=10,
        method="R",
        bootstrap="stationary",
        seed=seed,
    )

    mcs.compute()

    tabela = mcs.pvalues.copy()
    tabela.index.name = "Modelo"
    tabela = tabela.reset_index()

    # p-values acima de alpha indicam modelos que permanecem
    # no conjunto de confiança ao nível escolhido.
    tabela["Sobrevive_MCS"] = tabela["Pvalue"] >= alpha
    tabela["Perda"] = perda

    return tabela.sort_values(
        "Pvalue", ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parents[2]
    pasta_resultados = raiz / "results"

    arquivos = {
        "RF": "pred_rf.parquet",
        "XGBoost": "pred_xgb.parquet",
        "GARCH": "pred_garch.parquet",
        "EGARCH": "pred_egarch.parquet",
    }

    resultados = {
        nome: pd.read_parquet(pasta_resultados / arquivo)
        for nome, arquivo in arquivos.items()
    }

    for perda in ["MSE", "QLIKE"]:
        print(f"\n=== MODEL CONFIDENCE SET: {perda} ===")

        tabela = executar_mcs(
            resultados,
            perda=perda,
            alpha=0.05,
            reps=1000,
            seed=12345,
        )

        print(tabela.to_string(index=False))
