from pathlib import Path
import pandas as pd
import numpy as np
import sys

BASE_DIR = Path.cwd().resolve()

# Sobe de Backtesting para SCRIPTS e depois para TCC
BASE_DIR = BASE_DIR.parent.parent

sys.path.insert(0, str(BASE_DIR))

from SCRIPTS.Backtesting.mincer_zarnowitz import teste_mincer_zarnowitz
from SCRIPTS.Backtesting.diebold_mariano import teste_diebold_mariano
from SCRIPTS.Backtesting.model_confidence_set import executar_mcs


def carregar_resultados():
    BASE_DIR = Path(__file__).resolve().parents[2]
    RESULTS_DIR = BASE_DIR / "results"

    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Não encontrei a pasta results em {BASE_DIR}"
        )
    resultados = {
        "RF": pd.read_parquet(RESULTS_DIR / "pred_rf.parquet"),
        "XGBoost": pd.read_parquet(RESULTS_DIR / "pred_xgb.parquet"),
        "GARCH": pd.read_parquet(RESULTS_DIR / "pred_garch.parquet"),
        "EGARCH": pd.read_parquet(RESULTS_DIR / "pred_egarch.parquet"),
    }

    return resultados

# Métricas de avaliação das previsões

def calcular_metricas(resultados):
    linhas = []

    for nome, df in resultados.items():
        # Valores observados e previstos
        actual = df["actual"].to_numpy()
        pred_col = [c for c in df.columns if c.startswith("pred_")]

        if len(pred_col) != 1:
            raise ValueError(
                f"{nome}: esperava uma coluna pred_, encontrei {pred_col}"
            )

        pred = df[pred_col[0]].to_numpy()

        # Erros de previsão
        erro = actual - pred

        mse = np.mean(erro ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(erro))

        # QLIKE: protege contra previsões não positivas
        piso = max(np.mean(actual) * 0.01, np.finfo(float).tiny)
        pred_qlike = np.maximum(pred, piso)

        qlike = np.mean(
            np.log(pred_qlike) + actual / pred_qlike
        )

        linhas.append({
            "Modelo": nome,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "QLIKE": qlike,
            "Previsões <= 0": int(np.sum(pred <= 0))
        })

    return (
        pd.DataFrame(linhas)
        .set_index("Modelo")
        .sort_values("RMSE")
    )




# Função para fazer os backtestings
def executar_testes_backtesting(resultados):
    # 1. Mincer-Zarnowitz
    mz = teste_mincer_zarnowitz(resultados)

    # 2. Diebold-Mariano
    dm_mse = teste_diebold_mariano(
        resultados, perda="MSE"
    )
    dm_qlike = teste_diebold_mariano(
        resultados, perda="QLIKE"
    )

    # 3. Model Confidence Set
    mcs_mse = executar_mcs(
        resultados, perda="MSE"
    )
    mcs_qlike = executar_mcs(
        resultados, perda="QLIKE"
    )

    print("\n=== MINCER-ZARNOWITZ ===")
    print(mz)

    print("\n=== DIEBOLD-MARIANO: MSE ===")
    print(dm_mse)

    print("\n=== DIEBOLD-MARIANO: QLIKE ===")
    print(dm_qlike)

    print("\n=== MCS: MSE ===")
    print(mcs_mse)

    print("\n=== MCS: QLIKE ===")
    print(mcs_qlike)

    return {
        "mincer_zarnowitz": mz,
        "dm_mse": dm_mse,
        "dm_qlike": dm_qlike,
        "mcs_mse": mcs_mse,
        "mcs_qlike": mcs_qlike,
    }


if __name__ == "__main__":
    resultados = carregar_resultados()
    # Calcula e apresenta os resultados
    metricas = calcular_metricas(resultados)

    print("=== MÉTRICAS DE DESEMPENHO ===")
    print(metricas.round(8).to_string())
    executar_testes_backtesting(resultados)