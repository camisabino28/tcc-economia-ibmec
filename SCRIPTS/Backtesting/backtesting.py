from pathlib import Path
import pandas as pd
import numpy as np
import sys
from SCRIPTS.Backtesting.diebold_mariano import teste_diebold_mariano

BASE_DIR = Path.cwd().resolve()

# Sobe de Backtesting para SCRIPTS e depois para TCC
BASE_DIR = BASE_DIR.parent.parent

sys.path.insert(0, str(BASE_DIR))

def carregar_resultados(tipo_janela="expansiva"):
    BASE_DIR = Path(__file__).resolve().parents[2]
    RESULTS_DIR = BASE_DIR / "results"

    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Não encontrei a pasta results em {BASE_DIR}"
        )

    if tipo_janela == "expansiva":
        arquivos = {
            "RF": "pred_rf.parquet",
            "GARCH": "pred_garch.parquet",
            "EGARCH": "pred_egarch.parquet",
        }

    elif tipo_janela == "rolante":
        arquivos = {
            "RF": "pred_rf_rolling.parquet",
            "GARCH": "pred_garch_rolling.parquet",
        }

    else:
        raise ValueError("tipo_janela deve ser 'expansiva' ou 'rolante'")

    resultados = {
        nome: pd.read_parquet(RESULTS_DIR / arquivo)
        for nome, arquivo in arquivos.items()
    }

    return resultados

# Métricas de avaliação das previsões

def calcular_metricas(resultados, tipo_janela):
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
        
    df_metricas = pd.DataFrame(linhas).set_index("Modelo").sort_values("RMSE")
    
    # Diretórios
    BASE_DIR = Path(__file__).resolve().parents[2]
    OUTPUTS_DIR = BASE_DIR / "OUTPUTS" / "tables"
    RESULTS_DIR = BASE_DIR / "results"

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Salva a tabela de métricas
    caminho_metricas = OUTPUTS_DIR / f"metricas_{tipo_janela}.csv"
    df_metricas.to_csv(caminho_metricas)

    print(f"\nTabela de métricas salva em: {caminho_metricas}")
    
    # Registra e salva as datas exatas onde ocorreram previsões <= 0 (negativo ou nulo)
    casos_neg_nulo = []
    for nome, df in resultados.items():
        pred_col = [c for c in df.columns if c.startswith("pred_")][0]
        anomalias = df[df[pred_col] <= 0]
        for data_anomalia, row in anomalias.iterrows():
            casos_neg_nulo.append({
                "Modelo": nome,
                "Data": data_anomalia,
                "Valor_Previsto": row[pred_col],
            })
            
    if casos_neg_nulo:
        df_neg_nulo = pd.DataFrame(casos_neg_nulo)
        caminho_neg_nulo = RESULTS_DIR / f"datas_previsoes_neg_nulo_{tipo_janela}.csv"
        df_neg_nulo.to_csv(caminho_neg_nulo, index=False)
        print(
            f"Foram encontradas {len(df_neg_nulo)} previsões <= 0. "
            f"Detalhes salvos em: {caminho_neg_nulo}"
    )
    else:
        print(
            "\nNenhuma previsão <= 0 foi encontrada. Todas foram estritamente"
            " positivas!"
        )

    return df_metricas



# Função para fazer os backtestings
def executar_testes_backtesting(resultados, tipo_janela):
    # 1. Diebold-Mariano
    dm_mse = teste_diebold_mariano(
        resultados, perda="MSE"
    )
    dm_qlike = teste_diebold_mariano(
        resultados, perda="QLIKE"
    )
    print("\n=== DIEBOLD-MARIANO: MSE ===")
    print(dm_mse)

    print("\n=== DIEBOLD-MARIANO: QLIKE ===")
    print(dm_qlike)
    
    # Salva os resultados
    BASE_DIR = Path(__file__).resolve().parents[2]
    OUTPUTS_DIR = BASE_DIR / "OUTPUTS" / "tables"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    dm_mse.to_csv(OUTPUTS_DIR / f"diebold_mariano_mse_{tipo_janela}.csv")
    dm_qlike.to_csv(OUTPUTS_DIR / f"diebold_mariano_qlike_{tipo_janela}.csv")

    return {
        "dm_mse": dm_mse,
        "dm_qlike": dm_qlike
    }


if __name__ == "__main__":
    resultados = carregar_resultados()
    # Calcula e apresenta os resultados
    metricas = calcular_metricas(resultados)

    print("=== MÉTRICAS DE DESEMPENHO ===")
    print(metricas.round(8).to_string())
    executar_testes_backtesting(resultados)