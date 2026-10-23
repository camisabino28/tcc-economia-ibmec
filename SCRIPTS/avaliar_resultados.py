from pathlib import Path
import pandas as pd
import numpy as np
from data import obter_serie_cambio

#Localize a raiz do projeto TCC

BASE_DIR = Path.cwd()

while not (BASE_DIR / "results").is_dir() and BASE_DIR != BASE_DIR.parent:
    BASE_DIR = BASE_DIR.parent

if not (BASE_DIR / "results").is_dir():
    raise FileNotFoundError(
    "Não encontrei a pasta results. Confira o diretório do notebook."
)

RESULTS_DIR = BASE_DIR / "results"

#Carrega os três arquivos

results_rf = pd.read_parquet(RESULTS_DIR / "pred_rf.parquet")
results_garch = pd.read_parquet(RESULTS_DIR / "pred_garch.parquet")
results_egarch = pd.read_parquet(RESULTS_DIR / "pred_egarch.parquet")

resultados = {
"RF": results_rf,
"GARCH": results_garch,
"EGARCH": results_egarch,
}


# DIAGNÓSTICO

for nome, df in resultados.items():
    print(f"\n--- {nome} ---")
    print(df.head())
    print("Colunas:", df.columns.tolist())
    print("Período:", df.index.min(), "até", df.index.max())
    print("Observações:", len(df))
    print("Valores ausentes:\n", df.isna().sum())
    
# Verifica as datas
for nome, df in resultados.items():
    assert df.index.equals(results_rf.index), (
    f"As datas de {nome} não coincidem com as do RF."
    )
    
#Verifica a variável observada

for nome, df in resultados.items():
    assert np.allclose(
    results_rf["actual"].to_numpy(),
    df["actual"].to_numpy(),
    equal_nan=True
    ), f"A coluna actual difere no modelo {nome}."

    print("\nTodos os arquivos foram carregados e as verificações passaram.")
    

# Verifica se actual corresponde ao retorno ao quadrado na data prevista

serie_cambio = obter_serie_cambio()

# Retornos ao quadrado na escala original
vol_realizada = serie_cambio["log_return"].dropna() ** 2

# Alinha com as datas do backtesting
actual_esperado = vol_realizada.reindex(results_rf.index)
actual_salvo = results_rf["actual"]

print("Datas sem correspondência:", actual_esperado.isna().sum())

print(
    "Maior diferença absoluta:",
    np.max(np.abs(
        actual_salvo.to_numpy() - actual_esperado.to_numpy()
    ))
)

print(
    "Actual corresponde à série original?",
    np.allclose(
        actual_salvo.to_numpy(),
        actual_esperado.to_numpy(),
        rtol=1e-7,
        atol=1e-12
    )
)

# Verifica se tem previsões negativas ou nulas
print("\n=== INVESTIGAÇÃO DE PREVISÕES NEGATIVAS OU NULAS ===")
for nome, df in resultados.items():
  pred_col = [c for c in df.columns if c.startswith("pred_")][0]
  preds = df[pred_col]

  n_negativos = (preds < 0).sum()
  n_zeros = (preds == 0).sum()

  print(f"Modelo {nome}:")
  print(f"  - Previsões negativas: {n_negativos}")
  print(f"  - Previsões nulas: {n_zeros}")

  if n_negativos > 0:
    print(
        f" O modelo {nome} gerou {n_negativos} valores negativos!"
        " Investigar origem."
    )