import pandas as pd
from SCRIPTS.data import obter_serie_cambio
from SCRIPTS.garch_model import ajustar_modelo

df_cambio = obter_serie_cambio()
retornos_pct = df_cambio["log_return"].dropna() * 100

data_falha = pd.Timestamp("2020-01-16")
data_limite = data_falha - pd.DateOffset(years=5)

train = retornos_pct.loc[
    (retornos_pct.index >= data_limite) &
    (retornos_pct.index < data_falha)
]

print(f"Observações: {len(train)}")
print(f"Período: {train.index[0].date()} até {train.index[-1].date()}")

res = ajustar_modelo(train, 'EGARCH', 't')

print("Código de convergência:", res.convergence_flag)
print("Mensagem:", res.optimization_result.message)
print("Parâmetros estimados:")
print(res.params)
print("Log-likelihood:", res.loglikelihood)

train_exp = retornos_pct.loc[retornos_pct.index < data_falha]

res_exp = ajustar_modelo(train_exp, 'EGARCH', 't')

print("\n=== JANELA EXPANSIVA ===")
print("Código:", res_exp.convergence_flag)
print(res_exp.params)

print("\n=== ESTATÍSTICAS DA JANELA ROLANTE ===")
print(train.describe())
print("Curtose:", train.kurtosis())
print("Assimetria:", train.skew())

print("\n=== ESTATÍSTICAS DA JANELA EXPANSIVA ===")
print(train_exp.describe())
print("Curtose:", train_exp.kurtosis())
print("Assimetria:", train_exp.skew())

fc = res.forecast(horizon=1)
print("\nVariância prevista:", fc.residual_variance.iloc[-1, 0])
print("Variância condicional máxima:", res.conditional_volatility.max() ** 2)

if res.convergence_flag != 0:
    n_falhas += 1
...
print(f"Reestimações sem convergência: {n_falhas} de {len(preds)}")

