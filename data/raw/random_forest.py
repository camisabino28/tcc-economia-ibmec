import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from teste_cambio import obter_serie_cambio
from tqdm import tqdm # Mostra o progresso do codigo 

df_cambio = obter_serie_cambio()

# Engenharia de features
def build_features(df, lags=5, vol_lags=5, ma_windows=[5, 21]):
    # df deve ter coluna log_return.
    d = df.copy()

    # Lags do retorno
    for i in range(1, lags + 1):
        d[f"ret_lag{i}"] = d["log_return"].shift(i)

    # Volatilidade realizada (target e feature)
    d["vol_realizada"] = d["log_return"] ** 2

    # Lags da volatilidade realizada
    for i in range(1, vol_lags + 1):
        d[f"vol_lag{i}"] = d["vol_realizada"].shift(i)

    # Médias móveis do retorno absoluto
    for w in ma_windows:
        d[f"ma_abs_{w}"] = d["log_return"].abs().rolling(w).mean()

    # RSI (14 dias)
    delta = d["log_return"]
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # Volatilidade histórica janela 21
    d["vol_hist_21"] = d["log_return"].rolling(21).std()

    d = d.dropna()
    return d

# Separa x e y
df_feat = build_features(df_cambio)

feature_cols = [c for c in df_feat.columns if c not in ["log_return", "vol_realizada", "Adj Close"]]
X = df_feat[feature_cols].values
y = df_feat["vol_realizada"].values
dates = df_feat.index

# WALK-FORWARD (expanding window)

min_train = 252   # mínimo de 1 ano pra iniciar

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=5,
    max_features="sqrt",
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

preds  = []
actuals = []
pred_dates = []



for t in tqdm(range(min_train, len(X))):
    X_train = X[:t]
    y_train = y[:t]
    X_test  = X[t].reshape(1, -1)
    y_test  = y[t]

    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)[0]

    preds.append(pred)
    actuals.append(y_test)
    pred_dates.append(dates[t])

preds   = np.array(preds)
actuals = np.array(actuals)

# Calcula métricas
mse  = mean_squared_error(actuals, preds)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(actuals, preds)
r2   = r2_score(actuals, preds)

print("=== Random Forest — Walk-Forward ===")
print(f"MSE:  {mse:.8f}")
print(f"RMSE: {rmse:.8f}")
print(f"MAE:  {mae:.8f}")
print(f"R²:   {r2:.4f}")

# séries de previsão
results_rf = pd.DataFrame({
    "date":    pred_dates,
    "actual":  actuals,
    "pred_rf": preds
}).set_index("date")

# Mostra as variáveis importantes
importances = pd.Series(rf.feature_importances_, index=feature_cols)
print("\n=== Top 10 features ===")
print(importances.sort_values(ascending=False).head(10))

