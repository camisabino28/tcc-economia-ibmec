import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data import obter_serie_cambio
from tqdm import tqdm


def build_features(df, lags=5, vol_lags=5, ma_windows=[5, 21]):
    d = df.copy()

    for i in range(1, lags + 1):
        d[f"ret_lag{i}"] = d["log_return"].shift(i)

    d["vol_realizada"] = d["log_return"] ** 2

    for i in range(1, vol_lags + 1):
        d[f"vol_lag{i}"] = d["vol_realizada"].shift(i)

    for w in ma_windows:
        d[f"ma_abs_{w}"] = d["log_return"].abs().rolling(w).mean()

    delta = d["log_return"]
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    d["vol_hist_21"] = d["log_return"].rolling(21).std()

    return d.dropna()


def rf_walk_forward(df_cambio, min_train=252):
    df_feat = build_features(df_cambio)

    feature_cols = [c for c in df_feat.columns if c not in ["log_return", "vol_realizada", "Adj Close"]]
    X = df_feat[feature_cols].values
    y = df_feat["vol_realizada"].values
    dates = df_feat.index

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=5,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    preds = []
    actuals = []
    pred_dates = []

    for t in tqdm(range(min_train, len(X))):
        X_train = X[:t]
        y_train = y[:t]
        X_test = X[t].reshape(1, -1)
        y_test = y[t]

        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)[0]

        preds.append(pred)
        actuals.append(y_test)
        pred_dates.append(dates[t])

    preds = np.array(preds)
    actuals = np.array(actuals)

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)

    print("=== Random Forest — Walk-Forward ===")
    print(f"MSE:  {mse:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"MAE:  {mae:.8f}")
    print(f"R²:   {r2:.4f}")

    results_rf = pd.DataFrame({
        "date": pred_dates,
        "actual": actuals,
        "pred_rf": preds
    }).set_index("date")

    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    print("\n=== Top 10 features ===")
    print(importances.sort_values(ascending=False).head(10))

    return results_rf, importances


if __name__ == "__main__":
    df_cambio = obter_serie_cambio()
    results_rf, importances = rf_walk_forward(df_cambio)