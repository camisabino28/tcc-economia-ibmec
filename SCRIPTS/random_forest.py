import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from SCRIPTS.data import obter_serie_cambio
from tqdm import tqdm
from SCRIPTS.features import build_features

def rf_walk_forward(df_cambio, data_inicio_prev="2020-01-01", refit_every=1):
    df_feat = build_features(df_cambio)
    feature_cols = [c for c in df_feat.columns if c not in ["log_return", "vol_realizada", "Adj Close"]]
    X = df_feat[feature_cols].values
    y = df_feat["vol_realizada"].values
    dates = df_feat.index
    
    min_train = np.where(dates >= pd.to_datetime(data_inicio_prev))[0][0]
    print(
        f"Treinando com dados até: {dates[min_train - 1].date()} | "
        f"Iniciando previsões em: {dates[min_train].date()}"
    )

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
        if (t - min_train) % refit_every == 0:
            rf.fit(X[:t], y[:t])

        X_test = X[t].reshape(1, -1)
        y_test = y[t]

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