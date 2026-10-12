from xgboost import XGBRegressor
from SCRIPTS.features import build_features
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from tqdm import tqdm
from SCRIPTS.data import obter_serie_cambio

#https://www.datacamp.com/tutorial/xgboost-in-python

def xgb_walk_forward(df_cambio, min_train=252, refit_every=1):
    df_feat = build_features(df_cambio)
    feature_cols = [c for c in df_feat.columns if c not in ["log_return", "vol_realizada", "Adj Close"]]
    X = df_feat[feature_cols].values
    y = df_feat["vol_realizada"].values
    dates = df_feat.index

    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    preds = []
    actuals = []
    pred_dates = []

    for t in tqdm(range(min_train, len(X))):
        if (t - min_train) % refit_every == 0:
            xgb.fit(X[:t], y[:t])

        X_test = X[t].reshape(1, -1)
        y_test = y[t]

        pred = xgb.predict(X_test)[0]

        preds.append(pred)
        actuals.append(y_test)
        pred_dates.append(dates[t])

    preds = np.array(preds)
    actuals = np.array(actuals)

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)

    print("=== XGBoost — Walk-Forward ===")
    print(f"MSE:  {mse:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"MAE:  {mae:.8f}")
    print(f"R²:   {r2:.4f}")

    results_xgb = pd.DataFrame({
        "date": pred_dates,
        "actual": actuals,
        "pred_xgb": preds
    }).set_index("date")

    importances = pd.Series(xgb.feature_importances_, index=feature_cols)
    print("\n=== Top 10 features ===")
    print(importances.sort_values(ascending=False).head(10))

    return results_xgb, importances


if __name__ == "__main__":
    df_cambio = obter_serie_cambio()
    results_xgb, importances = xgb_walk_forward(df_cambio)