from arch import arch_model
from data import obter_serie_cambio
import numpy as np
import pandas as pd
from tqdm import tqdm

def ajustar_garch(retornos, dist='t'):
    am = arch_model(retornos, mean='Zero', vol='Garch', p=1, q=1, dist=dist)
    res = am.fit(disp='off')
    return res

def forecast_garch(res, horizon=5):
    forecasts = res.forecast(horizon=horizon)
    volatilidade_prevista = np.sqrt(forecasts.variance.iloc[-1, :])
    return volatilidade_prevista

def evaluate_model(retornos_series):
    am = arch_model(retornos_series, mean='Zero', vol='Garch', p=1, q=1, dist='t')
    res = am.fit(disp='off')

    aic = res.aic
    bic = res.bic

    residuals = retornos_series - res.conditional_volatility
    res_t = residuals / res.conditional_volatility
    backtest = (res_t**2).sum()

    data_length = len(retornos_series)
    train_size = int(0.8 * data_length)
    train_data = retornos_series[:train_size]
    test_data = retornos_series[train_size:]

    ultima_data_treino = train_data.index[-1]
    res_oos = am.fit(last_obs=ultima_data_treino, disp='off')
    forecast = res_oos.forecast(start=ultima_data_treino, horizon=len(test_data))

    forecast_var = forecast.residual_variance.loc[ultima_data_treino].values
    actual_var = test_data.values ** 2
    error = np.mean((actual_var - forecast_var) ** 2)

    return {"aic": aic, "bic": bic, "backtest": backtest, "forecast_error": error}


def garch_walk_forward(retornos_series, min_train=252):
    retornos = retornos_series.values
    dates = retornos_series.index

    preds_garch = []
    actuals_garch = []
    pred_dates = []

    for t in tqdm(range(min_train, len(retornos))):
        train = retornos_series.iloc[:t]

        am = arch_model(train, mean='Zero', vol='Garch', p=1, q=1, dist='t')
        res = am.fit(disp='off')

        fc = res.forecast(horizon=1)
        pred_var = fc.residual_variance.iloc[-1, 0]

        actual_var = retornos[t] ** 2

        preds_garch.append(pred_var)
        actuals_garch.append(actual_var)
        pred_dates.append(dates[t])

    preds_garch = np.array(preds_garch)
    actuals_garch = np.array(actuals_garch)

    mse = np.mean((actuals_garch - preds_garch) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actuals_garch - preds_garch))

    print("=== GARCH(1,1) — Walk-Forward ===")
    print(f"MSE:  {mse:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"MAE:  {mae:.8f}")

    return pd.DataFrame({
        "actual": actuals_garch,
        "pred_garch": preds_garch
    }, index=pred_dates)


if __name__ == "__main__":
    df_cambio = obter_serie_cambio()
    retornos = df_cambio["log_return"].dropna()

    res_normal = ajustar_garch(retornos, dist='normal')
    print(res_normal.summary())

    res_t = ajustar_garch(retornos, dist='t')
    print(res_t.summary())

    print(forecast_garch(res_t))

    metrics = evaluate_model(retornos)
    print(metrics)

    results_garch = garch_walk_forward(retornos)