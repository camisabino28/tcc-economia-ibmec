from arch import arch_model
from SCRIPTS.data import obter_serie_cambio
import numpy as np
import pandas as pd
from tqdm import tqdm

def criar_modelo(retornos, vol_type='Garch', dist='t'):
    if vol_type.upper() == 'EGARCH':
        return arch_model(retornos, mean='Zero', vol='EGARCH', p=1, o=1, q=1, dist=dist)
    return arch_model(retornos, mean='Zero', vol='Garch', p=1, q=1, dist=dist)

def ajustar_modelo(retornos, vol_type='Garch', dist='t'):
    am = criar_modelo(retornos, vol_type, dist)
    res = am.fit(disp='off')
    return res

def metodo_forecast(vol_type, horizon):
    # EGARCH só tem forecast analítico para horizon=1
    if vol_type.upper() == 'EGARCH' and horizon > 1:
        return 'simulation'
    return 'analytic'

def forecast_garch(res, vol_type='Garch', horizon=5):
    forecasts = res.forecast(horizon=horizon, method=metodo_forecast(vol_type, horizon))
    volatilidade_prevista = np.sqrt(forecasts.variance.iloc[-1, :])
    return volatilidade_prevista

def evaluate_model(retornos_series, vol_type='Garch', dist='t'):
    res = ajustar_modelo(retornos_series, vol_type, dist)
    return {"aic": res.aic, "bic": res.bic}


def garch_walk_forward(retornos_series, vol_type='Garch', min_train=252, start_date=None):
    """retornos_series: log-retornos em % (já multiplicados por 100)."""
    retornos = retornos_series.values
    dates = retornos_series.index

    if start_date is not None:
        idx_start = np.where(dates >= pd.to_datetime(start_date))[0][0]
        min_train = max(min_train, idx_start)
        print(
            f"[{vol_type.upper()}] Treinando até:"
            f" {dates[min_train - 1].date()} | Iniciando previsões em:"
            f" {dates[min_train].date()}"
    )

    preds, actuals, pred_dates = [], [], []

    for t in tqdm(range(min_train, len(retornos))):
        train = retornos_series.iloc[:t]
        res = ajustar_modelo(train, vol_type, 't')
        fc = res.forecast(horizon=1)
        pred_var = fc.residual_variance.iloc[-1, 0]

        preds.append(pred_var / 1e4)            # volta p/ escala de log_return2
        actuals.append(retornos[t] ** 2 / 1e4)
        pred_dates.append(dates[t])

    preds = np.array(preds)
    actuals = np.array(actuals)

    mse = np.mean((actuals - preds) ** 2)
    print(f"=== {vol_type.upper()} — Walk-Forward ===")
    print(f"MSE:  {mse:.12f}")
    print(f"RMSE: {np.sqrt(mse):.6f}")
    print(f"MAE:  {np.mean(np.abs(actuals - preds)):.6f}")

    return pd.DataFrame(
        {"actual": actuals, f"pred_{vol_type.lower()}": preds},
        index=pred_dates,
    )


if __name__ == "__main__":
    df_cambio = obter_serie_cambio()
    retornos = df_cambio["log_return"].dropna() * 100

    for vol_type in ['Garch', 'EGARCH']:
        res_normal = ajustar_modelo(retornos, vol_type, dist='normal')
        print(res_normal.summary())

        res_t = ajustar_modelo(retornos, vol_type, dist='t')
        print(res_t.summary())

        print(forecast_garch(res_t, vol_type))

        metrics = evaluate_model(retornos, vol_type)
        print(metrics)

    results_garch = garch_walk_forward(retornos, 'Garch', start_date="2020-01-01")
    results_egarch = garch_walk_forward(retornos, 'EGARCH', start_date="2020-01-01")
