from arch import arch_model
from teste_cambio import obter_serie_cambio
import numpy as np
import pandas as pd

df_cambio = obter_serie_cambio()
retornos = df_cambio["log_return"].dropna()

# Estima GARCH model
am = arch_model(retornos, mean='Zero', vol='Garch', p=1, q=1)
res = am.fit(disp='off')
print(res.summary())
# o alpha ser igual a 4,4% indica que a volatilidade reage de forma suave a notícias e choques diários imediatos.
# o beta de93,73% mostra que a volatilidade do câmbio tem muita memória. Se o mercado entrar em um período de estresse hoje, esse efeito vai demorar muitos dias para se dissipar.
# Soma alpha e beta menor que 1, o modelo é estatisticamente estável e estacionário na variância.
# Como a soma está muito próxima de 1 (0,9818), confirma-se que os choques na volatilidade do USD/BRL demoram bastante tempo para sumir 

# Teste distribuição t student
model = arch_model(retornos, mean='Zero', vol='Garch', p=1, q=1, dist='t')
res = model.fit(disp='off')
print(res.summary())
#A distribuição t-Student foi feita para lidar com léptocurtose, o parâmetro nu (9.18) provou matematicamente que a série tem caudas pesadas.
# Isso significa que o modelo sabe que o USD/BRL passa por longos períodos de calmaria, mas que choques grandes acontecem com mais frequência do que a teoria clássica prevê.
# AIC e BIC mais negativos

# Forecast volatility
forecasts = res.forecast(horizon=5)

print(forecasts.mean.iloc[-1, :])
# Printa a VOLATILIDADE condicional prevista
print(np.sqrt(forecasts.variance.iloc[-1, :]))

def evaluate_model(retornos_series):
    # Estima GARCH model
    am = arch_model(retornos_series, mean='Zero', vol='Garch', p=1, q=1, dist='t')
    res = am.fit(disp='off')

    # Calcula AIC and BIC
    aic = res.aic
    bic = res.bic

    # Performa backtesting
    residuals = retornos_series - res.conditional_volatility
    res_t = residuals / res.conditional_volatility
    backtest = (res_t**2).sum()

    # Out-of-sample testing 80/20
    data_length = len(retornos_series)
    train_size = int(0.8 * data_length)
    train_data = retornos_series[:train_size]
    test_data = retornos_series[train_size:]

    ultima_data_treino = train_data.index[-1]
    res_oos = am.fit(last_obs=train_data.index[-1], disp='off')
    forecast = res_oos.forecast(start=train_data.index[-1], horizon=len(test_data))

    # Calcula out-of-sample forecast error
    forecast_var = forecast.residual_variance.loc[ultima_data_treino].values
    actual_var = test_data.values ** 2
    error = np.mean((actual_var - forecast_var) ** 2)

    return aic, bic, backtest, error

aic, bic, backtest, forecast_error = evaluate_model(retornos)
print(f'AIC: {aic}')
print(f'BIC: {bic}')
print(f'Backtesting Result: {backtest}')
print(f'MSE do Erro de Previsão da Volatilidade: {forecast_error:.9f}')

def garch_walk_forward(retornos_series, min_train=252):
    retornos = retornos_series.values
    dates    = retornos_series.index
    
    preds_garch  = []
    actuals_garch = []
    pred_dates   = []
    
    from tqdm import tqdm
    for t in tqdm(range(min_train, len(retornos))):
        train = retornos_series.iloc[:t]
        
        am  = arch_model(train, mean='Zero', vol='Garch', p=1, q=1, dist='t')
        res = am.fit(disp='off')
        
        # Previsão 1 passo à frente
        fc = res.forecast(horizon=1)
        pred_var = fc.residual_variance.iloc[-1, 0]
        
        # Target: retorno² do próximo dia
        actual_var = retornos[t] ** 2
        
        preds_garch.append(pred_var)
        actuals_garch.append(actual_var)
        pred_dates.append(dates[t])

    preds_garch   = np.array(preds_garch)
    actuals_garch = np.array(actuals_garch)

    mse  = np.mean((actuals_garch - preds_garch) ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(actuals_garch - preds_garch))
    
    print("=== GARCH(1,1) — Walk-Forward ===")
    print(f"MSE:  {mse:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"MAE:  {mae:.8f}")

    return pd.DataFrame({
        "actual":     actuals_garch,
        "pred_garch": preds_garch
    }, index=pred_dates)

results_garch = garch_walk_forward(retornos)