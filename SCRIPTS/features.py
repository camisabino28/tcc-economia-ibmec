import numpy as np

"""
df: DataFrame com coluna 'log_return'.
Retorna df + colunas de features, com dropna() no final.
Todas as features usam apenas informação até t-1 (sem vazamento).
"""
    
def build_features(df, lags=5, vol_lags=5, ma_windows=(5, 21)):
    d = df.copy()

    # lags do retorno
    for i in range(1, lags + 1):
        d[f"ret_lag{i}"] = d["log_return"].shift(i)

    # alvo (não é feature, mas fica aqui porque todas as outras derivam dele)
    d["vol_realizada"] = d["log_return"] ** 2

    # lags da volatilidade realizada
    for i in range(1, vol_lags + 1):
        d[f"vol_lag{i}"] = d["vol_realizada"].shift(i)

    # médias móveis do |retorno| — shift(1) para não usar o retorno do dia t
    for w in ma_windows:
        d[f"ma_abs_{w}"] = d["log_return"].abs().rolling(w).mean().shift(1)

    # RSI(14) — shift(1) pelo mesmo motivo
    delta = d["log_return"]
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi_14"] = (100 - (100 / (1 + rs))).shift(1)

    # vol histórica de 21 dias — shift(1)
    d["vol_hist_21"] = d["log_return"].rolling(21).std().shift(1)

    return d.dropna()