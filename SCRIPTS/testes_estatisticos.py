from statsmodels.tsa.stattools import adfuller
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from SCRIPTS.data import obter_serie_cambio
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Teste de Estacionariedade ADF
    # Hipótese nula (H0): A série tem raiz unitária (é não-estacionária);
    # Hipótese alternativa (H1): A série não tem raiz unitária (é estacionária).

def teste_adf(serie):
    resultado = adfuller(serie.dropna())
    return {"estatistica": float(resultado[0]), "p_valor": float(resultado[1])}


# Teste de normalidade Jarque-Bera
    # Hipótese nula (H0): Os dados possuem distribuição normal.
    # Hipótese alternativa (H1): Os dados não possuem distribuição normal.

def teste_jarque_bera(retornos):
    stat, p = stats.jarque_bera(retornos)
    return {"estatistica": float(stat), "p_valor": float(p)}

# Teste de autocorrelação Ljung-Box dos retornos e dos retornos ao quadrado
    # Hipótese nula (H0): Não há dependência temporal entre os erros passados e os atuais.
    # Hipótese alternativa (H1): Os dados apresentam autocorrelação significativa

def teste_ljung_box(retornos, lags=[10, 20]):
    return acorr_ljungbox(retornos, lags=lags, return_df=True)


# Teste ARCH LM 
    # Hipótese nula (H0):  não há efeito ARCH (variância constante)
    # Hipótese alternativa (H1): há efeito ARCH, variância condicional heterocedásticaq

def teste_arch(retornos, nlags=10):
    stat, p, _, _ = het_arch(retornos, nlags=nlags)
    return {"estatistica": float(stat), "p_valor": float(p)}


# Teste t para verificar se a média é estatisticamente diferente de zero
    # Hipótese nula (H0): A média dos retornos é igual a zero
    # Hipótese alternativa (H1): A média dos retornos é diferente de zero

def teste_t_media_zero(retornos):
    t_stat, p_valor = stats.ttest_1samp(retornos, 0)
    return {"estatistica": float(t_stat), "p_valor": float(p_valor)}


def calcular_momentos(retornos):
    return {
        "media": retornos.mean(),
        "variancia": retornos.var(),
        "desvio_padrao": retornos.std(),
        "assimetria": retornos.skew(),
        "curtose": retornos.kurtosis()
    }
    

if __name__ == "__main__":
    serie_cambio = obter_serie_cambio()
    retornos = serie_cambio["log_return"].dropna()

    print("--- ADF (nível) ---")
    print(teste_adf(serie_cambio["Adj Close"]))

    print("\n--- ADF (retornos) ---")
    print(teste_adf(retornos))

    print("\n--- Jarque-Bera ---")
    print(teste_jarque_bera(retornos))

    print("\n--- Teste ARCH ---")
    print(teste_arch(retornos))
    
    print("\n--- Teste de Autocorrelação (Ljung-Box) ---")
    print("Ljung-Box (retornos):")
    print(teste_ljung_box(retornos))
    
    print("\nLjung-Box (retornos²):")
    print(teste_ljung_box(retornos**2))

    print("\n=== Momentos ===")
    m = calcular_momentos(retornos)
    print(f"Média:             {m['media']:.6f}")
    print(f"Variância:         {m['variancia']:.6f}")
    print(f"Desvio Padrão:     {m['desvio_padrao']:.6f}")
    print(f"Assimetria:        {m['assimetria']:.4f}")
    print(f"Excesso de Curtose: {m['curtose']:.4f}")

 
SCALE = 1e4     # só para estabilidade numérica nas regressões (log_return² é ~1e-5)
FLOOR_REL = 0.01  # piso das previsões no QLIKE = 1% da média de r² (RF/XGB podem prever <= 0)
 
 
# Alinhamento
def alinhar(resultados: dict) -> pd.DataFrame:
    #Une tudo em um DataFrame com 'actual' + uma coluna por modelo (mesma amostra).
    partes = {}
    for nome, df in resultados.items():
        col = [c for c in df.columns if c.startswith("pred_")]
        if len(col) != 1:
            raise ValueError(f"{nome}: esperava exatamente 1 coluna pred_*, achei {col}")
        partes[nome] = df[col[0]]
    preds = pd.DataFrame(partes).dropna()
 
    # 'actual' tem que ser idêntico entre modelos: se não for, há problema de escala/alinhamento
    ref = next(iter(resultados.values()))["actual"].reindex(preds.index)
    for nome, df in resultados.items():
        outro = df["actual"].reindex(preds.index)
        if not np.allclose(ref.values, outro.values, rtol=1e-6, atol=1e-12):
            raise ValueError(f"'actual' de {nome} difere dos demais (escala ou datas desalinhadas)")
 
    out = preds.copy()
    out.insert(0, "actual", ref)
    return out
 
 
# Funções de perda
def perda_mse(actual, pred):
    return (np.asarray(actual) - np.asarray(pred)) ** 2
 
 
def perda_qlike(actual, pred):
    """QLIKE = log(h) + r²/h. Robusta a ruído no proxy (Patton, 2011)."""
    actual = np.asarray(actual)
    h = np.maximum(np.asarray(pred), FLOOR_REL * actual.mean())
    return np.log(h) + actual / h
 
 
def tabela_perdas(df: pd.DataFrame) -> pd.DataFrame:
    modelos = [c for c in df.columns if c != "actual"]
    linhas = {}
    for m in modelos:
        linhas[m] = {
            "MSE": perda_mse(df["actual"], df[m]).mean(),
            "MAE": np.abs(df["actual"] - df[m]).mean(),
            "QLIKE": perda_qlike(df["actual"], df[m]).mean(),
            "% previsões <= 0": (df[m] <= 0).mean() * 100,
        }
    return pd.DataFrame(linhas).T.sort_values("QLIKE")
 
 
# Mincer-Zarnowitz
def mincer_zarnowitz(actual, pred, maxlags=5):
    """
    Regride actual = a + b * pred + e. Previsão eficiente => a=0 e b=1.
    Teste conjunto (Wald) com erros HAC (Newey-West), pois r² é heterocedástico.
    """
    y = np.asarray(actual) * SCALE
    x = np.asarray(pred) * SCALE
    res = sm.OLS(y, sm.add_constant(x, has_constant="add")).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    diff = res.params - np.array([0.0, 1.0])
    W = float(diff @ np.linalg.pinv(res.cov_params()) @ diff)   # Wald ~ chi2(2) sob H0
    p_conjunto = 1 - stats.chi2.cdf(W, df=2)
    return {
        "alpha": res.params[0],
        "beta": res.params[1],
        "p(alpha=0)": res.pvalues[0],
        "p(beta=1)": 2 * (1 - stats.norm.cdf(abs((res.params[1] - 1) / res.bse[1]))),
        "p(conjunto)": p_conjunto,
        "R2": res.rsquared,
    }
 
 
def tabela_mz(df: pd.DataFrame, maxlags=5) -> pd.DataFrame:
    modelos = [c for c in df.columns if c != "actual"]
    return pd.DataFrame(
        {m: mincer_zarnowitz(df["actual"], df[m], maxlags) for m in modelos}
    ).T
 
 
# Diebold-Mariano
def diebold_mariano(l1, l2, h=1):
    """
    H0: mesma acurácia. Estatística < 0 => modelo 1 tem perda menor (é melhor).
    Correção de Harvey, Leybourne & Newbold (1997) e t de Student.
    """
    d = np.asarray(l1) - np.asarray(l2)
    T = len(d)
    dbar = d.mean()
    var = np.var(d, ddof=0)
    for k in range(1, h):
        var += 2 * np.cov(d[k:], d[:-k], ddof=0)[0, 1]
    dm = dbar / np.sqrt(var / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p = 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))
    return stat, p
 
 
def matriz_dm(df: pd.DataFrame, perda="QLIKE", h=1):
    #Retorna (estatísticas, p-valores). Linha i vs coluna j; estat < 0 => i melhor que j.
    f = perda_qlike if perda.upper() == "QLIKE" else perda_mse
    modelos = [c for c in df.columns if c != "actual"]
    perdas = {m: f(df["actual"], df[m]) for m in modelos}
    stat = pd.DataFrame(np.nan, index=modelos, columns=modelos)
    pval = stat.copy()
    for a in modelos:
        for b in modelos:
            if a != b:
                stat.loc[a, b], pval.loc[a, b] = diebold_mariano(perdas[a], perdas[b], h)
    return stat, pval
 
 
# Model Confidence Set
def model_confidence_set(df: pd.DataFrame, perda="QLIKE", size=0.10, reps=1000, seed=42):
    #Conjunto de modelos que não se distinguem do melhor ao nível `size`.
    from arch.bootstrap import MCS
 
    f = perda_qlike if perda.upper() == "QLIKE" else perda_mse
    modelos = [c for c in df.columns if c != "actual"]
    perdas = pd.DataFrame({m: f(df["actual"], df[m]) for m in modelos}, index=df.index)
    mcs = MCS(perdas, size=size, reps=reps, block_size=10, seed=seed)  # bootstrap em blocos
    mcs.compute()
    return mcs.pvalues, mcs.included
 
 
# Orquestração
def backtest_completo(resultados: dict, maxlags=5, com_mcs=True):
    df = alinhar(resultados)
    print(f"Amostra comum: {df.index[0].date()} a {df.index[-1].date()} ({len(df)} obs.)\n")
 
    perdas = tabela_perdas(df)
    print("=== Perdas médias (menor é melhor; ordenado por QLIKE) ===")
    print(perdas.to_string(float_format=lambda x: f"{x:.6g}"), "\n")
 
    mz = tabela_mz(df, maxlags)
    print("=== Mincer-Zarnowitz (esperado: alpha=0, beta=1) ===")
    print(mz.to_string(float_format=lambda x: f"{x:.4f}"), "\n")
 
    out = {"perdas": perdas, "mz": mz}
    for nome_perda in ["QLIKE", "MSE"]:
        st, p = matriz_dm(df, nome_perda)
        print(f"=== Diebold-Mariano ({nome_perda}) — p-valores (linha vs coluna) ===")
        print(p.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"    estatística < 0 => linha melhor que coluna\n")
        out[f"dm_{nome_perda.lower()}_stat"], out[f"dm_{nome_perda.lower()}_p"] = st, p
 
    if com_mcs:
        pv, incl = model_confidence_set(df, "QLIKE")
        print("=== Model Confidence Set (QLIKE, 10%) ===")
        print(pv.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"Modelos no MCS: {incl}\n")
        out["mcs_pvalues"], out["mcs_included"] = pv, incl
 
    return out
 