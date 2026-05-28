# Importa as bibliotecas
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch

def obter_serie_cambio():
    df = yf.download(
        "BRL=X",
        start="2006-01-01",
        interval="1d",
        auto_adjust=False
    )
    df.columns = df.columns.get_level_values(0)
    
    serie_cambio = df.loc[:, ["Adj Close"]].dropna()
    serie_cambio["log_return"] = np.log(serie_cambio["Adj Close"] / serie_cambio["Adj Close"].shift(1))
    
    return serie_cambio

if __name__ == "__main__":
    df = yf.download(
        "BRL=X",
        start="2006-01-01",
        interval="1d",
        auto_adjust=False
    )
    df.columns = df.columns.get_level_values(0)
    print("--- Verificação de Nulos ---")
    print(df.isnull().sum())
    print(df[df.isnull().any(axis=1)])

    serie_cambio = df.loc[:, ["Adj Close"]]
    serie_cambio["log_return"] = np.log(serie_cambio["Adj Close"] / serie_cambio["Adj Close"].shift(1))
    
    # Plota gráfico da série original
    plt.title("Câmbio", fontsize=25, family="Arial", fontweight="bold")
    plt.xlabel("Data", fontsize=10, family="Arial", fontweight="bold")
    plt.ylabel("Valor", fontsize=10, family="Arial", fontweight="bold")
    plt.tick_params(axis="both", colors="#242121")
    plt.plot(serie_cambio["Adj Close"], color="darkred", linewidth=1)
    plt.show()

    # Decompondo a série
    result = seasonal_decompose(serie_cambio["Adj Close"].dropna(), model="additive", period=252)
    fig = result.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.show()
    # Série claramente não é estacionária tem tendência, resíduos autocorrelacionados e 
    # heterocedasticidade condicional

    retornos = serie_cambio["log_return"].dropna()
    
    # Verificando se tem outliers nos retornos
    print("\n--- Outliers (Retornos Absolutos > 10%) ---")
    print(retornos[retornos.abs() > 0.10])

    # Plota gráfico do Retorno
    plt.title("Retorno - Câmbio", fontsize=25, family="Arial", fontweight="bold")
    plt.xlabel("Data", fontsize=10, family="Arial", fontweight="bold")
    plt.ylabel("Valor", fontsize=10, family="Arial", fontweight="bold")
    plt.tick_params(axis="both", colors="#242121")
    plt.plot(retornos, color="darkred", linewidth=1)
    plt.show()

    # Teste de Estacionariedade ADF
    # Hipótese nula (H0): A série tem raiz unitária (é não-estacionária);
    # Hipótese alternativa (H1): A série não tem raiz unitária (é estacionária).
    print("\n--- Testes de Raiz Unitária (ADF) ---")
    adf_level = adfuller(serie_cambio["Adj Close"].dropna())
    print(f"Nível — p-valor ADF: {adf_level[1]:.4f}")
    
    adf_return = adfuller(retornos)
    print(f"Retorno — p-valor ADF: {adf_return[1]:.4f}")
    # Desse modo, a série original em nível é I(1)

    # Diagnósticos Descritivos
    print("\n=== Momentos ===")
    print(f"Média:        {retornos.mean():.6f}")
    print(f"Variância:    {retornos.var():.6f}")
    print(f"Desvio Padrão:{retornos.std():.6f}")
    print(f"Assimetria:   {retornos.skew():.4f}")
    print(f"Excesso de Curtose:{retornos.kurtosis():.4f}")
    # é assimetrica e há excesso de curtose 
    # Teste t para verificar se a média é estatisticamente diferente de zero
    # Hipótese nula (H0): A média dos retornos é igual a zero
    # Hipótese alternativa (H1): A média dos retornos é diferente de zero
    print("\n--- Teste t (Média = 0) ---")
    t_stat, t_pvalue = stats.ttest_1samp(retornos, 0)
    print(f"Teste t para Média=0: stat={t_stat:.4f}, p-valor={t_pvalue:.4f}")

    # Teste de normalidade Jarque-Bera
    # Hipótese nula (H0): Os dados possuem distribuição normal.
    # Hipótese alternativa (H1): Os dados não possuem distribuição normal.
    print("\n--- Teste de Normalidade (Jarque-Bera) ---")
    jb_stat, jb_p = stats.jarque_bera(retornos)
    print(f"Jarque-Bera: stat={jb_stat:.4f}, p-valor={jb_p:.4f}")
    # Os dados não possuem distribuição normal.

    # Estima densidade de Kernel
    x = np.linspace(retornos.min(), retornos.max(), 300)
    kde = stats.gaussian_kde(retornos)
    normal = stats.norm.pdf(x, retornos.mean(), retornos.std())

    plt.figure(figsize=(10, 5))
    plt.plot(x, kde(x), label="KDE empírica", color="steelblue", linewidth=2)
    plt.plot(x, normal, label="Normal teórica", color="red", linewidth=2, linestyle="--")
    plt.hist(retornos, bins=80, density=True, alpha=0.3, color="darkred")
    plt.legend()
    plt.title("Densidade dos Retornos Log — USD/BRL")
    plt.xlabel("Retorno log diário")
    plt.ylabel("Densidade")
    plt.tight_layout()
    plt.show()
    
    # Série claramente como visto no calculo dos momentos tem léptocurtose (excesso de curtose)

    # Teste de autocorrelação Ljung-Box dos retornos e dos retornos ao quadrado
    # Hipótese nula (H0): Não há dependência temporal entre os erros passados e os atuais.
    # Hipótese alternativa (H1): Os dados apresentam autocorrelação significativa
    print("\n--- Teste de Autocorrelação (Ljung-Box) ---")
    lb_ret = acorr_ljungbox(retornos, lags=[10, 20], return_df=True)
    print("Ljung-Box (retornos):")
    print(lb_ret)
    # os retornos logarítmicos apresentam autocorrelação linear significativa

    lb_sq = acorr_ljungbox(retornos**2, lags=[10, 20], return_df=True)
    print("\nLjung-Box (retornos²):")
    print(lb_sq)
    # existe uma forte autocorrelação nos retornos elevados ao quadrado.

    # Plot gráficos ACF PACF
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plot_acf(retornos, lags=20, ax=axes[0, 0], title="ACF — Retornos", color = "darkred")
    plot_pacf(retornos, lags=20, ax=axes[0, 1], title="PACF — Retornos", color = "darkred")
    plot_acf(retornos**2, lags=20, ax=axes[1, 0], title="ACF — Retornos²", color = "darkred")
    plot_pacf(retornos**2, lags=20, ax=axes[1, 1], title="PACF — Retornos²", color = "darkred")
    plt.tight_layout()
    plt.show()

    # Teste ARCH LM 
    # Hipótese nula (H0):  não há efeito ARCH (variância constante)
    # Hipótese alternativa (H1): há efeito ARCH, variância condicional heterocedástica
    print("\n--- Teste de Efeito ARCH (Heterocedasticidade Condicional) ---")
    arch_stat, arch_p, _, _ = het_arch(retornos, nlags=10)
    print(f"Teste ARCH(10): stat={arch_stat:.4f}, p-valor={arch_p:.4f}")
    # A série possui hetecedasticidade condicional 

    # Gráfico da Volatilidade de Janela Móvel (21 dias úteis)
    plt.figure(figsize=(10, 4))
    plt.plot(retornos.rolling(window=21).std(), color="darkred", linewidth=1)
    plt.title("Volatilidade Histórica Móvel (Janela de 21 dias) — USD/BRL")
    plt.xlabel("Data")
    plt.ylabel("Desvio Padrão Móvel")
    plt.tight_layout()
    plt.show()