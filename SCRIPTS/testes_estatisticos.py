from statsmodels.tsa.stattools import adfuller
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from data import obter_serie_cambio


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


