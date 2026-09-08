from data import obter_serie_cambio
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PASTA_OUTPUTS = os.path.abspath(os.path.join(DIRETORIO_SCRIPT, "..", "OUTPUTS"))


def _salvar_figura(nome_arquivo, salvar=True):
    """Salva a figura atual em OUTPUTS/figures/nome_arquivo.png, se salvar=True."""
    if salvar:
        pasta_figures = os.path.join(PASTA_OUTPUTS, "figures")
        os.makedirs(pasta_figures, exist_ok=True)
        caminho = os.path.join(pasta_figures, f"{nome_arquivo}.png")
        plt.savefig(caminho, dpi=300, bbox_inches="tight")
        print(f"Salvo em: {caminho}")


# Plota gráfico da série em nível

def plot_serie(serie_cambio, salvar=True):
    plt.title("Câmbio", fontsize=25, family="Arial", fontweight="bold")
    plt.xlabel("Data", fontsize=10, family="Arial", fontweight="bold")
    plt.ylabel("Valor", fontsize=10, family="Arial", fontweight="bold")
    plt.tick_params(axis="both", colors="#242121")
    plt.plot(serie_cambio["Adj Close"], color="darkred", linewidth=1)
    _salvar_figura("serie_cambio", salvar)
    plt.show()


# Decompondo a série

def plot_decomposicao(serie_cambio, period=252, salvar=True):
    result = seasonal_decompose(
        serie_cambio["Adj Close"].dropna(), model="additive", period=period
    )
    fig = result.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    _salvar_figura("decomposicao", salvar)
    plt.show()


# Plota gráfico do Retorno

def plot_retornos(retornos, salvar=True):
    plt.title("Retorno - Câmbio", fontsize=25, family="Arial", fontweight="bold")
    plt.xlabel("Data", fontsize=10, family="Arial", fontweight="bold")
    plt.ylabel("Valor", fontsize=10, family="Arial", fontweight="bold")
    plt.tick_params(axis="both", colors="#242121")
    plt.plot(retornos, color="darkred", linewidth=1)
    _salvar_figura("retornos", salvar)
    plt.show()


# Estima densidade de Kernel

def plot_densidade(retornos, salvar=True):
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
    _salvar_figura("densidade_retornos", salvar)
    plt.show()


# Plot gráficos ACF PACF

def plot_acf_pacf(retornos, lags=20, salvar=True):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plot_acf(retornos, lags=lags, ax=axes[0, 0], title="ACF — Retornos", color="darkred")
    plot_pacf(retornos, lags=lags, ax=axes[0, 1], title="PACF — Retornos", color="darkred")
    plot_acf(retornos**2, lags=lags, ax=axes[1, 0], title="ACF — Retornos²", color="darkred")
    plot_pacf(retornos**2, lags=lags, ax=axes[1, 1], title="PACF — Retornos²", color="darkred")
    plt.tight_layout()
    _salvar_figura("acf_pacf", salvar)
    plt.show()


# Gráfico da Volatilidade de Janela Móvel (21 dias úteis)

def plot_volatilidade_movel(retornos, window=21, salvar=True):
    plt.figure(figsize=(10, 4))
    plt.plot(retornos.rolling(window=window).std(), color="darkred", linewidth=1)
    plt.title(f"Volatilidade Histórica Móvel (Janela de {window} dias) — USD/BRL")
    plt.xlabel("Data")
    plt.ylabel("Desvio Padrão Móvel")
    plt.tight_layout()
    _salvar_figura(f"volatilidade_movel_{window}d", salvar)
    plt.show()

if __name__ == "__main__":
    serie_cambio = obter_serie_cambio()
    retornos = serie_cambio["log_return"].dropna()

    plot_serie(serie_cambio)
    plot_decomposicao(serie_cambio)
    plot_retornos(retornos)
    plot_densidade(retornos)
    plot_acf_pacf(retornos)
    plot_volatilidade_movel(retornos)