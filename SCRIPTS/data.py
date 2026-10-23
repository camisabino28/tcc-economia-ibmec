from fredapi import Fred
import numpy as np
import pandas as pd
import os
from datetime import date
from pathlib import Path

serie_fred = "DEXBZUS"
data_inicio = "2000-01-01"
data_fim = "2026-05-31"

api_key ='a5c8b00d7ea149d2b7afe1e463a9ed51'

ROOT_DIR = Path(__file__).resolve().parent.parent

caminho_bruto = ROOT_DIR / "data/raw/cambio_usdbrl.csv"
caminho_tratado = ROOT_DIR / "data/processed/cambio_usdbrl.csv"
caminho_metadados = ROOT_DIR / "data/raw/cambio_usdbrl_metadados.txt"
caminho_ausentes = ROOT_DIR / "data/raw/datas_sem_cotacao.csv"

def baixar_serie_bruta():
    fred = Fred(api_key=api_key)
    serie = fred.get_series(
        serie_fred,
        observation_start=data_inicio,
        observation_end=data_fim,
    )

    bruta = serie.to_frame(name="Adj Close")
    bruta.index.name = "date"

    caminho_bruto.parent.mkdir(parents=True, exist_ok=True)
    bruta.to_csv(caminho_bruto) 

    ultima_obs = bruta["Adj Close"].dropna().index.max().date()
    caminho_metadados.write_text(
        f"Fonte: FRED (Federal Reserve Bank of St. Louis)\n"
        f"Serie: {serie_fred}\n"
        f"Data do download: {date.today().isoformat()}\n"
        f"Inicio solicitado: {data_inicio}\n"
        f"Fim solicitado: {data_fim}\n"
        f"Ultima observacao disponivel: {ultima_obs}\n",
        encoding="utf-8",
    )
    return bruta

def tratar_serie(bruta: pd.DataFrame) -> pd.DataFrame:
    df = bruta.copy()
    df.index = pd.to_datetime(df.index)

    # Conferências de ordem e duplicidade
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        print(f"Aviso: {n_dup} datas duplicadas; mantendo a última de cada.")
        df = df[~df.index.duplicated(keep="last")]
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Registro das ausências
    ausentes = df[df["Adj Close"].isna()].index
    print(f"{len(ausentes)} observações ausentes registradas.")
    ausentes.to_series().to_csv(caminho_ausentes, index=False, header=["date"])

    # Exclui ausentes e calcula retorno entre observações consecutivas disponíveis
    df = df.dropna(subset=["Adj Close"])
    df = df.loc[:data_fim]
    df["log_return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    df["dias_corridos"] = df.index.to_series().diff().dt.days

    print(f"Última observação disponível: {df.index.max().date()}")
    return df

def preparar_dados():
    bruta = baixar_serie_bruta()
    tratada = tratar_serie(bruta)
    caminho_tratado.parent.mkdir(parents=True, exist_ok=True)
    tratada.to_csv(caminho_tratado)
    return tratada

def obter_serie_cambio():
    if not caminho_tratado.exists():
        raise FileNotFoundError(
            f"{caminho_tratado} não encontrado."
        )
    return pd.read_csv(caminho_tratado, index_col="date", parse_dates=True)


if __name__ == "__main__":
    preparar_dados()

