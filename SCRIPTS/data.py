from fredapi import Fred
import numpy as np
import pandas as pd


API_KEY ='a5c8b00d7ea149d2b7afe1e463a9ed51'

def obter_serie_cambio():
    fred = Fred(api_key= API_KEY)
    
    serie = fred.get_series(
    'DEXBZUS',
    observation_start='2000-01-01'
    )

    serie_cambio = serie.to_frame(name="Adj Close").interpolate(method='linear').dropna()
    serie_cambio["log_return"] = np.log(serie_cambio["Adj Close"] / serie_cambio["Adj Close"].shift(1))
    
    return serie_cambio
