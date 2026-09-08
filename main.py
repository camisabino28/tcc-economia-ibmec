from SCRIPTS.data import obter_serie_cambio
from SCRIPTS.testes_estatisticos import (
    teste_adf, teste_jarque_bera, teste_arch,
    teste_ljung_box, teste_t_media_zero, calcular_momentos,
)
from SCRIPTS.visualizacoes import (
    plot_serie, plot_decomposicao, plot_retornos,
    plot_densidade, plot_acf_pacf, plot_volatilidade_movel,
)
from SCRIPTS.garch_model import ajustar_garch, evaluate_model, garch_walk_forward
from SCRIPTS.random_forest import rf_walk_forward

if __name__ == "__main__":
    serie_cambio = obter_serie_cambio()
    retornos = serie_cambio["log_return"].dropna()

    print("--- ADF (nível) ---")
    print(teste_adf(serie_cambio["Adj Close"]))

    print("\n--- ADF (retornos) ---")
    print(teste_adf(retornos))

    print("\n--- Teste t (Média = 0) ---")
    print(teste_t_media_zero(retornos))

    print("\n--- Jarque-Bera ---")
    print(teste_jarque_bera(retornos))

    print("\n--- Ljung-Box (retornos) ---")
    print(teste_ljung_box(retornos))

    print("\n--- Ljung-Box (retornos²) ---")
    print(teste_ljung_box(retornos**2))

    print("\n--- Teste ARCH ---")
    print(teste_arch(retornos))

    print("\n--- Momentos ---")
    print(calcular_momentos(retornos))

    # Plots
    plot_serie(serie_cambio)
    plot_decomposicao(serie_cambio)
    plot_retornos(retornos)
    plot_densidade(retornos)
    plot_acf_pacf(retornos)
    plot_volatilidade_movel(retornos)

    # GARCH
    res_t = ajustar_garch(retornos, dist='t')
    print(res_t.summary())
    print(evaluate_model(retornos))

    results_garch = garch_walk_forward(retornos)

    # RF
    results_rf, importances = rf_walk_forward(serie_cambio)