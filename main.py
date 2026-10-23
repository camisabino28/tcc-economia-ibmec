from SCRIPTS.data import obter_serie_cambio
from SCRIPTS.testes_estatisticos import (
    teste_adf, teste_jarque_bera, teste_arch,
    teste_ljung_box, teste_t_media_zero, calcular_momentos,
)
from SCRIPTS.visualizacoes import (
    plot_serie, plot_decomposicao, plot_retornos,
    plot_densidade, plot_acf_pacf, plot_volatilidade_movel,
)
from SCRIPTS.garch_model import ajustar_modelo, evaluate_model, garch_walk_forward
from SCRIPTS.random_forest import rf_walk_forward
import os
os.makedirs("results", exist_ok=True)
from SCRIPTS.Backtesting.backtesting import (
        carregar_resultados,
        calcular_metricas,
        executar_testes_backtesting,
    )

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
    
    # RF primeiro para a data de início dele para alinhar o GARCH e EGARCH
    results_rf, importances = rf_walk_forward(serie_cambio, refit_every=5, data_inicio_prev="2020-01-01")

    # GARCH e EGARCH, alinhados a partir da mesma data que o RF
    retornos_pct = retornos * 100
    start = results_rf.index[0]
    resultados_wf = {}

    for vol_type in ['Garch', 'EGARCH']:
        print(f"\n##### {vol_type.upper()} #####")

        res_t = ajustar_modelo(retornos_pct, vol_type, dist='t')
        print(res_t.summary())
        print(evaluate_model(retornos_pct, vol_type))

        resultados_wf[vol_type] = garch_walk_forward(retornos_pct, vol_type, start_date=start)

    results_garch = resultados_wf['Garch']
    results_egarch = resultados_wf['EGARCH']

    # Salva os resultados
    results_rf.to_parquet("results/pred_rf.parquet")
    results_garch.to_parquet("results/pred_garch.parquet")
    results_egarch.to_parquet("results/pred_egarch.parquet")
    
    # Backtesting
    resultados = carregar_resultados() 
    metricas = calcular_metricas(resultados)
    print("=== MÉTRICAS DE DESEMPENHO ===")
    print(metricas.round(8).to_string())
    executar_testes_backtesting(resultados)
