from SCRIPTS.data import obter_serie_cambio, preparar_dados
from SCRIPTS.testes_estatisticos import (
    teste_adf, teste_jarque_bera, teste_arch,
    teste_ljung_box, teste_t_media_zero, calcular_momentos,
)
from SCRIPTS.visualizacoes import (
    plot_serie, plot_decomposicao, plot_retornos,
    plot_densidade, plot_acf_pacf, plot_volatilidade_movel, plot_actual_vs_pred,
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
    # Caso seja a primeira vez usando o código, rodar:
    # preparar_dados()
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
    # Para rodar esses é preciso rodar o código de baixo antes
    # Janela expansiva
    plot_actual_vs_pred("garch", tipo_janela="expansiva")
    plot_actual_vs_pred("egarch", tipo_janela="expansiva")
    plot_actual_vs_pred("rf", tipo_janela="expansiva")
    # Janela rolante
    plot_actual_vs_pred("garch", tipo_janela="rolante")
    plot_actual_vs_pred("rf", tipo_janela="rolante")

    
    # Janela Expansiva
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
    
    
    # Teste de robustez: janela rolante de 5 anos

    results_rf_rolling, importances_rolling = rf_walk_forward(serie_cambio, refit_every=5, data_inicio_prev="2020-01-01", window_years=5)
    
    retornos_pct = retornos * 100
    resultados_wf_rolling = {}

    for vol_type in ['Garch', 'EGARCH']:
        print(f"\n##### {vol_type.upper()} - JANELA ROLANTE #####")
        resultados_wf_rolling[vol_type] = garch_walk_forward(retornos_pct, vol_type, start_date=results_rf_rolling.index[0], window_years=5)

    results_garch_rolling = resultados_wf_rolling['Garch']

    # Salva resultados
    results_rf_rolling.to_parquet("results/pred_rf_rolling.parquet")
    results_garch_rolling.to_parquet("results/pred_garch_rolling.parquet")
        
    # Backtesting
    # Janela expansiva
    resultados_expansiva = carregar_resultados("expansiva")
    metricas_expansiva = calcular_metricas(resultados_expansiva, "expansiva")

    print("\n=== MÉTRICAS DE DESEMPENHO — JANELA EXPANSIVA ===")
    print(metricas_expansiva.round(12).to_string())
    executar_testes_backtesting(resultados_expansiva, "expansiva")

    # Janela rolante
    resultados_rolante = carregar_resultados("rolante")
    metricas_rolante = calcular_metricas(resultados_rolante, "rolante")

    print("\n=== MÉTRICAS DE DESEMPENHO — JANELA ROLANTE ===")
    print(metricas_rolante.round(12).to_string())
    executar_testes_backtesting(resultados_rolante, "rolante")

