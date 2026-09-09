from simfolio_forecasting_methodology.seeds import deterministic_seed


def test_seed_derivation_known_vectors():
    assert deterministic_seed("forecast_final_fixed") == 2991566072
    assert (
        deterministic_seed(
            "copula_alternatives",
            "asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced",
            "2000-01-03",
            252,
            240,
        )
        == 1878859831
    )
    assert (
        deterministic_seed(
            "asset_level_naive",
            "forecast_oos_all_daily_coherent",
            "2000-01-03",
            252,
            240,
        )
        == 1688747045
    )


def test_seed_parts_are_null_delimited_and_order_sensitive():
    assert deterministic_seed("ab", "c") != deterministic_seed("a", "bc")
    assert deterministic_seed("a", "b") != deterministic_seed("b", "a")
