from simfolio_forecasting_methodology.protocol import (
    CANONICAL_DENSE_PROTOCOL,
    ORIGIN_SELECTION_HORIZONS_DAYS,
    TEMPORAL_SPLITS,
)


def test_dense_protocol_invariants():
    p = CANONICAL_DENSE_PROTOCOL
    p.validate()
    assert p.total_origin_tasks == 4080
    assert p.portfolio_count == 80
    assert p.rolling_origins_per_portfolio == 48
    assert p.temporal_origins_per_portfolio == 3
    assert p.simulations_per_origin == 240
    assert p.panel_seed == 20260528
    assert p.origin_policy == "full_history_even"
    assert p.scored_cells_per_model == 701280


def test_dense_selection_inputs_are_pinned():
    assert len(ORIGIN_SELECTION_HORIZONS_DAYS) == 19
    assert ORIGIN_SELECTION_HORIZONS_DAYS[0] == 21
    assert ORIGIN_SELECTION_HORIZONS_DAYS[-1] == 7560
    assert TEMPORAL_SPLITS == (
        ("train_first_quarter_test_remaining", 0.25),
        ("train_first_half_test_remaining", 0.50),
        ("train_first_three_quarters_test_final_quarter", 0.75),
    )
