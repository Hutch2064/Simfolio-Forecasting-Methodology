from simfolio_forecasting_methodology.protocol import (
    CANONICAL_DENSE_PROTOCOL,
    CANONICAL_PROTOCOL_FINGERPRINT,
    ORIGIN_SELECTION_HORIZONS_DAYS,
    TEMPORAL_SPLITS,
    canonical_protocol_identity,
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
    assert p.rolling_min_training_observations == 504
    assert p.rolling_horizon_rule == "daily_1_to_min_forward_and_floor_training_over_4"
    assert p.failure_policy == "fail_closed_nonfinite_or_missing_cells"


def test_dense_selection_inputs_are_pinned():
    assert len(ORIGIN_SELECTION_HORIZONS_DAYS) == 19
    assert ORIGIN_SELECTION_HORIZONS_DAYS[0] == 21
    assert ORIGIN_SELECTION_HORIZONS_DAYS[-1] == 7560
    assert TEMPORAL_SPLITS == (
        ("train_first_quarter_test_remaining", 0.25),
        ("train_first_half_test_remaining", 0.50),
        ("train_first_three_quarters_test_final_quarter", 0.75),
    )


def test_protocol_identity_resource_matches_contract():
    assert CANONICAL_DENSE_PROTOCOL.protocol_fingerprint == CANONICAL_PROTOCOL_FINGERPRINT
    identity = canonical_protocol_identity()
    assert identity["protocol_fingerprint"] == CANONICAL_PROTOCOL_FINGERPRINT
    assert identity["identity_status"] == "verified_against_retained_dense_20260823_artifact"
