from simfolio_forecasting_methodology.catalog import FRONTIER_SOURCE_ID
from simfolio_forecasting_methodology.harnesses import (
    canonical_plan,
    load_historical_master_ids,
    master_plan,
)


def test_canonical_harness_is_exact_175_with_frontier_first():
    plan = canonical_plan()
    assert plan.model_count == 175
    assert plan.model_ids[0] == FRONTIER_SOURCE_ID
    assert plan.expected_origin_tasks == 4080
    assert plan.protocol.simulations_per_origin == 240


def test_master_harness_is_canonical_superset_plus_historical_snapshot():
    canonical = canonical_plan()
    historical = load_historical_master_ids()
    master = master_plan()
    assert len(historical) == 369
    assert len(set(historical)) == 369
    assert set(canonical.model_ids).issubset(set(master.model_ids))
    assert set(historical).issubset(set(master.model_ids))
    assert len(master.model_ids) == len(set(master.model_ids))
    assert master.model_count >= 369
    assert FRONTIER_SOURCE_ID in master.model_ids
    assert "naive_iid_historical_portfolio_bootstrap" in master.model_ids
    assert master.expected_origin_tasks == 4080
    assert master.protocol.simulations_per_origin == 240
