from simfolio_forecasting_methodology.catalog import FRONTIER_SOURCE_ID
from simfolio_forecasting_methodology.harnesses import canonical_plan, master_plan


def test_canonical_harness_is_exact_175_with_frontier_first():
    plan = canonical_plan()
    assert plan.model_count == 175
    assert plan.model_ids[0] == FRONTIER_SOURCE_ID
    assert plan.expected_origin_tasks == 4080
    assert plan.protocol.simulations_per_origin == 240


def test_master_harness_materializes_exact_369_historical_ids():
    plan = master_plan()
    assert plan.model_count == 369
    assert len(set(plan.model_ids)) == 369
    assert "naive_iid_historical_portfolio_bootstrap" in plan.model_ids
    assert plan.expected_origin_tasks == 4080
    assert plan.protocol.simulations_per_origin == 240
