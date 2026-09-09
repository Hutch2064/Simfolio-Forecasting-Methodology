from simfolio_forecasting_methodology.origins import (
    expected_origin_tasks,
    temporal_holdouts,
)
from simfolio_forecasting_methodology.panel import (
    generate_equal_class_history_panel,
    rebalance_counts,
)


def test_canonical_panel_shape_and_rebalance_counts():
    panel = generate_equal_class_history_panel()
    assert len(panel) == 80
    assert len({ticker for p in panel for ticker in p.tickers}) == 38
    assert rebalance_counts(panel) == {
        "annually": 18,
        "monthly": 20,
        "none": 24,
        "quarterly": 18,
    }


def test_first_canonical_portfolios_are_stable():
    panel = generate_equal_class_history_panel()
    assert panel[0].name == "equal_class_history_001"
    assert panel[0].rebalance == "annually"
    assert panel[0].tickers == (
        "URTHSIM",
        "VXUSSIM",
        "IEISIM",
        "ZROZSIM",
        "GLDSIM",
        "REITSIM",
    )
    assert panel[1].tickers == (
        "VOOSIM",
        "VVSIM",
        "SHYSIM",
        "TLTSIM",
        "GSGSIM",
        "SLVSIM",
    )


def test_dense_origin_contract():
    assert expected_origin_tasks() == 4080
    holdouts = temporal_holdouts(11687)
    assert len(holdouts) == 3
    assert [h.train_fraction for h in holdouts] == [0.25, 0.50, 0.75]
