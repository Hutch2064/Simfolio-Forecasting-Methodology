import numpy as np

from simfolio_forecasting_methodology.scoring import (
    ScoredOrigin,
    aggregate_equal_portfolio_horizon,
    empirical_crps,
)


def _definition(samples: np.ndarray, realized: float) -> float:
    return float(
        np.mean(np.abs(samples - realized))
        - 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
    )


def test_exact_empirical_crps_matches_definition() -> None:
    samples = np.array([-0.4, -0.1, 0.0, 0.2, 0.9, 1.3], dtype=np.float64)
    assert np.isclose(empirical_crps(samples, 0.15), _definition(samples, 0.15), atol=1e-15)


def test_degenerate_empirical_distribution() -> None:
    assert empirical_crps([0.2] * 20, -0.3) == 0.5


def test_cell_first_aggregation() -> None:
    rows = [
        ScoredOrigin("a", 21, 1.0),
        ScoredOrigin("a", 21, 3.0),
        ScoredOrigin("a", 42, 10.0),
        ScoredOrigin("b", 21, 4.0),
    ]
    assert aggregate_equal_portfolio_horizon(rows) == 16.0 / 3.0
