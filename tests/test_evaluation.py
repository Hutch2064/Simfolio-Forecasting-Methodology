import numpy as np

from simfolio_forecasting_methodology.evaluation import (
    CellAccumulator,
    empirical_crps_by_horizon,
)
from simfolio_forecasting_methodology.scoring import empirical_crps


def test_vectorized_daily_crps_matches_scalar_definition():
    samples = np.array(
        [
            [0.1, 0.2, 0.3],
            [-0.1, 0.4, 0.8],
            [0.0, 0.5, 0.6],
            [0.2, -0.1, 0.9],
        ],
        dtype=float,
    )
    realized = np.array([0.05, 0.3, 0.7], dtype=float)
    vector = empirical_crps_by_horizon(samples, realized)
    scalar = np.array(
        [empirical_crps(samples[:, horizon], realized[horizon]) for horizon in range(3)]
    )
    assert np.allclose(vector, scalar, atol=1e-15, rtol=0.0)


def test_streaming_accumulator_is_cell_first_not_origin_pooled():
    accumulator = CellAccumulator()
    accumulator.add_vector("p1", np.array([1.0, 10.0]))
    accumulator.add_vector("p1", np.array([3.0, np.nan]))
    accumulator.add_vector("p2", np.array([4.0]))

    assert accumulator.cell_means() == {
        ("p1", 1): 2.0,
        ("p1", 2): 10.0,
        ("p2", 1): 4.0,
    }
    assert accumulator.aggregate_score() == 16.0 / 3.0
    assert accumulator.cell_count == 3


def test_strict_daily_crps_rejects_nonfinite_horizon():
    samples = np.array([[0.0, np.nan], [0.1, 0.2]])
    realized = np.array([0.0, 0.1])
    try:
        empirical_crps_by_horizon(samples, realized, strict=True)
    except ValueError as exc:
        assert "finite samples" in str(exc)
    else:
        raise AssertionError("nonfinite strict CRPS inputs must fail closed")


def test_fixed_denominator_accumulator_gate_rejects_failed_task():
    accumulator = CellAccumulator()
    accumulator.add_vector("p", np.array([1.0]))
    try:
        accumulator.fixed_denominator_score(
            expected_cells=1,
            expected_tasks=1,
            completed_tasks=1,
            failed_tasks=1,
        )
    except ValueError as exc:
        assert "failed tasks" in str(exc)
    else:
        raise AssertionError("failed tasks must fail closed")
